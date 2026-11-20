"""WP-A 向量 + 关键词双路索引：sqlite-vec(vec0) + bm25s + jieba。

职责边界（严格）：
  * 成熟库负责的部分：
      - sqlite-vec 0.1.9  → vec0 虚表，float32 向量存储 + KNN 余弦相似度检索
      - bm25s 0.3.16      → BM25 关键词检索（中文分词由 jieba 0.42.1 提供）
      - jieba             → 中文分词；import 失败时退化为字符级（CJK 单字 + ascii 词）
  * 本模块自研的只有胶水层：索引表 DDL、写入/删除、结果整形。

【embedding 口径声明 —— 必须如实转述，不许包装成"语义模型"】
  本机没有 fastembed，项目也没有任何现成 embedding 能力
  （memory_stream.embedding 列从未被写入）。
  所以 embed_text() 是一个 **确定性的本地 hash embedding + 极小的手工情感概念词典扩展**：
    - 它不是训练出来的模型，不具备通用语义泛化能力；
    - 概念词典只让"难过/失眠"与"痛苦/睡不着"这类**人工枚举的同义组**落进同一维度，
      目的仅是让 vec0 语义路在本项目的疗愈语料上可跑通、可验收；
    - 若要真正的语义召回，把 embed_text() 换成 fastembed / 本地 sentence-transformers
      即可，返回维度固定为 EMBED_DIM，其余代码零改动。

索引落在独立的 SQLite 文件里（默认 = 业务库路径 + ".vecidx.sqlite3"），
不碰 SQLAlchemy 管理的任何 ORM 表，也不碰 store.py（WP-B 所有权）。
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import threading
import zlib
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import sqlite_vec
import bm25s

try:  # 中文分词：jieba 在 requirements.txt 里，但缺依赖时按字符级兜底
    import jieba

    jieba.setLogLevel(60)
except Exception:  # pragma: no cover - 仅在缺依赖环境触发
    jieba = None

logger = logging.getLogger(__name__)

EMBED_DIM = 64
_CONCEPT_WEIGHT = 2.0
_TOKEN_RE = re.compile(r"[一-鿿]|[a-zA-Z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

# 手工情感概念词典（非训练模型，见文件头声明）。key 顺序即 embedding 的前 N 个维度。
# 刻意不放"汤圆/豆豆"等专名：专名精确匹配交给 bm25s 关键词路，避免污染语义路。
CONCEPT_LEXICON: Dict[str, tuple[str, ...]] = {
    "sad": ("难过", "伤心", "悲伤", "难受", "低落", "痛苦", "心碎", "失落", "忧郁", "委屈", "孤独", "想念", "思念", "哭", "眼泪"),
    "sleep": ("失眠", "睡不着", "睡不著", "彻夜", "整晚", "整夜", "夜不能寐", "难以入睡", "无法入睡", "熬夜", "睡不好", "睡眠", "半夜", "惊醒"),
    "loss": ("去世", "离开", "走了", "离别", "失去", "丧失", "逝去", "死亡", "没了", "永别", "告别", "不在了"),
    "pet": ("猫", "狗", "宠物", "小狗", "小猫", "喵", "汪", "晒太阳"),
    "food": ("桂花糕", "做饭", "蛋糕", "吃饭", "食物", "面包", "咖啡"),
    "weather": ("天气", "晴天", "下雨", "刮风", "阳光", "气温"),
}
_CONCEPT_DIMS = len(CONCEPT_LEXICON)
_HASH_DIMS = EMBED_DIM - _CONCEPT_DIMS


# ---------------------------------------------------------------------------
# 分词 / embedding（确定性，非语义模型）
# ---------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """中文分词；jieba 缺失时退化为 CJK 单字 + ascii 词（字符级兜底）。"""
    raw = (text or "").strip().lower()
    if not raw:
        return []
    if jieba is not None:
        toks: Iterable[str] = jieba.lcut(raw)
    else:  # 字符级兜底
        toks = _TOKEN_RE.findall(raw)
    out: List[str] = []
    for t in toks:
        t = t.strip()
        if not t or not (_CJK_RE.search(t) or re.search(r"[a-z0-9]", t)):
            continue
        out.append(t)
    return out


def embed_text(text: str) -> List[float]:
    """确定性本地 embedding（**非语义模型**，见文件头声明）。

    前 _CONCEPT_DIMS 维：手工同义概念词典命中即置 _CONCEPT_WEIGHT；
    其余维度：token / CJK bigram 的 CRC32 特征哈希；最后 L2 归一化。
    同一输入永远得到同一向量（无随机、无网络、无模型下载）。
    """
    vec = [0.0] * EMBED_DIM
    lowered = (text or "").lower()
    for ci, terms in enumerate(CONCEPT_LEXICON.values()):
        if any(term in lowered for term in terms):
            vec[ci] += _CONCEPT_WEIGHT

    for tok in tokenize(text):
        h = zlib.crc32(tok.encode("utf-8")) % _HASH_DIMS
        vec[_CONCEPT_DIMS + h] += 1.0
        if len(tok) >= 2 and all(_CJK_RE.search(ch) for ch in tok):
            for i in range(len(tok) - 1):
                bg = tok[i : i + 2]
                hbg = zlib.crc32(("bg:" + bg).encode("utf-8")) % _HASH_DIMS
                vec[_CONCEPT_DIMS + hbg] += 0.5

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


# ---------------------------------------------------------------------------
# 索引主体
# ---------------------------------------------------------------------------
class DualIndex:
    """一个 SQLite 文件内的 vec0 向量表 + bm25s 语料元数据表。"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        with self._lock:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0("
                "memory_id TEXT PRIMARY KEY, user_id TEXT, embedding float[%d])" % EMBED_DIM
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS memory_index_meta ("
                "memory_id TEXT PRIMARY KEY, user_id TEXT, text TEXT, tokens TEXT, "
                "ts TEXT, importance REAL, embedding TEXT)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS ix_memory_index_meta_user ON memory_index_meta(user_id)"
            )
            self._conn.commit()

    # ---- 写入 / 删除 ----
    def upsert(self, memory_id: str, user_id: str, text: str, *, ts: Optional[datetime] = None,
               importance: float = 5.0, embedding: Optional[List[float]] = None) -> None:
        """幂等 upsert：先删旧行再插新行（vec0 不支持对向量列直接 UPDATE）。"""
        mid = str(memory_id)
        vec = embedding if embedding is not None else embed_text(text)
        if len(vec) != EMBED_DIM:
            raise ValueError("embedding dim %d != %d" % (len(vec), EMBED_DIM))
        toks = tokenize(text)
        blob = sqlite_vec.serialize_float32([float(x) for x in vec])
        with self._lock:
            self._conn.execute("DELETE FROM memory_vec WHERE memory_id = ?", (mid,))
            self._conn.execute(
                "INSERT INTO memory_vec(memory_id, user_id, embedding) VALUES (?,?,?)",
                (mid, str(user_id), blob),
            )
            self._conn.execute("DELETE FROM memory_index_meta WHERE memory_id = ?", (mid,))
            self._conn.execute(
                "INSERT INTO memory_index_meta(memory_id,user_id,text,tokens,ts,importance,embedding) "
                "VALUES (?,?,?,?,?,?,?)",
                (mid, str(user_id), text or "", json.dumps(toks, ensure_ascii=False),
                 ts.isoformat() if ts is not None else None, float(importance),
                 json.dumps(vec)),
            )
            self._conn.commit()

    def purge(self, memory_id: str) -> bool:
        """级联删除 vec0 行 + bm25s 元数据行，返回是否删到过任何一行。"""
        mid = str(memory_id)
        with self._lock:
            c1 = self._conn.execute("DELETE FROM memory_vec WHERE memory_id = ?", (mid,)).rowcount
            c2 = self._conn.execute("DELETE FROM memory_index_meta WHERE memory_id = ?", (mid,)).rowcount
            self._conn.commit()
        return bool(c1 or c2)

    def clear_user(self, user_id: str) -> int:
        """按用户清空索引（重建用）。"""
        with self._lock:
            mids = [r[0] for r in self._conn.execute(
                "SELECT memory_id FROM memory_index_meta WHERE user_id = ?", (str(user_id),)).fetchall()]
            for mid in mids:
                self._conn.execute("DELETE FROM memory_vec WHERE memory_id = ?", (mid,))
            self._conn.execute("DELETE FROM memory_index_meta WHERE user_id = ?", (str(user_id),))
            self._conn.commit()
        return len(mids)

    # ---- 检索 ----
    def vector_search(self, query_vec: List[float], user_id: str, k: int) -> List[dict]:
        """sqlite-vec vec0 KNN（余弦距离），按 user_id 过滤。"""
        if len(query_vec) != EMBED_DIM or k <= 0:
            return []
        blob = sqlite_vec.serialize_float32([float(x) for x in query_vec])
        with self._lock:
            rows = self._conn.execute(
                "SELECT memory_id, distance FROM memory_vec "
                "WHERE embedding MATCH ? AND user_id = ? ORDER BY distance LIMIT ?",
                (blob, str(user_id), int(k)),
            ).fetchall()
        return [
            {"id": r[0], "distance": float(r[1]), "score": 1.0 / (1.0 + float(r[1]))}
            for r in rows
        ]

    def keyword_search(self, query: str, user_id: str, k: int) -> List[dict]:
        """bm25s 关键词检索；语料每次从 memory_index_meta 重建，purge 后天然一致。"""
        qtok = tokenize(query)
        if not qtok or k <= 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT memory_id, text FROM memory_index_meta WHERE user_id = ? ORDER BY memory_id",
                (str(user_id),),
            ).fetchall()
        if not rows:
            return []
        ids = [r[0] for r in rows]
        corpus = [tokenize(r[1]) or ["<empty>"] for r in rows]
        retriever = bm25s.BM25()
        retriever.index(corpus, show_progress=False)
        res, scores = retriever.retrieve([qtok], k=min(int(k), len(ids)), show_progress=False)
        hits = [
            {"id": ids[int(i)], "score": float(sc)}
            for i, sc in zip(res[0], scores[0])
            if float(sc) > 0.0
        ]
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits

    def get_meta(self, memory_ids: Iterable[str]) -> Dict[str, dict]:
        """按 id 批量取索引元数据（点查，不是 list_memories 全量读取）。"""
        ids = [str(x) for x in memory_ids]
        if not ids:
            return {}
        ph = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._conn.execute(
                "SELECT memory_id,text,tokens,ts,importance,embedding FROM memory_index_meta "
                "WHERE memory_id IN (%s)" % ph,
                ids,
            ).fetchall()
        out: Dict[str, dict] = {}
        for r in rows:
            ts = None
            if r[3]:
                try:
                    ts = datetime.fromisoformat(r[3])
                except ValueError:
                    ts = None
            out[r[0]] = {
                "id": r[0],
                "text": r[1],
                "tokens": set(json.loads(r[2] or "[]")),
                "ts": ts,
                "importance": float(r[4]) if r[4] is not None else 5.0,
                "embedding": json.loads(r[5] or "[]"),
            }
        return out

    # ---- 可观测性（V4 用）----
    def diagnostics(self, memory_id: str) -> dict:
        mid = str(memory_id)
        with self._lock:
            vec_n = self._conn.execute(
                "SELECT count(*) FROM memory_vec WHERE memory_id = ?", (mid,)).fetchone()[0]
            bm25_n = self._conn.execute(
                "SELECT count(*) FROM memory_index_meta WHERE memory_id = ?", (mid,)).fetchone()[0]
            total_vec = self._conn.execute("SELECT count(*) FROM memory_vec").fetchone()[0]
        return {"vec_rows": int(vec_n), "bm25_rows": int(bm25_n), "vec_total": int(total_vec)}

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ---------------------------------------------------------------------------
# 连接工厂 / 路径解析
# ---------------------------------------------------------------------------
_indexes: Dict[str, DualIndex] = {}
_index_lock = threading.Lock()


def resolve_db_path(db: Any = None) -> str:
    """索引文件路径：环境变量优先，其次由业务库路径派生。"""
    env = os.getenv("DUAL_RECALL_INDEX_DB", "").strip()
    if env:
        return os.path.abspath(env)
    if db is not None:
        try:
            database = db.get_bind().url.database
            if database and database != ":memory:":
                return os.path.abspath(database) + ".vecidx.sqlite3"
        except Exception:
            pass
    return os.path.abspath(os.path.join("work", "dual_recall_index.sqlite3"))


def get_index(db_path: str) -> DualIndex:
    path = os.path.abspath(db_path)
    with _index_lock:
        idx = _indexes.get(path)
        if idx is None:
            idx = DualIndex(path)
            _indexes[path] = idx
        return idx


def get_index_for(db: Any = None) -> DualIndex:
    return get_index(resolve_db_path(db))


def reset_index(db_path: Optional[str] = None) -> None:
    """关闭并清空连接缓存；测试切库用。不删文件（调用方可自行删除）。"""
    with _index_lock:
        targets = [db_path] if db_path else list(_indexes)
        for path in targets:
            idx = _indexes.pop(os.path.abspath(path), None) if path else None
            if idx is not None:
                try:
                    idx.close()
                except Exception:
                    pass
        if db_path is None:
            _indexes.clear()
