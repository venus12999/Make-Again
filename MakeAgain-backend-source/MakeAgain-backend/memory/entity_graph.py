"""实体双向链接图（WP-C，记忆内核深度替换第二批）。

资产来源（gh api 实读 mem0 原文，见 work/r4_mem0_main_raw.txt）：
- mem0/memory/main.py:582-584  ``_normalize_entity_text``
- mem0/memory/main.py:586-603  ``_existing_entities_by_text``（先精确去重）
- mem0/memory/main.py:605-650  ``_upsert_entity``（精确 or 余弦相似度 >= 0.95 命中后追加 linked_memory_ids）
- mem0/memory/main.py:652-705  ``_remove_memory_from_entity_store``（删除记忆时清反链 / 空则删实体）
- mem0/memory/main.py:707-728  ``_link_entities_for_memory``（单条记忆实体链接）

替换点：mem0 的 vector_store 换成已装成熟库 **sqlite-vec v0.1.9 的 vec0 虚表**，
相似度匹配、KNN、级联删除全部由 vec0 完成；本模块只保留 mem0 的三件套业务规则
（双向 linked_memory_ids、0.95 阈值、删除清反链）作为胶水。

设计红线（与 store.py 的 USE_PROVENANCE 同一口径）：
- 开关 ``USE_ENTITY_PORTRAIT`` 默认关闭；关闭时本模块不参与任何写入路径。
- 任何实体写失败都 swallow（debug 级），绝不破坏主记忆写入/删除。
- ``link_facts()`` 是给 lead 接到 extract 路径的 R2 遗留接口：extract_turn 产出的
  facts 里 linked_memory_ids 恒为空，调用本接口后按实体图回填真实关联 id。
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import threading
import uuid

import sqlite_vec

logger = logging.getLogger(__name__)

# ---- 开关（默认关闭） ----
_ENABLED_VALUES = {"1", "true", "yes", "on"}


def enabled() -> bool:
    """USE_ENTITY_PORTRAIT 开关：os.getenv，默认关闭。"""
    return (os.getenv("USE_ENTITY_PORTRAIT") or "").strip().lower() in _ENABLED_VALUES


# ---- sqlite-vec vec0 连接（按 db_path 缓存，:memory: 进程内共享） ----
_DB_PATH_ENV = "ENTITY_GRAPH_DB_PATH"
_EMBED_DIM = 32             # 默认（本地 hash）维度
_active_dim = _EMBED_DIM    # 当前 vec0 表维度；set_embedder 按 embedder 维度重建
_SIM_THRESHOLD = 0.95  # mem0 main.py:621/1166 原值


def _create_sql(dim: int) -> str:
    """按 embedder 维度生成 vec0 DDL（真语义 512 维时重建表，避免维度不匹配报错）。"""
    return f"""
CREATE VIRTUAL TABLE IF NOT EXISTS entities USING vec0(
    entity_id TEXT PRIMARY KEY,
    embedding float[{dim}] distance_metric=cosine,
    data TEXT,
    data_norm TEXT,
    entity_type TEXT,
    linked_memory_ids TEXT,
    user_id TEXT
)
"""

_connections: dict[str, sqlite3.Connection] = {}
_conn_lock = threading.RLock()
_embedder = None  # 可注入真实 embedding 模型；默认本地确定性哈希向量（零外部依赖）


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """取（必要时新建）带 vec0 扩展的 SQLite 连接。"""
    path = db_path if db_path is not None else (os.getenv(_DB_PATH_ENV) or ":memory:")
    with _conn_lock:
        conn = _connections.get(path)
        if conn is None:
            conn = sqlite3.connect(path, check_same_thread=False, timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            conn.execute(_create_sql(_active_dim))
            conn.commit()
            _connections[path] = conn
        return conn


def _infer_dim(fn) -> int | None:
    """推断 embedder 维度：优先 .dim 属性（fastembed 单例带此属性），否则探针一次。"""
    if fn is None:
        return _EMBED_DIM
    dim = getattr(fn, "dim", None)
    if isinstance(dim, int) and dim > 0:
        return dim
    try:
        n = len(list(fn("__dim_probe__")))
        return n if n > 0 else None
    except Exception:
        return None


def _rebuild_for_dim(dim: int) -> None:
    """维度变化时重建 vec0 表（切换 embedding 模型=配置变更，旧实体数据丢弃）。"""
    global _active_dim
    _active_dim = dim
    with _conn_lock:
        for path, conn in list(_connections.items()):
            try:
                conn.execute("DROP TABLE IF EXISTS entities")
                conn.execute(_create_sql(dim))
                conn.commit()
            except Exception as exc:  # noqa: BLE001 - 单连接失败不影响其它
                logger.warning("entity table rebuild failed for %s: %s", path, exc)


def set_embedder(fn) -> None:
    """注入 (text: str) -> list[float] 的 embedding 函数；传 None 恢复默认 hash。

    与 vector_index.set_embedder 联动：复用同一函数对象（fastembed 只加载一次）。
    维度变化时自动重建 vec0 表，避免 32/512 维混用导致 KNN 查询报错。
    """
    global _embedder
    dim = _infer_dim(fn)
    _embedder = fn
    if dim is not None and dim != _active_dim:
        _rebuild_for_dim(dim)


def enable_semantic_embedding() -> bool:
    """复用 vector_index 的共享 fastembed 单例（不再加载第二个模型）。"""
    from memory import vector_index
    ok = vector_index.enable_semantic_embedding()
    fn = vector_index.get_semantic_embedder()
    if fn is not None:
        set_embedder(fn)
    return bool(ok and fn is not None)


def current_dim() -> int:
    return _active_dim


def _local_embed(text: str) -> list[float]:
    """本地确定性字符 unigram+bigram 哈希向量（L2 归一化），零模型零网络。

    只用于「同一实体的文本完全一致 / 极接近」场景；精确匹配路径（data_norm）先于
    相似度路径，所以「豆豆」两轮对话必命中。需要语义泛化时由 set_embedder 注入。
    """
    vec = [0.0] * _EMBED_DIM
    s = _normalize_entity_text(text)
    tokens = list(s) + [s[i:i + 2] for i in range(len(s) - 1)]
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest()[:8], 16)
        vec[h % _EMBED_DIM] += 1.0
        vec[(h >> 9) % _EMBED_DIM] += 1.0 if (h >> 31) & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _embed(text: str) -> list[float]:
    fn = _embedder or _local_embed
    vec = [float(x) for x in fn(text)]
    if not vec:
        raise ValueError("embedder returned empty vector")
    # 维度由 set_embedder 的 _rebuild_for_dim 保证与 vec0 表一致（32 hash / 512 语义）
    return vec


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


# ---- mem0 main.py:582-584，原样 ----
def _normalize_entity_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _load_linked(raw) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [str(x) for x in parsed] if isinstance(parsed, list) else []


def _dump_linked(linked: list[str]) -> str:
    return json.dumps(linked, ensure_ascii=False)


# ---- mem0 main.py:605-650 ``_upsert_entity`` 的 vec0 版 ----
def upsert_entity(entity_text: str, entity_type: str, memory_id: str,
                  user_id: str = "", conn: sqlite3.Connection | None = None) -> dict | None:
    """实体 upsert：精确文本命中优先，否则 vec0 余弦相似度 >= 0.95 命中；
    命中则双向追加 memory_id，未命中则新建实体行。失败返回 None（不抛）。"""
    if not isinstance(entity_text, str) or not entity_text.strip() or not memory_id:
        return None
    try:
        conn = conn or get_connection()
        norm = _normalize_entity_text(entity_text)
        embedding = _embed(entity_text)

        # mem0 L610：先按规范化文本做精确去重（vec0 元数据 WHERE 过滤，实读可支持）
        exact = conn.execute(
            "SELECT entity_id, data, linked_memory_ids FROM entities "
            "WHERE data_norm = ? AND user_id = ?",
            (norm, user_id),
        ).fetchone()
        match = exact

        # mem0 L613-622：精确未命中 → vec0 KNN，相似度 = 1 - cosine_distance >= 0.95
        if match is None:
            rows = conn.execute(
                "SELECT entity_id, data, linked_memory_ids, distance FROM entities "
                "WHERE embedding MATCH ? AND k = 5 AND user_id = ? ORDER BY distance",
                (_vec_literal(embedding), user_id),
            ).fetchall()
            for row in rows:
                if (1.0 - float(row["distance"])) >= _SIM_THRESHOLD:
                    match = row
                    break

        if match is not None:
            linked = _load_linked(match["linked_memory_ids"])
            if memory_id not in linked:
                linked.append(memory_id)
                conn.execute(
                    "UPDATE entities SET linked_memory_ids = ? WHERE entity_id = ?",
                    (_dump_linked(linked), match["entity_id"]),
                )
                conn.commit()
            return {
                "entity_id": match["entity_id"],
                "data": match["data"],
                "linked_memory_ids": linked,
                "created": False,
            }

        entity_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO entities(entity_id, embedding, data, data_norm, entity_type, "
            "linked_memory_ids, user_id) VALUES (?,?,?,?,?,?,?)",
            (entity_id, _vec_literal(embedding), entity_text, norm,
             entity_type or "entity", _dump_linked([memory_id]), user_id),
        )
        conn.commit()
        return {
            "entity_id": entity_id,
            "data": entity_text,
            "linked_memory_ids": [memory_id],
            "created": True,
        }
    except Exception as exc:  # noqa: BLE001 - 与 mem0 一致：实体写失败不破坏主路径
        logger.warning("entity upsert failed for %r: %s", entity_text, exc)
        return None


# ---- mem0 main.py:652-705 ``_remove_memory_from_entity_store`` 的 vec0 版 ----
def remove_memory(memory_id: str, user_id: str | None = None,
                  conn: sqlite3.Connection | None = None) -> int:
    """删除记忆时清所有实体的反链；反链空则删实体行。返回被清理的实体数。"""
    if not memory_id:
        return 0
    try:
        conn = conn or get_connection()
        if user_id is None:
            rows = conn.execute(
                "SELECT entity_id, linked_memory_ids FROM entities"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT entity_id, linked_memory_ids FROM entities WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        cleaned = 0
        for row in rows:
            linked = _load_linked(row["linked_memory_ids"])
            if memory_id not in linked:
                continue
            remaining = [mid for mid in linked if mid != memory_id]
            try:
                if not remaining:
                    conn.execute("DELETE FROM entities WHERE entity_id = ?", (row["entity_id"],))
                else:
                    conn.execute(
                        "UPDATE entities SET linked_memory_ids = ? WHERE entity_id = ?",
                        (_dump_linked(remaining), row["entity_id"]),
                    )
                cleaned += 1
            except Exception as exc:  # noqa: BLE001 - 单实体失败不影响整体
                logger.debug("entity cleanup failed id=%s: %s", row["entity_id"], exc)
        if cleaned:
            conn.commit()
        return cleaned
    except Exception as exc:  # noqa: BLE001 - outer swallow，删除主路径永不被破坏
        logger.warning("entity cleanup failed for memory_id=%s: %s", memory_id, exc)
        return 0


# ---- 查询辅助（验收/调试用） ----
def get_entity(entity_text: str, user_id: str = "",
               conn: sqlite3.Connection | None = None) -> dict | None:
    conn = conn or get_connection()
    row = conn.execute(
        "SELECT entity_id, data, data_norm, entity_type, linked_memory_ids, user_id "
        "FROM entities WHERE data_norm = ? AND user_id = ?",
        (_normalize_entity_text(entity_text), user_id),
    ).fetchone()
    if row is None:
        return None
    return {
        "entity_id": row["entity_id"],
        "data": row["data"],
        "entity_type": row["entity_type"],
        "linked_memory_ids": _load_linked(row["linked_memory_ids"]),
        "user_id": row["user_id"],
    }


def list_entities(user_id: str | None = None,
                  conn: sqlite3.Connection | None = None) -> list[dict]:
    conn = conn or get_connection()
    if user_id is None:
        rows = conn.execute(
            "SELECT entity_id, data, entity_type, linked_memory_ids, user_id FROM entities"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT entity_id, data, entity_type, linked_memory_ids, user_id "
            "FROM entities WHERE user_id = ?", (user_id,)
        ).fetchall()
    return [{
        "entity_id": r["entity_id"],
        "data": r["data"],
        "entity_type": r["entity_type"],
        "linked_memory_ids": _load_linked(r["linked_memory_ids"]),
        "user_id": r["user_id"],
    } for r in rows]


def reset(conn: sqlite3.Connection | None = None) -> None:
    """清空实体图（仅测试用）。"""
    conn = conn or get_connection()
    conn.execute("DELETE FROM entities")
    conn.commit()


# ---- 默认实体抽取器（可注入替代） ----
_QUOTED_RE = re.compile(r"[「\"'《]([^」\"'》]{1,24})[」\"'》]")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,8}")


def _default_extract_entities(text: str) -> list[tuple[str, str]]:
    """零依赖确定性抽取：引号内实体 + 同一句内重复的 2-4 字中文词。
    需要真 NER 时由 link_facts(entity_extractor=...) 注入。"""
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for m in _QUOTED_RE.finditer(text or ""):
        name = m.group(1).strip()
        if name:
            key = _normalize_entity_text(name)
            if key and key not in seen:
                seen.add(key)
                found.append(("name", name))
    run = _CJK_RUN_RE.findall(text or "")
    counts: dict[str, int] = {}
    for piece in run:
        for n in (2, 3, 4):
            for i in range(len(piece) - n + 1):
                gram = piece[i:i + n]
                counts[gram] = counts.get(gram, 0) + 1
    for gram, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if cnt >= 2 and gram not in seen:
            seen.add(gram)
            found.append(("name", gram))
    return found


# ---- R2 遗留接口：给 extract 路径回填 linked_memory_ids ----
def link_facts(facts: list[dict], *, memory_id: str, user_id: str = "",
               entities: list[tuple[str, str]] | None = None,
               entity_extractor=None,
               conn: sqlite3.Connection | None = None) -> list[dict]:
    """把 extract_turn 产出的 facts 接进实体图，并回填每条 fact 的 linked_memory_ids。

    - entities：显式实体列表 (entity_type, entity_text)，对所有 fact 生效；
    - entity_extractor：callable(text) -> [(type, text)]，用于替换默认抽取器；
    - 返回新的 fact dict 列表（不改入参），每条 linked_memory_ids 非空当且仅当
      实体图里存在与该 fact 共享实体、且 memory_id 不同的其它记忆。
    失败一律 swallow：原样返回 facts（linked_memory_ids 保持原值）。
    """
    try:
        conn = conn or get_connection()
        extractor = entity_extractor or _default_extract_entities
        out: list[dict] = []
        for fact in facts:
            if not isinstance(fact, dict):
                out.append(fact)
                continue
            text = str(fact.get("fact") or fact.get("text") or "")
            pairs = list(entities) if entities is not None else extractor(text)
            related: set[str] = {str(x) for x in (fact.get("linked_memory_ids") or [])}
            for entity_type, entity_text in pairs:
                res = upsert_entity(entity_text, entity_type, memory_id, user_id, conn=conn)
                if res:
                    related.update(mid for mid in res["linked_memory_ids"] if mid != memory_id)
            new_fact = dict(fact)
            new_fact["linked_memory_ids"] = sorted(related)
            out.append(new_fact)
        return out
    except Exception as exc:  # noqa: BLE001 - R2 接口失败不破坏抽取结果
        logger.warning("link_facts failed: %s", exc)
        return list(facts)
