"""WP-A 双路召回行为验收 V1-V6（不是"测试全绿"就算过）。

V1 开关开启后 list_memories 调用次数 = 0（100 条记忆，mock 计数）
V2 vec0 语义路：查询"失眠难过"召回无字面重叠的"彻夜睡不着…痛苦"
V3 bm25s 关键词路：专名"豆豆"排在语义相近但无关项之前
V4 purge 级联：vec0 行 + bm25s 元数据行同步消失
V5 开关关闭：与基线 cde75f8 逐字节一致（内嵌 oracle = git show 原文）
V6 冲突用例："养猫 → 猫去世" 新事实仍排前

注意：V2 的"语义"来自确定性 hash + 手工情感概念词典（**非训练语义模型**），
在 V2 断言里显式核对 embedding_kind，报告须如实说明这一边界。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from memory import recall as recall_mod
from memory import store
from memory.vector_index import get_index_for, reset_index


# 基线 oracle：git show cde75f8:MakeAgain-backend/memory/recall.py 中
# _recall_legacy 的函数体原文（= 开关全关时 cde75f8 的实际行为）。
_BASELINE_SOURCE = '''"""召回流程:recency + importance + relevance 混合打分。"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from config import get_settings
from memory.facts import facts_text
from memory.store import list_memories

_TOKEN_RE = re.compile(r"[一-鿿]|[a-zA-Z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text or ""))


def _relevance(query_tokens: set[str], entry_tokens: set[str]) -> float:
    """余弦式关键词重叠,无重叠为 0。"""
    if not query_tokens or not entry_tokens:
        return 0.0
    inter = query_tokens & entry_tokens
    if not inter:
        return 0.0
    return len(inter) / math.sqrt(len(query_tokens) * len(entry_tokens))


def _overlap_fraction(entry_tokens: set[str], recent_tokens: set[str]) -> float:
    """记忆自身词汇里已有多少比例出现在最近对话中(用于克制重复引用)。"""
    if not entry_tokens:
        return 0.0
    return len(entry_tokens & recent_tokens) / len(entry_tokens)


def recall(db, user_id: str, query: str, top_k: int | None = None, recent_text: str | None = None) -> list[dict]:
    s = get_settings()
    top_k = top_k or s.recall_top_k
    entries = list_memories(db, user_id)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    q_tokens = _tokens(query)
    recent_tokens = _tokens(recent_text) if recent_text else set()

    scored: list[dict] = []
    for e in entries:
        entry_tokens = _tokens(f"{facts_text(e.facts)} {e.summary or ''} {e.content}")
        if recent_tokens and _overlap_fraction(entry_tokens, recent_tokens) >= s.recall_recent_overlap_threshold:
            continue
        days = max(0.0, (now - e.ts).total_seconds() / 86400.0)
        recency = math.exp(-days / 30.0)
        importance = (e.importance or 5.0) / 10.0
        relevance = _relevance(q_tokens, entry_tokens)
        score = s.recall_alpha * recency + s.recall_beta * importance + s.recall_gamma * relevance
        scored.append({"entry": e, "score": score, "recency": recency, "relevance": relevance})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
'''


_FIXED_NOW = datetime(2026, 11, 21, 12, 0, 0)


class _FixedDatetime:
    @staticmethod
    def now(tz=None):
        return _FIXED_NOW


def _baseline_recall(fixed_datetime=None):
    ns = {"__name__": "recall_baseline_cde75f8"}
    exec(compile(_BASELINE_SOURCE, "cde75f8:memory/recall.py", "exec"), ns)
    if fixed_datetime is not None:
        ns["datetime"] = fixed_datetime
    return ns["recall"]


def _dump(result: list[dict]) -> bytes:
    payload = [
        {
            "id": str(r["entry"].id),
            "content": r["entry"].content,
            "score": repr(r["score"]),
            "recency": repr(r["recency"]),
            "relevance": repr(r["relevance"]),
        }
        for r in result
    ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")


@pytest.fixture
def idx_env(tmp_path, monkeypatch):
    """每个用例独立索引文件，避免跨用例污染。"""
    path = str(tmp_path / "dual_idx.sqlite3")
    monkeypatch.setenv("DUAL_RECALL_INDEX_DB", path)
    reset_index()
    monkeypatch.delenv("USE_DUAL_RECALL", raising=False)
    yield path
    reset_index()


@pytest.fixture
def db(isolation):
    from memory.db import SessionLocal

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _add(db, user_id, content, summary, *, facts=None, importance=5.0, days_ago=0):
    m = store.add_memory(db, user_id, content=content, summary=summary,
                         facts=facts or [], importance=importance)
    m.ts = _FIXED_NOW - timedelta(days=days_ago)
    db.commit()
    return m


def _index_all(db, user_id):
    """仅供测试 setup：把 store 里的记忆显式喂给索引（write-through API）。"""
    entries = store.list_memories(db, user_id)
    n = sum(1 for e in entries if recall_mod.index_memory(e, db=db))
    return n, len(entries)


# ---------------------------------------------------------------------------
# V1 —— 开启后 list_memories 调用 = 0
# ---------------------------------------------------------------------------
def test_v1_list_memories_call_count_zero(db, idx_env, monkeypatch):
    uid = "A-V1-U"
    for i in range(100):
        _add(db, uid, f"第{i}条记忆：今天做了事情编号{i}", f"记忆{i}",
             facts=[{"kind": "memory", "fact": f"记忆{i}", "confidence": 0.9}])
    indexed, total = _index_all(db, uid)
    assert indexed == total == 100

    real_list = recall_mod.list_memories
    counter = {"n": 0}

    def counting(*a, **kw):
        counter["n"] += 1
        return real_list(*a, **kw)

    monkeypatch.setattr(recall_mod, "list_memories", counting)
    monkeypatch.setenv("USE_DUAL_RECALL", "1")
    res = recall_mod.recall(db, uid, "记忆", top_k=5)
    assert counter["n"] == 0, f"list_memories 被调用 {counter['n']} 次"
    assert len(res) == 5
    assert all(isinstance(r["entry"], store.MemoryEntry) for r in res)


# ---------------------------------------------------------------------------
# V2 —— vec0 语义路（字面零重叠）
# ---------------------------------------------------------------------------
def test_v2_vector_path_recalls_no_literal_overlap(db, idx_env, monkeypatch):
    uid = "A-V2-U"
    # summary 不能含"失眠"，否则 bm25 会字面命中，无法证明语义路单独生效
    target = _add(db, uid, "那晚彻夜睡不着，心里很痛苦", "回忆", days_ago=3)
    _add(db, uid, "今天去超市买了苹果和香蕉", "买水果", days_ago=1)
    _add(db, uid, "天气很好，出门散了步", "天气", days_ago=2)
    _add(db, uid, "桂花糕的做法记下来了", "桂花糕", days_ago=4)
    assert _index_all(db, uid)[0] == 4

    monkeypatch.setenv("USE_DUAL_RECALL", "1")
    res = recall_mod.recall(db, uid, "失眠难过", top_k=4)
    assert res, "vec0 路无任何召回"
    top = res[0]
    assert top["entry"].id == target.id, (
        f"语义路未召回目标: {[r['entry'].content for r in res]}"
    )
    sd = top["score_details"]
    assert sd["vec_score"] > 0
    # 查询与目标的 jieba token 零重叠 → bm25 原始分为 0，证明命中完全来自 vec0 路
    assert sd["bm25_raw"] == 0.0
    # 非语义模型必须如实标注
    assert sd["embedding_kind"] == "deterministic_hash_nonsemantic"


# ---------------------------------------------------------------------------
# V3 —— bm25s 关键词路：专名精确匹配优先
# ---------------------------------------------------------------------------
def test_v3_keyword_exact_proper_noun_ranks_first(db, idx_env, monkeypatch):
    uid = "A-V3-U"
    doudou = _add(db, uid, "豆豆是一只小狗，很可爱", "豆豆", days_ago=7)
    _add(db, uid, "汤圆是一只猫，很喜欢晒太阳，非常可爱的小猫", "汤圆", days_ago=0)
    assert _index_all(db, uid)[0] == 2

    monkeypatch.setenv("USE_DUAL_RECALL", "1")
    res = recall_mod.recall(db, uid, "豆豆", top_k=2)
    assert res and res[0]["entry"].id == doudou.id, (
        f"专名未排第一: {[r['entry'].content for r in res]}"
    )
    assert res[0]["score_details"]["bm25_raw"] > 0, "bm25 路未命中专名"


# ---------------------------------------------------------------------------
# V4 —— purge 级联一致
# ---------------------------------------------------------------------------
def test_v4_purge_cascades_vec_and_bm25(db, idx_env, monkeypatch):
    uid = "A-V4-U"
    a = _add(db, uid, "豆豆喜欢在院子里跑来跑去", "豆豆")
    b = _add(db, uid, "红枣桂花糕的配方", "桂花糕")
    assert _index_all(db, uid)[0] == 2
    idx = get_index_for(db)

    before_a = idx.diagnostics(a.id)
    before_b = idx.diagnostics(b.id)
    assert before_a["vec_rows"] == 1 and before_a["bm25_rows"] == 1
    assert before_b["vec_rows"] == 1 and before_b["bm25_rows"] == 1

    assert recall_mod.purge_memory(a.id, db=db) is True
    after_a = idx.diagnostics(a.id)
    after_b = idx.diagnostics(b.id)
    assert after_a["vec_rows"] == 0 and after_a["bm25_rows"] == 0, after_a
    assert after_b["vec_rows"] == 1 and after_b["bm25_rows"] == 1, after_b

    monkeypatch.setenv("USE_DUAL_RECALL", "1")
    res = recall_mod.recall(db, uid, "豆豆 跑", top_k=5)
    assert all(r["entry"].id != a.id for r in res), "purge 后仍被召回"


# ---------------------------------------------------------------------------
# V5 —— 开关关闭逐字节回退（oracle = git show cde75f8）
# ---------------------------------------------------------------------------
def test_v5_off_is_byte_for_byte_identical_to_baseline(db, idx_env, monkeypatch):
    # 即使索引里有数据，开关关闭也必须走原样 legacy 路径
    uid = "A-V5-U"
    _add(db, uid, "汤圆是一只猫，很爱晒太阳", "养猫",
         facts=[{"kind": "object", "fact": "汤圆是一只猫", "confidence": 0.9}],
         importance=6.0, days_ago=1)
    _add(db, uid, "今天早上汤圆去世了，我很伤心", "猫去世",
         facts=[{"kind": "memory", "fact": "汤圆去世了", "confidence": 0.9}],
         importance=8.0, days_ago=0)
    _add(db, uid, "做了桂花糕", "桂花糕", facts=[], importance=5.0, days_ago=10)
    _add(db, uid, "今天天气不错", "天气", facts=[], importance=4.0, days_ago=3)
    _index_all(db, uid)

    monkeypatch.delenv("USE_DUAL_RECALL", raising=False)
    monkeypatch.delenv("USE_RECALL_FUSION", raising=False)
    monkeypatch.setattr(recall_mod, "datetime", _FixedDatetime)
    baseline = _baseline_recall(_FixedDatetime)
    for q, recent in [("汤圆", None), ("猫 去世", "汤圆 猫"), ("桂花糕", "天气 不错")]:
        base = _dump(baseline(db, uid, q, top_k=5, recent_text=recent))
        actual = _dump(recall_mod.recall(db, uid, q, top_k=5, recent_text=recent))
        assert actual == base, f"query={q!r}\nbase={base!r}\nactual={actual!r}"


# ---------------------------------------------------------------------------
# V6 —— 冲突新事实排前
# ---------------------------------------------------------------------------
def test_v6_conflict_new_fact_still_ranks_first(db, idx_env, monkeypatch):
    uid = "A-V6-U"
    _add(db, uid, "养了一只猫，叫汤圆", "养猫",
         facts=[{"kind": "object", "fact": "汤圆是一只猫", "confidence": 0.9}],
         importance=9.0, days_ago=120)
    new = _add(db, uid, "汤圆今天去世了", "猫去世",
               facts=[{"kind": "memory", "fact": "汤圆去世了", "confidence": 0.9}],
               importance=6.0, days_ago=0)
    assert _index_all(db, uid)[0] == 2

    monkeypatch.setenv("USE_DUAL_RECALL", "1")
    res = recall_mod.recall(db, uid, "汤圆 去世 了", top_k=2)
    assert res and res[0]["entry"].id == new.id, (
        f"冲突新事实未排前: {[r['entry'].content for r in res]}"
    )
    assert res[0]["relevance"] > res[1]["relevance"]
