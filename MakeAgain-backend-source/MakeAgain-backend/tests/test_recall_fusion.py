"""R1 召回融合验收:开关关逐字节一致 / 开关开多路融合 / MMR 去重 / 冲突新事实。

基线来源:main@45f3351:MakeAgain-backend/memory/recall.py(62 行,逐字嵌入下方常量)。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from memory import recall as recall_mod
from memory import store

# main@45f3351:MakeAgain-backend/memory/recall.py 原文(逐字),用作逐字节对照 oracle。
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
    """记忆自身词汇里已有多少比例出现在最近对话中(用于克制重复引用)。

    不对称:分母是记忆自身的词数。若记忆的大半关键词都已在最近对话里出现,
    说明这件事正在被聊,无需再作为「相关记忆」注入——否则刚说完又提,显得机械。
    """
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
        # 克制:已出现在最近几轮对话里的记忆不再注入(它已在会话历史里,模型本就看得见)
        if recent_tokens and _overlap_fraction(entry_tokens, recent_tokens) >= s.recall_recent_overlap_threshold:
            continue
        days = max(0.0, (now - e.ts).total_seconds() / 86400.0)
        recency = math.exp(-days / 30.0)  # 30 天半衰期
        importance = (e.importance or 5.0) / 10.0
        relevance = _relevance(q_tokens, entry_tokens)
        score = s.recall_alpha * recency + s.recall_beta * importance + s.recall_gamma * relevance
        scored.append({"entry": e, "score": score, "recency": recency, "relevance": relevance})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
'''


# 逐字节对照必须同一时刻口径:两侧都注入同一个固定 now,消除微秒级 recency 漂移。
_FIXED_NOW = datetime(2026, 11, 21, 12, 0, 0)


class _FixedDatetime:
    @staticmethod
    def now(tz=None):
        return _FIXED_NOW


def _baseline_recall(fixed_datetime=None):
    ns = {"__name__": "recall_baseline_45f3351"}
    exec(compile(_BASELINE_SOURCE, "main@45f3351:memory/recall.py", "exec"), ns)
    if fixed_datetime is not None:
        ns["datetime"] = fixed_datetime
    return ns["recall"]


def _dump(result: list[dict]) -> bytes:
    """结果序列化为 UTF-8 字节:包含 id/内容/三个浮点位的精确 repr。"""
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
def db(isolation):
    from memory.db import SessionLocal

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _add(db, user_id, content, summary, facts, importance, days_ago):
    m = store.add_memory(db, user_id, content=content, summary=summary,
                         facts=facts, importance=importance)
    m.ts = _FIXED_NOW - timedelta(days=days_ago)
    db.commit()
    return m


def _seed_corpus(db, uid="R1-U"):
    _add(db, uid, "汤圆是一只猫，很爱晒太阳", "养猫",
         [{"kind": "object", "fact": "汤圆是一只猫", "confidence": 0.9}], 6.0, 1)
    _add(db, uid, "今天早上汤圆去世了，我很伤心", "猫去世",
         [{"kind": "memory", "fact": "汤圆去世了", "confidence": 0.9}], 8.0, 0)
    _add(db, uid, "做了桂花糕", "桂花糕", [], 5.0, 10)
    _add(db, uid, "今天天气不错", "天气", [], 4.0, 3)
    _add(db, uid, "他送我的围巾还在", "围巾", [], 7.0, 20)
    _add(db, uid, "计划去旅行", "旅行", [], 5.0, 45)


def test_off_is_byte_for_byte_identical_to_baseline(db, monkeypatch):
    """① 6 个查询:开关关闭时输出字节 == 基线 main@45f3351。"""
    monkeypatch.delenv("USE_RECALL_FUSION", raising=False)
    # 基线是 R1 之前的 legacy 打分；dual/semantic/intent 已默认开，需显式关以保持契约
    for key in ("USE_DUAL_RECALL", "USE_SEMANTIC_EMBEDDING", "USE_QUERY_INTENT"):
        monkeypatch.setenv(key, "0")
    monkeypatch.setattr(recall_mod, "datetime", _FixedDatetime)
    baseline = _baseline_recall(_FixedDatetime)
    uid = "R1-BYTE-U"
    _seed_corpus(db, uid)
    queries = [
        ("汤圆", None),
        ("猫 去世", "汤圆 猫"),
        ("桂花糕", "天气 不错"),
        ("天气", None),
        ("围巾", "他 围巾"),
        ("旅行 计划", "桂花糕"),
    ]
    for q, recent in queries:
        base = _dump(baseline(db, uid, q, top_k=5, recent_text=recent))
        actual = _dump(recall_mod.recall(db, uid, q, top_k=5, recent_text=recent))
        assert actual == base, f"query={q!r} recent={recent!r}\nbase={base!r}\nactual={actual!r}"


def test_fusion_failure_falls_back_byte_for_byte(db, monkeypatch):
    """② 新路径抛异常时自动回退旧路径,输出字节 == 基线。"""
    monkeypatch.setenv("USE_RECALL_FUSION", "1")
    # 让 fusion 成为唯一新路径（dual 已默认开会抢先），测它抛异常后的 legacy fallback
    for key in ("USE_DUAL_RECALL", "USE_SEMANTIC_EMBEDDING", "USE_QUERY_INTENT"):
        monkeypatch.setenv(key, "0")
    monkeypatch.setattr(recall_mod, "datetime", _FixedDatetime)
    baseline = _baseline_recall(_FixedDatetime)
    uid = "R1-FALLBACK-U"
    _seed_corpus(db, uid)

    def boom(*args, **kwargs):
        raise RuntimeError("simulated fusion failure")

    monkeypatch.setattr(recall_mod, "_recall_fusion", boom)
    base = _dump(baseline(db, uid, "汤圆", top_k=5))
    assert _dump(recall_mod.recall(db, uid, "汤圆", top_k=5)) == base


def test_fusion_promotes_relevant_memory(db, monkeypatch):
    """③ 开关开:多路融合把相关记忆排前(基线会把新近无关项排前)。"""
    uid = "R1-FUSE-U"
    old = _add(db, uid, "汤圆 猫 陪伴 我", "养猫",
               [{"kind": "object", "fact": "汤圆是一只猫"}], 10.0, 69)
    recent = _add(db, uid, "今天天气很好出门散步", "天气", [], 9.0, 0)
    q = "汤圆 猫"

    monkeypatch.delenv("USE_RECALL_FUSION", raising=False)
    baseline_top = _baseline_recall()(db, uid, q, top_k=2)[0]["entry"].content
    assert baseline_top == recent.content  # 基线口径:recency+importance 压过 relevance

    monkeypatch.setenv("USE_RECALL_FUSION", "1")
    fused = recall_mod.recall(db, uid, q, top_k=2)
    assert fused[0]["entry"].content == old.content
    assert {"entry", "score", "recency", "relevance"} <= set(fused[0])
    assert fused[0]["score_details"]["semantic_score"] > 0


def test_mmr_removes_semantic_duplicate(db, monkeypatch):
    """④ MMR 去掉语义重复项:两条同文记忆只保留一条,腾出的位置给不同内容。"""
    uid = "R1-MMR-U"
    a = _add(db, uid, "汤圆是一只猫，很喜欢晒太阳", "猫", [], 5.0, 0)
    b = _add(db, uid, "汤圆是一只猫，很喜欢晒太阳", "猫", [], 5.0, 1)
    c = _add(db, uid, "今天去超市买了桂花糕", "桂花糕", [], 5.0, 2)

    monkeypatch.delenv("USE_RECALL_FUSION", raising=False)
    base_ids = [r["entry"].id for r in _baseline_recall()(db, uid, "汤圆", top_k=2)]
    assert set(base_ids) == {a.id, b.id}  # 基线:重复项占满 top2

    monkeypatch.setenv("USE_RECALL_FUSION", "1")
    fused = recall_mod.recall(db, uid, "汤圆", top_k=2)
    ids = [r["entry"].id for r in fused]
    assert len(ids) == 2
    assert not ({a.id, b.id} <= set(ids)), f"MMR 未去掉重复项: {ids}"
    assert c.id in ids, f"去重后应让不同内容上位: {ids}"


def test_conflict_new_fact_still_ranks_first(db, monkeypatch):
    """⑤ 冲突用例「养猫 → 猫去世」:新事实仍排前。"""
    uid = "R1-CONFLICT-U"
    _add(db, uid, "养了一只猫，叫汤圆", "养猫",
         [{"kind": "object", "fact": "汤圆是一只猫", "confidence": 0.9}], 9.0, 120)
    new = _add(db, uid, "汤圆今天去世了", "猫去世",
               [{"kind": "memory", "fact": "汤圆去世了", "confidence": 0.9}], 6.0, 0)

    monkeypatch.setenv("USE_RECALL_FUSION", "1")
    fused = recall_mod.recall(db, uid, "汤圆 去世 了", top_k=2)
    assert fused[0]["entry"].content == new.content
    assert fused[0]["relevance"] > fused[1]["relevance"]
