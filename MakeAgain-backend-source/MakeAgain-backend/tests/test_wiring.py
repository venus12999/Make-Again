"""包 G2 接线验收：中文词形归一接召回 / mute_topic 读源与过滤钩子。

口径：synthetic-handwritten + 离线 fixture（隔离 SQLite + monkeypatch + 固定时钟），
不访问网络、不读真 key、不启动后端、不跑真模型评测。自造用例不冒充真人/金标验证。
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from agent import companion
from memory import recall as recall_mod
from memory import store, tokenize_zh

_FIXED_NOW = datetime(2026, 11, 24, 12, 0, 0)


class _FixedDatetime:
    @staticmethod
    def now(tz=None):
        return _FIXED_NOW


@pytest.fixture
def db():
    from memory.db import SessionLocal

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _force_legacy(monkeypatch, normalize: str = "1") -> None:
    """固定走 _recall_legacy（隔离 dual/fusion/intent 的干扰），固定时钟。"""
    for key in ("USE_DUAL_RECALL", "USE_SEMANTIC_EMBEDDING", "USE_QUERY_INTENT", "USE_RECALL_FUSION"):
        monkeypatch.setenv(key, "0")
    monkeypatch.setenv("USE_TOKENIZE_NORMALIZE", str(normalize))
    monkeypatch.setattr(recall_mod, "datetime", _FixedDatetime)


def _add(db, uid: str, content: str, summary: str = "", *, days_ago: int = 0,
         importance: float = 5.0) -> str:
    m = store.add_memory(db, uid, content=content, summary=summary, facts=[], importance=importance)
    m.ts = _FIXED_NOW - timedelta(days=days_ago)
    db.commit()
    return str(m.id)


def _dump(result: list[dict]) -> list[tuple]:
    """结果序列化：id + 三个浮点位的精确 repr（与历史逐字节 oracle 同口径）。"""
    return [
        (str(r["entry"].id), repr(r["score"]), repr(r["recency"]), repr(r["relevance"]))
        for r in result
    ]


def _legacy_oracle(db, uid: str, query: str, top_k: int = 5, recent_text: str | None = None) -> list[tuple]:
    """改动前的 _recall_legacy 打分公式（复刻，不调用 G2 新函数）。"""
    s = recall_mod.get_settings()
    top_k = top_k or s.recall_top_k
    entries = recall_mod.list_memories(db, uid)
    q_tokens = recall_mod._tokens(query)
    recent_tokens = recall_mod._tokens(recent_text) if recent_text else set()
    scored: list[tuple] = []
    for e in entries:
        entry_tokens = recall_mod._tokens(f"{recall_mod.facts_text(e.facts)} {e.summary or ''} {e.content}")
        if recent_tokens and recall_mod._overlap_fraction(entry_tokens, recent_tokens) >= s.recall_recent_overlap_threshold:
            continue
        days = max(0.0, (_FIXED_NOW - e.ts).total_seconds() / 86400.0)
        recency = math.exp(-days / 30.0)
        importance = (e.importance or 5.0) / 10.0
        relevance = recall_mod._relevance(q_tokens, entry_tokens)
        score = s.recall_alpha * recency + s.recall_beta * importance + s.recall_gamma * relevance
        scored.append((str(e.id), score, recency, relevance))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(i, repr(sc), repr(rc), repr(rv)) for i, sc, rc, rv in scored[:top_k]]


class _FakeEntry:
    def __init__(self, summary: str, eid: int, ts=None, facts=None):
        self.summary = summary
        self.content = ""
        self.id = eid
        self.ts = ts
        self.facts = facts or []


# ---------------------------------------------------------------------------
# 任务1：词形归一接线
# ---------------------------------------------------------------------------
def test_recall_switch_off_byte_identical_to_legacy_oracle(db, monkeypatch):
    """① 开关=0：6 个查询的输出字节 == 改动前 legacy 公式（固定夹具 oracle）。"""
    _force_legacy(monkeypatch, normalize="0")
    uid = "WIRING-OFF-U"
    _add(db, uid, "汤圆是一只猫，很爱晒太阳", "养猫", days_ago=1, importance=6.0)
    _add(db, uid, "今天早上汤圆去世了，我很伤心", "猫去世", days_ago=0, importance=8.0)
    _add(db, uid, "做了桂花糕", "桂花糕", days_ago=10)
    _add(db, uid, "今天天气不错", "天气", days_ago=3)
    for q, recent in [("汤圆", None), ("猫 去世", "汤圆 猫"), ("桂花糕", "天气 不错"),
                      ("天气", None), ("晒太阳", None), ("伤心 猫", "汤圆")]:
        base = _legacy_oracle(db, uid, q, top_k=5, recent_text=recent)
        actual = _dump(recall_mod.recall(db, uid, q, top_k=5, recent_text=recent))
        assert actual == base, f"query={q!r} recent={recent!r}\nbase={base!r}\nactual={actual!r}"


def test_recall_switch_off_zero_tokenize_calls(db, monkeypatch):
    """① 补充证据：开关=0 时 normalize 一次都不调用（抛错探针不被触发）。"""
    _force_legacy(monkeypatch, normalize="0")
    uid = "WIRING-OFF-ZERO-U"
    _add(db, uid, "最近很难过", "难过")

    def boom(_text):
        raise AssertionError("USE_TOKENIZE_NORMALIZE=0 时不应调用 normalize")

    monkeypatch.setattr(tokenize_zh, "normalize", boom)
    recall_mod.recall(db, uid, "难过", top_k=5)  # 不抛错即证明零调用


def test_recall_switch_on_calls_normalize(db, monkeypatch):
    """② 开关=1：归一函数被调用（monkeypatch 计数 > 0）。"""
    _force_legacy(monkeypatch, normalize="1")
    uid = "WIRING-ON-CALL-U"
    _add(db, uid, "最近很难过", "难过")
    calls = {"n": 0}
    real = tokenize_zh.normalize

    def counting(text):
        calls["n"] += 1
        return real(text)

    monkeypatch.setattr(tokenize_zh, "normalize", counting)
    recall_mod.recall(db, uid, "难过", top_k=5)
    assert calls["n"] > 0, "开关=1 时 normalize 应被调用"


def test_recall_variant_relevance_only_when_on(db, monkeypatch):
    """③ 繁简/全半角变体：开关=0 基线相关度为 0；开关=1 变体能匹配。"""
    _force_legacy(monkeypatch, normalize="0")
    uid = "WIRING-VARIANT-U"
    trad = _add(db, uid, "難過的日子", "難過")
    fullwidth = _add(db, uid, "ＷＩＦＩ 很好用", "ＷＩＦＩ")
    _add(db, uid, "今天天气不错", "天气")

    def relevance_map(query):
        return {str(r["entry"].id): r["relevance"]
                for r in recall_mod.recall(db, uid, query, top_k=10)}

    off_zh = relevance_map("难过")
    off_en = relevance_map("wifi")
    assert off_zh[trad] == 0.0, "基线（关）繁简字面不相交，相关度应为 0"
    assert off_en[fullwidth] == 0.0, "基线（关）全半角字面不相交，相关度应为 0"

    monkeypatch.setenv("USE_TOKENIZE_NORMALIZE", "1")
    on_zh = relevance_map("难过")
    on_en = relevance_map("wifi")
    assert on_zh[trad] > 0.0, "开：难过→難過 归一后应能匹配"
    assert on_en[fullwidth] > 0.0, "开：wifi→ＷＩＦＩ NFKC 后应能匹配"


def test_recall_variant_promoted_into_top_k(db, monkeypatch):
    """③ 端到端短词召回：开=1 时变体记忆进入 top1，关=0 时进不去。"""
    _force_legacy(monkeypatch, normalize="0")
    uid = "WIRING-PROMOTE-U"
    variant = _add(db, uid, "難過", "難過", days_ago=0, importance=5.0)
    other = _add(db, uid, "今天天气不错", "天气", days_ago=0, importance=10.0)

    off_top = recall_mod.recall(db, uid, "难过", top_k=1)
    assert str(off_top[0]["entry"].id) == other, "关：变体相关度 0，重要度更高的无关项排前"
    monkeypatch.setenv("USE_TOKENIZE_NORMALIZE", "1")
    on_top = recall_mod.recall(db, uid, "难过", top_k=1)
    assert str(on_top[0]["entry"].id) == variant, "开：变体相关度补点后进入 top1"


# ---------------------------------------------------------------------------
# 任务2：mute_topic 接线（读源 + 过滤钩子）
# ---------------------------------------------------------------------------
class _DB:
    def __init__(self, **info):
        self.info = info


@pytest.fixture
def _stub_temporal(monkeypatch):
    monkeypatch.setattr(companion.temporal, "recall_daily_summaries", lambda *a, **k: "")
    monkeypatch.setattr(companion.temporal, "recall_by_time", lambda *a, **k: "")


def test_mute_no_state_field_honest_marker(monkeypatch):
    """无专用字段/写入方：开关默认关，无来源返回 []，并如实标注 NO_STATE_FIELD。"""
    monkeypatch.delenv("USE_MUTE_TOPIC", raising=False)
    assert companion._mute_topic_enabled() is False
    assert companion._mute_topics(object(), "u-1") == []
    assert companion.MUTE_STATE_SOURCE == "NO_STATE_FIELD: 未接真实来源"


def test_mute_reads_existing_db_info_source(monkeypatch):
    """从现有 db.info 读 mute 列表（请求作用域注入点，不新增 DB 列）。"""
    monkeypatch.setenv("USE_MUTE_TOPIC", "1")
    assert companion._mute_topics(_DB(mute_topics=["海", "", "   ", "橘子"]), "u-1") == ["海", "橘子"]
    assert companion._mute_topics(_DB(), "u-1") == []


def test_mute_filter_hook_filters_from_db_info(monkeypatch, _stub_temporal):
    """开关=1 且有来源：记忆里命中话题的行被过滤，稳定区画像不受影响。"""
    monkeypatch.setenv("USE_MUTE_TOPIC", "1")
    portraits = {"user": {"称呼": "小满"}, "object": {"称呼": "林夏"}}
    memories = [{"entry": _FakeEntry("她喜欢海", 7)}]
    out = companion._context_block(_DB(mute_topics=["海"]), "u-1", "hi", portraits, memories, None)
    assert "她喜欢海" not in out
    assert "[思念对象画像]" in out


def test_mute_switch_off_ignores_source(monkeypatch, _stub_temporal):
    """开关=0：即便 db.info 注入了列表也不过滤（逐字节零行为变化）。"""
    monkeypatch.setenv("USE_MUTE_TOPIC", "0")
    portraits = {"user": {"称呼": "小满"}, "object": {"称呼": "林夏"}}
    memories = [{"entry": _FakeEntry("她喜欢海", 7)}]
    out = companion._context_block(_DB(mute_topics=["海"]), "u-1", "hi", portraits, memories, None)
    assert "她喜欢海" in out


def test_d070_compaction_chain_not_broken():
    """D-070 守卫：cc.maybe_compact 链仍在（本次改动未触碰）。"""
    src = (Path(__file__).resolve().parents[1] / "agent" / "companion.py").read_text(encoding="utf-8")
    assert "from memory import context_compaction as cc" in src
    assert "turns, summary_block = cc.maybe_compact(" in src
    assert 'msgs.append({"role": "system", "content": "[此前对话摘要]' in src
    assert "turns[-1:]" in src
