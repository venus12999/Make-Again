"""WP-1 行为验收：linked_memory_ids 真正接进抽取链路（V1-V4 + 越权/边界）。

验收纪律：每条 V 打印可复现的真实命令/输出；全绿不是标准。
运行：pytest -s tests/test_linked_ids_wired.py -v

背景：基线 29cda50 中 entity_graph.link_facts 只被单测直接调用，extract/async_write
引用 0 次 → facts 里 linked_memory_ids 恒为空。本文件验证接线后抽取落库即回填。
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import select

from memory import async_write, entity_graph, extract, store
from memory.db import SessionLocal
from memory.models import MemoryEntry, User

REPO_ROOT = Path(__file__).resolve().parents[2]
FACT = "豆豆很可爱，豆豆陪着我"
FACT2 = "又想起豆豆，豆豆是我的家人"


@pytest.fixture(autouse=True)
def _wp1_isolation(monkeypatch, tmp_path):
    """每条用例独立实体图 SQLite 文件 + 业务库由全局 conftest 隔离。"""
    eg_path = str(tmp_path / "wp1-entity-graph.sqlite3")
    monkeypatch.setenv("ENTITY_GRAPH_DB_PATH", eg_path)
    monkeypatch.setenv("USE_ENTITY_PORTRAIT", "0")
    monkeypatch.setenv("USE_MEMORY_CONFIRM", "0")
    entity_graph.set_embedder(None)
    entity_graph.reset()
    yield eg_path
    entity_graph.reset()
    monkeypatch.delenv("USE_ENTITY_PORTRAIT", raising=False)


@pytest.fixture
def db():
    s = SessionLocal()
    s.add(User(id="u-wp1"))
    s.add(User(id="u-other"))
    s.commit()
    yield s
    s.close()


@pytest.fixture
def wired(monkeypatch):
    """隔离 _do_write 的外部依赖（LLM 客户端 / 队列 / 回声壁），不动被测接线本身。"""
    class _DummyClient:
        mock = True

    monkeypatch.setattr(async_write, "LLMClient", lambda: _DummyClient())
    import memory.jobs as jobs
    monkeypatch.setattr(jobs, "enqueue", lambda *a, **k: None)
    monkeypatch.setattr(jobs, "flush_intents", lambda *a, **k: None)
    from agent import echo as echo_mod
    monkeypatch.setattr(echo_mod, "maybe_capture_echo", lambda *a, **k: None)


def _turn(fact_text: str) -> dict:
    """最小合法 turn dict：facts 带空的 linked_memory_ids（抽取链路的原始形态）。"""
    return {
        "facts": [{"kind": "user", "fact": fact_text, "confidence": 0.9,
                   "linked_memory_ids": []}],
        "emotion": {"emotion": "想念", "valence": -0.5, "arousal": 0.4},
        "summary": fact_text[:120],
        "importance": 5.0,
        "place_tag": None,
        "time_tag": None,
    }


def _extractor_for(fact_text: str):
    return lambda client, text: _turn(fact_text)


def _business_db_path() -> str:
    return os.environ["DATABASE_URL"].replace("sqlite:///", "", 1)


def _raw_facts(user_id: str) -> list[str]:
    raw = sqlite3.connect(_business_db_path())
    try:
        rows = raw.execute(
            "SELECT facts FROM memory_stream WHERE user_id = ? ORDER BY id", (user_id,)
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        raw.close()


# ==================== V1：两轮对话提同一实体 → facts.linked_memory_ids 非空 ====================
def test_v1_two_turns_same_entity_linked(db, wired, monkeypatch):
    monkeypatch.setenv("USE_ENTITY_PORTRAIT", "1")
    async_write._do_write("u-wp1", "第一轮", "chat", extractor=_extractor_for(FACT))
    async_write._do_write("u-wp1", "第二轮", "chat", extractor=_extractor_for(FACT2))

    entries = db.execute(
        select(MemoryEntry).where(MemoryEntry.user_id == "u-wp1").order_by(MemoryEntry.id)
    ).scalars().all()
    assert len(entries) == 2
    # UUID 主键的字典序 != 插入序，按 fact 正文识别两轮，避免测试自身误判
    first = next(e for e in entries if e.facts[0]["fact"] == FACT)
    second = next(e for e in entries if e.facts[0]["fact"] == FACT2)
    print("\n[V1] cmd: pytest -s tests/test_linked_ids_wired.py -v")
    print("[V1] first(turn1) id:", str(first.id), "| linked_memory_ids:",
          first.facts[0]["linked_memory_ids"])
    print("[V1] second(turn2) id:", str(second.id), "| linked_memory_ids:",
          second.facts[0]["linked_memory_ids"], "<- 非空即接线生效")

    # 真实 SQLite 查询（业务库 facts JSON + 实体图 vec0 表）
    raw_facts = _raw_facts("u-wp1")
    eg = entity_graph.get_connection(os.environ["ENTITY_GRAPH_DB_PATH"])
    eg_rows = eg.execute(
        "SELECT data, linked_memory_ids FROM entities WHERE data_norm = ? AND user_id = ?",
        ("豆豆", "u-wp1"),
    ).fetchall()
    parsed = [json.loads(r) for r in raw_facts]
    raw_first = next(x for x in parsed if x[0]["fact"] == FACT)
    raw_second = next(x for x in parsed if x[0]["fact"] == FACT2)
    print("[V1] raw SQLite memory_stream.facts:", raw_facts)
    print("[V1] raw SQLite entities(豆豆):", [dict(r) for r in eg_rows])
    assert first.facts[0]["linked_memory_ids"] == []
    assert second.facts[0]["linked_memory_ids"] == [str(first.id)]
    assert raw_first[0]["linked_memory_ids"] == []
    assert raw_second[0]["linked_memory_ids"] == [str(first.id)]
    assert len(eg_rows) == 1
    assert set(json.loads(eg_rows[0]["linked_memory_ids"])) == {
        str(first.id), str(second.id),
    }


# ==================== V2：开关关闭 → 逐字节回退（以 store.add_memory 基线为 oracle） ====================
def test_v2_switch_off_byte_for_byte(db, wired):
    # oracle：未经任何新接线的基线构造器
    store.add_memory(db, "u-other", content="基线", facts=_turn(FACT)["facts"])
    # 新链路，开关关
    async_write._do_write("u-wp1", "基线", "chat", extractor=_extractor_for(FACT))

    baseline_raw, wired_raw = _raw_facts("u-other")[0], _raw_facts("u-wp1")[0]
    facts = _turn(FACT)["facts"]
    before = json.dumps(facts, sort_keys=True, ensure_ascii=False).encode("utf-8")
    out = extract.link_facts_for_memory(facts, memory_id="m-would-link", user_id="u-wp1")
    after = json.dumps(out, sort_keys=True, ensure_ascii=False).encode("utf-8")

    print("\n[V2] baseline raw facts :", baseline_raw)
    print("[V2] switch-off raw facts:", wired_raw)
    print("[V2] byte-identical (DB):", baseline_raw.encode("utf-8") == wired_raw.encode("utf-8"))
    print("[V2] same object / bytes (extract helper):", out is facts, before == after)
    print("[V2] entity graph entities:", entity_graph.list_entities())
    assert baseline_raw.encode("utf-8") == wired_raw.encode("utf-8")
    assert out is facts and before == after
    assert entity_graph.list_entities() == []  # 开关关时实体图零写入


# ==================== V3：失败 swallow，不破坏主链路 ====================
def test_v3_failure_swallowed(db, wired, monkeypatch):
    monkeypatch.setenv("USE_ENTITY_PORTRAIT", "1")

    def boom(*args, **kwargs):
        raise RuntimeError("entity store down")

    monkeypatch.setattr(entity_graph, "link_facts", boom)
    facts = _turn(FACT)["facts"]
    out = extract.link_facts_for_memory(facts, memory_id="m-fail", user_id="u-wp1")
    print("\n[V3] helper returns same object on failure:", out is facts)

    # 主链路照常落库（异常不外泄）
    async_write._do_write("u-wp1", "失败也照样落库", "chat", extractor=_extractor_for(FACT))
    entry = db.execute(select(MemoryEntry).where(MemoryEntry.user_id == "u-wp1")).scalar_one()
    print("[V3] entry.facts after failure:", entry.facts)
    print("[V3] raw SQLite facts:", _raw_facts("u-wp1"))
    assert out is facts
    assert entry.facts == facts
    assert len(_raw_facts("u-wp1")) == 1


# ==================== 越权/边界：跨用户实体隔离 ====================
def test_boundary_cross_user_entity_isolation(db, wired, monkeypatch):
    monkeypatch.setenv("USE_ENTITY_PORTRAIT", "1")
    async_write._do_write("u-wp1", "我的豆豆", "chat", extractor=_extractor_for(FACT))
    async_write._do_write("u-other", "别人的豆豆", "chat", extractor=_extractor_for(FACT2))

    u1 = db.execute(select(MemoryEntry).where(MemoryEntry.user_id == "u-wp1")).scalar_one()
    u2 = db.execute(select(MemoryEntry).where(MemoryEntry.user_id == "u-other")).scalar_one()
    print("\n[BOUNDARY] u-wp1 linked:", u1.facts[0]["linked_memory_ids"])
    print("[BOUNDARY] u-other linked:", u2.facts[0]["linked_memory_ids"])
    print("[BOUNDARY] entities:", entity_graph.list_entities())
    assert u1.facts[0]["linked_memory_ids"] == []      # 首个记忆本就没有可链对象
    assert u2.facts[0]["linked_memory_ids"] == []      # 绝不跨用户链到 u-wp1
    assert len(entity_graph.list_entities("u-wp1")) == 1
    assert len(entity_graph.list_entities("u-other")) == 1


# ==================== 边界：confirm pending 无 memory_id → 跳过链接 ====================
def test_boundary_confirm_pending_skips_linking(db, wired, monkeypatch):
    monkeypatch.setenv("USE_ENTITY_PORTRAIT", "1")
    monkeypatch.setenv("USE_MEMORY_CONFIRM", "1")
    monkeypatch.setattr(extract, "extract_turn", lambda client, text: _turn(FACT))

    async_write._do_write("u-wp1", "待确认", "chat")  # extractor=None → submit_candidate

    facts = _turn(FACT)["facts"]
    out = extract.link_facts_for_memory(facts, memory_id=None, user_id="u-wp1")
    print("\n[BOUNDARY] pending: no MemoryEntry, no entity; helper(memory_id=None) same object:",
          out is facts)
    assert db.execute(select(MemoryEntry).where(MemoryEntry.user_id == "u-wp1")).scalars().all() == []
    assert entity_graph.list_entities() == []
    assert out is facts


# ==================== 边界：confirm 路由补做接口（给 lead） ====================
def test_boundary_confirm_route_interface(db, monkeypatch):
    monkeypatch.setenv("USE_ENTITY_PORTRAIT", "1")
    # 模拟 confirm_candidate 已落库两笔，路由拿 memory_id + facts 调接口补做
    e1 = store.add_memory(db, "u-wp1", content="确认1", facts=_turn(FACT)["facts"])
    e2 = store.add_memory(db, "u-wp1", content="确认2", facts=_turn(FACT2)["facts"])
    out1 = async_write.link_facts_for_entry(db, str(e1.id), "u-wp1", _turn(FACT)["facts"])
    out2 = async_write.link_facts_for_entry(db, str(e2.id), "u-wp1", _turn(FACT2)["facts"])
    db.refresh(e1); db.refresh(e2)

    print("\n[BOUNDARY] confirm-route补做 out1:", out1[0]["linked_memory_ids"])
    print("[BOUNDARY] confirm-route补做 out2:", out2[0]["linked_memory_ids"])
    print("[BOUNDARY] persisted raw facts:", _raw_facts("u-wp1"))
    assert out1[0]["linked_memory_ids"] == []
    assert out2[0]["linked_memory_ids"] == [str(e1.id)]
    assert e2.facts[0]["linked_memory_ids"] == [str(e1.id)]
    # 无 memory_id / 非本用户记忆均不抛异常
    assert async_write.link_facts_for_entry(db, None, "u-wp1", _turn(FACT)["facts"]) == _turn(FACT)["facts"]


# ==================== 边界：空 facts / 无可抽取实体 / 非 dict 项 ====================
def test_boundary_empty_and_no_entity(monkeypatch):
    monkeypatch.setenv("USE_ENTITY_PORTRAIT", "1")
    out_empty = extract.link_facts_for_memory([], memory_id="m1", user_id="u-wp1")
    out_no_entity = extract.link_facts_for_memory(
        [{"fact": "今天天气不错", "linked_memory_ids": []}], memory_id="m2", user_id="u-wp1")
    mixed = [{"fact": "豆豆", "linked_memory_ids": []}, "not-a-dict"]
    out_mixed = extract.link_facts_for_memory(mixed, memory_id="m3", user_id="u-wp1")
    print("\n[BOUNDARY] empty:", out_empty, "| no entity:", out_no_entity, "| mixed:", out_mixed)
    assert out_empty == []
    assert out_no_entity[0]["linked_memory_ids"] == []
    assert out_mixed[1] == "not-a-dict"
