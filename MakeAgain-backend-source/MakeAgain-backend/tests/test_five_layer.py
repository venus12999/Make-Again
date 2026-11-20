"""WP-B 行为验收：agent-memory-core 五层内核（supersede/软删/衰减/溯源/双写/回退）。

全绿不是标准：每个 V 都断言具体行为并 print 原始证据（用 -s 采集）。
"""
from __future__ import annotations

import datetime as dt

import pytest

from memory.five_layer import FiveLayerAdapter
from memory import store as mstore
from memory.db import SessionLocal
from memory.models import MemoryEntry

START = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)


def _adapter(tmp_path, *, start=START):
    from agent_memory.core.clock import FrozenClock
    from agent_memory.core.config import Config

    clock = FrozenClock(start)
    config = Config()
    config.manage.git_commit = False  # 不让库的 sleep 在宿主仓库里 git commit
    return FiveLayerAdapter(tmp_path / "five-layer", clock=clock, config=config), clock


# ---------------------------------------------------------------- V1 supersede 当前/历史
def test_v1_supersede_current_and_history(tmp_path):
    adapter, clock = _adapter(tmp_path)
    adapter.add("cat-alive", "养猫", provenance=["用户原话：我养了一只猫"])
    clock.advance(days=10)
    adapter.add("cat-deceased", "猫去世", provenance=["用户原话：我的猫昨天去世了"])
    flipped = adapter.supersede("cat-alive", "cat-deceased")
    current = adapter.current("猫")
    history = adapter.current("猫", as_of="2026-01-05T00:00:00Z")

    print("\n[V1] superseded record:", flipped)
    print("[V1] current(猫):", [(r["name"], r["abstract"], r["status"]) for r in current])
    print("[V1] history(猫, as_of=2026-01-05):", [(r["name"], r["abstract"], r["status"]) for r in history])

    assert flipped["superseded_by"] == "cat-deceased"
    assert flipped["invalid_at"] == "2026-01-11T00:00:00Z"
    assert [r["name"] for r in current] == ["cat-deceased"]
    assert [r["name"] for r in history] == ["cat-alive"]


# ---------------------------------------------------------------- V2 invalidate 软删可追溯
def test_v2_invalidate_soft_delete_traceable(tmp_path):
    adapter, _ = _adapter(tmp_path)
    original = "用户原话：这条记忆以后要删掉"
    adapter.add("del-me", "要删除的记忆", provenance=[original])
    assert len(adapter.current("删除")) == 1

    invalid = adapter.invalidate("del-me")
    trace = adapter.trace("del-me")
    from agent_memory.core.database import Database

    with Database(adapter._store.layout).connect() as connection:
        row = connection.execute(
            "SELECT name, invalid_at FROM records WHERE name='del-me'"
        ).fetchone()

    print("\n[V2] after invalidate:", invalid)
    print("[V2] current(删除):", adapter.current("删除"))
    print("[V2] records row still present:", dict(row) if row else None)
    print("[V2] trace text:", trace["text"].strip().splitlines()[-1])

    assert invalid["status"] == "invalid" and invalid["invalid_at"]
    assert adapter.current("删除") == []
    assert row is not None and row["invalid_at"] == invalid["invalid_at"]
    assert original in trace["text"]


# ---------------------------------------------------------------- V3 衰减到 floor
def test_v3_decay_to_floor(tmp_path):
    from agent_memory.core.config import Config

    adapter, clock = _adapter(tmp_path)
    adapter.add("old-low", "低权重旧记忆", weight=0.1,
                provenance=["用户原话：很久以前的一件小事"])
    clock.advance(days=45)
    report = adapter.settle()
    after = adapter.record("old-low")
    floor = Config().weight.floor

    print("\n[V3] settle actions:", report["actions"])
    print("[V3] weight:", after["weight"], "floor:", floor, "still present:", bool(after))

    assert any(a["kind"] == "weight-settled" and a["target"] == "old-low"
               for a in report["actions"])
    assert after["weight"] == floor
    assert after["name"] == "old-low"  # 衰减≠删除，仍在库


# ---------------------------------------------------------------- V4 provenance 溯源原消息
def test_v4_provenance_trace_original_message(tmp_path):
    adapter, _ = _adapter(tmp_path)
    message = "用户原话：我叫小明，我最喜欢的花是茉莉"
    adapter.add("user-x", "喜欢茉莉", provenance=[message])
    trace = adapter.trace("user-x")
    current = adapter.current("茉莉")[0]

    print("\n[V4] provenance pointer:", trace["provenance"])
    print("[V4] trace text:", trace["text"].strip().splitlines()[-1])
    print("[V4] record provenance:", current["provenance"])

    assert trace["provenance"] and current["provenance"]
    assert message in trace["text"]
    assert "小明" in trace["message_texts"][0] or message in trace["message_texts"][0]


# ---------------------------------------------------------------- V5 与 store.add_memory 双写一致
def test_v5_dual_write_consistency(tmp_path, monkeypatch):
    root = tmp_path / "dual-write-five-layer"
    monkeypatch.setenv("FIVE_LAYER_STORE_DIR", str(root))
    monkeypatch.delenv("USE_FIVE_LAYER", raising=False)
    mstore.reset_five_layer_cache()
    db = SessionLocal()
    try:
        off_entry = mstore.add_memory(db, "u-off", content="关开关写入")
        assert not root.exists(), "开关关闭时不得创建第二存储"
        print("\n[V5] switch OFF -> second store created:", root.exists(),
              "(ORM entry:", off_entry.id, off_entry.content, ")")

        monkeypatch.setenv("USE_FIVE_LAYER", "1")
        on_entry = mstore.add_memory(db, "u-on", content="我养了一只猫")
        hits = mstore.five_layer_current("猫")
        trace = mstore.five_layer_trace(on_entry.id)
        orm = db.get(MemoryEntry, on_entry.id)

        print("[V5] switch ON  -> second store created:", root.exists())
        print("[V5] ORM entry:", orm.id, orm.content, "importance", orm.importance)
        print("[V5] five_layer_current(猫):", [(h["name"], h["abstract"]) for h in hits])
        print("[V5] five_layer_trace text:", trace.get("text", "").strip().splitlines()[-1])

        assert off_entry is not None and orm is not None
        assert orm.content == "我养了一只猫"
        assert [h["abstract"] for h in hits] == ["我养了一只猫"]
        assert hits[0]["name"] == on_entry.id
        assert "我养了一只猫" in trace["text"]
        mstore.reset_five_layer_cache()
    finally:
        db.close()


# ---------------------------------------------------------------- V6 关闭开关逐字节回退
def test_v6_switch_off_rollback(tmp_path, monkeypatch):
    monkeypatch.delenv("USE_FIVE_LAYER", raising=False)
    root = tmp_path / "must-not-exist"
    monkeypatch.setenv("FIVE_LAYER_STORE_DIR", str(root))
    mstore.reset_five_layer_cache()

    db = SessionLocal()
    try:
        entry = mstore.add_memory(db, "u-v6", content="基线一致", summary="s",
                                  importance=7.0, time_tag="t")
        print("\n[V6] enabled:", mstore.five_layer_enabled())
        print("[V6] current/trace/settle:", mstore.five_layer_current("任意"),
              mstore.five_layer_trace("x"), mstore.five_layer_settle())
        print("[V6] ORM row:", entry.id, entry.content, entry.summary,
              entry.importance, entry.time_tag)
        print("[V6] second store created:", root.exists())

        assert mstore.five_layer_enabled() is False
        assert mstore.five_layer_current("任意") == []
        assert mstore.five_layer_trace("x") == {}
        assert mstore.five_layer_settle() == {}
        assert not root.exists()
        assert (entry.content, entry.summary, entry.importance, entry.time_tag) == (
            "基线一致", "s", 7.0, "t")
    finally:
        db.close()
