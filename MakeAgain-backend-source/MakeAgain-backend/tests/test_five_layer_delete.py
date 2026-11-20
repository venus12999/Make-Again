"""G1 (U-011) 五层库删除同步——行为验收（不是"全绿"）。

G1-V1 写→删→current 不返已删项
G1-V2 软删可追溯（trace 仍含 provenance）
G1-V3 invalidate_at 双时间（as_of 可见 / current 不可见）
G1-V4 开关关 + invalidate 失败均回退基线（逐字节 oracle 见 work/g1-byte-compare.py）
G1-V5 更新触发 supersede，current 只返最新

每个断言都 print 原始证据；用 `pytest -s` 采集。
"""
from __future__ import annotations

import datetime as dt

import pytest

from memory import store as mstore
from memory.db import SessionLocal
from memory.models import MemoryEntry


@pytest.fixture
def fl(monkeypatch, tmp_path):
    """隔离的五层第二存储 + 业务库会话；开关显式打开。"""
    root = tmp_path / "five-layer"
    monkeypatch.setenv("FIVE_LAYER_STORE_DIR", str(root))
    monkeypatch.setenv("USE_FIVE_LAYER", "1")
    monkeypatch.delenv("USE_DUAL_RECALL", raising=False)
    monkeypatch.delenv("USE_PROVENANCE", raising=False)
    mstore.reset_five_layer_cache()
    db = SessionLocal()
    try:
        yield db, root, mstore._get_five_layer_adapter()
    finally:
        db.close()
        mstore.reset_five_layer_cache()


def _frozen(adapter, moment: dt.datetime):
    """把 adapter 与其底层 Store 的时钟一起冻结，保证双时间断言确定性。"""
    from agent_memory.core.clock import FrozenClock

    clock = FrozenClock(moment)
    adapter._clock = clock
    adapter._store.clock = clock
    return clock


# ---------------------------------------------------------------- G1-V1
def test_g1_v1_delete_removes_from_current(fl):
    """写"豆豆"→删除→five_layer_current("豆豆") 不再返回它。"""
    db, root, adapter = fl
    e = mstore.add_memory(db, "u-g1-v1", type="chat", content="豆豆是一只小狗")

    before = mstore.five_layer_current("豆豆")
    assert [h["name"] for h in before] == [e.id]

    deleted = mstore.prov_delete_memory(db, str(e.id), actor_id="u-g1-v1")
    after = mstore.five_layer_current("豆豆")
    rec = adapter.record(e.id)

    print("\n[G1-V1] entry id:", e.id)
    print("[G1-V1] before delete, five_layer_current(豆豆):",
          [(h["name"], h["abstract"], h["status"]) for h in before])
    print("[G1-V1] prov_delete_memory returned:", deleted)
    print("[G1-V1] after delete, five_layer_current(豆豆):", after)
    print("[G1-V1] record kept (soft delete):", rec)

    assert deleted is True
    assert db.get(MemoryEntry, str(e.id)) is None, "业务库必须真的删掉"
    assert after == [], "current 不得再返回已删项"
    assert adapter.present_in_current(e.id) is False
    assert rec["name"] == e.id and rec["invalid_at"], "记录本体仍在、仅区间结束"


# ---------------------------------------------------------------- G1-V2
def test_g1_v2_soft_delete_still_traceable(fl):
    """软删后 five_layer_trace(record_id) 仍能查到记录 + provenance。"""
    db, root, adapter = fl
    e = mstore.add_memory(db, "u-g1-v2", type="chat", content="豆豆很可爱，陪了我很多年")

    assert mstore.prov_delete_memory(db, str(e.id)) is True
    trace = mstore.five_layer_trace(str(e.id))
    rec = adapter.record(e.id)

    print("\n[G1-V2] trace:", trace)
    print("[G1-V2] record after soft delete:", rec)

    assert rec and rec["invalid_at"], "记录本体保留"
    assert rec["provenance"], "provenance 指针保留"
    assert trace.get("provenance"), "trace 仍能读到 provenance"
    assert "豆豆" in trace.get("text", ""), "trace 仍能回读原消息文本"
    assert mstore.five_layer_current("豆豆") == []


# ---------------------------------------------------------------- G1-V3
def test_g1_v3_invalidate_at_dual_time(fl):
    """invalidate_at 双时间正确：as_of 历史时点可见，current 不可见。"""
    db, root, adapter = fl
    e = mstore.add_memory(db, "u-g1-v3", type="chat", content="豆豆还在我身边")
    created = e.ts  # naive UTC，秒精度

    delete_at = created.replace(tzinfo=dt.UTC) + dt.timedelta(days=2)
    _frozen(adapter, delete_at)
    assert mstore.prov_delete_memory(db, str(e.id)) is True

    rec = adapter.record(e.id)
    as_of = (created.replace(tzinfo=dt.UTC) + dt.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    current = mstore.five_layer_current("豆豆")
    historic = mstore.five_layer_current("豆豆", as_of=as_of)

    print("\n[G1-V3] created(valid_from):", rec["valid_from"])
    print("[G1-V3] invalid_at:", rec["invalid_at"])
    print("[G1-V3] as_of:", as_of)
    print("[G1-V3] five_layer_current(豆豆):", current)
    print("[G1-V3] five_layer_current(豆豆, as_of):",
          [(h["name"], h["abstract"], h["status"]) for h in historic])
    print("[G1-V3] present_in_current now / as_of:",
          adapter.present_in_current(e.id), adapter.present_in_current(e.id, as_of=as_of))

    assert rec["invalid_at"] is not None
    assert rec["invalid_at"] == delete_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert current == []
    assert adapter.present_in_current(e.id) is False
    assert [h["name"] for h in historic] == [e.id], "历史时点应能看到被软删的旧事实"
    assert adapter.present_in_current(e.id, as_of=as_of) is True


# ---------------------------------------------------------------- G1-V4
def test_g1_v4_switch_off_byte_baseline(monkeypatch, tmp_path):
    """开关关闭：不创建五层库、删除行为与基线一致（源级 oracle 见 byte-compare 脚本）。"""
    root = tmp_path / "must-not-exist"
    monkeypatch.setenv("FIVE_LAYER_STORE_DIR", str(root))
    monkeypatch.delenv("USE_FIVE_LAYER", raising=False)
    monkeypatch.delenv("USE_DUAL_RECALL", raising=False)
    monkeypatch.delenv("USE_PROVENANCE", raising=False)
    mstore.reset_five_layer_cache()
    db = SessionLocal()
    try:
        e = mstore.add_memory(db, "u-g1-v4", content="开关关闭也要能删")
        print("\n[G1-V4] enabled:", mstore.five_layer_enabled(), "root created on add:", root.exists())
        assert mstore.five_layer_enabled() is False
        assert not root.exists()
        deleted = mstore.prov_delete_memory(db, str(e.id))
        print("[G1-V4] prov_delete_memory returned:", deleted,
              "ORM row:", db.get(MemoryEntry, str(e.id)), "root created on delete:", root.exists())
        assert deleted is True
        assert db.get(MemoryEntry, str(e.id)) is None
        assert not root.exists(), "开关关闭时删除路径不得创建/触碰五层库"
        assert mstore.five_layer_current("开关关闭也要能删") == []
    finally:
        db.close()
        mstore.reset_five_layer_cache()


def test_g1_v4_invalidate_failure_swallowed(monkeypatch, tmp_path):
    """五层软删抛异常时 swallow：业务库删除仍成功、返回 True（fallback 回基线）。"""
    root = tmp_path / "five-layer-fail"
    monkeypatch.setenv("FIVE_LAYER_STORE_DIR", str(root))
    monkeypatch.setenv("USE_FIVE_LAYER", "1")
    monkeypatch.delenv("USE_DUAL_RECALL", raising=False)
    monkeypatch.delenv("USE_PROVENANCE", raising=False)
    mstore.reset_five_layer_cache()
    db = SessionLocal()
    try:
        e = mstore.add_memory(db, "u-g1-v4b", content="删我时五层会抛错")
        adapter = mstore._get_five_layer_adapter()

        def boom(*args, **kwargs):
            raise RuntimeError("simulated five-layer invalidate failure")

        monkeypatch.setattr(adapter, "invalidate", boom)
        deleted = mstore.prov_delete_memory(db, str(e.id))
        print("\n[G1-V4b] boom patched, prov_delete_memory returned:", deleted,
              "ORM row:", db.get(MemoryEntry, str(e.id)))
        assert deleted is True
        assert db.get(MemoryEntry, str(e.id)) is None, "五层失败不得阻断业务库删除"
    finally:
        db.close()
        mstore.reset_five_layer_cache()


# ---------------------------------------------------------------- G1-V5
def test_g1_v5_update_triggers_supersede(fl, monkeypatch):
    """更新记忆触发 supersede：current 只返最新，as_of 历史时点仍见旧版本。"""
    db, root, adapter = fl
    e = mstore.add_memory(db, "u-g1-v5", type="chat", content="豆豆是一只小狗")
    created = e.ts

    update_at = created.replace(tzinfo=dt.UTC) + dt.timedelta(days=2)
    _frozen(adapter, update_at)
    monkeypatch.setattr(mstore, "utcnow", lambda: update_at.replace(tzinfo=None))

    updated = mstore.prov_update_memory(db, str(e.id), content="豆豆是一只年老的小狗")
    cur = mstore.five_layer_current("豆豆")
    old_rec = adapter.record(e.id)
    succ_name = old_rec["superseded_by"]
    succ_rec = adapter.record(succ_name)

    as_of = (created.replace(tzinfo=dt.UTC) + dt.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    historic = mstore.five_layer_current("豆豆", as_of=as_of)

    print("\n[G1-V5] updated ORM content:", updated.content)
    print("[G1-V5] old record after supersede:", old_rec)
    print("[G1-V5] successor record:", succ_rec)
    print("[G1-V5] five_layer_current(豆豆):",
          [(h["name"], h["abstract"], h["status"]) for h in cur])
    print("[G1-V5] five_layer_current(豆豆, as_of=" + as_of + "):",
          [(h["name"], h["abstract"], h["status"]) for h in historic])

    assert updated.content == "豆豆是一只年老的小狗"
    assert old_rec["invalid_at"] and succ_name, "旧记录区间必须终止且指向后继"
    assert old_rec["invalid_at"] == update_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert [h["name"] for h in cur] == [succ_name], "current 只返最新"
    assert [h["abstract"] for h in cur] == ["豆豆是一只年老的小狗"]
    assert [h["name"] for h in historic] == [e.id], "历史时点仍见旧版本"
