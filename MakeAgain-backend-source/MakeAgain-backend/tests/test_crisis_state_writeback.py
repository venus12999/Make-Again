# -*- coding: utf-8 -*-
"""R3 · 危机确认轮状态回写 crisis_store（U-004 最后一公里）。

口径：纯离线（conftest 提供 MOCK_LLM / 隔离 SQLite / 无真实 key / 不起 8000 端口）。
语料为 synthetic 自造短句（"我想死"level3 / "我撑不下去了,想解脱"level2），
只证明确认结果 → 持久化表的回写机制、开关关逐字节不变、用户隔离、失败 swallow。
真后端 API / 真实模型危机评测 NOT_RUN。

覆盖：
  ① 开关关：confirm_crisis 返回 dict 与旧版逐字段+序列化逐字节全等，且回写零调用；
  ② 开关开：safe=False → DB state=escalated 且 review_required 0→1；safe=True → confirmed_safe；
  ③ 用户隔离：A 的确认不改 B 的 DB 行（含兜底定位也带 user_id 约束）；
  ④ 回写失败 swallow：话术/返回结构正常；
  ⑤ api/crisis GET ?state= 过滤（向后兼容）。
全量零回归由 `python -m pytest tests -q` 单独验证（基线 678）。
"""
from __future__ import annotations

import json

import pytest

from conftest import account
from emotion import crisis_flow
from memory import crisis_store
from memory.db import SessionLocal

T0 = 1_700_000_000.0  # 固定 epoch，确定性控制冷却期（不依赖真实时钟）


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    """每个用例从开关未设置 + 内存态干净开始；conftest 已逐例重建 DB。"""
    monkeypatch.delenv("USE_CRISIS_FLOW", raising=False)
    crisis_flow.reset_crisis_state()
    yield
    crisis_flow.reset_crisis_state()


def _seed_pending(user_id: str, *, text: str = "我想死", ts: float = T0) -> str:
    """开关开：造一条内存 pending confirm + 一条 DB pending_confirm 行，返回 event id。

    直接调 crisis_flow / crisis_store（不经过 HTTP），聚焦 store-flow 边界；
    save_crisis_event 的 id 口径 = 事件 dict 的 'crisis-N'（与 DB 主键同值）。
    """
    r = crisis_flow.handle_crisis(text, user_id, session_id=f"s-{user_id}", now=ts)
    assert r["need_confirm"] is True
    ev = r["event"]
    crisis_store.save_crisis_event(ev)
    return ev["id"]


def _db_row(user_id: str, event_id: str) -> dict | None:
    with SessionLocal() as db:
        rows = crisis_store.list_crisis_events(db, user_id)
    return next((x for x in rows if x["id"] == event_id), None)


# ===========================================================================
# ① 开关关：逐字节不变 + 回写零调用
# ===========================================================================
def test_switch_off_byte_identical_and_zero_writeback(monkeypatch):
    monkeypatch.setenv("USE_CRISIS_FLOW", "1")
    eid = _seed_pending("u-off")
    before_ev = crisis_flow.get_pending_confirm("u-off")
    assert before_ev is not None and before_ev["state"] == "pending_confirm"

    # 关开关；注入哨兵：一旦被调用立刻失败，证明"零调用"
    monkeypatch.setenv("USE_CRISIS_FLOW", "0")
    calls: list = []

    def sentinel(*a, **kw):
        calls.append(kw)
        raise AssertionError("writeback must NOT be called when USE_CRISIS_FLOW=0")

    got = crisis_flow.confirm_crisis("u-off", safe=False, now=T0 + 1, writeback=sentinel)

    # 用旧版算法（D-073 H1 之前的 4 行逻辑）逐字段重建期望，再比序列化字节
    expected_ev = dict(before_ev)
    expected_ev["state"] = "escalated"
    expected_ev["confirmed_at"] = T0 + 1
    expected_ev["review_required"] = True
    expected_ev["review_entry"] = crisis_flow._review_entry(expected_ev["level"], forced=True)
    expected = {"ok": True, "event": expected_ev}

    assert got == expected
    assert json.dumps(got, sort_keys=True, ensure_ascii=False) == \
           json.dumps(expected, sort_keys=True, ensure_ascii=False)  # 逐字节
    assert calls == []  # 回写零调用
    # DB 行也没被动：仍是 pending_confirm
    row = _db_row("u-off", eid)
    assert row["state"] == "pending_confirm" and row["review_required"] is True

    # no_pending_confirm 早退路径同样逐字节不变
    none_got = crisis_flow.confirm_crisis("u-never", safe=True, writeback=sentinel)
    assert none_got == {"ok": False, "reason": "no_pending_confirm", "user_id": "u-never"}
    assert calls == []


# ===========================================================================
# ② 开关开：确认结果真落到 DB（回写前后对照）
# ===========================================================================
def test_safe_false_writes_back_escalated_and_review(monkeypatch):
    monkeypatch.setenv("USE_CRISIS_FLOW", "1")
    u = "u-esc"
    eid = _seed_pending(u, text="我撑不下去了,想解脱")  # level2：初始 review_required=False

    before = _db_row(u, eid)
    assert before["state"] == "pending_confirm"
    assert before["review_required"] is False

    got = crisis_flow.confirm_crisis(u, safe=False)

    after = _db_row(u, eid)
    # 回写前后对照：pending_confirm/0 → escalated/1
    assert (before["state"], before["review_required"]) == ("pending_confirm", False)
    assert (after["state"], after["review_required"]) == ("escalated", True)
    assert got["event"]["state"] == "escalated"
    # review_entry 无 DB 列：只在内存事件与返回 dict 中（诚实边界）
    assert got["event"]["review_entry"]["reason"] == "escalated_by_user_answer"


def test_safe_true_writes_back_confirmed_safe(monkeypatch):
    monkeypatch.setenv("USE_CRISIS_FLOW", "1")
    u = "u-safe"
    eid = _seed_pending(u, text="我想死")

    before = _db_row(u, eid)
    assert before["state"] == "pending_confirm"

    got = crisis_flow.confirm_crisis(u, safe=True)

    after = _db_row(u, eid)
    assert (before["state"], after["state"]) == ("pending_confirm", "confirmed_safe")
    assert got["event"]["state"] == "confirmed_safe"
    # safe=True 不改 review_required（level3 原本 True，保持）
    assert after["review_required"] is True


# ===========================================================================
# ③ 用户隔离（主定位 + 兜底定位都带 user_id）
# ===========================================================================
def test_confirm_a_does_not_touch_b(monkeypatch):
    monkeypatch.setenv("USE_CRISIS_FLOW", "1")
    eid_a = _seed_pending("u-A", text="我想死")
    eid_b = _seed_pending("u-B", text="我想死")

    crisis_flow.confirm_crisis("u-A", safe=False)

    row_a, row_b = _db_row("u-A", eid_a), _db_row("u-B", eid_b)
    assert row_a["state"] == "escalated"
    assert row_b["state"] == "pending_confirm"  # B 完全没被 A 的确认改
    assert row_b["review_required"] is True     # B 的初始 level3 值，保持


def test_fallback_locator_is_user_scoped(monkeypatch):
    monkeypatch.setenv("USE_CRISIS_FLOW", "1")
    # B 有 DB pending 行；A 只有内存 pending（event_ref 在 DB 不存在）
    eid_b = _seed_pending("u-B2", text="我想死")
    crisis_flow.handle_crisis("我想死", "u-A2", now=T0)

    # A 用一个不存在的 ref → 兜底也找不到 A 的 DB 行，返回 0
    assert crisis_store.update_event_state(
        "u-A2", "crisis-does-not-exist", "escalated", {"review_required": True}
    ) == 0
    assert _db_row("u-B2", eid_b)["state"] == "pending_confirm"  # B 没被误伤

    # 正例：ref 不存在但同用户有 pending 行 → 兜底命中该用户最近一条
    assert crisis_store.update_event_state(
        "u-B2", "bogus-ref", "escalated", {"review_required": True}
    ) == 1
    assert _db_row("u-B2", eid_b)["state"] == "escalated"


# ===========================================================================
# ④ 回写失败 swallow
# ===========================================================================
def test_explicit_hook_failure_does_not_break_confirm(monkeypatch):
    monkeypatch.setenv("USE_CRISIS_FLOW", "1")
    eid = _seed_pending("u-fail", text="我想死")

    def boom(*a, **kw):
        raise RuntimeError("simulated writeback failure")

    got = crisis_flow.confirm_crisis("u-fail", safe=False, writeback=boom)
    assert got["ok"] is True
    assert got["event"]["state"] == "escalated"
    assert got["event"]["review_required"] is True
    assert "你现在安全吗" in got["event"]["confirm_prompt"]  # 确认轮话术仍在
    assert _db_row("u-fail", eid)["state"] == "pending_confirm"  # 只是没落库


def test_default_hook_failure_is_swallowed(monkeypatch):
    """不传 writeback 时走懒加载默认实现；它抛异常也不能破坏返回。"""
    monkeypatch.setenv("USE_CRISIS_FLOW", "1")
    _seed_pending("u-fail2", text="我想死")

    def boom(*a, **kw):
        raise RuntimeError("default hook boom")

    monkeypatch.setattr(crisis_store, "update_event_state", boom)
    got = crisis_flow.confirm_crisis("u-fail2", safe=True)
    assert got["ok"] is True and got["event"]["state"] == "confirmed_safe"


# ===========================================================================
# ⑤ API ?state= 过滤（SecureRoute 鉴权不变，不传时契约不变）
# ===========================================================================
def test_api_state_filter_backward_compatible(client):
    u, h, _ = account(client, "crisis-filter")
    base = {"user_id": u, "level": 3, "hit": ["想死"], "text": "合成事件", "ts": T0}
    crisis_store.save_crisis_event({**base, "id": "flt-1", "state": "confirmed_safe"})
    crisis_store.save_crisis_event({**base, "id": "flt-2", "state": "escalated", "ts": T0 + 1})

    r = client.get("/api/crisis/events?state=escalated", headers=h)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 1 and body["events"][0]["id"] == "flt-2"

    r_all = client.get("/api/crisis/events", headers=h)
    assert r_all.status_code == 200, r_all.text
    assert r_all.json()["count"] == 2  # 不传 state：旧契约两条全回
