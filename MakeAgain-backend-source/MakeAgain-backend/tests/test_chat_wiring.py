# -*- coding: utf-8 -*-
"""H1 · chat 主链路集成验收：危机闭环接线 + entitlement 门禁接线。

口径：纯离线（conftest 提供 MOCK_LLM / 无真实 key / 不起 8000 端口），语料为
synthetic 自造短句，只证明"开关关逐字节不变 + 开关开写边界 + 失败 swallow"。
真后端 API / 真实模型危机评测 NOT_RUN；确认轮升级目前只落 crisis_flow 内存表
（crisis_store 只提供 insert，无状态回写接口），见汇报的诚实边界。
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from api import entitlement as ent
from conftest import account
from emotion import crisis, crisis_flow
from emotion.crisis import CRISIS_MESSAGE
from memory import crisis_store
from memory import models as m
from memory.db import SessionLocal, _RawSessionLocal

T0 = 1_700_000_000.0  # 固定 epoch，确定性控制冷却期（不依赖真实时钟）


@pytest.fixture(autouse=True)
def _wiring_defaults(monkeypatch):
    """每个用例从两个开关都关的干净状态开始。"""
    monkeypatch.delenv("USE_CRISIS_FLOW", raising=False)
    monkeypatch.delenv(ent.FLAG, raising=False)
    crisis_flow.reset_crisis_state()
    yield
    crisis_flow.reset_crisis_state()


def _chat(client, h, sid, msg, rid):
    return client.post(
        "/api/chat", headers=h,
        json={"message": msg, "session_id": sid,
              "request_id": rid, "client_request_id": rid},
    )


def _set_membership(user, *, expired: bool):
    """直接改库造权限态（synthetic，不用支付/真 key）。"""
    with _RawSessionLocal() as db:
        u = db.get(m.User, user)
        u.membership = "member"
        delta = timedelta(days=-1) if expired else timedelta(days=30)
        u.membership_expires_at = m.utcnow() + delta
        u.membership_source = "trial"
        db.commit()


# ===========================================================================
# 任务 1-A：开关关时 detect_with_flow ≡ detect（字段全等）
# ===========================================================================
def test_flow_off_detect_with_flow_equals_detect(monkeypatch):
    monkeypatch.setenv("USE_CRISIS_FLOW", "0")
    for text in ("我想死", "我撑不下去了,想解脱", "我不想死", "今天天气不错"):
        expected = crisis.detect(text)
        got = crisis.detect_with_flow(text, "u-off", now=T0)
        # detect 的三个字段逐一全等；唯一多出的是 legacy 的 reply 呈现字段
        assert {k: v for k, v in got.items() if k != "reply"} == expected
        assert set(got) == set(expected) | {"reply"}
        assert got["reply"] == (CRISIS_MESSAGE if expected["is_crisis"] else None)
    assert crisis_flow.get_events() == []  # 关闭时零事件、零落库


# ===========================================================================
# 任务 1-B：开关关时 /api/chat 危机分支逐字节不变
# ===========================================================================
def test_flow_off_chat_crisis_response_byte_identical(client):
    u, h, s = account(client)
    text = "我想死"
    r = _chat(client, h, s["chat_session_id"], text, "wire-off")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reply"] == CRISIS_MESSAGE
    assert body["crisis"] == crisis.detect(text)   # hit/is_crisis/level 三字段全等
    assert "flow" not in body["crisis"]             # 无新增扩展字段
    with SessionLocal() as db:
        assert crisis_store.count_crisis_events(db, u) == 0

    # 与 crisis_flow 的 legacy 回滚输出交叉比对（同一硬门槛、无扩展字段）
    legacy = crisis_flow.handle_crisis(text, u, now=T0)
    assert {k: v for k, v in legacy.items() if k != "reply"} == crisis.detect(text)
    assert legacy["reply"] == CRISIS_MESSAGE


# ===========================================================================
# 任务 1-C：开关开时高危事件落 crisis_store 可查 + 冷却 + 确认轮 + 用户隔离
# ===========================================================================
def test_flow_on_persists_event_cooldown_confirm_and_isolation(client, monkeypatch):
    monkeypatch.setenv("USE_CRISIS_FLOW", "1")
    u, h, s = account(client)
    u2, h2, s2 = account(client, "wire-other")

    # 首次高危：硬门槛结果与用户话术不变，扩展字段只挂在 crisis["flow"]
    r1 = _chat(client, h, s["chat_session_id"], "我想死", "wire-on-1")
    assert r1.status_code == 200, r1.text
    b1 = r1.json()
    assert b1["reply"] == CRISIS_MESSAGE
    assert b1["crisis"]["level"] == 3
    assert b1["crisis"]["is_crisis"] is True
    f1 = b1["crisis"]["flow"]
    assert f1["need_confirm"] is True
    assert f1["review_required"] is True
    assert f1["review_entry"]["queue"] == "human_review"
    assert f1["graded_script"] == "level3"
    assert f1["cooldown_active"] is False
    assert "400-161-9995" in f1["hotline"]
    assert f1["event"]["state"] == "pending_confirm"
    eid1 = f1["event_id"]

    # 独立 session 真提交后可读回（D-070 的 managed-session 坑规避）
    with SessionLocal() as db:
        rows = crisis_store.list_crisis_events(db, u)
        assert [r["id"] for r in rows] == [eid1]
        assert rows[0]["keywords"] == ["想死"]
        assert rows[0]["level"] == 3 and rows[0]["review_required"] is True
        assert rows[0]["text"] == "我想死"
        assert crisis_store.count_crisis_events(db, u2) == 0  # 用户隔离

    # 冷却期：1 小时内重复高危只改 flow 呈现，用户 reply 仍是 CRISIS_MESSAGE
    r2 = _chat(client, h, s["chat_session_id"], "我又想死了", "wire-on-2")
    assert r2.status_code == 200, r2.text
    b2 = r2.json()
    assert b2["reply"] == CRISIS_MESSAGE
    f2 = b2["crisis"]["flow"]
    assert f2["cooldown_active"] is True
    assert f2["reply"] == crisis_flow.COOLDOWN_MESSAGE
    with SessionLocal() as db:
        assert crisis_store.count_crisis_events(db, u) == 2  # 冷却期也落记录
        assert crisis_store.count_crisis_events(db, u2) == 0

    # 确认轮：回答"不安全"升级人工复核（in-memory 闭环）
    pending = crisis_flow.get_pending_confirm(u)
    assert pending is not None and pending["id"] == f2["event_id"]
    confirmed = crisis_flow.confirm_crisis(u, safe=False)
    assert confirmed["ok"] is True
    assert confirmed["event"]["state"] == "escalated"
    assert confirmed["event"]["review_required"] is True
    assert confirmed["event"]["review_entry"]["reason"] == "escalated_by_user_answer"
    assert crisis_flow.get_pending_confirm(u) is None


# ===========================================================================
# 任务 1-D：失败 swallow —— 持久化 / 闭环探测失败都不得破坏危机话术
# ===========================================================================
def test_persistence_failure_does_not_break_crisis_reply(client, monkeypatch):
    monkeypatch.setenv("USE_CRISIS_FLOW", "1")

    def boom(*a, **kw):
        raise RuntimeError("simulated DB failure")

    monkeypatch.setattr(crisis_store, "save_crisis_event", boom)
    u, h, s = account(client)
    r = _chat(client, h, s["chat_session_id"], "我想死", "wire-fail-save")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["reply"] == CRISIS_MESSAGE
    assert b["crisis"]["level"] == 3
    assert b["crisis"]["flow"]["need_confirm"] is True


def test_flow_detection_failure_falls_back_to_hard_gate(client, monkeypatch):
    monkeypatch.setenv("USE_CRISIS_FLOW", "1")

    def boom(*a, **kw):
        raise RuntimeError("simulated flow failure")

    monkeypatch.setattr(crisis, "detect_with_flow", boom)
    u, h, s = account(client)
    r = _chat(client, h, s["chat_session_id"], "我想死", "wire-fail-flow")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["reply"] == CRISIS_MESSAGE
    assert b["crisis"] == crisis.detect("我想死")  # 逐字节回退旧硬门槛
    assert "flow" not in b["crisis"]


# ===========================================================================
# 任务 2-A：entitlement 门禁默认关时零影响（即使会员过期）
# ===========================================================================
def test_entitlement_off_chat_unchanged(client):
    u, h, s = account(client)
    _set_membership(u, expired=True)  # 过期会员：只有门开时才该被拒
    r = _chat(client, h, s["chat_session_id"], "今天有点想他", "ent-off")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["status"] == "completed" and b["reply"]
    assert b["crisis"]["is_crisis"] is False


# ===========================================================================
# 任务 2-B：门开时无权限被拒；有权限时正常路径回执仍逐字段不变
# ===========================================================================
def test_entitlement_on_denies_expired_and_allows_active(client, monkeypatch):
    monkeypatch.setenv(ent.FLAG, "1")
    u, h, s = account(client)

    _set_membership(u, expired=True)
    denied = _chat(client, h, s["chat_session_id"], "今天有点想他", "ent-on-deny")
    assert denied.status_code == 403, denied.text
    body = denied.json()
    assert body["code"] == "ENTITLEMENT_EXPIRED"
    assert body["message"] == "试用已到期，无法继续生成；已生成的纪念资产仍可查看"
    assert body["retryable"] is False

    _set_membership(u, expired=False)
    ok = _chat(client, h, s["chat_session_id"], "今天有点想他", "ent-on-allow")
    assert ok.status_code == 200, ok.text
    b = ok.json()
    assert b["status"] == "completed"
    assert set(b) == {
        "reply", "crisis", "emotion", "recalled", "tone", "tool", "usage",
        "memory_context", "ok", "request_id", "client_request_id",
        "session_id", "companion_id", "status", "created_at",
        "user_message_id", "assistant_message_id", "disclosure",
    }


# ===========================================================================
# D-083 安全修复：危机消息必须绕过 entitlement 门禁（过期会员也不能被付费墙拦掉热线）
# ===========================================================================
def test_crisis_bypasses_entitlement_gate(client, monkeypatch):
    monkeypatch.setenv(ent.FLAG, "1")
    u, h, s = account(client)
    _set_membership(u, expired=True)  # 过期会员

    # 危机消息：即使门禁开+会员过期，也必须放行并拿到热线话术
    crisis_resp = _chat(client, h, s["chat_session_id"], "我想死", "ent-crisis-bypass")
    assert crisis_resp.status_code == 200, crisis_resp.text
    body = crisis_resp.json()
    assert body["reply"] == CRISIS_MESSAGE
    assert body["crisis"]["is_crisis"] is True

    # 同一用户非危机消息：仍被门禁拒绝（证明不是门禁整体失效）
    denied = _chat(client, h, s["chat_session_id"], "今天有点想他", "ent-normal-deny")
    assert denied.status_code == 403, denied.text
    assert denied.json()["code"] == "ENTITLEMENT_EXPIRED"


def test_entitlement_router_mounted_in_production_app(client):
    """D-083：/api/entitlement 视图端点在生产 main.py 已挂载（此前仅测试自挂）。

    新版 FastAPI 的 include_router 用 _IncludedRouter 聚合，r.path 拿不到内部路径，
    因此用真实请求判定：挂载了→端点存在（401 鉴权失败或 200），没挂→404。
    """
    r = client.get("/api/entitlement")
    assert r.status_code != 404, "entitlement router 未挂载到生产 app"
