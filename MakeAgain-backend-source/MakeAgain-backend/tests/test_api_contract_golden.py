# -*- coding: utf-8 -*-
"""WP8 · 前后端 API 契约 golden test（新建文件，不改业务代码）。

所有断言均来自对真实端点的实测（work/wp8_probe.py 的原始响应），
用 TestClient 打真实路由。任何字段名 / 类型 / 值 / 错误码 / 幂等行为的变动
都会让本文件失败 —— 这正是 golden 的意义：无声破坏前端契约会在 CI 被拦下。

锁定的契约面：
  1. /api/chat 的 tool payload（item_keep / photo_upload / todo_list / self_note）
  2. api/security.py fail() 的错误码与 HTTP status（非 500）
  3. 幂等回执：同 client_request_id 重放同结果 + DB 一条记录 + GET 回执结构
  4. 鉴权：无 token / 过期 token / 越权 user_id

conftest.py 提供 client fixture 与 autouse 的 DB 隔离；本文件只新增一个本地
fixture 把 USE_PYDANTIC_AI 复位为默认关闭，保证走 v1 关键词兜底链路
（卡片字段稳定、与实测一致）。_account 为本文件自包含的注册助手，避免重复
导入 conftest 触发模块级环境重置。
"""
from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import select

from memory import models as m
from memory.b07_models import Operation
from memory.db import _RawSessionLocal


@pytest.fixture(autouse=True)
def _contract_v1_default(monkeypatch):
    """锁定默认链路：USE_PYDANTIC_AI 关闭时走关键词兜底，卡片字段与 golden 一致。"""
    monkeypatch.delenv("USE_PYDANTIC_AI", raising=False)


# ---------------------------------------------------------------------------
# helpers（全部只调用真实 HTTP 端点）
# ---------------------------------------------------------------------------
def _account(client, name="golden_user"):
    """注册 + 同意 + 读状态，返回 (user_id, headers, state)。镜像 conftest.account。"""
    r = client.post("/api/auth/register", json={"username": name, "password": "secure-test-password"})
    assert r.status_code == 200, r.text
    j = r.json()
    u = j["user_id"]
    h = {"Authorization": "Bearer " + j["token"]}
    cr = client.post("/api/consent", headers=h, json={"user_id": u, "version": 1})
    assert cr.status_code == 200, cr.text
    s = client.get("/api/state", headers=h).json()
    return u, h, s


def _chat(client, h, sid, msg, rid):
    return client.post(
        "/api/chat", headers=h,
        json={"message": msg, "session_id": sid, "request_id": rid,
              "client_request_id": rid},
    )


def _assert_error(response, status, code, message, request_id=None):
    """fail() 的统一错误信封：顶层字段 + 等价的 detail，retryable 仅 429/503 为 True。"""
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"code", "message", "retryable", "request_id", "detail"}, body
    assert body["code"] == code, body
    assert body["message"] == message, body
    assert body["retryable"] is False, body
    assert body["request_id"] == request_id, body
    assert body["detail"] == {
        "code": code, "message": message, "retryable": False, "request_id": request_id,
    }, body
    return body


# ===========================================================================
# 1. tool payload —— /api/chat 的 tool 字段逐字段锁定
# ===========================================================================
class TestToolPayloadGolden:
    """实测响应（MOCK_LLM=true，关键词兜底）逐字段锁定，前端可直接消费。"""

    def test_item_payload_exact_fields(self, client):
        u, h, s = _account(client)
        r = _chat(client, h, s["chat_session_id"], "他送我的那条围巾还在", "golden-item")
        assert r.status_code == 200, r.text
        body = r.json()
        tool = body["tool"]
        # 精确键集合：新增/删除/改名字段都会失败
        assert set(tool) == {
            "type", "item_name", "intent", "item_description",
            "upload", "copy", "suggestion_id", "status",
        }, tool
        assert tool["type"] == "item_keep"
        assert tool["intent"] == "keep"
        assert tool["item_name"] == "" and isinstance(tool["item_name"], str)
        assert tool["item_description"] == "" and isinstance(tool["item_description"], str)
        assert tool["upload"] is True
        assert tool["copy"] == body["reply"] and isinstance(tool["copy"], str)
        assert tool["suggestion_id"] == "golden-item"
        assert tool["status"] == "awaiting_confirmation"

    def test_photo_payload_exact_fields(self, client):
        u, h, s = _account(client)
        r = _chat(client, h, s["chat_session_id"], "翻到一张老照片，突然很想他", "golden-photo")
        assert r.status_code == 200, r.text
        body = r.json()
        tool = body["tool"]
        assert set(tool) == {
            "type", "photo_title", "scene_description",
            "upload", "copy", "suggestion_id", "status",
        }, tool
        assert tool["type"] == "photo_upload"
        assert tool["photo_title"] == "" and isinstance(tool["photo_title"], str)
        assert tool["scene_description"] == "" and isinstance(tool["scene_description"], str)
        assert tool["upload"] is True
        assert tool["copy"] == body["reply"]
        assert tool["suggestion_id"] == "golden-photo"
        assert tool["status"] == "awaiting_confirmation"

    def test_todo_payload_exact_fields(self, client):
        u, h, s = _account(client)
        r = _chat(client, h, s["chat_session_id"], "我最近睡不好，走不出来", "golden-todo")
        assert r.status_code == 200, r.text
        body = r.json()
        tool = body["tool"]
        assert set(tool) == {"type", "tasks", "copy", "suggestion_id", "status"}, tool
        assert tool["type"] == "todo_list"
        tasks = tool["tasks"]
        assert isinstance(tasks, list) and 1 <= len(tasks) <= 3
        for t in tasks:
            assert set(t) == {"content", "done", "status"}, t
            assert isinstance(t["content"], str) and t["content"]
            assert t["done"] is False
            assert t["status"] == "suggested"
        assert tool["copy"] == body["reply"]
        assert tool["suggestion_id"] == "golden-todo"
        assert tool["status"] == "awaiting_confirmation"

    def test_note_payload_exact_fields(self, client):
        u, h, s = _account(client)
        r = _chat(client, h, s["chat_session_id"], "有好多话没说出口", "golden-note")
        assert r.status_code == 200, r.text
        body = r.json()
        tool = body["tool"]
        assert set(tool) == {
            "type", "target", "opening", "copy", "suggestion_id", "status",
        }, tool
        assert tool["type"] == "self_note"
        assert tool["target"] == "self"
        assert tool["opening"] == "" and isinstance(tool["opening"], str)
        assert tool["copy"] == body["reply"]
        assert tool["suggestion_id"] == "golden-note"
        assert tool["status"] == "awaiting_confirmation"

    def test_chat_envelope_exact_fields(self, client):
        """/api/chat 顶层回执信封：request_id 双写 + status=completed + tool 透传。"""
        u, h, s = _account(client)
        r = _chat(client, h, s["chat_session_id"], "他送我的那条围巾还在", "golden-envelope")
        assert r.status_code == 200, r.text
        body = r.json()
        assert set(body) == {
            "reply", "crisis", "emotion", "recalled", "tone", "tool", "usage",
            "memory_context", "ok", "request_id", "client_request_id",
            "session_id", "companion_id", "status", "created_at",
            "user_message_id", "assistant_message_id",
        }, body
        assert body["ok"] is True
        assert body["request_id"] == "golden-envelope"
        assert body["client_request_id"] == "golden-envelope"
        assert body["status"] == "completed"
        assert body["tool"]["suggestion_id"] == "golden-envelope"
        assert set(body["crisis"]) == {"hit", "is_crisis", "level"}
        assert body["crisis"]["is_crisis"] is False


# ===========================================================================
# 2. fail() 错误码契约 —— 全部为规定 code，不是 500
# ===========================================================================
class TestErrorCodeGolden:
    def test_unauthenticated_no_token(self, client):
        r = client.post("/api/chat", json={"message": "你好"})
        _assert_error(r, 401, "UNAUTHENTICATED", "请先登录")

    def test_session_expired_bad_token(self, client):
        r = client.get("/api/state", headers={"Authorization": "Bearer nosuchtoken"})
        _assert_error(r, 401, "SESSION_EXPIRED", "登录已失效")

    def test_not_found(self, client):
        u, h, s = _account(client)
        r = client.post("/api/todo/done", headers=h,
                        json={"todo_id": "no-such-todo", "request_id": "golden-404"})
        _assert_error(r, 404, "NOT_FOUND", "记录不存在", request_id="golden-404")

    def test_idempotency_mismatch(self, client):
        u, h, s = _account(client)
        body = {"message": "第一句", "session_id": s["chat_session_id"],
                "request_id": "golden-mismatch", "client_request_id": "golden-mismatch"}
        assert client.post("/api/chat", headers=h, json=body).status_code == 200
        body2 = dict(body); body2["message"] = "不同的内容"
        r = client.post("/api/chat", headers=h, json=body2)
        _assert_error(r, 409, "IDEMPOTENCY_MISMATCH", "相同请求标识对应不同内容",
                      request_id="golden-mismatch")

    def test_invalid_json(self, client):
        u, h, s = _account(client)
        r = client.post("/api/chat", headers={**h, "Content-Type": "application/json"},
                        content=b"{not-json")
        _assert_error(r, 422, "INVALID_JSON", "请求格式错误")

    def test_request_id_mismatch(self, client):
        u, h, s = _account(client)
        r = client.post("/api/chat", headers=h,
                        json={"message": "你好", "request_id": "A", "client_request_id": "B"})
        _assert_error(r, 422, "REQUEST_ID_MISMATCH", "请求标识不一致")

    def test_error_never_500(self, client):
        """回归护栏：错误路径的 status 必须 < 500 且带规定 code。"""
        cases = [
            client.post("/api/chat", json={"message": "你好"}),
            client.get("/api/state", headers={"Authorization": "Bearer bad"}),
        ]
        for r in cases:
            assert r.status_code < 500, (r.status_code, r.text)
            assert r.json()["code"]


# ===========================================================================
# 3. 幂等回执 —— 同 client_request_id 重放同结果，DB 一条记录
# ===========================================================================
class TestIdempotencyGolden:
    def test_replay_returns_same_result_and_single_row(self, client):
        u, h, s = _account(client)
        body = {"message": "今天的心情有点复杂", "session_id": s["chat_session_id"],
                "request_id": "golden-idem", "client_request_id": "golden-idem"}
        first = client.post("/api/chat", headers=h, json=body)
        replay = client.post("/api/chat", headers=h, json=body)
        assert first.status_code == 200 and replay.status_code == 200
        assert replay.json() == first.json(), "重放必须返回逐字节一致的结果"
        assert first.json()["request_id"] == "golden-idem"
        assert first.json()["client_request_id"] == "golden-idem"

        with _RawSessionLocal() as db:
            rows = db.scalars(select(Operation).where(Operation.request_id == "golden-idem")).all()
            assert len(rows) == 1, "同一 request_id 只能有一条 Operation"
            assert rows[0].user_id == u
            assert rows[0].status == "completed"
            assert rows[0].result == first.json()

    def test_operation_receipt_shape(self, client):
        """GET /api/operations/{request_id} 的回执信封是前端轮询的依据。"""
        u, h, s = _account(client)
        body = {"message": "你好呀", "session_id": s["chat_session_id"],
                "request_id": "golden-receipt", "client_request_id": "golden-receipt"}
        first = client.post("/api/chat", headers=h, json=body).json()
        r = client.get("/api/operations/golden-receipt", headers=h)
        assert r.status_code == 200, r.text
        receipt = r.json()
        assert set(receipt) == {
            "ok", "request_id", "client_request_id", "status",
            "resource_id", "companion_id", "result", "error",
        }, receipt
        assert receipt["ok"] is True
        assert receipt["request_id"] == "golden-receipt"
        assert receipt["client_request_id"] == "golden-receipt"
        assert receipt["status"] == "completed"
        assert receipt["error"] is None
        assert receipt["result"] == first


# ===========================================================================
# 4. 鉴权契约 —— 无 token / 过期 token / 越权 user_id
# ===========================================================================
class TestAuthGolden:
    def test_missing_token(self, client):
        _assert_error(client.post("/api/chat", json={"message": "你好"}),
                      401, "UNAUTHENTICATED", "请先登录")

    def test_expired_token(self, client):
        u, h, s = _account(client)
        db = _RawSessionLocal()
        db.add(m.AuthSession(token="golden-expired-token", user_id=u,
                             expires_at=m.utcnow() - timedelta(days=1)))
        db.commit(); db.close()
        r = client.get("/api/state", headers={"Authorization": "Bearer golden-expired-token"})
        _assert_error(r, 401, "SESSION_EXPIRED", "登录已失效")

    def test_owner_mismatch(self, client):
        u, h, s = _account(client, "golden_owner")
        u2, h2, s2 = _account(client, "golden_other")
        r = client.post("/api/chat", headers=h,
                        json={"message": "你好", "user_id": u2,
                              "session_id": s2["chat_session_id"]})
        _assert_error(r, 403, "OWNER_MISMATCH", "无法访问此账号数据")
