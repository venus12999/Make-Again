"""WP7 · 合规基线行为验收测试。

逐项对应 MASTER_PLAN_PARALLEL.md 第 5 节 WP7 行为验收表：
1. AI 标识：披露响应含明确结构化 AI 生成披露字段
2. 数据删除：删除请求被记录为事件（不物理删除），可开关回滚
3. 敏感数据：自伤/哀伤内容单独标记 + 严格访问审计

不设未成年人门槛（本产品是疗愈陪伴，不是虚拟亲密关系）。
所有开关用 USE_COMPLIANCE_GATE，默认关闭；测试用 monkeypatch 切换。
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import compliance as c

@pytest.fixture
def gate_on(monkeypatch):
    monkeypatch.setenv("USE_COMPLIANCE_GATE", "1")
    c.reset_compliance_state()
    yield
    c.reset_compliance_state()


@pytest.fixture
def gate_off(monkeypatch):
    monkeypatch.setenv("USE_COMPLIANCE_GATE", "0")
    c.reset_compliance_state()
    yield
    c.reset_compliance_state()


@pytest.fixture
def api_client():
    """独立挂载合规路由到临时 app；不改 main.py（非所有权文件）。"""
    app = FastAPI()
    app.include_router(c.router)
    with TestClient(app) as client:
        yield client


# ---- 1. AI 标识 --------------------------------------------------------------

def test_disclosure_fields_present(gate_on, api_client):
    r = api_client.get("/api/compliance/disclosure?content_type=simulated_call")
    print("\n[2a] GET /api/compliance/disclosure ->", r.status_code, r.json())
    assert r.status_code == 200
    body = r.json()
    for field in ("is_ai_generated", "ai_disclosure", "generated_by",
                  "content_type", "explicit_label", "implicit_marker"):
        assert body[field], field
    assert body["is_ai_generated"] is True
    assert body["content_type"] == "simulated_call"
    assert body["enforced"] is True

    audit = api_client.get("/api/compliance/disclosure/audit").json()
    print("[2b] GET disclosure/audit ->", audit)
    assert audit["gaps"]  # 记录现有披露缺口
    locs = {g["location"] for g in audit["gaps"]}
    assert {"ending.py", "call.py", "routes.py(chat)"} <= locs


def test_disclosure_switch_off_still_readable(gate_off, api_client):
    body = api_client.get("/api/compliance/disclosure").json()
    assert body["enforced"] is False
    assert body["is_ai_generated"] is True  # 模板本身可读，只是未强制


# ---- 3. 数据删除请求记录 -----------------------------------------------------

def test_data_delete_records_event_not_physical(gate_on, api_client):
    r = api_client.post("/api/compliance/data-delete", json={
        "user_id": "u-del-1", "scope": ["memories", "photos"],
        "reason": "用户主动注销", "request_id": "req-del-1",
    })
    print("\n[3a] POST /api/compliance/data-delete ->", r.status_code, r.json())
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "pending_erasure"
    assert body["physical_delete"] is False
    assert body["enforced"] is True
    assert body["deletion_id"].startswith("del-")

    evs = api_client.get("/api/compliance/data-delete/requests?user_id=u-del-1").json()["requests"]
    print("[3b] GET data-delete/requests ->", evs)
    assert len(evs) == 1
    assert evs[0]["scope"] == ["memories", "photos"]
    assert evs[0]["status"] == "pending_erasure"
    assert evs[0]["physical_delete"] is False

    bad = api_client.post("/api/compliance/data-delete", json={
        "user_id": "u-del-1", "scope": ["unknown_scope"],
    })
    print("[3c] 非法 scope ->", bad.status_code, bad.json())
    assert bad.json()["reason"] == "invalid_scope"


def test_data_delete_switch_off_no_event(gate_off, api_client):
    r = api_client.post("/api/compliance/data-delete", json={"user_id": "u-del-2"})
    print("\n[3d] 开关关闭 data-delete ->", r.json())
    assert r.json()["status"] == "not_enforced"
    assert r.json()["enforced"] is False
    assert c.get_deletion_requests() == []


# ---- 4. 敏感数据标记 ---------------------------------------------------------

def test_sensitive_mark_and_strict_audit(gate_on, api_client):
    r = api_client.post("/api/compliance/sensitive/mark", json={
        "user_id": "u-sens-1", "text": "我想死", "source": "chat",
    })
    print("\n[4a] 自伤文本 ->", r.json())
    body = r.json()
    assert body["sensitive"] is True
    assert "self_harm" in body["categories"]
    assert body["access_policy"] == "restricted"
    assert body["strict_audit"] is True
    assert body["enforced"] is True
    assert "text" not in body  # 原始正文不复制到审计返回

    r2 = api_client.post("/api/compliance/sensitive/mark", json={
        "user_id": "u-sens-1", "text": "我想念去世的妈妈", "source": "chat",
    })
    print("[4b] 哀伤文本 ->", r2.json())
    assert "grief" in r2.json()["categories"]
    assert r2.json()["level"] == "medium"

    r3 = api_client.post("/api/compliance/sensitive/mark", json={
        "user_id": "u-sens-1", "text": "今天天气不错", "source": "chat",
    })
    print("[4c] 普通文本 ->", r3.json())
    assert r3.json()["sensitive"] is False

    audit = api_client.get("/api/compliance/sensitive/audit?user_id=u-sens-1").json()["audit"]
    print("[4d] 严格访问审计 ->", audit)
    assert len(audit) == 2  # 自伤 + 哀伤，普通文本不记
    assert all(a["access_policy"] == "restricted" and a["access_log"] == "strict" for a in audit)
    assert all(a["user_id"] == "u-sens-1" for a in audit)


def test_sensitive_switch_off_no_audit(gate_off, api_client):
    body = api_client.post("/api/compliance/sensitive/mark", json={
        "user_id": "u-sens-2", "text": "我想死",
    }).json()
    print("\n[4e] 开关关闭敏感标记 ->", body)
    assert body["sensitive"] is True
    assert body["enforced"] is False
    assert body["audit_id"] is None
    assert c.get_sensitive_audit() == []


# ---- 5. 状态端点 -------------------------------------------------------------

def test_status_endpoint(gate_on, api_client):
    s = api_client.get("/api/compliance/status").json()
    print("\n[5] GET /api/compliance/status ->", s)
    assert s["enabled"] is True and s["switch"] == "USE_COMPLIANCE_GATE"
    assert "min_age" not in s  # 不设未成年人门槛
