"""UC-2 记忆纠错闭环 U2-V1~V5 行为验收（全绿不是标准：逐条 print 原始证据）。

U2-V1 纠错后 current 只返新版
U2-V2 as_of 历史时点仍见旧版
U2-V3 纠错后聊天上下文不再含旧事实
U2-V4 完整变更链可查（supersede 关系 + 两版 trace）
U2-V5 越权被拒（跨用户读写 + 伪造 user_id）

用 `pytest -s tests/test_memory_correct.py` 采集证据。
"""
from __future__ import annotations

import datetime as dt

import pytest
from agent_memory.core.clock import FrozenClock
from fastapi.testclient import TestClient

from api.memory_correct import router as correct_router
from main import app
from memory import correct as correct_mod
from memory import recall as recall_mod
from memory import store as mstore
from memory.db import SessionLocal
from memory.models import MemoryEntry

# 本分支内自注册；lead 在 main.py 正式注册后 guard 会跳过，避免重复路由。
if not any(getattr(r, "path", "") == "/api/memory/{memory_id}/correct" for r in app.routes):
    app.include_router(correct_router)

OLD = "豆豆最喜欢的颜色是蓝色"
NEW = "豆豆最喜欢的颜色是绿色"


@pytest.fixture
def env(monkeypatch, tmp_path):
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


def _account(c: TestClient, name: str):
    r = c.post("/api/auth/register", json={"username": name, "password": "secure-test-password"})
    assert r.status_code == 200, r.json()
    data = r.json()
    headers = {"Authorization": "Bearer " + data["token"]}
    consent = c.post("/api/consent", headers=headers,
                     json={"user_id": data["user_id"], "version": 1})
    assert consent.status_code == 200, consent.json()
    return data["user_id"], headers


def _add(db, user_id: str, content: str, *, companion_id: str | None = None):
    """带 companion_id 写入，使 SecureRoute 的业务域作用域能查到该条记忆。"""
    if companion_id:
        db.info["companion_id"] = companion_id
    try:
        return mstore.add_memory(db, user_id, type="chat", content=content)
    finally:
        db.info.pop("companion_id", None)


def _freeze(adapter, moment_utc: dt.datetime):
    clock = FrozenClock(moment_utc)
    adapter._clock = clock
    adapter._store.clock = clock
    return clock


def _pin_now(monkeypatch, moment_utc: dt.datetime):
    monkeypatch.setattr(correct_mod, "_now", lambda: moment_utc.replace(tzinfo=None))


def _stamp(moment_utc: dt.datetime) -> str:
    return moment_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- U2-V1
def test_u2_v1_current_only_new(env, client, monkeypatch):
    """POST /api/memory/{id}/correct 后，current 只返新版本。"""
    db, _, adapter = env
    uid, headers = _account(client, "u2-v1")
    cid = mstore.get_current_companion(db, uid).id
    entry = _add(db, uid, OLD, companion_id=cid)

    before = mstore.five_layer_current("豆豆")
    t2 = entry.ts.replace(tzinfo=dt.UTC) + dt.timedelta(days=2)
    _freeze(adapter, t2)
    _pin_now(monkeypatch, t2)

    resp = client.post(f"/api/memory/{entry.id}/correct", headers=headers,
                       json={"content": NEW, "reason": "颜色记错了"})
    after = mstore.five_layer_current("豆豆")
    old_record = adapter.record(str(entry.id))

    print("\n[U2-V1] POST status/body:", resp.status_code, resp.json())
    print("[U2-V1] before current:", [(h["name"], h["abstract"]) for h in before])
    print("[U2-V1] after current:", [(h["name"], h["abstract"]) for h in after])
    print("[U2-V1] old record:", old_record)
    print("[U2-V1] old present_in_current:", adapter.present_in_current(str(entry.id)))

    assert resp.status_code == 200, resp.json()
    assert resp.json()["content"] == NEW
    assert [h["name"] for h in before] == [str(entry.id)]
    assert [h["abstract"] for h in after] == [NEW]
    assert str(entry.id) not in [h["name"] for h in after]
    assert old_record["invalid_at"] == _stamp(t2)
    assert old_record["superseded_by"] == resp.json()["superseded_by"]
    assert adapter.present_in_current(str(entry.id)) is False


# ---------------------------------------------------------------- U2-V2
def test_u2_v2_as_of_sees_old(env, monkeypatch):
    """历史时点 as_of 仍看到旧事实；当前查询看不到。"""
    db, _, adapter = env
    entry = _add(db, "u2-v2", OLD)
    t2 = entry.ts.replace(tzinfo=dt.UTC) + dt.timedelta(days=2)
    as_of = entry.ts.replace(tzinfo=dt.UTC) + dt.timedelta(days=1)
    _freeze(adapter, t2)
    _pin_now(monkeypatch, t2)

    result = correct_mod.correct_memory(db, str(entry.id), NEW, "颜色记错了", user_id="u2-v2")
    current = mstore.five_layer_current("蓝色")
    historic = mstore.five_layer_current("蓝色", as_of=_stamp(as_of))
    succ = adapter.record(result["superseded_by"])

    print("\n[U2-V2] correction result:", result)
    print("[U2-V2] current(蓝色):", current)
    print("[U2-V2] as_of(" + _stamp(as_of) + ") 蓝色:", [(h["name"], h["abstract"]) for h in historic])
    print("[U2-V2] successor record:", succ)
    print("[U2-V2] present_in_current old now / as_of:",
          adapter.present_in_current(str(entry.id)), adapter.present_in_current(str(entry.id), as_of=_stamp(as_of)))

    assert current == [], "当前不应再有旧事实"
    assert [h["name"] for h in historic] == [str(entry.id)]
    assert [h["abstract"] for h in historic] == [OLD]
    assert adapter.present_in_current(str(entry.id)) is False
    assert adapter.present_in_current(str(entry.id), as_of=_stamp(as_of)) is True
    assert succ["abstract"] == NEW and succ["invalid_at"] is None


# ---------------------------------------------------------------- U2-V3
def test_u2_v3_chat_context_no_old_fact(env, monkeypatch):
    """纠错后业务库内容/召回/五层 current 都不再含旧事实。"""
    db, _, adapter = env
    entry = _add(db, "u2-v3", OLD)
    before_hits = recall_mod.recall(db, "u2-v3", "蓝", top_k=5)
    # 立刻取字符串/分数快照：recall 返回的是 ORM 对象，纠错后会被同一身份映射改写。
    before_text = " | ".join(h["entry"].content or "" for h in before_hits)
    before_scores = [round(h["score"], 3) for h in before_hits]

    t2 = entry.ts.replace(tzinfo=dt.UTC) + dt.timedelta(days=2)
    _freeze(adapter, t2)
    _pin_now(monkeypatch, t2)
    correct_mod.correct_memory(db, str(entry.id), NEW, "颜色记错了", user_id="u2-v3")

    after_hits = recall_mod.recall(db, "u2-v3", "蓝", top_k=5)
    after_text = " | ".join(h["entry"].content or "" for h in after_hits)
    orm = db.get(MemoryEntry, str(entry.id))
    five_cur = mstore.five_layer_current("豆豆")

    print("\n[U2-V3] recall before (snapshot):", before_text, before_scores)
    print("[U2-V3] recall after:", after_text, [round(h["score"], 3) for h in after_hits])
    print("[U2-V3] ORM content/summary/facts:", orm.content, orm.summary, orm.facts)
    print("[U2-V3] five_layer_current(豆豆):", [(h["name"], h["abstract"]) for h in five_cur])

    assert OLD in before_text
    assert "蓝" not in after_text, "纠错后上下文不得再注入旧事实"
    assert "蓝" not in (orm.content or "") and "蓝" not in (orm.summary or "")
    assert [h["abstract"] for h in five_cur] == [NEW]


# ---------------------------------------------------------------- U2-V4
def test_u2_v4_history_change_chain(env, client, monkeypatch):
    """GET /api/memory/{id}/history 给出完整 supersede 链，两版 trace 都在。"""
    db, _, adapter = env
    uid, headers = _account(client, "u2-v4")
    cid = mstore.get_current_companion(db, uid).id
    entry = _add(db, uid, OLD, companion_id=cid)
    t2 = entry.ts.replace(tzinfo=dt.UTC) + dt.timedelta(days=2)
    _freeze(adapter, t2)
    _pin_now(monkeypatch, t2)

    posted = client.post(f"/api/memory/{entry.id}/correct", headers=headers,
                         json={"content": NEW, "reason": "颜色记错了"})
    resp = client.get(f"/api/memory/{entry.id}/history", headers=headers)
    data = resp.json()
    versions = data["versions"]

    trace_old = mstore.five_layer_trace(str(entry.id))
    trace_new = mstore.five_layer_trace(posted.json()["superseded_by"])

    print("\n[U2-V4] history status/body:", resp.status_code, data)
    print("[U2-V4] chain:", data["chain"])
    print("[U2-V4] old trace text:", trace_old.get("text", "").strip().splitlines()[-1])
    print("[U2-V4] new trace text:", trace_new.get("text", "").strip().splitlines()[-1])

    assert posted.status_code == 200
    assert resp.status_code == 200, data
    assert data["chain"][0] == str(entry.id) and len(data["chain"]) == 2
    succ_name = data["chain"][1]
    assert succ_name.startswith(str(entry.id) + "-")
    assert versions[0]["content"] == OLD and versions[1]["content"] == NEW
    assert versions[0]["superseded_by"] == succ_name
    assert versions[0]["invalid_at"] == _stamp(t2)
    assert versions[1]["invalid_at"] is None
    assert versions[1]["reason"] == "颜色记错了"
    assert versions[0]["trace_present"] and versions[1]["trace_present"]
    assert OLD in trace_old.get("text", "") and NEW in trace_new.get("text", "")


# ---------------------------------------------------------------- U2-V5
def test_u2_v5_cross_user_denied(env, client, monkeypatch):
    """越权必含：跨用户纠错/查历史拒绝，伪造 user_id 拒绝，且数据未被改动。"""
    db, _, adapter = env
    uid_a, h_a = _account(client, "u2-v5-a")
    uid_b, h_b = _account(client, "u2-v5-b")
    cid_a = mstore.get_current_companion(db, uid_a).id
    entry = _add(db, uid_a, OLD, companion_id=cid_a)

    other_correct = client.post(f"/api/memory/{entry.id}/correct", headers=h_b,
                                json={"content": NEW, "reason": "越权纠错"})
    other_history = client.get(f"/api/memory/{entry.id}/history", headers=h_b)
    forged = client.post(f"/api/memory/{entry.id}/correct", headers=h_a,
                         json={"content": NEW, "reason": "伪造归属", "user_id": uid_b})
    orm = db.get(MemoryEntry, str(entry.id))
    cur = mstore.five_layer_current("蓝色")

    print("\n[U2-V5] user B POST correct:", other_correct.status_code, other_correct.json())
    print("[U2-V5] user B GET history:", other_history.status_code, other_history.json())
    print("[U2-V5] forged user_id POST:", forged.status_code, forged.json())
    print("[U2-V5] ORM after denials:", orm.content, "current:", [(h["name"], h["abstract"]) for h in cur])

    assert other_correct.status_code in (403, 404)
    assert other_history.status_code in (403, 404)
    assert forged.status_code == 403 and forged.json()["code"] == "OWNER_MISMATCH"
    assert orm.content == OLD, "越权请求不得修改任何数据"
    assert [h["name"] for h in cur] == [str(entry.id)]
    assert adapter.record(str(entry.id))["superseded_by"] in (None, "")
