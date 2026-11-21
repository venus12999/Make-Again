"""UC-1 记忆确认流验收：U1-V1 ~ U1-V5（逐条真实命令 + 输出，越权必含）。

跑法：cd MakeAgain-backend && python -m pytest tests/test_memory_confirm.py -q -s
每条用例打印 `U1-Vn-EVIDENCE ...`，可直接作为台账证据。
"""
from __future__ import annotations

import json
import sqlite3

import pytest
from sqlalchemy import select

from api.memory_confirm import router as confirm_router
from main import app
from memory import confirm, store
from memory.db import _RawSessionLocal
from memory.models import MemoryEntry

# lead 尚未把路由注册进 main；本模块自挂载（若 lead 已注册则跳过，避免重复路由）。
if not any(getattr(r, "path", "").startswith("/api/memory/candidates") for r in app.routes):
    app.include_router(confirm_router)


def account(c, name="tester"):
    r = c.post("/api/auth/register", json={"username": name, "password": "secure-test-password"})
    assert r.status_code == 200, r.json()
    token = r.json()["token"]
    headers = {"Authorization": "Bearer " + token}
    cr = c.post("/api/consent", headers=headers, json={"user_id": r.json()["user_id"], "version": 1})
    assert cr.status_code == 200, cr.json()
    return r.json()["user_id"], headers


@pytest.fixture(autouse=True)
def isolate_confirm(tmp_path, monkeypatch):
    monkeypatch.setenv("USE_MEMORY_CONFIRM", "1")
    monkeypatch.setenv("USE_FIVE_LAYER", "1")
    monkeypatch.setenv("MEMORY_CONFIRM_DB_PATH", str(tmp_path / "candidates.sqlite3"))
    monkeypatch.setenv("FIVE_LAYER_STORE_DIR", str(tmp_path / "five-layer"))
    store.reset_five_layer_cache()
    confirm.reset_confirm_cache()
    yield
    store.reset_five_layer_cache()
    confirm.reset_confirm_cache()


def _db_rows(user_id: str) -> list[MemoryEntry]:
    with _RawSessionLocal() as db:
        return list(db.scalars(select(MemoryEntry).where(MemoryEntry.user_id == user_id)).all())


def _snapshot(entry: MemoryEntry) -> str:
    """基线逐字节对比口径：排除随机 id / ts / user_id，其余全部字段进规范化 JSON。"""
    return json.dumps(
        {
            "type": entry.type, "content": entry.content, "summary": entry.summary,
            "facts": entry.facts, "emotion": entry.emotion, "cbt": entry.cbt,
            "importance": entry.importance, "place_tag": entry.place_tag,
            "time_tag": entry.time_tag,
        },
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )


# ------------------------------------------------------------ 开关默认值（产品地基）
def test_confirm_flag_defaults_on(monkeypatch):
    monkeypatch.delenv("USE_MEMORY_CONFIRM", raising=False)
    assert confirm.confirmation_enabled() is True
    print("FLAG-DEFAULT-EVIDENCE USE_MEMORY_CONFIRM unset -> confirmation_enabled()=True")


# ------------------------------------------------------------------ U1-V1
def test_u1_v1_extract_is_pending_not_fact(client):
    uid, h = account(client, "v1_user")
    r = client.post("/api/memory/candidates", headers=h,
                    json={"user_id": uid, "content": "我养了一只叫团子的猫", "type": "chat"})
    assert r.status_code == 200, r.json()
    body = r.json()
    assert body["review_status"] == "pending", body
    assert body["memory_id"] is None, body
    assert body["candidate_id"], body

    rows = _db_rows(uid)
    recalled = store.five_layer_current("团子")
    listed = client.get("/api/memory/candidates", headers=h).json()
    assert rows == [], rows
    assert recalled == [], recalled
    assert any(x["id"] == body["candidate_id"] for x in listed["candidates"]), listed

    print("U1-V1-EVIDENCE review_status=%s memory_id=%s business_rows=%d five_layer_current=%s listed=%d"
          % (body["review_status"], body["memory_id"], len(rows), recalled, listed["count"]))


# ------------------------------------------------------------------ U1-V2
def test_u1_v2_confirm_recallable_reject_not_recallable(client):
    uid, h = account(client, "v2_user")
    a = client.post("/api/memory/candidates", headers=h,
                    json={"user_id": uid, "content": "团子喜欢睡在窗台上"}).json()
    b = client.post("/api/memory/candidates", headers=h,
                    json={"user_id": uid, "content": " rejected-secret-token "}).json()

    ca = client.post("/api/memory/candidates/%s/confirm" % a["candidate_id"], headers=h,
                     json={"user_id": uid}).json()
    rb = client.post("/api/memory/candidates/%s/reject" % b["candidate_id"], headers=h,
                     json={"user_id": uid, "reason": "抽取错误"}).json()
    assert ca["candidate"]["review_status"] == "confirmed", ca
    assert rb["candidate"]["review_status"] == "rejected", rb

    confirm_rows = [x for x in _db_rows(uid) if x.content == "团子喜欢睡在窗台上"]
    confirm_recall = store.five_layer_current("团子喜欢睡在窗台上")
    reject_rows = [x for x in _db_rows(uid) if "rejected-secret-token" in x.content]
    reject_recall = store.five_layer_current("rejected-secret-token")
    assert confirm_rows and confirm_recall, (confirm_rows, confirm_recall)
    assert not reject_rows and reject_recall == [], (reject_rows, reject_recall)

    print("U1-V2-EVIDENCE confirm: rows=%d recall=%s | reject: rows=%d recall=%s"
          % (len(confirm_rows), [x["abstract"] for x in confirm_recall],
             len(reject_rows), reject_recall))


# ------------------------------------------------------------------ U1-V3
def test_u1_v3_edit_uses_supersede_old_fact_invalid(client):
    uid, h = account(client, "v3_user")
    a = client.post("/api/memory/candidates", headers=h,
                    json={"user_id": uid, "content": "团子的名字是团子"}).json()
    ca = client.post("/api/memory/candidates/%s/confirm" % a["candidate_id"], headers=h,
                     json={"user_id": uid}).json()
    old_id = ca["candidate"]["record_id"]
    assert old_id, ca
    adapter = store._get_five_layer_adapter()
    assert adapter.present_in_current(old_id) is True

    ed = client.post("/api/memory/candidates/%s/edit" % a["candidate_id"], headers=h,
                     json={"user_id": uid, "content": "团子的名字是小团子", "reason": "用户纠正"}).json()
    assert ed["candidate"]["review_status"] == "edited", ed
    old = adapter.record(old_id)
    successor = ed["superseded_by"] or old.get("superseded_by")
    assert old.get("invalid_at"), old
    assert old.get("superseded_by") == successor, (old, successor)
    assert adapter.present_in_current(old_id) is False, old
    assert adapter.record(successor).get("abstract") == "团子的名字是小团子"
    assert any(x["name"] == successor for x in store.five_layer_current("团子的名字是小团子"))
    assert [x.content for x in _db_rows(uid)] == ["团子的名字是小团子"]

    print("U1-V3-EVIDENCE old=%s invalid_at=%s superseded_by=%s old_in_current=%s successor=%s new_current=%s"
          % (old_id, old["invalid_at"], old["superseded_by"],
             adapter.present_in_current(old_id), successor,
             [x["abstract"] for x in store.five_layer_current("团子的名字是小团子")]))


# ------------------------------------------------------------------ U1-V4（越权必含）
def test_u1_v4_owner_check_rejects_cross_user(client):
    alice, ha = account(client, "v4_alice")
    bob, hb = account(client, "v4_bob")
    a = client.post("/api/memory/candidates", headers=ha,
                    json={"user_id": alice, "content": "alice 的私密记忆"}).json()

    forbidden = {}
    for action in ("confirm", "reject"):
        r = client.post("/api/memory/candidates/%s/%s" % (a["candidate_id"], action),
                        headers=hb, json={"user_id": bob})
        forbidden[action] = (r.status_code, r.json().get("detail", {}).get("code"))
        assert r.status_code == 404, r.json()
    r = client.post("/api/memory/candidates/%s/edit" % a["candidate_id"], headers=hb,
                    json={"user_id": bob, "content": "篡改"})
    forbidden["edit"] = (r.status_code, r.json().get("detail", {}).get("code"))
    assert r.status_code == 404, r.json()

    # SecureRoute 外层：body 里塞他人 user_id 直接 403 OWNER_MISMATCH
    r = client.post("/api/memory/candidates", headers=hb,
                    json={"user_id": alice, "content": "伪装成 alice"})
    forbidden["body_user_id"] = (r.status_code, r.json().get("detail", {}).get("code"))
    assert r.status_code == 403, r.json()
    assert r.json()["detail"]["code"] == "OWNER_MISMATCH"

    # 记忆层纵深防御：直接调函数带错 owner 也必须拒
    with pytest.raises(confirm.PermissionDenied):
        confirm.get_candidate(a["candidate_id"], bob)
    with pytest.raises(confirm.PermissionDenied):
        confirm.reject_candidate(a["candidate_id"], bob)

    assert confirm.get_candidate(a["candidate_id"], alice)["review_status"] == "pending"
    assert confirm.list_candidates(bob) == []
    print("U1-V4-EVIDENCE cross_user=%s body_user_id=%s alice_candidate_still=%s bob_list=%s"
          % (forbidden, forbidden["body_user_id"],
             confirm.get_candidate(a["candidate_id"], alice)["review_status"],
             confirm.list_candidates(bob)))


# ------------------------------------------------------------------ U1-V5
def test_u1_v5_flag_off_byte_for_byte_baseline(client, monkeypatch):
    u1, h1 = account(client, "v5_off")
    u2, _ = account(client, "v5_baseline")
    monkeypatch.setenv("USE_MEMORY_CONFIRM", "0")
    monkeypatch.setenv("USE_FIVE_LAYER", "0")
    store.reset_five_layer_cache()

    kwargs = dict(type="chat", content="基线逐字节对照", summary="s", facts=["f"],
                  emotion={"e": 1}, importance=6.0, place_tag="p", time_tag="t", cbt={"c": 2})
    with _RawSessionLocal() as db:
        result = confirm.submit_candidate(db, u1, **kwargs)
        baseline = store.add_memory(db, u2, **kwargs)
        snap_confirm = _snapshot(db.get(MemoryEntry, result["memory_id"]))
        snap_baseline = _snapshot(baseline)

    assert confirm.confirmation_enabled() is False
    assert result["mode"] == "baseline" and result["candidate_id"] is None
    assert confirm.list_candidates(u1) == []
    assert snap_confirm == snap_baseline
    assert store.five_layer_current("基线逐字节对照") == []  # 两层都未写五层

    # 开关关时 API 也直接落 Fact（不产生 pending 候选）
    r = client.post("/api/memory/candidates", headers=h1,
                    json={"user_id": u1, "content": "关开关的 API 写入"})
    assert r.status_code == 200 and r.json()["review_status"] == "auto", r.json()
    assert client.get("/api/memory/candidates", headers=h1).json()["count"] == 0
    assert any(x.content == "关开关的 API 写入" for x in _db_rows(u1))

    print("U1-V5-EVIDENCE flag=off snapshot_confirm=%s snapshot_baseline=%s equal=%s candidates=%d"
          % (snap_confirm, snap_baseline, snap_confirm == snap_baseline,
             client.get("/api/memory/candidates", headers=h1).json()["count"]))
