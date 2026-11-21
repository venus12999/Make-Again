"""WP-2 · 隔离守卫接主链路行为验收（V1~V4 + 越权/边界必含）。

全绿不是标准：每条 V 用 pytest -s 打印真实命令可复现的原始证据，并断言拒绝行为。
被测对象：memory/jobs.py(开关) + memory/async_write.py(worker 接线)
          + api/memory_scope.py(scoped 端点)。不改 routes.py（lead 注册）。

真实命令：
  pytest -s tests/test_scope_wiring.py                 # V1~V4 + 边界
  pytest -q                                            # V5 全量
"""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from api.memory_scope import router as scope_router
from conftest import account
from memory import isolation as iso
from memory.b07_models import Job
from memory.db import SessionLocal, _RawSessionLocal
from memory.jobs import scope_guard_enabled
from memory.models import Companion, MemoryEntry, UserState, utcnow
from memory.scope import current_db

import admin_cli


# 独立 app 只挂 scoped 路由；main.py 的注册由 lead 负责，本文件不改任何路由源码。
_scope_app = FastAPI()
_scope_app.include_router(scope_router)


@pytest.fixture
def scope_client():
    with TestClient(_scope_app) as c:
        yield c


def _create_memory_job(user_id: str, companion_id: str, dedupe: str,
                       message: str = "我的小狗豆豆今天陪我散步，我很开心",
                       extractor: str | None = "memory.pet.extract") -> str:
    with _RawSessionLocal() as db:
        job = Job(kind="memory", user_id=user_id, companion_id=companion_id,
                  dedupe_key=dedupe,
                  payload={"message": message, "type": "chat", "extractor": extractor})
        db.add(job)
        db.commit()
        return job.id


def _job_status(job_id: str) -> str:
    with _RawSessionLocal() as db:
        return db.get(Job, job_id).status


def _call_do_write(user_id: str, companion_id: str, *, extractor=None):
    """模拟 admin_cli worker 的受管上下文，直接调用被接线的 _do_write。"""
    db = _RawSessionLocal()
    db.info.update(managed=True, borrowed=True, user_id=user_id, companion_id=companion_id)
    ctx = current_db.set(db)
    try:
        from memory.async_write import _do_write
        _do_write(user_id, "测试写入", "chat", extractor)
    finally:
        current_db.reset(ctx)
        db.info["borrowed"] = False
        db.close()


# ---------------------------------------------------------------------------
# V1 · 用户 A worker 写入带 A companion scope，B 通过 scoped 端点查不到
# ---------------------------------------------------------------------------
def test_v1_worker_writes_scoped_memory_b_cannot_see(client, scope_client, monkeypatch):
    monkeypatch.delenv("USE_SCOPE_GUARD", raising=False)
    assert scope_guard_enabled() is True, "USE_SCOPE_GUARD 默认必须开"

    ua, ha, sa = account(client, "scope-v1-A")
    ub, hb, sb = account(client, "scope-v1-B")
    ca, cb = sa["current_companion_id"], sb["current_companion_id"]

    # spy：证明 worker 真的调用了隔离守卫（不是只建了库层能力没人用）
    calls: list[tuple] = []
    original_build = iso.build_scope

    def spy_build(db, uid, cid, **kw):
        calls.append((uid, cid))
        return original_build(db, uid, cid, **kw)

    monkeypatch.setattr(iso, "build_scope", spy_build)

    jid = _create_memory_job(ua, ca, "v1-worker-A")
    assert admin_cli.run_one() is True
    status = _job_status(jid)

    db = SessionLocal()
    try:
        rows = list(db.execute(
            select(MemoryEntry).where(MemoryEntry.user_id == ua)
            .execution_options(unscoped=True)
        ).scalars().all())
    finally:
        db.close()
    assert status == "completed" and len(rows) == 1
    assert rows[0].companion_id == ca
    aid = str(rows[0].id)

    ra = scope_client.get("/api/memory/scoped/recall",
                          params={"query": "豆豆", "companion_id": ca}, headers=ha)
    rb = scope_client.get("/api/memory/scoped/recall",
                          params={"query": "豆豆", "companion_id": cb}, headers=hb)
    ja, jb = ra.json(), rb.json()
    ids_a = {h["id"] for h in ja["hits"]}
    ids_b = {h["id"] for h in jb["hits"]}

    print("\n[SCOPE-V1] guard default:", scope_guard_enabled(), "| job status:", status)
    print("[SCOPE-V1] worker scope calls:", calls)
    print("[SCOPE-V1] A memory:", aid, "companion_id:", rows[0].companion_id)
    print("[SCOPE-V1] A recall:", json.dumps(ja, ensure_ascii=False))
    print("[SCOPE-V1] B recall:", json.dumps(jb, ensure_ascii=False))

    assert (ua, ca) in calls, "worker 必须用当前 companion scope 走守卫"
    assert ja["path"] == "isolation.scoped_recall" and aid in ids_a
    assert jb["path"] == "isolation.scoped_recall" and aid not in ids_b
    assert jb["count"] == 0, "B 的 scoped 端点不得查到 A 的记忆"


# ---------------------------------------------------------------------------
# V2 · 越权：assert_scope 返回 SCOPE_DENIED；scoped 端点 403 且不泄漏存在性
# ---------------------------------------------------------------------------
def test_v2_cross_owner_assert_scope_and_endpoint_deny(client, scope_client):
    ua, ha, sa = account(client, "scope-v2-A")
    ub, hb, sb = account(client, "scope-v2-B")
    ca, cb = sa["current_companion_id"], sb["current_companion_id"]

    db = SessionLocal()
    try:
        ma = iso.scoped_add_memory(db, ua, ca, content="A 私密：对桂花糕过敏", importance=9.0)
        mb = iso.scoped_add_memory(db, ub, cb, content="B 的秘密：汤圆去世了", importance=9.0)
    finally:
        db.close()

    # 库层守卫：双向越权（A 读 B、B 读 A 都被拒）
    d1, d2 = SessionLocal(), SessionLocal()
    try:
        with pytest.raises(iso.ScopeDenied) as ea:
            iso.assert_scope(d1, ua, ca, mb.id)
        with pytest.raises(iso.ScopeDenied) as eb:
            iso.assert_scope(d2, ub, cb, ma.id)
    finally:
        d1.close()
        d2.close()

    # 端点越权：A 拿 B 的 memory_id（query 里带 A 自己的 companion，通过 owner 校验）
    cross = scope_client.get(f"/api/memory/scoped/{mb.id}",
                             params={"companion_id": ca}, headers=ha)
    # 不存在的 id：必须与越权同码同文案（无存在性 Oracle）
    missing = scope_client.get("/api/memory/scoped/not-exist-id",
                               params={"companion_id": ca}, headers=ha)
    # 伪造 owner：B 的 token + A 的 companion_id → SecureRoute 归属拒绝
    forged = scope_client.get(f"/api/memory/scoped/{mb.id}",
                              params={"companion_id": ca}, headers=hb)

    cj, mj, fj = cross.json(), missing.json(), forged.json()
    print("\n[SCOPE-V2] assert_scope A->B:", ea.value.code, "| B->A:", eb.value.code)
    print("[SCOPE-V2] cross-owner endpoint:", cross.status_code,
          json.dumps(cj, ensure_ascii=False))
    print("[SCOPE-V2] missing endpoint:", missing.status_code,
          json.dumps(mj, ensure_ascii=False))
    print("[SCOPE-V2] forged owner endpoint:", forged.status_code,
          json.dumps(fj, ensure_ascii=False))

    assert ea.value.code == "SCOPE_DENIED" and eb.value.code == "SCOPE_DENIED"
    assert cross.status_code == 403 and cj["code"] == "SCOPE_DENIED"
    assert "汤圆去世" not in cross.text, "越权响应不得回显他人记忆内容"
    assert missing.status_code == 403 and missing.json()["code"] == "SCOPE_DENIED"
    assert cj["code"] == mj["code"] and cj["message"] == mj["message"], \
        "存在/不存在必须同响应，杜绝存在性 Oracle"
    assert forged.status_code in (403, 404) and fj["code"] in ("COMPANION_NOT_FOUND", "OWNER_MISMATCH")


# ---------------------------------------------------------------------------
# V3 · 未 consent 默认拒绝（worker + scoped 端点）
# ---------------------------------------------------------------------------
def test_v3_no_consent_denies_worker_and_endpoint(client, scope_client, monkeypatch):
    monkeypatch.setenv("USE_SCOPE_GUARD", "1")
    u, h, s = account(client, "scope-v3-noconsent")
    cid = s["current_companion_id"]
    with _RawSessionLocal() as db:
        st = db.get(UserState, u)
        st.consent_accepted_at = None
        db.commit()

    # worker：守卫在 LLM/落库之前就拒绝，且错误码明确
    with pytest.raises(iso.ScopeDenied) as exc:
        _call_do_write(u, cid)
    print("\n[SCOPE-V3] worker denied code:", exc.value.code)

    db = SessionLocal()
    try:
        n = db.query(MemoryEntry).filter(MemoryEntry.user_id == u).count()
    finally:
        db.close()

    # 端点：默认拒绝（不是返回空数据的"软失败"）
    r = scope_client.get("/api/memory/scoped/recall",
                         params={"query": "任何", "companion_id": cid}, headers=h)
    print("[SCOPE-V3] endpoint:", r.status_code, json.dumps(r.json(), ensure_ascii=False))
    print("[SCOPE-V3] memory rows after worker attempt:", n)

    assert exc.value.code == "CONSENT_REQUIRED"
    assert n == 0, "未 consent 时 worker 不得落任何记忆"
    assert r.status_code == 403 and r.json()["code"] == "CONSENT_REQUIRED"


# ---------------------------------------------------------------------------
# V4 · 开关关：逐字节回退（守卫代码零执行，走接线前基线路径）
# ---------------------------------------------------------------------------
def test_v4_switch_off_byte_for_byte_fallback(client, scope_client, monkeypatch):
    monkeypatch.delenv("USE_SCOPE_GUARD", raising=False)
    assert scope_guard_enabled() is True
    monkeypatch.setenv("USE_SCOPE_GUARD", "0")
    assert scope_guard_enabled() is False

    def boom(*args, **kwargs):
        raise AssertionError("USE_SCOPE_GUARD=0 时守卫代码不得执行")

    monkeypatch.setattr(iso, "build_scope", boom)

    u, h, s = account(client, "scope-v4-off")
    cid = s["current_companion_id"]
    jid = _create_memory_job(u, cid, "v4-off-worker", extractor=None)
    assert admin_cli.run_one() is True
    status = _job_status(jid)

    # 端点也走基线：recall.recall 被调用，守卫路径不被调用
    monkeypatch.setattr("memory.recall.recall", lambda *a, **k: [])
    r = scope_client.get("/api/memory/scoped/recall",
                         params={"query": "豆豆", "companion_id": cid}, headers=h)
    jr = r.json()
    print("\n[SCOPE-V4] guard enabled:", scope_guard_enabled(), "| job status:", status)
    print("[SCOPE-V4] endpoint:", r.status_code, json.dumps(jr, ensure_ascii=False))

    assert status == "completed", "关开关时 worker 必须不受守卫影响，照常完成"
    assert jr["guard"] is False and jr["path"] == "recall.recall"


# ---------------------------------------------------------------------------
# 边界（必含）· 归档 Journey 写入拒绝 / scope 内省只含本人 / 默认开时开关键三态
# ---------------------------------------------------------------------------
def test_extra_archived_journey_worker_denied(client, monkeypatch):
    monkeypatch.setenv("USE_SCOPE_GUARD", "1")
    u, _, s = account(client, "scope-extra-archived")
    cid = s["current_companion_id"]
    with _RawSessionLocal() as db:
        db.get(Companion, cid).archived_at = utcnow()
        db.commit()
    with pytest.raises(iso.ScopeDenied) as exc:
        _call_do_write(u, cid)
    print("\n[SCOPE-EXTRA] archived worker denied:", exc.value.code)
    assert exc.value.code == "ARCHIVED_READ_ONLY"


def test_extra_switch_tristate(monkeypatch):
    monkeypatch.delenv("USE_SCOPE_GUARD", raising=False)
    assert scope_guard_enabled() is True, "未设置=默认开"
    for v in ("0", "false", "NO", "Off"):
        monkeypatch.setenv("USE_SCOPE_GUARD", v)
        assert scope_guard_enabled() is False, f"{v} 应识别为关"
    monkeypatch.setenv("USE_SCOPE_GUARD", "1")
    assert scope_guard_enabled() is True
    print("\n[SCOPE-EXTRA] switch tristate OK: default=on, 0/false/NO/Off=off, 1=on")


def test_extra_scope_introspection_owner_only(client, scope_client):
    u, h, s = account(client, "scope-extra-introspect")
    cid = s["current_companion_id"]
    r = scope_client.get("/api/memory/scoped/scope", params={"companion_id": cid}, headers=h)
    jr = r.json()
    print("\n[SCOPE-EXTRA] scope introspection:", r.status_code, json.dumps(jr, ensure_ascii=False))
    assert r.status_code == 200
    assert jr["scope"]["user_id"] == u and jr["scope"]["companion_id"] == cid
    assert cid in jr["scope"]["allowed_companion_ids"]
