# -*- coding: utf-8 -*-
"""UC-3 · 记忆删除与导出 —— 行为验收（不是「全绿即通过」）。

U3-V1 导出 JSON 含全部类别 + provenance
U3-V2 单条删除后：业务库 / vec0 / bm25 / 实体反链 / 五层 current 全部真实查询无残留
U3-V3 全量删除回执数字准确（planned / deleted / verification 三方对齐）
U3-V4 越权删除被拒（他人 id / user_id 伪造 / 无 token / 导出不串数据 / 全删不动他人）
U3-V5 删除后不可恢复但有审计（业务库物理删 + provenance DELETE + admin_audit + trace）

真实命令：
    cd MakeAgain-backend
    python -m pytest tests/test_memory_export.py -q -s

每条断言前都 print 原始证据；失败时输出即为排查依据。
"""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select, text

from api.memory_export import router as memory_export_router
from memory import entity_graph as eg
from memory import export as mexport
from memory import store as mstore
from memory import vector_index as vi
from memory.b07_models import Audit
from memory.db import _RawSessionLocal, SessionLocal
from memory.models import (
    DailyMood, DiaryNote, EchoEntry, EmotionNode, MemoryEntry, Portrait, utcnow,
)

import main as main_module


# --------------------------------------------------------------------------- fixtures
@pytest.fixture()
def api_client():
    """在真实 main.app 上注册本工作包路由（lead 后续集成后不会重复注册）。"""
    app = main_module.app
    if not any(getattr(r, "path", "").startswith("/api/memory/") for r in app.routes):
        app.include_router(memory_export_router)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def layered(monkeypatch, tmp_path):
    """四层机制全开 + 全部落隔离文件，返回 (会话, tmp_path)。"""
    monkeypatch.setenv("USE_PROVENANCE", "1")
    monkeypatch.setenv("USE_FIVE_LAYER", "1")
    monkeypatch.setenv("USE_DUAL_RECALL", "1")
    monkeypatch.setenv("USE_ENTITY_PORTRAIT", "1")
    monkeypatch.setenv("PROVENANCE_DB_PATH", str(tmp_path / "prov.sqlite3"))
    monkeypatch.setenv("FIVE_LAYER_STORE_DIR", str(tmp_path / "five-layer"))
    monkeypatch.setenv("DUAL_RECALL_INDEX_DB", str(tmp_path / "idx.sqlite3"))
    monkeypatch.setenv("ENTITY_GRAPH_DB_PATH", str(tmp_path / "entities.sqlite3"))

    mstore.reset_provenance_cache()
    mstore.reset_five_layer_cache()
    vi.reset_index()
    for conn in list(eg._connections.values()):
        try:
            conn.close()
        except Exception:
            pass
    eg._connections.clear()

    db = SessionLocal()
    try:
        yield db, tmp_path
    finally:
        db.close()
        mstore.reset_provenance_cache()
        mstore.reset_five_layer_cache()
        vi.reset_index()
        for conn in list(eg._connections.values()):
            try:
                conn.close()
            except Exception:
                pass
        eg._connections.clear()


# --------------------------------------------------------------------------- helpers
def _account(client, name: str):
    r = client.post("/api/auth/register", json={"username": name, "password": "secure-test-password"})
    assert r.status_code == 200, r.text
    j = r.json()
    u = j["user_id"]
    h = {"Authorization": "Bearer " + j["token"]}
    cr = client.post("/api/consent", headers=h, json={"user_id": u, "version": 1})
    assert cr.status_code == 200, cr.text
    s = client.get("/api/state", headers=h).json()
    return u, h, s


def _seed_memories(db, user_id: str, n: int = 1, text: str = "豆豆陪我散步") -> list[MemoryEntry]:
    entries = []
    for i in range(n):
        e = mstore.add_memory(
            db, user_id, type="chat", content=f"{text} 第{i + 1}次",
            summary=f"第{i + 1}次回忆", facts=[{"fact": f"事实-{i + 1}"}],
        )
        eg.upsert_entity("豆豆", "name", str(e.id), user_id)
        entries.append(e)
    return entries


def _seed_side_data(db, user_id: str) -> None:
    mstore.upsert_portrait(db, user_id, "user", {"name": "小美"}, status="confirmed")
    db.add(EmotionNode(user_id=user_id, trigger="散步", emotion="平静", intensity=0.7))
    db.add(DailyMood(user_id=user_id, date_key="2026-11-21", mood="平静", score=70))
    db.add(DiaryNote(user_id=user_id, content="今天很好"))
    db.add(EchoEntry(user_id=user_id, content="那句话还在"))
    db.commit()


def _live(db, memory_id: str) -> bool:
    return db.execute(
        select(MemoryEntry).where(MemoryEntry.id == str(memory_id)).execution_options(unscoped=True)
    ).scalars().first() is not None


# =========================================================================== U3-V1
def test_u3_v1_export_all_categories_with_provenance(layered, api_client):
    db, _ = layered
    u, h, _ = _account(api_client, "u3-v1")
    entries = _seed_memories(db, u, n=2)
    _seed_side_data(db, u)
    mstore.prov_update_memory(db, str(entries[0].id), content="豆豆陪我散步（更正）", actor_id=u)

    r = api_client.get("/api/memory/export", headers=h)
    assert r.status_code == 200, r.text
    body = r.json()

    print("\n[U3-V1] GET /api/memory/export status:", r.status_code)
    print("[U3-V1] top-level keys:", sorted(body))
    print("[U3-V1] categories:", body["categories"])
    print("[U3-V1] source ids:", [e["id"] for e in body["source"]])
    print("[U3-V1] fact sample:", body["fact"][:2])
    print("[U3-V1] provenance of first memory:",
          json.dumps(body["provenance_graph"][str(entries[0].id)], ensure_ascii=False)[:600])

    assert body["schema_version"] == "1.0"
    assert body["user_id"] == u
    for key in ("source", "candidate", "fact", "portrait", "emotion",
                "mood", "diary", "echo", "chat_message", "provenance_graph"):
        assert key in body, f"导出缺类别 {key}"
    assert len(body["source"]) == 2
    assert body["categories"]["memory"] == 2
    assert body["categories"]["fact"] == 2
    assert body["categories"]["portrait"] == 1
    assert body["categories"]["emotion"] == 1
    assert len(body["portrait"]) == 1 and len(body["emotion"]) == 1
    # provenance 必须是真实事件链（含 ADD + UPDATE），不是空壳字段
    prov0 = body["provenance_graph"][str(entries[0].id)]
    events = [x["event"] for x in prov0["history"]]
    assert "ADD" in events and "UPDATE" in events, events
    assert all("provenance" in e for e in body["source"])
    assert body["fact"][0]["provenance"]["history"], "fact 也要带 provenance"
    # 不得串到他人数据
    assert all(e["user_id"] == u for e in body["source"])


# =========================================================================== U3-V2
def test_u3_v2_single_delete_cascade_no_residue(layered, api_client, monkeypatch):
    # 本用例断言 64 维表诊断；语义embedding已默认开，显式关以保持原契约
    monkeypatch.setenv("USE_SEMANTIC_EMBEDDING", "0")
    db, _ = layered
    u, h, _ = _account(api_client, "u3-v2")
    entry = _seed_memories(db, u, n=1)[0]
    mid = str(entry.id)
    # 制造 supersede 后继：删除必须沿图一起软删
    mstore.prov_update_memory(db, mid, content="豆豆老了", actor_id=u)
    succ = mstore._get_five_layer_adapter().record(mid)["superseded_by"]

    idx = vi.get_index_for(db)
    before = {
        "business": _live(db, mid),
        "diag": idx.diagnostics(mid),
        "entities": eg.list_entities(u),
        "record": mstore._get_five_layer_adapter().record(mid),
        "current": mstore.five_layer_current("豆豆"),
    }
    r = api_client.request("DELETE", f"/api/memory/{mid}", headers=h)
    assert r.status_code == 200, r.text
    body = r.json()

    # ---- 真实查询验证（不是只读回执自述）----
    fresh = _RawSessionLocal()
    try:
        orm_row = fresh.execute(
            select(MemoryEntry).where(MemoryEntry.id == mid).execution_options(unscoped=True)
        ).scalars().first()
        audits = fresh.scalars(select(Audit).where(Audit.target_id == mid)).all()
    finally:
        fresh.close()
    after_diag = vi.get_index_for(db).diagnostics(mid)
    after_entities = eg.list_entities(u)
    adapter = mstore._get_five_layer_adapter()
    after_rec = adapter.record(mid)
    after_succ = adapter.record(succ) if succ else {}
    after_current = mstore.five_layer_current("豆豆")
    history = mstore.get_history(mid)

    print("\n[U3-V2] memory id:", mid, "supersede successor:", succ)
    print("[U3-V2] BEFORE diag:", before["diag"], "entities:", before["entities"])
    print("[U3-V2] BEFORE five_layer_current:",
          [(x["name"], x["status"]) for x in before["current"]])
    print("[U3-V2] DELETE response:", json.dumps(body, ensure_ascii=False)[:700])
    print("[U3-V2] AFTER orm_row:", orm_row, "diag:", after_diag)
    print("[U3-V2] AFTER entities:", after_entities)
    print("[U3-V2] AFTER record:", after_rec)
    print("[U3-V2] AFTER successor record:", after_succ)
    print("[U3-V2] AFTER five_layer_current:", after_current)
    print("[U3-V2] provenance history events:",
          [(x["event"], x["is_deleted"]) for x in history])
    print("[U3-V2] admin_audit rows:", [(a.actor_id, a.action, a.target_id) for a in audits])

    assert before["business"] and before["diag"]["vec_rows"] >= 1
    assert before["diag"]["bm25_rows"] >= 1 and before["entities"]
    assert body["deleted"] is True and body["cascade"]["business_row"] is True
    assert orm_row is None, "业务库行必须物理删除"
    assert after_diag["vec_rows"] == 0, "vec0 不得残留"
    assert after_diag["bm25_rows"] == 0, "bm25 元数据不得残留"
    assert all(mid not in e["linked_memory_ids"] for e in after_entities), "实体反链不得残留"
    assert body["cascade"]["entity_backlinks_clean"] is True
    # 五层：主记录和后继都 invalid_at；current 不返任何相关记录
    assert after_rec["invalid_at"], "五层主记录必须软删"
    assert succ and after_succ["invalid_at"], "supersede 后继必须沿图一起软删"
    assert after_current == []
    assert body["cascade"]["five_layer_current_clean"] is True
    assert any(x["event"] == "DELETE" and x["is_deleted"] for x in history)
    assert body["cascade"]["provenance_delete_event"] is True
    assert audits and audits[0].actor_id == u


# ============================================================================== U3-V2b
def test_u3_v2b_user_control_not_gated_by_experimental_switches(monkeypatch, tmp_path, api_client):
    """地基默认可用：四开关全关时，删除依然真删业务库+vec0+bm25+实体反链。"""
    for key in ("USE_PROVENANCE", "USE_FIVE_LAYER", "USE_DUAL_RECALL", "USE_ENTITY_PORTRAIT",
                "USE_SEMANTIC_EMBEDDING", "USE_QUERY_INTENT"):
        monkeypatch.setenv(key, "0")
    monkeypatch.setenv("DUAL_RECALL_INDEX_DB", str(tmp_path / "idx-off.sqlite3"))
    monkeypatch.setenv("ENTITY_GRAPH_DB_PATH", str(tmp_path / "entities-off.sqlite3"))
    mstore.reset_provenance_cache()
    mstore.reset_five_layer_cache()
    vi.reset_index()
    for conn in list(eg._connections.values()):
        try:
            conn.close()
        except Exception:
            pass
    eg._connections.clear()

    u, h, _ = _account(api_client, "u3-v2b")
    db = SessionLocal()
    try:
        entry = mstore.add_memory(db, u, type="chat", content="默认开关也要能删")
        mid = str(entry.id)
        # 手动建索引（USE_DUAL_RECALL 关闭时 add_memory 不会自动建）
        from memory.recall import index_memory

        index_memory(entry, db=db)
        eg.upsert_entity("豆豆", "name", mid, u)
        before = vi.get_index_for(db).diagnostics(mid)
        r = api_client.request("DELETE", f"/api/memory/{mid}", headers=h)
        after = vi.get_index_for(db).diagnostics(mid)
        entities = eg.list_entities(u)
        alive = db.execute(
            select(MemoryEntry).where(MemoryEntry.id == mid).execution_options(unscoped=True)
        ).scalars().first()
        print("\n[U3-V2b] switches off; before diag:", before, "after diag:", after)
        print("[U3-V2b] DELETE status:", r.status_code, "orm row:", alive, "entities:", entities)
        assert r.status_code == 200, r.text
        assert before["vec_rows"] == 1 and before["bm25_rows"] == 1
        assert alive is None
        assert after["vec_rows"] == 0 and after["bm25_rows"] == 0
        assert all(mid not in e["linked_memory_ids"] for e in entities)
    finally:
        db.close()
        vi.reset_index()
        for conn in list(eg._connections.values()):
            try:
                conn.close()
            except Exception:
                pass
        eg._connections.clear()


# =========================================================================== U3-V3
def test_u3_v3_full_delete_receipt_counts(layered, api_client):
    db, _ = layered
    u, h, _ = _account(api_client, "u3-v3")
    _seed_memories(db, u, n=3)
    _seed_side_data(db, u)
    planned = mexport.count_user_memory_data(db, u)

    # 另一用户的数据不能被顺带删除
    u2, h2, _ = _account(api_client, "u3-v3-other")
    other = _seed_memories(db, u2, n=2)
    other_ids = [str(e.id) for e in other]

    r = api_client.request("DELETE", "/api/memory/all", headers=h)
    assert r.status_code == 200, r.text
    receipt = r.json()

    fresh = _RawSessionLocal()
    try:
        after_u = mexport.count_user_memory_data(fresh, u)
        after_u2 = mexport.count_user_memory_data(fresh, u2)
        alive_other = fresh.execute(
            select(MemoryEntry).where(MemoryEntry.id.in_(other_ids)).execution_options(unscoped=True)
        ).scalars().all()
        audit = fresh.scalars(select(Audit).where(Audit.action == "DELETE /api/memory/all")).first()
    finally:
        fresh.close()

    print("\n[U3-V3] planned (pre-delete):", planned)
    print("[U3-V3] receipt deleted:", receipt["deleted"])
    print("[U3-V3] receipt verification:", receipt["verification"])
    print("[U3-V3] receipt index/five-layer:",
          receipt["index"], len(receipt["five_layer_soft_deleted"]))
    print("[U3-V3] after delete counts (user A):", after_u)
    print("[U3-V3] after delete counts (user B, must be intact):", after_u2)
    print("[U3-V3] audit metadata:", audit.metadata_safe if audit else None)

    assert receipt["planned"] == planned
    for category, count in planned.items():
        assert receipt["deleted"].get(category) == count, f"{category} 回执数字不符"
    assert receipt["verification"] == after_u
    assert all(v == 0 for v in after_u.values()), f"删后本用户必须归零: {after_u}"
    assert receipt["index"]["vector_and_bm25_memories_cleared"] >= 3
    assert len(receipt["five_layer_soft_deleted"]) >= 3
    assert after_u2["memory"] == 2 and len(alive_other) == 2, "不得删除他人记忆"
    assert audit and audit.actor_id == u and audit.metadata_safe["scope"] == "all"


# =========================================================================== U3-V4
def test_u3_v4_cross_user_and_forged_owner_denied(layered, api_client):
    db, _ = layered
    u, h, _ = _account(api_client, "u3-v4-a")
    u2, h2, _ = _account(api_client, "u3-v4-b")
    victim = _seed_memories(db, u2, n=1)[0]
    vid = str(victim.id)

    r_foreign = api_client.request("DELETE", f"/api/memory/{vid}", headers=h)
    r_forged = api_client.request("DELETE", f"/api/memory/{vid}", headers=h, json={"user_id": u2})
    r_anon = api_client.request("DELETE", f"/api/memory/{vid}")
    own_export = api_client.get("/api/memory/export", headers=h)
    api_client.request("DELETE", "/api/memory/all", headers=h)  # A 全删也不能碰 B

    fresh = _RawSessionLocal()
    try:
        victim_alive = _live(fresh, vid)
    finally:
        fresh.close()

    print("\n[U3-V4] foreign id DELETE:", r_foreign.status_code, r_foreign.json())
    print("[U3-V4] forged user_id DELETE:", r_forged.status_code, r_forged.json())
    print("[U3-V4] anonymous DELETE:", r_anon.status_code, r_anon.json())
    print("[U3-V4] A export status:", own_export.status_code,
          "contains victim id:", vid in own_export.text)
    print("[U3-V4] victim alive after A full-delete:", victim_alive)

    assert r_foreign.status_code == 404 and r_foreign.json()["code"] == "NOT_FOUND"
    assert r_forged.status_code == 403 and r_forged.json()["code"] == "OWNER_MISMATCH"
    assert r_anon.status_code == 401 and r_anon.json()["code"] == "UNAUTHENTICATED"
    assert own_export.status_code == 200 and vid not in own_export.text
    assert victim_alive is True


# =========================================================================== U3-V5
def test_u3_v5_irreversible_with_audit(layered, api_client):
    db, _ = layered
    u, h, _ = _account(api_client, "u3-v5")
    entry = _seed_memories(db, u, n=1, text="豆豆陪我走过雨雪")[0]
    mid = str(entry.id)

    r = api_client.request("DELETE", f"/api/memory/{mid}", headers=h)
    assert r.status_code == 200, r.text

    fresh = _RawSessionLocal()
    try:
        orm_row = fresh.execute(
            select(MemoryEntry).where(MemoryEntry.id == mid).execution_options(unscoped=True)
        ).scalars().first()
        audits = fresh.scalars(select(Audit).where(Audit.target_id == mid)).all()
    finally:
        fresh.close()
    after_export = api_client.get("/api/memory/export", headers=h).json()
    history = mstore.get_history(mid)
    trace = mstore.five_layer_trace(mid)
    record = mstore._get_five_layer_adapter().record(mid)
    restore_routes = [
        getattr(route, "path", "") for route in api_client.app.routes
        if "memory" in getattr(route, "path", "")
        and any(word in getattr(route, "path", "") for word in ("restore", "undelete", "recover"))
    ]

    print("\n[U3-V5] business row after delete:", orm_row)
    print("[U3-V5] id in export:", mid in json.dumps(after_export, ensure_ascii=False))
    print("[U3-V5] provenance DELETE event:",
          [(x["event"], x["is_deleted"], x["actor_id"]) for x in history if x["event"] == "DELETE"])
    print("[U3-V5] five-layer record (soft delete keeps trace):", record)
    print("[U3-V5] trace text (audit evidence):", trace.get("text", "")[:120])
    print("[U3-V5] audit rows:",
          [(a.actor_id, a.action, a.target_id, a.metadata_safe) for a in audits])
    print("[U3-V5] restore/undelete routes:", restore_routes)

    # 不可恢复：业务库物理删除；导出不再出现；没有任何恢复端点
    assert orm_row is None
    assert mid not in json.dumps(after_export, ensure_ascii=False)
    assert restore_routes == []
    # 有审计：provenance DELETE 事件 + admin_audit 元数据（且不含正文/PII）
    delete_events = [x for x in history if x["event"] == "DELETE"]
    assert delete_events and delete_events[-1]["is_deleted"] is True
    assert delete_events[-1]["actor_id"] == u
    assert audits and audits[0].actor_id == u and audits[0].action == "DELETE /api/memory/{id}"
    audit_blob = json.dumps(audits[0].metadata_safe, ensure_ascii=False)
    assert "豆豆陪我走过雨雪" not in audit_blob, "审计不得写记忆正文"
    # 五层只保留软删记录与 trace 作审计，current 不再返回
    assert record["invalid_at"]
    assert "豆豆" in trace.get("text", "")
    assert mstore.five_layer_current("豆豆") == []
