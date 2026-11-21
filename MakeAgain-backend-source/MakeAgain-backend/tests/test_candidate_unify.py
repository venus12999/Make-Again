"""F2 · Candidate 并入五层一体化验收（F2-V1 ~ F2-V4）。

跑法：cd MakeAgain-backend && python -m pytest tests/test_candidate_unify.py -q -s
每条用例打印 `F2-Vn-EVIDENCE ...`，可直接作为台账证据。

口径：
  F2-V1 新 Candidate 写入五层库 index.db（真实 sqlite3 查询，证明不是独立 sqlite）；
  F2-V2 旧登记册数据读时迁移 + confirm/reject/edit 仍可用；
  F2-V3 confirm/reject/edit 现有行为不变（本文件冒烟 + 复用 test_memory_confirm.py）；
  F2-V4 全量 pytest（命令证据，见交付报告）。
"""
from __future__ import annotations

import pathlib
import sqlite3
import uuid

import pytest
from sqlalchemy import select

from api.memory_confirm import router as confirm_router
from main import app
from memory import confirm, store
from memory.db import _RawSessionLocal
from memory.models import MemoryEntry

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
def isolate_f2(tmp_path, monkeypatch):
    monkeypatch.setenv("USE_MEMORY_CONFIRM", "1")
    monkeypatch.setenv("USE_FIVE_LAYER", "1")
    monkeypatch.setenv("FIVE_LAYER_STORE_DIR", str(tmp_path / "five-layer"))
    # 老登记册路径指向一个不存在的文件；新代码不应为写入而创建它（迁移只读）。
    monkeypatch.setenv("MEMORY_CONFIRM_DB_PATH", str(tmp_path / "legacy-candidates.sqlite3"))
    store.reset_five_layer_cache()
    confirm.reset_confirm_cache()
    yield
    store.reset_five_layer_cache()
    confirm.reset_confirm_cache()


def _five_layer_db_path() -> pathlib.Path:
    adapter = store._get_five_layer_adapter()
    return pathlib.Path(adapter._store.layout.index_db)


def _query(db_path: pathlib.Path, sql: str, params=()) -> list[dict]:
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(sql, params).fetchall()]
    finally:
        connection.close()


def _business_rows(user_id: str) -> list[MemoryEntry]:
    with _RawSessionLocal() as db:
        return list(db.scalars(select(MemoryEntry).where(MemoryEntry.user_id == user_id)).all())


def _seed_legacy(user_id: str, content: str, *, status: str = "pending") -> dict:
    """在老独立登记册里种一条历史数据，模拟迁移前的存量。"""
    path = pathlib.Path(confirm.default_candidate_db_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    connection.execute(
        """CREATE TABLE IF NOT EXISTS candidates (
               id TEXT PRIMARY KEY, user_id TEXT NOT NULL, content TEXT NOT NULL,
               edited_content TEXT, review_status TEXT NOT NULL, memory_id TEXT,
               record_id TEXT, successor_id TEXT, reason TEXT,
               created_at TEXT NOT NULL, reviewed_at TEXT)"""
    )
    row = {
        "id": uuid.uuid4().hex, "user_id": str(user_id), "content": content,
        "edited_content": None, "review_status": status, "memory_id": None,
        "record_id": None, "successor_id": None, "reason": None,
        "created_at": "2026-01-01T00:00:00", "reviewed_at": None,
    }
    connection.execute(
        "INSERT INTO candidates (id, user_id, content, edited_content, review_status, "
        "memory_id, record_id, successor_id, reason, created_at, reviewed_at) "
        "VALUES (:id, :user_id, :content, :edited_content, :review_status, "
        ":memory_id, :record_id, :successor_id, :reason, :created_at, :reviewed_at)",
        row,
    )
    connection.commit()
    connection.close()
    return row


# ------------------------------------------------------------------ F2-V1
def test_f2_v1_candidate_in_five_layer_db(client):
    uid, h = account(client, "v1_unify")
    created = client.post("/api/memory/candidates", headers=h,
                          json={"user_id": uid, "content": "一体化候选：养了只猫叫团子",
                                "type": "chat"})
    assert created.status_code == 200, created.json()
    body = created.json()
    cid = body["candidate_id"]
    assert body["review_status"] == "pending", body

    db_path = _five_layer_db_path()
    assert db_path.exists(), db_path
    names = {t["name"] for t in _query(db_path, "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "candidates" in names and "records" in names, names  # 同库：candidates 与五层 records 并存

    rows = _query(db_path, "SELECT * FROM candidates WHERE id = ?", (cid,))
    assert len(rows) == 1 and rows[0]["review_status"] == "pending", rows
    assert rows[0]["user_id"] == uid, rows

    rec_cols = [c["name"] for c in _query(db_path, "PRAGMA table_info(records)")]
    assert "review_status" in rec_cols and "candidate_id" in rec_cols, rec_cols

    legacy = pathlib.Path(confirm.default_candidate_db_path())
    assert not legacy.exists(), legacy  # 不再创建独立 sqlite（消除两套存储）

    assert _business_rows(uid) == [], _business_rows(uid)
    assert store.five_layer_current("团子") == []  # pending 不落 Fact → 不可召回

    print("F2-V1-EVIDENCE five_layer_db=%s tables=%s candidate=%s records_ext_cols=%s legacy_exists=%s business=%d current=%s"
          % (db_path, sorted(names), rows[0],
             [c for c in rec_cols if c in ("review_status", "candidate_id")],
             legacy.exists(), len(_business_rows(uid)), store.five_layer_current("团子")))


# ------------------------------------------------------------------ F2-V2
def test_f2_v2_legacy_registry_migrated_and_actionable(client):
    uid, h = account(client, "v2_legacy")
    leg_confirm = _seed_legacy(uid, "老登记册：团子喜欢晒太阳")
    leg_reject = _seed_legacy(uid, "老登记册：拒绝我")
    leg_edit = _seed_legacy(uid, "老登记册：待纠错")

    # 读：老行可读，且读时回填到五层库候选表
    got = confirm.get_candidate(leg_confirm["id"], uid)
    assert got["content"] == leg_confirm["content"], got
    db_path = _five_layer_db_path()
    migrated = _query(db_path, "SELECT * FROM candidates WHERE id = ?", (leg_confirm["id"],))
    assert len(migrated) == 1 and migrated[0]["review_status"] == "pending", migrated

    # list：把该用户全部老行回填，统一从五层库查
    listed = confirm.list_candidates(uid)
    listed_ids = {c["id"] for c in listed}
    assert {leg_confirm["id"], leg_reject["id"], leg_edit["id"]} <= listed_ids, listed_ids

    # 老数据 confirm → 落 Fact
    cf = client.post("/api/memory/candidates/%s/confirm" % leg_confirm["id"],
                     headers=h, json={"user_id": uid})
    assert cf.status_code == 200, cf.json()
    assert cf.json()["candidate"]["review_status"] == "confirmed", cf.json()
    assert any(x.content == leg_confirm["content"] for x in _business_rows(uid))

    # 老数据 reject → 不落 Fact
    rj = client.post("/api/memory/candidates/%s/reject" % leg_reject["id"],
                     headers=h, json={"user_id": uid, "reason": "旧数据拒绝"})
    assert rj.status_code == 200, rj.json()
    assert rj.json()["candidate"]["review_status"] == "rejected", rj.json()

    # 老数据 edit（pending → 仍 pending，改内容不改状态）
    ed = client.post("/api/memory/candidates/%s/edit" % leg_edit["id"],
                     headers=h, json={"user_id": uid, "content": "老登记册：已纠正"})
    assert ed.status_code == 200, ed.json()
    assert ed.json()["candidate"]["review_status"] == "pending", ed.json()
    assert ed.json()["candidate"]["edited_content"] == "老登记册：已纠正", ed.json()

    statuses = {r["id"]: r["review_status"]
                for r in _query(db_path, "SELECT id, review_status FROM candidates")}
    assert statuses[leg_confirm["id"]] == "confirmed"
    assert statuses[leg_reject["id"]] == "rejected"
    assert statuses[leg_edit["id"]] == "pending"

    print("F2-V2-EVIDENCE legacy_read=%s migrated_into=%s statuses=%s edited=%s"
          % (got["content"], db_path, statuses, ed.json()["candidate"]["edited_content"]))


# ------------------------------------------------------------------ F2-V3
def test_f2_v3_confirm_reject_edit_behavior_unchanged(client):
    uid, h = account(client, "v3_unify")
    a = client.post("/api/memory/candidates", headers=h,
                    json={"user_id": uid, "content": "团子的名字是团子"}).json()
    b = client.post("/api/memory/candidates", headers=h,
                    json={"user_id": uid, "content": "永不召回的拒绝文本"}).json()

    cf = client.post("/api/memory/candidates/%s/confirm" % a["candidate_id"],
                     headers=h, json={"user_id": uid}).json()
    rj = client.post("/api/memory/candidates/%s/reject" % b["candidate_id"],
                     headers=h, json={"user_id": uid}).json()
    assert cf["candidate"]["review_status"] == "confirmed", cf
    assert rj["candidate"]["review_status"] == "rejected", rj

    old_id = cf["candidate"]["record_id"]
    adapter = store._get_five_layer_adapter()
    assert adapter.present_in_current(old_id) is True
    assert store.five_layer_current("永不召回的拒绝文本") == []

    ed = client.post("/api/memory/candidates/%s/edit" % a["candidate_id"], headers=h,
                     json={"user_id": uid, "content": "团子的名字是小团子"}).json()
    assert ed["candidate"]["review_status"] == "edited", ed
    successor = ed["superseded_by"]
    assert old_id and successor, (old_id, ed)
    assert adapter.present_in_current(old_id) is False
    assert adapter.record(successor).get("abstract") == "团子的名字是小团子"
    assert [x.content for x in _business_rows(uid)] == ["团子的名字是小团子"]

    print("F2-V3-EVIDENCE confirm=%s reject=%s old=%s invalid_after_edit=%s successor=%s business=%s"
          % (cf["candidate"]["review_status"], rj["candidate"]["review_status"], old_id,
             adapter.present_in_current(old_id), successor,
             [x.content for x in _business_rows(uid)]))
