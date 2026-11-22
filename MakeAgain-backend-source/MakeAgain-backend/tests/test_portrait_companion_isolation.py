"""U-018 GAP-V3 修复证据：画像按 companion_id 隔离（向后兼容 API）。

口径：全部离线，不读 key、不发网络。真实命令 + 输出见 docs/PROGRESS.md 本轮台账。

- B-V1：传 companion_id 只读该 Journey 画像（读写都带归属，互不串味）。
- B-V2：不传 companion_id 时行为完全不变（兼容所有既有 3 参数调用方）。
- B-V3：新 Journey 主链读取不到旧 companion 的对象画像（原 V3 xfail 场景等价复现）。
- B-V4：存量 SQLite 的 portraits 表缺 companion_id 时，init_db 自动补列且不破坏旧数据。
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from sqlalchemy import select

from memory import isolation as iso
from memory import store
from memory.db import _RawSessionLocal
from memory.models import Portrait, utcnow

ROOT = Path(__file__).resolve().parents[1]


def _mkuser_with_two_journeys(uid: str) -> tuple[str, str]:
    """建用户 + consent + 旧 Journey A（已归档）/ 新 Journey B（当前），返回 (A id, B id)。"""
    db = _RawSessionLocal()
    try:
        store.get_or_create_user(db, uid, loss_type="breakup")
        st = store.get_or_create_user_state(db, uid)
        st.consent_accepted_at = utcnow()
        db.commit()
        a = store.create_companion(db, uid, relationship_type="breakup", subject_name="小夏")
        store.archive_companion(db, a.id, ritual="scene_transition")
        b = store.create_companion(db, uid, relationship_type="pet", subject_name="团子")
        return a.id, b.id
    finally:
        db.close()


# ---------------------------------------------------------------------------
# B-V1 · 传 companion_id 只读该画像
# ---------------------------------------------------------------------------
def test_b_v1_get_portrait_filters_by_companion_id():
    db = _RawSessionLocal()
    uid = "b-v1-filter"
    a, b = _mkuser_with_two_journeys(uid)
    try:
        store.upsert_portrait(db, uid, "object", {"称呼": "小夏", "关系": "前任"}, companion_id=a)
        store.upsert_portrait(db, uid, "object", {"称呼": "团子", "关系": "猫"}, companion_id=b)
        store.upsert_portrait(db, uid, "user", {"昵称": "林薇"}, companion_id=a)

        oa = store.get_portrait(db, uid, "object", companion_id=a)
        ob = store.get_portrait(db, uid, "object", companion_id=b)
        print("\n[B-V1] A(object) =", oa)
        print("[B-V1] B(object) =", ob)
        assert oa == {"称呼": "小夏", "关系": "前任"}
        assert ob == {"称呼": "团子", "关系": "猫"}

        # 同一 companion 再写一次只更新自己那条（version+1），不串到另一个 Journey
        row = store.upsert_portrait(db, uid, "object", {"称呼": "小夏(新版)"}, companion_id=a)
        assert row.companion_id == a and row.version == 2
        assert store.get_portrait(db, uid, "object", companion_id=a) == {"称呼": "小夏(新版)"}
        assert store.get_portrait(db, uid, "object", companion_id=b) == {"称呼": "团子", "关系": "猫"}
        print("[B-V1] 覆写后 A =", store.get_portrait(db, uid, "object", companion_id=a),
              "| B =", store.get_portrait(db, uid, "object", companion_id=b))
    finally:
        db.close()


# ---------------------------------------------------------------------------
# B-V2 · 不传参数行为不变（向后兼容）
# ---------------------------------------------------------------------------
def test_b_v2_omitting_companion_id_keeps_legacy_behavior():
    db = _RawSessionLocal()
    uid = "b-v2-compat"
    try:
        # 老调用方式：无 companion（老库/脚本/评测的既有形态）
        store.upsert_portrait(db, uid, "object", {"称呼": "旧称呼"}, status="draft")
        store.upsert_portrait(db, uid, "user", {"昵称": "林薇"}, status="draft")

        # 参照旧实现：user_id + kind 取 version 最大一行，完全不看归属
        legacy_row = db.execute(
            select(Portrait).where(Portrait.user_id == uid, Portrait.kind == "object")
            .order_by(Portrait.version.desc()).limit(1)
        ).scalar_one_or_none()
        obj = store.get_portrait(db, uid, "object")
        print("\n[B-V2] 旧实现参照 =", legacy_row.content, "| 3参数 get_portrait =", obj)
        assert obj == legacy_row.content == {"称呼": "旧称呼"}

        # 3参数 upsert 仍按 user_id+kind 覆盖同一行、version 递增、companion_id 保持 None
        row = store.upsert_portrait(db, uid, "object", {"称呼": "新称呼"}, status="confirmed")
        assert row.version == 2 and row.companion_id is None
        assert store.get_portrait(db, uid, "object") == {"称呼": "新称呼"}
        print("[B-V2] 覆写后 version =", row.version, "companion_id =", row.companion_id,
              "| 内容 =", store.get_portrait(db, uid, "object"))
    finally:
        db.close()


# ---------------------------------------------------------------------------
# B-V3 · 新 Journey 主链不读旧 companion 画像（原 V3 xfail 场景等价复现）
# ---------------------------------------------------------------------------
def test_b_v3_new_journey_does_not_read_old_companion_portrait():
    db = _RawSessionLocal()
    uid = "b-v3-isolation"
    a, b = _mkuser_with_two_journeys(uid)
    try:
        # 旧 Journey 报告落库（interview._persist_report 修复后的同款：带 companion_id）
        store.upsert_portrait(db, uid, "object", {"称呼": "小夏", "关系": "前任"}, companion_id=a)
        store.upsert_portrait(db, uid, "user", {"昵称": "林薇"}, companion_id=a)
        # 新 Journey 报告落库：新对象画像
        store.upsert_portrait(db, uid, "object", {"称呼": "团子", "关系": "猫"}, companion_id=b)

        obj_new = store.get_portrait(db, uid, "object", companion_id=b)
        obj_old = store.get_portrait(db, uid, "object", companion_id=a)
        print("\n[B-V3] 新 Journey object =", obj_new)
        print("[B-V3] 旧 Journey object =", obj_old)
        assert "小夏" not in str(obj_new), "新 Journey 不应读到旧对象小夏的画像"
        assert obj_new["称呼"] == "团子"
        assert "小夏" in str(obj_old), "旧 Journey 自己的画像仍可读（未被误删）"

        # 与隔离层口径一致：新 Journey scoped_portraits 不含旧画像
        scoped = iso.scoped_portraits(db, uid, b)
        print("[B-V3] 新 Journey scoped_portraits =", [(p.companion_id, p.content) for p in scoped])
        assert all("小夏" not in str(p.content) for p in scoped)
        assert all(p.companion_id == b for p in scoped)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# B-V4 · 存量库自动补列，旧数据不破坏（子进程跑真实 init_db）
# ---------------------------------------------------------------------------
def test_b_v4_legacy_portraits_table_auto_adds_companion_id_preserving_rows(tmp_path):
    db_path = tmp_path / "legacy_portraits.sqlite3"
    with sqlite3.connect(db_path) as c:
        c.execute(
            "CREATE TABLE portraits (id TEXT PRIMARY KEY, user_id TEXT, kind TEXT, "
            "content JSON, status TEXT, version INTEGER, updated_at DATETIME)"
        )
        c.execute(
            "INSERT INTO portraits(id,user_id,kind,content,status,version,updated_at) "
            "VALUES (?,?,?,?,?,?,?)",
            ("p-old", "u-legacy", "object", '{"称呼":"小夏"}', "draft", 1, "2024-01-02 03:04:05"),
        )
        c.commit()

    env = dict(os.environ, DATABASE_URL="sqlite:///" + str(db_path))
    env.pop("USE_ALEMBIC", None)
    code = (
        "from memory.db import init_db, engine, _RawSessionLocal\n"
        "from sqlalchemy import text\n"
        "from memory import store\n"
        "init_db()\n"
        "with engine.connect() as c:\n"
        "    cols = [r[1] for r in c.execute(text('PRAGMA table_info(portraits)'))]\n"
        "    print('COLS=' + repr(cols))\n"
        "    print('ROW=' + repr(c.execute(text(\"SELECT content FROM portraits WHERE id='p-old'\")).scalar()))\n"
        "db = _RawSessionLocal()\n"
        "print('GET_LEGACY=' + repr(store.get_portrait(db, 'u-legacy', 'object')))\n"
        "print('GET_SCOPED=' + repr(store.get_portrait(db, 'u-legacy', 'object', companion_id='c-new')))\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True
    )
    print("\n[B-V4] stdout:\n" + r.stdout + "[B-V4] stderr:\n" + r.stderr)
    assert r.returncode == 0, r.stderr
    assert "'companion_id'" in r.stdout, "存量 portraits 表应自动补 companion_id 列"
    assert "小夏" in r.stdout, "旧画像数据行必须原样保留"
    assert "GET_LEGACY={'称呼': '小夏'}" in r.stdout, "不传 companion_id 仍读到旧数据（兼容）"
    assert "GET_SCOPED={}" in r.stdout, "传新 companion_id 时旧无归属行不应被读出，且不崩"
