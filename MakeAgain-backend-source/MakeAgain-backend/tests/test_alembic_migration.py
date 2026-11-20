"""PR-A behaviour evidence: opt-in Alembic route (USE_ALEMBIC=1).

All scenarios run in a fresh subprocess so DATABASE_URL / USE_ALEMBIC are read
from scratch, exactly like a real first start or an operator-triggered upgrade.
The default path (USE_ALEMBIC unset) is asserted separately to stay legacy.
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "legacy-schema.sql"
REVISION = "b0001_baseline"
STAMP = "2024-01-02 03:04:05.000000"


def run_backend(db_path: Path, *, use_alembic: bool, extra_code: str = ""):
    env = dict(os.environ, DATABASE_URL="sqlite:///" + str(db_path), PYTHONDONTWRITEBYTECODE="1")
    env.pop("USE_ALEMBIC", None)
    if use_alembic:
        env["USE_ALEMBIC"] = "1"
    code = "from memory.db import init_db; init_db()\n" + extra_code
    return subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True
    )


def assert_ok(result):
    assert result.returncode == 0, result.stderr


def make_legacy_db(path: Path) -> None:
    """B07-pre schema plus rows that must survive the upgrade verbatim."""
    with sqlite3.connect(path) as db:
        db.executescript(FIXTURE.read_text(encoding="utf-8"))

        def insert(table, values):
            for _, name, kind, required, _, _ in db.execute("PRAGMA table_info(" + table + ")"):
                if required and name not in values:
                    values[name] = "{}" if kind == "JSON" else 0 if kind in ("BOOLEAN", "FLOAT", "INTEGER") else ""
            names = ",".join(values)
            marks = ",".join("?" for _ in values)
            db.execute(f"INSERT INTO {table}({names}) VALUES ({marks})", tuple(values.values()))

        insert("users", dict(id="legacy", membership="member", created_at=STAMP))
        insert(
            "user_states",
            dict(user_id="legacy", relationship_type="relative", subject_name="奶奶",
                 created_at=STAMP, updated_at=STAMP),
        )
        insert("diary_notes", dict(id="old-1", user_id="legacy", content="  原始文本\n\n保留  ", ts=STAMP))
        insert("diary_notes", dict(id="old-2", user_id="legacy", content="second", ts=STAMP))
        db.commit()


def columns(path: Path, table: str) -> set[str]:
    with sqlite3.connect(path) as db:
        return {row[1] for row in db.execute(f"PRAGMA table_info({table})")}


def snapshot(path: Path):
    """Everything that must not change between two consecutive init_db() calls."""
    with sqlite3.connect(path) as db:
        tables = sorted(r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'"))
        counts = [(t, db.execute(f"SELECT count(*) FROM {t}").fetchone()[0]) for t in tables]
        cols = {t: sorted(columns(path, t)) for t in tables}
        alembic = db.execute("SELECT version_num FROM alembic_version ORDER BY version_num").fetchall()
        legacy = db.execute("SELECT version, applied_at FROM schema_migrations ORDER BY version").fetchall()
        diary = db.execute(
            "SELECT id, content, companion_id, is_current, canonical_diary_id FROM diary_notes ORDER BY id"
        ).fetchall()
    return counts, cols, alembic, legacy, diary


def test_fresh_db_with_alembic_has_full_schema_seed_and_version(tmp_path):
    path = tmp_path / "fresh.sqlite3"
    assert_ok(run_backend(path, use_alembic=True))

    from memory.db import _SQLITE_ADD_COLUMNS

    with sqlite3.connect(path) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"users", "user_states", "companions", "diary_notes", "welcome_letters",
                "alembic_version", "schema_migrations"} <= tables
        assert db.execute("SELECT version_num FROM alembic_version").fetchall() == [(REVISION,)]
        assert db.execute("SELECT version FROM schema_migrations").fetchall() == [("B07",)]
        assert db.execute("SELECT count(*) FROM welcome_letters").fetchone()[0] == 1
    for (table, column), _ in _SQLITE_ADD_COLUMNS.items():
        assert column in columns(path, table), (table, column)


def test_legacy_upgrade_preserves_rows_and_business_backfill(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    make_legacy_db(path)
    assert_ok(run_backend(path, use_alembic=True))

    with sqlite3.connect(path) as db:
        assert db.execute("SELECT membership, membership_source FROM users WHERE id='legacy'").fetchone() == \
               ("member", "legacy")
        cid = db.execute("SELECT current_companion_id FROM user_states WHERE user_id='legacy'").fetchone()[0]
        assert cid
        assert db.execute("SELECT count(*) FROM companions WHERE id=?", (cid,)).fetchone()[0] == 1
        rows = db.execute(
            "SELECT id, content, companion_id, is_current, canonical_diary_id FROM diary_notes ORDER BY id"
        ).fetchall()
        assert db.execute("SELECT version_num FROM alembic_version").fetchall() == [(REVISION,)]
    assert len(rows) == 2
    assert rows[0][1] == "  原始文本\n\n保留  "  # original text/whitespace untouched
    assert all(r[2] == cid for r in rows)
    assert sum(r[3] for r in rows) == 1
    assert all(r[4] == "old-2" for r in rows)
    assert "cbt" in columns(path, "memory_stream")  # post-B07 column added


def test_init_db_twice_is_idempotent(tmp_path):
    path = tmp_path / "repeat.sqlite3"
    make_legacy_db(path)
    assert_ok(run_backend(path, use_alembic=True))
    first = snapshot(path)
    assert_ok(run_backend(path, use_alembic=True))
    assert snapshot(path) == first


def test_previously_migrated_db_is_stamped_without_rebuilding(tmp_path):
    """A DB already carrying legacy schema_migrations B07 must be stamped, not re-run."""
    path = tmp_path / "already.sqlite3"
    assert_ok(run_backend(path, use_alembic=False))
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO users(id, membership, created_at) VALUES ('kept','free',?)", (STAMP,))
        db.commit()

    extra = """
from unittest.mock import patch
from memory import db as mdb
with patch.object(mdb.Base.metadata, 'create_all', side_effect=AssertionError('create_all RE-RUN')):
    mdb.init_db()
print('STAMPED_WITHOUT_CREATE_ALL')
"""
    result = run_backend(path, use_alembic=True, extra_code=extra)
    assert_ok(result)
    assert "STAMPED_WITHOUT_CREATE_ALL" in result.stdout
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT version_num FROM alembic_version").fetchall() == [(REVISION,)]
        assert db.execute("SELECT count(*) FROM users WHERE id='kept'").fetchone()[0] == 1


def test_switch_off_stays_on_legacy_path(tmp_path):
    path = tmp_path / "off.sqlite3"
    assert_ok(run_backend(path, use_alembic=False))
    with sqlite3.connect(path) as db:
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "alembic_version" not in tables
        assert "schema_migrations" in tables
        assert db.execute("SELECT count(*) FROM welcome_letters").fetchone()[0] == 1


def test_revision_is_queryable():
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini")))
    assert REVISION in script.get_heads()
    revision = script.get_revision(REVISION)
    assert revision.down_revision is None
    assert revision.module is not None


def test_alembic_current_reports_head_after_upgrade(tmp_path):
    path = tmp_path / "current.sqlite3"
    assert_ok(run_backend(path, use_alembic=True))
    env = dict(os.environ, DATABASE_URL="sqlite:///" + str(path), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "current"], cwd=ROOT, env=env,
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert REVISION in result.stdout


def test_downgrade_is_explicit_noop():
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(Config(str(ROOT / "alembic.ini")))
    revision = script.get_revision(REVISION)
    assert revision.module.downgrade() is None
