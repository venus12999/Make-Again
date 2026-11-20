"""R3 · 溯源存储替换验收（mem0 SQLiteManager 移植）。

覆盖：
1. 开关默认关闭 → add_memory 行为不变、不建溯源库、get_history 返回 []；
2. 开关打开 → write→update→delete 的真实 SQLite 事件链（ADD/UPDATE/DELETE）；
3. `_migrate_history_table` 对旧库（多 group-chat 列）安全迁移且不丢数据；
4. `save_messages`/`get_last_messages` 会话环形缓冲保留最近 10 条；
5. 溯源写失败时 fallback 到基线路径，add_memory 不抛异常。
"""
import hashlib
import os
import sqlite3
import time

import pytest

from memory import store
from memory.db import _RawSessionLocal
from memory.provenance_store import SQLiteManager

HISTORY_FIELDS = {
    "id", "memory_id", "old_memory", "new_memory", "event",
    "created_at", "updated_at", "is_deleted", "actor_id", "role",
}


@pytest.fixture
def prov_env(monkeypatch, tmp_path):
    """独立的溯源 SQLite 文件 + 干净的连接缓存；开关默认关闭。"""
    path = str(tmp_path / "provenance.sqlite3")
    monkeypatch.setenv("PROVENANCE_DB_PATH", path)
    monkeypatch.delenv("USE_PROVENANCE", raising=False)
    store.reset_provenance_cache()
    yield path
    store.reset_provenance_cache()


# ---------- 1. 开关默认关闭 ----------

def test_switch_off_does_not_touch_provenance(prov_env):
    assert store.provenance_enabled() is False
    db = _RawSessionLocal()
    try:
        entry = store.add_memory(db, "u-off", content="养猫")
        db.refresh(entry)
        assert entry.content == "养猫"
        assert store.get_history(entry.id) == []
        # 开关关：连接都不建立，溯源库文件不应存在
        assert not os.path.exists(prov_env)
        # 且原有读取路径照常工作
        assert [m.content for m in store.list_memories(db, "u-off")] == ["养猫"]
    finally:
        db.close()


def test_falsey_switch_values_are_off(prov_env, monkeypatch):
    for value in ("", "0", "false", "no", "OFF"):
        monkeypatch.setenv("USE_PROVENANCE", value)
        assert store.provenance_enabled() is False, value
    monkeypatch.setenv("USE_PROVENANCE", "1")
    assert store.provenance_enabled() is True


# ---------- 2. write → update → delete 事件链 ----------

def test_history_chain_add_update_delete(prov_env, monkeypatch):
    monkeypatch.setenv("USE_PROVENANCE", "1")
    assert store.provenance_enabled() is True
    db = _RawSessionLocal()
    try:
        entry = store.add_memory(db, "u-chain", content="养了一只猫")
        mid = str(entry.id)

        # mem0 get_history 的排序键含 DATETIME(updated_at)（秒级），
        # 事件间跨 1 秒保证顺序由数据本身而非 SQLite 行序决定。
        time.sleep(1.05)
        updated = store.prov_update_memory(
            db, mid, content="猫去世了", actor_id="u-chain", role="user"
        )
        assert updated.content == "猫去世了"
        time.sleep(1.05)
        assert store.prov_delete_memory(db, mid, actor_id="u-chain", role="user") is True

        history = store.get_history(mid)
        assert [h["event"] for h in history] == ["ADD", "UPDATE", "DELETE"]
        assert history[0]["old_memory"] is None
        assert history[0]["new_memory"] == "养了一只猫"
        assert history[1]["old_memory"] == "养了一只猫"
        assert history[1]["new_memory"] == "猫去世了"
        assert history[2]["old_memory"] == "猫去世了"
        assert history[2]["new_memory"] is None
        assert history[2]["is_deleted"] is True
        assert all(h["actor_id"] == "u-chain" for h in history[:2])
        assert history[2]["actor_id"] == "u-chain"
        assert all(h["role"] == "user" for h in history)
        # mem0 _update_memory 保留 created_at，只推进 updated_at
        assert history[0]["created_at"] == history[1]["created_at"] == history[2]["created_at"]
        assert history[0]["updated_at"] < history[1]["updated_at"] < history[2]["updated_at"]

        # mem0 main.py:2077 的 md5 hash 落点
        assert store._get_provenance_store().get_memory_hash(mid) == \
            hashlib.md5("猫去世了".encode()).hexdigest()

        # 真实 SQLite 查询（绕过 store.get_history 的 API，直接读表）
        conn = sqlite3.connect(prov_env)
        rows = conn.execute(
            "SELECT event, old_memory, new_memory, is_deleted FROM history "
            "WHERE memory_id=? ORDER BY created_at ASC, DATETIME(updated_at) ASC",
            (mid,),
        ).fetchall()
        conn.close()
        assert [r[0] for r in rows] == ["ADD", "UPDATE", "DELETE"]
        assert rows[2][3] == 1
    finally:
        db.close()


def test_history_table_schema_is_exactly_10_fields(prov_env):
    mgr = store._get_provenance_store()
    cols = {r[1] for r in mgr.connection.execute("PRAGMA table_info(history)")}
    assert cols == HISTORY_FIELDS
    assert len(cols) == 10


# ---------- 3. 旧库安全迁移不丢数据（用现有 client fixture） ----------

def test_migrate_old_history_table_preserves_data(prov_env, client):
    """模拟旧库：history 多两个 group-chat 列；迁移后交集数据必须保住。"""
    conn = sqlite3.connect(prov_env)
    conn.execute(
        """
        CREATE TABLE history (
            id TEXT PRIMARY KEY, memory_id TEXT, old_memory TEXT, new_memory TEXT,
            event TEXT, created_at DATETIME, updated_at DATETIME, is_deleted INTEGER,
            actor_id TEXT, role TEXT, session_id TEXT, convo_id TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO history VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        ("old-row", "mem-old", None, "它以前会喵喵叫", "ADD",
         "2026-01-01T00:00:00", "2026-01-01T00:00:00", 0, "legacy-user", "user", "s1", "c1"),
    )
    conn.commit()
    conn.close()

    mgr = SQLiteManager(prov_env)  # __init__ 自动触发 _migrate_history_table
    cols = {r[1] for r in mgr.connection.execute("PRAGMA table_info(history)")}
    assert cols == HISTORY_FIELDS  # 新表严格 10 字段
    rows = mgr.get_history("mem-old")
    assert len(rows) == 1
    assert rows[0]["new_memory"] == "它以前会喵喵叫"
    assert rows[0]["actor_id"] == "legacy-user"
    assert rows[0]["role"] == "user"
    assert rows[0]["is_deleted"] is False
    # 旧表已清理，没有残留
    assert mgr.connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='history_old'"
    ).fetchone() is None
    mgr.close()


# ---------- 4. 会话消息环形缓冲 ----------

def test_messages_ring_buffer_keeps_last_ten(prov_env, monkeypatch):
    monkeypatch.setenv("USE_PROVENANCE", "1")
    # 分次写入让 created_at 严格递增（mem0 原实现同一次调用内共享时间戳）
    for i in range(15):
        assert store.save_provenance_messages(
            [{"role": "user", "content": f"m{i}", "name": None}], "session-A"
        ) is True
    msgs = store.get_last_provenance_messages("session-A", 10)
    assert len(msgs) == 10
    assert [m["content"] for m in msgs] == [f"m{i}" for i in range(5, 15)]
    assert all(m["role"] == "user" for m in msgs)


def test_messages_api_off_returns_empty(prov_env):
    assert store.save_provenance_messages([{"role": "user", "content": "x"}], "s") is False
    assert store.get_last_provenance_messages("s") == []


# ---------- 5. 溯源失败 fallback ----------

def test_provenance_failure_falls_back_to_baseline(prov_env, monkeypatch, tmp_path):
    monkeypatch.setenv("USE_PROVENANCE", "1")
    # 目录路径无法作为 SQLite 文件打开 → SQLiteManager 构造失败
    monkeypatch.setenv("PROVENANCE_DB_PATH", str(tmp_path))
    store.reset_provenance_cache()
    db = _RawSessionLocal()
    try:
        entry = store.add_memory(db, "u-fallback", content="兜底也要能写")
        assert entry.content == "兜底也要能写"
        assert len(store.list_memories(db, "u-fallback")) == 1
        assert store.get_history(entry.id) == []
    finally:
        db.close()
        store.reset_provenance_cache()
