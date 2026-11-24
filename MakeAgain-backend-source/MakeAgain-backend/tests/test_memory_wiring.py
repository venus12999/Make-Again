"""H2 接线验收（D-073）：Fact 校验运维 CLI + checkpoint 接后台写入链。

口径：synthetic-handwritten 构造数据 + 本地 SQLite；零 LLM / 零真 key / 零网络。
验证的是「接线是否生效 + 开关关时逐字节回退」，不是产品效果（启发式非真值判定）。
状态：已验证（含证据）——接线行为；部分验证——未接用户确认流/crisis 门禁。
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import text

from memory import async_write
from memory.db import _RawSessionLocal, engine

BACKEND = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- fixtures

@pytest.fixture(autouse=True)
def _checkpoint_isolation(isolation, monkeypatch):
    """conftest 的 isolation 先跑（drop_all/init_db），再清掉不在 Base.metadata 里的 checkpoint 表。"""
    def drop():
        with engine.begin() as c:
            c.execute(text("DROP TABLE IF EXISTS session_checkpoints"))
    drop()
    monkeypatch.delenv("USE_CHECKPOINT", raising=False)
    yield
    drop()


@pytest.fixture
def wired(monkeypatch):
    """隔离 _do_write 的外部依赖（LLM/队列/回声壁），不动被测接线本身。"""
    class _DummyClient:
        mock = True

    monkeypatch.setattr(async_write, "LLMClient", lambda: _DummyClient())
    import memory.jobs as jobs
    monkeypatch.setattr(jobs, "enqueue", lambda *a, **k: None)
    monkeypatch.setattr(jobs, "flush_intents", lambda *a, **k: None)
    from agent import echo as echo_mod
    monkeypatch.setattr(echo_mod, "maybe_capture_echo", lambda *a, **k: None)
    monkeypatch.setenv("USE_ENTITY_PORTRAIT", "0")
    monkeypatch.setenv("USE_MEMORY_CONFIRM", "0")


def _turn(fact: str) -> dict:
    return {
        "facts": [{"kind": "user", "fact": fact, "confidence": 0.9, "linked_memory_ids": []}],
        "emotion": {"emotion": "想念", "valence": -0.5, "arousal": 0.4},
        "summary": fact[:120],
        "importance": 5.0,
        "place_tag": "家里",
        "time_tag": "最近",
        "cbt": None,
    }


def _extractor(fact: str):
    return lambda client, text: _turn(fact)


def _has_ckpt_table() -> bool:
    with engine.connect() as c:
        return bool(c.execute(text(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='session_checkpoints'"
        )).scalar())


def _memory_ids(user_id: str) -> list[str]:
    with engine.connect() as c:
        return [r[0] for r in c.execute(
            text("SELECT id FROM memory_stream WHERE user_id=:u"), {"u": user_id}
        ).all()]


def _load(user_id: str, session_id: str):
    from memory.checkpoint import load_checkpoint
    db = _RawSessionLocal()
    try:
        return load_checkpoint(db, user_id, session_id)
    finally:
        db.close()


# ===========================================================================
# 任务 1：Fact 校验运维 CLI（admin_cli.py fact-check --user-id <uid>）
# ===========================================================================

def _cli_env(db_path: Path, tmp_path: Path) -> dict:
    env = os.environ.copy()
    env.update(
        DATABASE_URL=f"sqlite:///{db_path}",
        ENVIRONMENT="test",
        MOCK_LLM="true", MOCK_WECHAT="true", MOCK_SPEECH="true", MOCK_VISION="true", MOCK_STORAGE="true",
        MEDIA_SIGNING_KEY="isolated-test-key-01234567890123456789",
        STORAGE_LOCAL_DIR=str(tmp_path / "media"),
        PUBLIC_BASE_URL="http://testserver",
    )
    return env


def _run_admin(env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(BACKEND / "admin_cli.py"), *args],
        env=env, capture_output=True, text=True, timeout=120,
    )


def _seed_cli_db(db_path: Path) -> None:
    con = sqlite3.connect(str(db_path))
    try:
        con.execute(
            "INSERT INTO memory_stream (id,user_id,type,facts,content,importance,ts) "
            "VALUES ('e-cli','u-cli','chat',?,'',5.0,'2026-11-01T00:00:00')",
            (json.dumps([{"kind": "object", "fact": "用户的猫叫橘子", "confidence": 0.9}],
                        ensure_ascii=False),),
        )
        con.execute(
            "INSERT INTO chat_messages (id,user_id,session_id,role,content,ts) "
            "VALUES (1,'u-cli','','user','它叫橘子，陪了我八年','2026-11-01T01:00:00')"
        )
        con.execute(
            "INSERT INTO chat_messages (id,user_id,session_id,role,content,ts) "
            "VALUES (2,'u-cli','','user','它不叫橘子了，我改叫它团子','2026-11-02T01:00:00')"
        )
        # 零命中用户：有事实也有消息，但无词形重叠/无纠错语气 → 应输出 0 candidates
        con.execute(
            "INSERT INTO memory_stream (id,user_id,type,facts,content,importance,ts) "
            "VALUES ('e-empty','u-empty','chat',?,'',5.0,'2026-11-01T00:00:00')",
            (json.dumps([{"fact": "用户喜欢喝咖啡"}], ensure_ascii=False),),
        )
        con.execute(
            "INSERT INTO chat_messages (id,user_id,session_id,role,content,ts) "
            "VALUES (3,'u-empty','','user','今天天气不错','2026-11-02T01:00:00')"
        )
        con.commit()
    finally:
        con.close()


def _snapshot_cli_db(db_path: Path):
    con = sqlite3.connect(str(db_path))
    try:
        return (
            con.execute("SELECT * FROM chat_messages ORDER BY id").fetchall(),
            con.execute("SELECT * FROM memory_stream ORDER BY id").fetchall(),
        )
    finally:
        con.close()


def test_fact_check_cli_candidates_zero_hit_and_readonly(tmp_path):
    """CLI 真跑：能报候选（fact id + 证据 msg id + 判据）、零命中空结果、不改任何 Fact。"""
    db_path = tmp_path / "cli.sqlite3"
    env = _cli_env(db_path, tmp_path)

    # 先让 CLI 自身的 init_db() 在干净临时库建好全部表（否则直接 INSERT 会缺列）
    init = _run_admin(env, "migrate")
    assert init.returncode == 0, init.stderr[-600:]

    _seed_cli_db(db_path)
    before = _snapshot_cli_db(db_path)

    hit = _run_admin(env, "fact-check", "--user-id", "u-cli")
    assert hit.returncode == 0, hit.stderr[-600:]
    assert "1 candidate(s)" in hit.stdout
    assert "fact_id=e-cli:0" in hit.stdout
    assert "evidence_msg_id=2" in hit.stdout
    assert "kind=conflict" in hit.stdout and "rule=correction" in hit.stdout
    assert "改叫它团子" in hit.stdout  # reason 之外的可追溯信息（判据上下文）

    empty = _run_admin(env, "fact-check", "--user-id", "u-empty")
    assert empty.returncode == 0, empty.stderr[-600:]
    assert "0 candidates" in empty.stdout
    assert "fact_id=" not in empty.stdout

    after = _snapshot_cli_db(db_path)
    assert after == before, "fact-check CLI 必须只读：chat_messages / memory_stream 逐行不变"


# ===========================================================================
# 任务 2：checkpoint 接后台写入链（USE_CHECKPOINT 默认关）
# ===========================================================================

def test_schedule_memory_write_switch_off_byte_for_byte(monkeypatch):
    """①开关关：schedule_memory_write 的 enqueue 调用/payload 与接线前逐字节一致。"""
    monkeypatch.delenv("USE_CHECKPOINT", raising=False)
    calls: list = []
    import memory.jobs as jobs
    monkeypatch.setattr(jobs, "enqueue", lambda *a, **k: calls.append((a, k)))

    async_write.schedule_memory_write("u-off", "你好")
    assert calls == [(
        ("memory", "u-off", {"message": "你好", "type": "chat", "extractor": None}),
        {},
    )]
    assert async_write.checkpoint_enabled() is False


def test_schedule_memory_write_threads_session_id_into_job_payload(monkeypatch):
    """session_id 是纯增量可选参数：传入才进 payload，透传给后台 job。"""
    monkeypatch.delenv("USE_CHECKPOINT", raising=False)
    calls: list = []
    import memory.jobs as jobs
    monkeypatch.setattr(jobs, "enqueue", lambda *a, **k: calls.append((a, k)))

    async_write.schedule_memory_write("u", "你好", session_id="s-9")
    payload = calls[0][0][2]
    assert payload["session_id"] == "s-9"
    assert payload["type"] == "chat"


def test_do_write_switch_off_zero_checkpoint_calls(wired, monkeypatch):
    """①开关关：_do_write 不 import/建表/写 checkpoint 一行，记忆写入照常。"""
    monkeypatch.setenv("USE_CHECKPOINT", "0")
    from memory import checkpoint as ckpt

    def boom(*a, **k):
        raise AssertionError("USE_CHECKPOINT=0 时不应调用任何 checkpoint 函数")

    monkeypatch.setattr(ckpt, "ensure_checkpoint_table", boom)
    monkeypatch.setattr(ckpt, "save_checkpoint", boom)

    async_write._do_write("u-off-write", "豆豆陪着我", "chat",
                          extractor=_extractor("豆豆陪着我"), session_id="s-off")
    assert _memory_ids("u-off-write"), "记忆写入不能被 checkpoint 开关影响"
    assert _has_ckpt_table() is False, "关闭时不应懒建 session_checkpoints 表"


def test_do_write_switch_on_saves_checkpoint_and_loads_back(wired, monkeypatch):
    """②开关开：后台写入 save 一份 checkpoint，可 load 回来，且懒建表。"""
    assert _has_ckpt_table() is False
    monkeypatch.setenv("USE_CHECKPOINT", "1")

    async_write._do_write("u-ck", "它叫橘子陪了我八年", "chat",
                          extractor=_extractor("用户的猫叫橘子"), session_id="s-1")

    assert _has_ckpt_table() is True, "表应由 ensure_checkpoint_table 懒建（未改 models.py）"
    ck = _load("u-ck", "s-1")
    assert ck is not None
    assert ck["user_id"] == "u-ck" and ck["session_id"] == "s-1"
    p = ck["payload"]
    assert p["stage"] == "chat"
    assert "家里" in p["topics"] and "最近" in p["topics"]
    assert p["key_fact_ids"] == _memory_ids("u-ck")
    assert p["key_fact_ids"], "关键事实 id 应指向刚落库的 memory_stream id"
    assert ck["summary"]


def test_do_write_checkpoint_user_and_session_isolation(wired, monkeypatch):
    """③用户/session 隔离：三份写入互不串扰，未写的组合 load 为 None。"""
    monkeypatch.setenv("USE_CHECKPOINT", "1")
    async_write._do_write("uA", "A1", "chat", extractor=_extractor("事实A1"), session_id="s1")
    async_write._do_write("uA", "A2", "chat", extractor=_extractor("事实A2"), session_id="s2")
    async_write._do_write("uB", "B1", "chat", extractor=_extractor("事实B1"), session_id="s1")

    a1, a2, b1 = _load("uA", "s1"), _load("uA", "s2"), _load("uB", "s1")
    assert a1["payload"]["key_fact_ids"] == _memory_ids("uA")[:1]
    assert a2["payload"]["key_fact_ids"] == _memory_ids("uA")[1:2]
    assert b1["payload"]["key_fact_ids"] == _memory_ids("uB")
    assert a1["payload"]["key_fact_ids"] != a2["payload"]["key_fact_ids"]
    assert _load("uA", "s3") is None and _load("uB", "s2") is None and _load("uC", "s1") is None


def test_do_write_checkpoint_lazy_creates_missing_table(wired, monkeypatch):
    """④表缺失时：开关开后由后台 job 懒建成功，不需要 models.py / 手工迁移。"""
    with engine.connect() as c:
        names = {r[0] for r in c.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
    assert "session_checkpoints" not in names

    monkeypatch.setenv("USE_CHECKPOINT", "on")
    async_write._do_write("u-lazy", "第一次", "chat",
                          extractor=_extractor("懒建表测试"), session_id="s-lazy")

    with engine.connect() as c:
        cols = {r[1] for r in c.execute(text("PRAGMA table_info(session_checkpoints)"))}
    assert {"id", "user_id", "session_id", "payload", "summary", "created_at", "updated_at"} <= cols
    assert _load("u-lazy", "s-lazy") is not None


def test_admin_cli_threads_session_id_to_do_write():
    """静态接线守卫：worker 把 job payload 的 session_id 透传到后台写入（D-073）。"""
    src = (BACKEND / "admin_cli.py").read_text(encoding="utf-8")
    assert "_do_write(job.user_id" in src
    assert "db.info['checkpoint_session_id']=job.payload['session_id']" in src


def test_worker_passes_session_id_to_background_write(client, monkeypatch):
    """行为证据：run_one 把 job payload 的 session_id 放进后台会话 db.info。"""
    from conftest import account
    from memory.b07_models import Job
    from memory.scope import current_db
    import admin_cli

    uid, _, state = account(client)
    with _RawSessionLocal() as db:
        job = Job(kind='memory', user_id=uid, companion_id=state['current_companion_id'],
                  payload={'message': '合成验证', 'type': 'chat', 'session_id': 's-worker'},
                  dedupe_key='ckpt-session-worker')
        db.add(job)
        db.commit()
        jid = job.id

    seen: dict = {}

    def fake_do_write(*args):  # 4 个位置参数——旧 monkeypatch 口径不能被新增 kwarg 破坏
        managed = current_db.get()
        seen['session_id'] = managed.info.get('checkpoint_session_id') if managed else None

    monkeypatch.setattr('memory.async_write._do_write', fake_do_write)
    assert admin_cli.run_one() is True
    assert seen['session_id'] == 's-worker'
    with _RawSessionLocal() as db:
        assert db.get(Job, jid).status == 'completed'
