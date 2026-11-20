"""PR-B Huey 试点行为验收：SQLite broker / outbox 原子性 / 指数退避 / 开关默认关闭。

所有断言都走真实 SqliteHuey + 真实 SqliteHuey storage，不使用 mock broker。
"""
import sqlite3
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select, text

from memory import jobs
from memory.b07_models import Job
from memory.db import _RawSessionLocal
from memory.scope import current_db


def _broker_file():
    return jobs.get_trial_huey().storage.filename


def _task_rows():
    """直接读 SQLite broker 的 task 表（真实持久化记录）。"""
    con = sqlite3.connect(_broker_file())
    try:
        return list(con.execute(
            "select id, queue, length(data) from task where queue=? order by id",
            (jobs.TRIAL_QUEUE,)).fetchall())
    finally:
        con.close()


def _kv_rows():
    con = sqlite3.connect(_broker_file())
    try:
        return list(con.execute(
            "select key, length(value) from kv where queue=? order by key",
            (jobs.TRIAL_QUEUE,)).fetchall())
    finally:
        con.close()


def _due_rows():
    con = sqlite3.connect(_broker_file())
    try:
        return con.execute(
            "select count(*) from schedule where queue=?",
            (jobs.TRIAL_QUEUE,)).fetchone()[0]
    finally:
        con.close()


def _enqueue_in_tx(kind, payload, *, dedupe=None, user_id='trial-user'):
    """模拟业务请求：事务内 enqueue -> flush_intents -> commit。"""
    db = _RawSessionLocal()
    token = current_db.set(db)
    try:
        jobs.enqueue(kind, user_id, payload, dedupe=dedupe)
        jobs.flush_intents(db)
        ids = list(db.info.get('huey_trial_ids', []))
        db.finish()
        return ids
    finally:
        current_db.reset(token)
        db.close()


@pytest.fixture
def huey_on(monkeypatch):
    monkeypatch.setenv('USE_HUEY_TRIAL', '1')
    huey = jobs.get_trial_huey()
    huey.flush()
    yield huey
    huey.flush()


def test_flag_default_off_durable_path_unchanged(monkeypatch):
    """开关关闭：逐字节走原有 durable_jobs 路径，broker 不产生任何行。"""
    monkeypatch.delenv('USE_HUEY_TRIAL', raising=False)
    assert jobs.huey_trial_enabled() is False
    jobs.get_trial_huey().flush()

    _enqueue_in_tx('memory', {'message': 'hi', 'type': 'chat'}, dedupe='off-path')

    with _RawSessionLocal() as check:
        job = check.scalar(select(Job).where(Job.dedupe_key == 'off-path'))
        assert job is not None
        assert (job.kind, job.status, job.attempts) == ('memory', 'pending', 0)
        assert job.payload == {'message': 'hi', 'type': 'chat'}
    assert _task_rows() == [], '开关关闭时不得写入 Huey broker'
    print('[EVIDENCE off] durable job pending, huey broker task rows = 0')


def test_enqueue_flush_consume_result_with_sqlite_broker(huey_on):
    """入队 -> SQLite broker 留痕 -> consumer 领取执行 -> 结果可查。"""
    ids = _enqueue_in_tx('trial_echo', {'message': 'hello-huey'})
    task_id = ids[0]
    rows = _task_rows()
    assert len(rows) == 1 and rows[0][1] == jobs.TRIAL_QUEUE
    print('[EVIDENCE broker] task rows after enqueue:', rows)

    assert jobs.run_trial_worker(1) == 1
    assert _task_rows() == [], 'consumer 领取后队列应清空'
    result = huey_on.result(task_id, preserve=True)
    assert result == {
        'echo': 'hello-huey', 'attempts': 1,
        'task_id': task_id, 'kind': 'trial_echo',
    }
    assert any(key == task_id for key, _ in _kv_rows()), '结果必须落在 broker kv 表'
    print('[EVIDENCE result] huey.result() =', result)
    print('[EVIDENCE broker] kv rows after consume:', _kv_rows())


def test_outbox_atomicity_rollback_leaves_no_broker_row(huey_on):
    """同一事务回滚时，Huey 入队随业务数据一起回滚（outbox 原子性）。"""
    before = _task_rows()
    db = _RawSessionLocal()
    token = current_db.set(db)
    try:
        jobs.enqueue('trial_echo', 'trial-user', {'message': 'rollback-me'})
        jobs.flush_intents(db)
        # 同一连接（事务内）能看到刚写入的 broker 行；外部连接还看不到 => 未提交。
        in_tx = db.execute(text(
            "select count(*) from task where queue='makeagain_trial'")).scalar()
        assert in_tx == len(before) + 1, in_tx
        assert _task_rows() == before, '未提交时外部连接不应看到 broker 行'
        print('[EVIDENCE atomicity] in-transaction broker rows =', in_tx,
              '; external connection rows =', len(_task_rows()))
        db.rollback()
        after_rollback = db.execute(text(
            "select count(*) from task where queue='makeagain_trial'")).scalar()
        assert after_rollback == len(before), after_rollback
    finally:
        current_db.reset(token)
        db.close()
    assert _task_rows() == before, '回滚后 broker 不得残留任务'
    print('[EVIDENCE atomicity] rollback restored broker task rows to', len(before))


def test_failure_retries_twice_with_exponential_backoff(huey_on):
    """retries=2 + 60/120 秒指数退避；第 3 次成功，结果可查。"""
    ids = _enqueue_in_tx('trial_echo', {'message': 'flaky', 'fail_times': 2})
    task_id = ids[0]

    # 第 1 次执行：失败 -> 排入 schedule，retries 2 -> 1，eta ~ now+60s
    assert jobs.run_trial_worker(1) == 1
    sched = huey_on.scheduled()
    assert len(sched) == 1 and sched[0].retries == 1
    eta1 = sched[0].eta
    delta1 = (eta1 - datetime.now()).total_seconds()
    assert 55 <= delta1 <= 65, delta1
    print('[EVIDENCE retry1] retries=%s backoff=%.1fs' % (sched[0].retries, delta1))

    # 第 2 次执行：取出到期的重试，失败 -> eta ~ now+120s，retries 1 -> 0
    due1 = huey_on.read_schedule(eta1 + timedelta(seconds=1))
    assert len(due1) == 1
    huey_on.execute(due1[0], timestamp=eta1 + timedelta(seconds=1))
    sched2 = huey_on.scheduled()
    assert len(sched2) == 1 and sched2[0].retries == 0
    eta2 = sched2[0].eta
    delta2 = (eta2 - datetime.now()).total_seconds()
    assert 115 <= delta2 <= 125, delta2
    print('[EVIDENCE retry2] retries=%s backoff=%.1fs' % (sched2[0].retries, delta2))

    # 第 3 次执行：成功
    due2 = huey_on.read_schedule(eta2 + timedelta(seconds=1))
    assert len(due2) == 1
    huey_on.execute(due2[0], timestamp=eta2 + timedelta(seconds=1))
    assert _due_rows() == 0
    result = huey_on.result(task_id)
    assert result['attempts'] == 3 and result['echo'] == 'flaky'
    print('[EVIDENCE retry-final] result =', result)


def test_durable_jobs_routing_unaffected_with_trial_on(huey_on):
    """开关打开时，memory 等原有 job 仍进 durable_jobs，不进 Huey broker。"""
    _enqueue_in_tx('memory', {'message': 'x', 'type': 'chat'},
                   dedupe='durable-with-trial-on')
    with _RawSessionLocal() as check:
        job = check.scalar(select(Job).where(Job.dedupe_key == 'durable-with-trial-on'))
        assert job is not None and job.kind == 'memory' and job.status == 'pending'
    assert _task_rows() == []
    print('[EVIDENCE isolation] memory job -> durable_jobs; huey broker task rows = 0')


def test_durable_memory_job_still_completes_with_trial_on(huey_on, client):
    """开关打开时 admin_cli.run_one 仍完整执行 durable memory job 状态机。"""
    from conftest import account
    uid, _, state = account(client)
    with _RawSessionLocal() as db:
        job = Job(kind='memory', user_id=uid, companion_id=state['current_companion_id'],
                  payload={'message': '今天整理照片，心情平静', 'type': 'chat'},
                  dedupe_key='durable-run-one-with-trial')
        db.add(job); db.commit(); jid = job.id
    import admin_cli
    assert admin_cli.run_one() is True
    with _RawSessionLocal() as db:
        job = db.get(Job, jid)
        assert job.status == 'completed' and job.attempts == 1
        portrait = db.scalar(select(Job).where(Job.user_id == uid, Job.kind == 'portrait'))
        assert portrait is not None and portrait.status == 'pending'
    assert _task_rows() == []
    print('[EVIDENCE state-machine] durable memory job completed with USE_HUEY_TRIAL=1')


def test_noop_kind_and_admin_cli_consumer(huey_on, monkeypatch, capsys):
    """trial_noop 走 Huey；admin_cli 试点 worker 在开/关两种状态下的行为。"""
    ids = _enqueue_in_tx('trial_noop', {'message': 'noop'})
    task_id = ids[0]
    jobs.run_trial_worker(1)
    result = huey_on.result(task_id, preserve=True)
    assert result['noop'] is True
    print('[EVIDENCE noop] huey.result() =', result)

    import sys
    import admin_cli
    _enqueue_in_tx('trial_noop', {'message': 'via-cli'})
    monkeypatch.setattr(sys, 'argv', ['admin_cli', 'huey-trial-worker', '--max-tasks', '1'])
    admin_cli.main()
    out = capsys.readouterr().out
    assert 'Huey trial processed 1 tasks' in out, out
    print('[EVIDENCE cli] ' + out.strip())

    monkeypatch.setattr('memory.jobs.huey_trial_enabled', lambda: False)
    with pytest.raises(RuntimeError, match='USE_HUEY_TRIAL is off'):
        admin_cli.main()
    print('[EVIDENCE guard] admin_cli huey-trial-worker refuses when switch is off')
