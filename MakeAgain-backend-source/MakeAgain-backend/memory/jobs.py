"""Queue intents with their originating transaction; no in-process fire-and-forget."""
import hashlib,json
import os
from sqlalchemy import select
from memory.scope import current_db
from memory.b07_models import Job

# ---- Opt-in Huey trial -------------------------------------------------------
# 这是一条**并行试点**通道，不替换 durable_jobs / Job.status 状态机。
# 开关默认关闭：关闭时下面所有新逻辑都不生效，原有入队/落库代码逐字节保持原样。
# 用 SQLite 自身作为 broker（SqliteHuey，无需 Redis），broker 表(task/schedule/kv)
# 直接建在应用同一个 SQLite 文件里；flush_intents 用**同一个事务的连接**写入
# Huey 的 task 表，因此业务数据与入队是真正原子提交/回滚（outbox 语义）。
USE_HUEY_TRIAL_ENV='USE_HUEY_TRIAL'
TRIAL_QUEUE='makeagain_trial'
TRIAL_KINDS=frozenset(('trial_echo','trial_noop'))
TRIAL_RETRIES=2
TRIAL_BACKOFF_BASE_SECONDS=60   # 与 durable job 的 60*2^(n-1) 退避一致
_huey=None
_trial_task=None

def huey_trial_enabled():
    """运行期读取开关，默认关闭（os.getenv，测试可 monkeypatch）。"""
    return os.getenv(USE_HUEY_TRIAL_ENV,'').strip().lower() in ('1','true','yes','on')

def _trial_broker_config():
    """SqliteHuey 配置：broker 文件默认复用应用 SQLite 文件，可用 HUEY_TRIAL_DB 覆盖。"""
    filename=os.getenv('HUEY_TRIAL_DB') or ''
    if not filename:
        from config import get_settings
        url=get_settings().database_url
        if url.startswith('sqlite:///'):
            filename=url.replace('sqlite:///','',1)
    if not filename or filename==':memory:':
        import tempfile
        filename=os.path.join(tempfile.gettempdir(),'makeagain_huey_trial.sqlite3')
    return dict(name=TRIAL_QUEUE,filename=filename,results=True,immediate=False,utc=False)

def get_trial_huey():
    """惰性创建 SqliteHuey 与 trial 任务：开关关闭时完全不触碰 broker。"""
    global _huey,_trial_task
    if _huey is None:
        from huey import SqliteHuey
        config=_trial_broker_config()
        _huey=SqliteHuey(**config)
        _trial_task=_huey.task(retries=TRIAL_RETRIES,retry_delay=TRIAL_BACKOFF_BASE_SECONDS,
                               context=True,name='trial_echo')(_trial_echo_task)
    return _huey

def _trial_echo_task(payload, task=None):
    """试点消费者：默认回显；fail_times=N 时前 N 次失败，走指数退避重试。"""
    payload=payload or {}
    kind=payload.get('kind') or 'trial_echo'
    fail_times=int(payload.get('fail_times',0) or 0)
    attempt=TRIAL_RETRIES-task.retries+1     # 1-based：2->1, 1->2, 0->3
    if attempt<=fail_times:
        # 普通异常由 Huey 消耗 retries；重试间隔在本轮设为 60*2^(attempt-1)。
        task.retry_delay=TRIAL_BACKOFF_BASE_SECONDS*(2**(attempt-1))
        raise RuntimeError(payload.get('error') or 'trial_echo forced failure')
    if kind=='trial_noop':
        return {'noop':True,'attempts':attempt,'task_id':task.id}
    return {'echo':payload.get('message'),'attempts':attempt,'task_id':task.id,'kind':kind}

def _dispatch_trial(db,args):
    """在调用方事务内写入 Huey broker 的 task 表（同一连接 => outbox 原子性）。"""
    from sqlalchemy import text
    huey=get_trial_huey()
    payload=dict(args['payload'] or {})
    payload.setdefault('kind',args['kind'])
    task=_trial_task.s(payload)
    data=huey.serialize_task(task)
    db.execute(text('INSERT INTO task (queue,data,priority) VALUES (:queue,:data,0)'),
               {'queue':TRIAL_QUEUE,'data':data})
    db.info.setdefault('huey_trial_ids',[]).append(task.id)
    return task.id

def run_trial_worker(max_tasks=10):
    """试点 consumer：领取并执行队内任务；不读取/改动 durable_jobs。"""
    huey=get_trial_huey()
    processed=0
    for _ in range(max_tasks):
        task=huey.dequeue()
        if task is None:break
        huey.execute(task);processed+=1
    return processed

def enqueue(kind,user_id,payload,dedupe=None):
    db=current_db.get()
    if db is None: raise RuntimeError('Background work requires a managed transaction')
    cid=db.info.get('companion_id')
    key=dedupe or f'{kind}:{user_id}:{cid}:{db.info.get("request_id") or hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()}'
    # Delay insertion until external calls are finished, keeping SQLite write locks short.
    db.info.setdefault('job_intents',[]).append(dict(kind=kind,user_id=user_id,companion_id=cid,payload=payload,dedupe_key=key))

def flush_intents(db):
    for args in db.info.pop('job_intents',[]):
        if huey_trial_enabled() and args['kind'] in TRIAL_KINDS:
            _dispatch_trial(db,args)
            continue
        existing = db.scalar(select(Job).where(Job.dedupe_key==args['dedupe_key']))
        # 仅在仍活跃/进行中时跳过；已失败/已取消的作业允许按新事务重新入队。
        if existing is None or existing.status in ('failed','cancelled'):
            if existing is not None:
                db.delete(existing)
                db.flush()
            db.add(Job(**args))
