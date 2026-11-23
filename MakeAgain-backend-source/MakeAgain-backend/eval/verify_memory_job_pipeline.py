#!/usr/bin/env python3
"""GAP-W1 端到端验证：记忆回写的真实产品链路（不绕过 Worker）。

真实链路：chat → enqueue() → flush_intents() → durable_jobs 表 → Worker run_one() → _do_write() → 记忆可召回。
之前所有验证都直接调 _do_write 绕过队列，本脚本验证真实链路通畅。

用 mock LLM（链路验证不需要真模型质量），不烧 key。
用法：python eval/verify_memory_job_pipeline.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))
ROOT = tempfile.mkdtemp(prefix="job-pipeline-")
os.environ.update(
    DATABASE_URL="sqlite:///" + str(Path(ROOT) / "app.sqlite3"),
    ENVIRONMENT="test", MOCK_LLM="true",
    MEDIA_SIGNING_KEY="x" * 40,
    STORAGE_LOCAL_DIR=str(Path(ROOT) / "media"),
    PUBLIC_BASE_URL="http://testserver",
    # 双路召回：store.add_memory 仅在 USE_DUAL_RECALL=1 时同步建索引；
    # 真实产品默认开（D-029），这里显式设上+独立索引库路径，验证"落库即可召回"。
    USE_DUAL_RECALL="1",
    DUAL_RECALL_INDEX_DB=str(Path(ROOT) / "recall.idx.sqlite3"),
)


def main():
    from memory.db import Base, engine, _RawSessionLocal
    from memory.scope import current_db
    from memory import store
    from memory.b07_models import Job
    from sqlalchemy import select

    Base.metadata.create_all(engine)
    result = {"steps": [], "verdict": "INCOMPLETE"}
    uid = "pipeline-u"
    message = "我养了一只叫橘子的猫，它昨天走了，我很想它"
    job_id = None
    comp_id = None

    # ===== 阶段A：模拟一次 HTTP 请求（enqueue → flush_intents），请求结束后关闭 session =====
    # 注意：不能用 borrowed=True——ManagedSession.close() 对 borrowed session 是 no-op，
    # 连接不会真释放，Worker 的 BEGIN IMMEDIATE 会锁失败。真实 HTTP 请求由 get_session
    # 依赖管理（非 borrowed），请求结束会真关闭，这里照此模拟。
    db = _RawSessionLocal()
    db.info.update(managed=True, borrowed=False)
    token = current_db.set(db)
    try:
        store.get_or_create_user(db, uid)
        comp = store.create_companion(db, uid, relationship_type="breakup")
        if comp is None:
            comp = store.get_current_companion(db, uid)
        # enqueue 用 db.info['companion_id'] 给 job 打 scope；真实 HTTP 请求由 get_session 注入
        if comp is not None:
            db.info["companion_id"] = comp.id
            comp_id = comp.id
        # 用户需接受记忆使用同意（真实用户走 /api/consent），否则 ScopeDenied 是正确行为
        from datetime import datetime as _dt
        st = store.get_or_create_user_state(db, uid)
        st.consent_accepted_at = _dt.utcnow()
        st.consent_version = 1
        db.flush()
        result["steps"].append({"step": 1, "name": "用户建立", "companion": bool(comp),
                                "consent": bool(st.consent_accepted_at), "ok": True})

        from memory.async_write import schedule_memory_write
        from memory.breakup import extract as breakup_extract
        # 用场景 extractor：直接落 Fact（"记得我"承诺的真实路径，不像普通聊天走 Candidate 确认流）
        schedule_memory_write(uid, message, extractor=breakup_extract.extract_breakup_turn)
        intents = db.info.get("job_intents", [])
        result["steps"].append({"step": 2, "name": "enqueue 产出 intent", "n": len(intents), "ok": len(intents) == 1})

        from memory.jobs import flush_intents
        flush_intents(db)
        # ManagedSession 的 commit() 对 managed session 只 flush 不真提交；
        # 真实 HTTP 请求由事务边界调用 finish() 真正落盘——这里照之。
        db.finish()
        jobs = db.scalars(select(Job).where(Job.user_id == uid)).all()
        job_id = jobs[0].id if jobs else None
        result["steps"].append({"step": 3, "name": "flush_intents 落 durable_jobs", "n": len(jobs),
                                "status": jobs[0].status if jobs else None, "ok": len(jobs) == 1})
    finally:
        # 真实部署里 HTTP 请求结束后 session 关闭——此处必须完全释放，Worker 才能拿到写锁
        current_db.reset(token)
        db.close()

    # ===== 阶段B：Worker 独立进程消费（真实部署就是这样）=====
    # 必须先 dispose 父进程 engine 的所有连接，否则 WAL -shm 锁会让子进程 BEGIN IMMEDIATE 失败
    engine.dispose()
    import subprocess
    worker_code = (
        "import sys; sys.path.insert(0, r'" + str(BACKEND) + "'); "
        "import admin_cli; ok=admin_cli.run_one(); print('WORKER_PROCESSED=' + str(ok))"
    )
    proc = subprocess.run([sys.executable, "-c", worker_code], env=dict(os.environ),
                          capture_output=True, text=True, timeout=120)
    worker_out = proc.stdout + proc.stderr
    processed = "WORKER_PROCESSED=True" in worker_out

    # ===== 阶段C：新 session 检查结果（设同样的 companion scope，模拟真实用户的召回上下文）=====
    db2 = _RawSessionLocal()
    db2.info.update(managed=True, borrowed=False, companion_id=comp_id)
    token2 = current_db.set(db2)
    try:
        job_after = db2.get(Job, job_id) if job_id else None
        result["steps"].append({"step": 4, "name": "Worker run_one 消费(独立子进程)", "processed": processed,
                                "status": job_after.status if job_after else None,
                                "attempts": job_after.attempts if job_after else None,
                                "error_code": job_after.error_code if job_after else None,
                                "worker_out": worker_out[-300:],
                                "ok": processed and job_after is not None and job_after.status == "completed"})

        from memory.models import MemoryEntry
        mems = db2.scalars(select(MemoryEntry).where(MemoryEntry.user_id == uid)).all()
        cat_mems = [m for m in mems if "橘子" in (m.content or "")]
        result["steps"].append({"step": 5, "name": "_do_write 落记忆", "n": len(mems), "cat_mem": len(cat_mems),
                                "ok": len(cat_mems) >= 1})

        from memory import recall
        rec = recall.recall(db2, uid, "橘子")
        result["steps"].append({"step": 6, "name": "记忆可被召回", "recalled": len(rec) if rec else 0,
                                "ok": bool(rec)})

        # 总判定
        all_ok = all(s["ok"] for s in result["steps"])
        result["verdict"] = "PIPELINE_OK" if all_ok else "PIPELINE_BROKEN"
    except Exception as exc:
        result["verdict"] = "ERROR"
        result["error"] = str(exc)[:300]
        import traceback
        result["traceback"] = traceback.format_exc()[-800:]
    finally:
        # 阶段A 的 token/db 已在自己的 finally 里释放；这里只清理阶段C
        try:
            current_db.reset(token2)
        except Exception:
            pass
        try:
            db2.close()
        except Exception:
            pass

    out = BACKEND.parent / "outputs" / "memory_pipeline_verification.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("\n=== GAP-W1 记忆回写真实链路端到端验证 ===")
    for s in result["steps"]:
        flag = "✅" if s.get("ok") else "❌"
        print(f"  {flag} Step{s['step']} {s['name']}: {json.dumps({k:v for k,v in s.items() if k not in ('ok','step','name')}, ensure_ascii=False)[:150]}")
    print(f"\n  总判定: {result['verdict']}")
    if result.get("error"):
        print(f"  错误: {result['error']}")
    print(f"  产物: {out}")


if __name__ == "__main__":
    main()
