"""GAP-W1 回归守卫：记忆回写的真实产品链路（不绕过 Worker）。

真实链路：chat → enqueue → flush_intents → durable_jobs → Worker run_one → _do_write → 记忆可召回。
之前所有记忆验证直接调 _do_write 绕过队列；本测试守住真实链路不退化。

用子进程跑 Worker（模拟真实独立部署）+ mock LLM（链路验证不需模型质量）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def test_memory_writeback_real_pipeline_end_to_end():
    """6 步全通：enqueue→durable_jobs→Worker→落库→召回。"""
    import tempfile
    root = tempfile.mkdtemp(prefix="job-pipeline-test-")
    script = BACKEND / "eval" / "verify_memory_job_pipeline.py"
    env = dict(os.environ)
    # 覆盖脚本内的 ROOT（脚本用 tempfile 自建，这里只确保环境干净）
    env["PYTHONPATH"] = str(BACKEND)
    proc = subprocess.run([sys.executable, str(script)], env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-500:]
    out = BACKEND.parent / "outputs" / "memory_pipeline_verification.json"
    assert out.exists()
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["verdict"] == "PIPELINE_OK", d
    assert len(d["steps"]) == 6
    assert all(s["ok"] for s in d["steps"])
    # 关键节点数字
    assert d["steps"][2]["status"] == "pending"
    assert d["steps"][3]["status"] == "completed"
    assert d["steps"][4]["cat_mem"] >= 1
    assert d["steps"][5]["recalled"] >= 1


def test_worker_begin_immediate_is_inside_try():
    """确认 run_one 的 BEGIN IMMEDIATE 在 try 块内——锁异常会被 except 接住走 retry。

    调试 GAP-W1 时曾误判为"在 try 外"，实际 try: 在 BEGIN 之前（pos 31 < 61），
    锁竞争会被捕获并按 retry 处理，这是正确的。固化此认知防止误改。
    """
    src = (BACKEND / "admin_cli.py").read_text(encoding="utf-8")
    fn = src.split("def run_one():")[1].split("def ")[0]
    begin_pos = fn.find("BEGIN IMMEDIATE")
    try_pos = fn.find("try:")
    assert begin_pos != -1
    assert try_pos < begin_pos, "BEGIN IMMEDIATE 应在 try 块内，锁异常才能走 retry"
