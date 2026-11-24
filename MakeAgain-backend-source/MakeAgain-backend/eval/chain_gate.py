#!/usr/bin/env python3
"""chain_complete 门禁探针封装（访谈→报告链路完成率；需要真 key，不进离线 CI）。

执行口径（诚实边界，逐条写清）：
- 底层探针 = ``python eval/e2_interview_chain.py``（默认 real 模式，**不加 --mock**；
  --mock 只构造 trace 验评测结构，不能当产品行为证据）。它是 E2 理解段链路评测，
  5 个检查点 CP1..CP5：reached_complete / report_generated / report_fields_complete /
  personalization_match / followup_turns_bounded（eval/e2_interview_chain.py:82-88）。
- key 只读 work/local-test-key.txt（探针 prepare_real_env/load_conf 内部读取），
  本封装不读取、不打印、不入库任何明文。
- 指标 = "链路完成率" = 通过的 (persona, checkpoint) 数 / (users × 检查点数)，
  取探针产物 outputs/e2/e2_chain_result.json 的 summary.checkpoint_pass_counts
  与 summary.users 现场计算。检查点数从产物键数动态取，不硬编码。
  这样 0.9 阈值有区分度；探针自身的 exit_code=1 要求"全用户全检查点通过"，
  是更严格信号，本门禁按 0.9 判定，两者分别打印、不混为一谈。
- 阈值 = 0.9（任务指定；非从历史实测数字导出）。
- 口径标签：真实 hy4 同步访谈→报告；persona=synthetic-handwritten（批次
  eval/batches/E2_理解段链路.json 自带 source_id，可追溯）。
- 离线行为：CI 纯离线只跑 tests/test_gate_inventory.py 静态检查，不跑本门禁。
  无 key / 探针无产物 → fail closed（exit 1，标 ENV_FAIL），绝不假装通过。

用法（在 MakeAgain-backend 目录，带真 key）：
  python eval/chain_gate.py                 # 用默认阈值 0.9 跑门禁
  python eval/chain_gate.py --gate 0.9      # 显式指定阈值（与 gate_all 一致）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
RESULT = BACKEND.parent / "outputs" / "e2" / "e2_chain_result.json"  # e2_interview_chain.py:51 默认 out
DEFAULT_THRESHOLD = 0.9  # 任务指定：链路完成率 ≥ 0.9

PROVENANCE = "e2_interview_chain.py real（真实 hy4 访谈→报告，CP1..CP5；persona=synthetic-handwritten，key 不入库）"


def evaluate(pass_counts: dict[str, int], users: int, threshold: float) -> tuple[bool, float, int]:
    """纯函数：从探针产物算链路完成率并判阈值。返回 (是否达标, 完成率, 总检查点数)。

    离线可单测，不碰 key。总检查点数从产物键数取（探针 CHECKPOINT_NAMES 的镜像）。
    """
    users = int(users)
    n_cp = len(pass_counts)
    total = users * n_cp
    passed = sum(int(v) for v in pass_counts.values()) if total > 0 else 0
    rate = passed / total if total > 0 else 0.0
    return rate >= float(threshold), rate, total


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="chain_complete 门禁（真实 hy4，需 key）")
    ap.add_argument("--gate", type=float, default=DEFAULT_THRESHOLD,
                    help=f"数值阈值，默认 {DEFAULT_THRESHOLD}")
    args = ap.parse_args(argv)

    cmd = [sys.executable, "eval/e2_interview_chain.py"]
    print(f"[chain_gate] 探针: {' '.join(cmd)} (cwd={BACKEND}, real 模式，非 --mock)")
    print(f"[chain_gate] 口径: {PROVENANCE}")
    try:
        r = subprocess.run(cmd, cwd=BACKEND, capture_output=True, text=True)
    except OSError as exc:
        print(f"[chain_gate] ENV_FAIL 探针无法启动: {exc}")
        return 1
    if r.stdout:
        print(r.stdout[-1500:])
    # 探针 exit=1 表示"有 persona 未全过"，是有效信号而非崩溃；仍读产物算完成率。
    print(f"[chain_gate] 探针 exit={r.returncode}（全通过信号；门禁按完成率独立判定）")

    try:
        payload = json.loads(RESULT.read_text(encoding="utf-8"))
        summary = payload["summary"]
        counts = summary["checkpoint_pass_counts"]
        users = int(summary["users"])
        overall = bool(summary.get("overall_pass"))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"[chain_gate] ENV_FAIL 读不到探针产物 {RESULT}: {exc}")
        return 1
    if not isinstance(counts, dict) or users <= 0 or not counts:
        print(f"[chain_gate] ENV_FAIL 探针产物无有效 checkpoints/users: {summary}")
        return 1

    ok, rate, total = evaluate(counts, users, args.gate)
    print(f"[chain_gate] 指标=链路完成率 实测={rate:.4f} ({sum(int(v) for v in counts.values())}/{total}) "
          f"阈值={args.gate} 来源={RESULT}")
    print(f"[chain_gate] 探针 overall_pass={overall}（更严格：全用户全检查点通过）")
    if ok:
        print("[chain_gate] GATE PASS（链路完成率达标）")
        return 0
    print("[chain_gate] GATE FAIL（链路完成率低于阈值，禁止合入）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
