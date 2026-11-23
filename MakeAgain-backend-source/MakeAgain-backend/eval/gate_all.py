#!/usr/bin/env python3
"""产品风险门禁总入口（漏斗的唯一入口）。

验收标准只有一条：**能不能拦住真实 bug**。
每个核心产品风险配一条门禁指标；任一不达标 exit 1，禁止合入。

用法（本地，带真 key）：
  python eval/gate_all.py                      # 跑全部门禁
  python eval/gate_all.py --only fact_retention  # 只跑某条
  python eval/gate_all.py --list               # 列出所有门禁

门禁清单（风险 → 指标 → 阈值 → 探针）：
  fact_retention  长对话丢事实（林夏/周明远）→ 事实保留率 ≥ 0.66 → long_context_eval.py --gate
  crisis_recall   危机漏检 → 危机维度 recall 不低于红线 → run_benchmarks 危机批（待接，见 TODO）
  chain_complete  访谈→报告链路断裂 → 链路完成率 ≥ 0.9 → e2_interview_chain（待接）

诚实边界：
- 目前只有 fact_retention 一条是真门禁（有实测证据 D-066/D-067）；其余标 TODO/未接入，
  不假装它们在工作——这正是过去"套壳"的毛病，这里明确标注
- 需要真 key（work/local-test-key.txt），所以不进 GitHub Actions 的纯离线 CI；
  CI 侧只做静态检查确保本文件被维护（见 tests/test_gate_inventory.py）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "outputs" / "gate_results.json"

# 风险 → 门禁定义（唯一真相；CI 静态检查读这个）
GATES = {
    "fact_retention": {
        "risk": "长对话丢失用户早期事实（真实bug：林夏/周明远丢失，用户'像她又走了一次'）",
        "metric": "事实保留率",
        "threshold": 0.66,
        "command": [sys.executable, "eval/long_context_eval.py", "--scenario", "all", "--rounds", "10", "--gate", "0.66"],
        "status": "ACTIVE",  # 真门禁，有 D-066/D-067 证据
    },
    "crisis_recall": {
        "risk": "危机信号漏检（用户求助时没识别）",
        "metric": "危机维度 recall",
        "threshold": "不低于 baseline（见 run_benchmarks 危机批）",
        "command": None,
        "status": "TODO",  # 未接入——不假装它在工作
    },
    "chain_complete": {
        "risk": "访谈→报告链路断裂",
        "metric": "链路完成率",
        "threshold": 0.9,
        "command": None,
        "status": "TODO",
    },
}


def list_gates():
    print("=== 产品风险门禁（验收标准：能不能拦住真实bug）===")
    for name, g in GATES.items():
        flag = "🟢" if g["status"] == "ACTIVE" else "⚪"
        print(f"  {flag} {name}: {g['metric']} ≥ {g['threshold']} [{g['status']}]")
        print(f"      风险: {g['risk']}")
    active = sum(1 for g in GATES.values() if g["status"] == "ACTIVE")
    print(f"\n活跃门禁 {active}/{len(GATES)} —— 其余是 TODO，不假装在工作")


def run_gates(only: str | None):
    results = {}
    failed = []
    names = [only] if only else [n for n, g in GATES.items() if g["status"] == "ACTIVE"]
    for name in names:
        g = GATES.get(name)
        if not g or g["status"] != "ACTIVE" or not g["command"]:
            print(f"[SKIP] {name}: 非活跃门禁")
            continue
        print(f"\n=== 跑门禁 {name}: {g['metric']} ≥ {g['threshold']} ===")
        r = subprocess.run(g["command"], cwd=HERE.parent, capture_output=True, text=True)
        print(r.stdout[-1200:])
        if r.stderr:
            print("STDERR:", r.stderr[-400:])
        ok = r.returncode == 0
        results[name] = {"ok": ok, "exit": r.returncode, "threshold": g["threshold"], "metric": g["metric"]}
        if not ok:
            failed.append(name)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果: {OUT}")
    if failed:
        print(f"[GATE_ALL FAIL] 未通过: {', '.join(failed)} —— 禁止合入")
        return 1
    print("[GATE_ALL PASS] 所有活跃门禁通过")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        list_gates()
        return 0
    return run_gates(args.only)


if __name__ == "__main__":
    sys.exit(main())
