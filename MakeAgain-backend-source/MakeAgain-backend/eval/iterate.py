#!/usr/bin/env python3
"""评测闭环工具：snapshot（存基线）/ compare（改后对比）/ history（迭代历史）。

把"评测数字 + prompt版本 + git commit"绑在一起，让每次改动有据可查。
判据见 docs/EVAL_ITERATION_LOOP.md。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))
HISTORY = BACKEND.parent / "outputs" / "benchmarks" / "iteration_history.jsonl"
BASELINE = BACKEND.parent / "outputs" / "benchmarks" / "baseline.json"


def git_info() -> dict:
    def run(args):
        try:
            return subprocess.check_output(args, cwd=str(BACKEND.parent), text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"
    return {"commit": run(["git", "rev-parse", "--short", "HEAD"]),
            "branch": run(["git", "branch", "--show-current"])}


def prompt_versions() -> dict:
    try:
        from agent import prompts_registry as pr
        return {pid: pr.get_version(pid) for pid in pr.ids()}
    except Exception:
        return {}


def metrics_from_result(batch_file: Path) -> dict:
    d = json.loads(batch_file.read_text(encoding="utf-8"))
    s = d["summary"]
    return {
        "n": s.get("n"),
        "judge_avg": s.get("judge_avg"),
        "emotion_hit_rate": s.get("emotion_hit_rate"),
        "crisis_detect_rate": s.get("crisis_detect_rate"),
    }


def run_batches(batch_names: list[str]) -> dict:
    """调用 run_benchmarks 的 run_batch，返回各批指标。"""
    import os
    # 确保 key/env（复用 run_benchmarks 的 setup）
    import run_benchmarks as rb
    conf = rb.load_key()
    rb.setup(conf)
    from gateway.client import LLMClient
    client = LLMClient()
    out = {}
    for name in batch_names:
        p = rb.BATCHDIR / f"{name}.json"
        if not p.exists():
            print(f"  ! batch not found: {name}")
            continue
        # 直接调 run_batch（它会写 outputs/benchmarks/<batch>.json）
        rb.run_batch(name, use_raw=False, client=client)
        out[name] = metrics_from_result(rb.OUTDIR / f"{name}.json")
    return out


def all_batch_names() -> list[str]:
    return sorted(p.stem for p in (HERE / "batches").glob("*.json") if not p.name.startswith("_"))


def cmd_snapshot(args):
    names = [args.batch] if args.batch else all_batch_names()
    print(f"[snapshot] running {len(names)} batches ...")
    metrics = run_batches(names)
    snap = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "git": git_info(), "prompt_versions": prompt_versions(),
        "batches": metrics, "note": args.note or "",
    }
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "snapshot", **snap}, ensure_ascii=False) + "\n")
    print(f"[snapshot] saved -> {BASELINE}")
    for k, v in metrics.items():
        print(f"  {k}: judge={v['judge_avg']} emo={v['emotion_hit_rate']} crisis={v['crisis_detect_rate']}")


# 判定阈值（docs/EVAL_ITERATION_LOOP.md）
THRESH = {
    "judge_avg": {"min_drop": 0.1, "improve": 0.1},
    "emotion_hit_rate": {"min_drop": 0.02, "improve": 0.02},
    "crisis_detect_rate": {"min_drop": 0.0, "improve": 0.01},  # 安全红线：不许退化
}


def verdict_for(key: str, before, after) -> str:
    if before is None or after is None:
        return "N/A（缺基线或当前值）"
    t = THRESH.get(key, {"min_drop": 0.05, "improve": 0.05})
    delta = after - before
    if delta < -t["min_drop"]:
        return "退化（建议不合入）"
    if delta > t["improve"]:
        return "改善"
    return "持平"


def cmd_compare(args):
    if not BASELINE.exists():
        print("[compare] 无基线，先跑 snapshot")
        return
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    names = [args.batch] if args.batch else list(base["batches"].keys())
    print(f"[compare] running {len(names)} batches ...")
    now = run_batches(names)
    print(f"\n{'='*70}\n对照基线: {base['ts']} commit={base['git']['commit']}\n{'='*70}")
    overall = []
    for name in names:
        b = base["batches"].get(name)
        n = now.get(name)
        if not b or not n:
            print(f"  {name}: 缺数据（基线或当前）")
            continue
        print(f"\n[{name}]")
        for key in ("judge_avg", "emotion_hit_rate", "crisis_detect_rate"):
            bv, nv = b.get(key), n.get(key)
            v = verdict_for(key, bv, nv)
            delta = (nv - bv) if (bv is not None and nv is not None) else None
            ds = f"{delta:+.3f}" if delta is not None else "N/A"
            print(f"  {key:20} {str(bv):>6} -> {str(nv):>6}  ({ds})  {v}")
            overall.append(v)
    # 总判定
    bad = any("退化" in v for v in overall)
    good = any(v == "改善" for v in overall)
    final = "❌ 有核心指标退化，建议不合入/回滚" if bad else ("✅ 改善，可合入" if good else "➖ 持平，可合入")
    print(f"\n{'='*70}\n总判定: {final}\n{'='*70}")
    snap = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "git": git_info(), "prompt_versions": prompt_versions(),
        "vs_baseline": base["ts"], "batches": now, "verdict": final,
        "note": args.note or "",
    }
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "compare", **snap}, ensure_ascii=False) + "\n")


def cmd_history(args):
    if not HISTORY.exists():
        print("[history] 暂无迭代历史")
        return
    print(f"{'时间':<26} {'事件':<10} {'commit':<10} 判定")
    print("-" * 80)
    for line in HISTORY.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
            ts = d["ts"][:19]
            print(f"{ts:<26} {d['event']:<10} {d['git']['commit']:<10} {d.get('verdict', d.get('note',''))}")
        except Exception:
            continue


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("snapshot"); s.add_argument("--batch"); s.add_argument("--note")
    c = sub.add_parser("compare"); c.add_argument("--batch"); c.add_argument("--note")
    h = sub.add_parser("history")
    args = ap.parse_args()
    if args.cmd == "snapshot":
        cmd_snapshot(args)
    elif args.cmd == "compare":
        cmd_compare(args)
    elif args.cmd == "history":
        cmd_history(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
