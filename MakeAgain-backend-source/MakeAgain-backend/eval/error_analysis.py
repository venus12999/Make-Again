#!/usr/bin/env python3
"""误差分析（评测板块第3项）：把失败 case 从评测结果里抽出来，归类失败模式，区分真失败 vs 评测集错配。

业界依据：Hamel FAQ 2026-09（open coding→axial coding→taxonomy）；Airbnb "When in doubt, look at your data"；
Anthropic Step6 "读 transcripts，失败应该看起来公平"。

用法：
  python eval/error_analysis.py collect                 # 从 outputs/benchmarks/ 抽所有失败 case
  python eval/error_analysis.py report                  # 打印归类报告（按 taxonomy + 真失败/评测错配）
  python eval/error_analysis.py annotate <id> <cat> --verdict real|mismatch [--note ...]   # 人工标注
  python eval/error_analysis.py to-regression           # 把标注为 real 的失败回流成回归用例（第10项产物回流）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
OUT = BACKEND.parent / "outputs" / "error_analysis"
OUT.mkdir(parents=True, exist_ok=True)
CASES = OUT / "failure_cases.json"
REGRESSION = OUT / "regression_cases.json"
RESULTS = BACKEND.parent / "outputs" / "benchmarks"

# 失败模式 taxonomy（open coding 后归纳的初始类，标注时可加）
TAXONOMY = {
    "crisis_mismatch": "危机集错配：gold标高危但内容是关系困扰/倾诉，非我们产品的自杀危机场景",
    "emotion_label_issue": "情绪标注问题：英文社交媒体讽刺/游戏语境，翻译后gold标签不适用",
    "emotion_boundary": "情绪边界分歧：gold细粒度(grief)与中文枚举(想念)的合理语义差异",
    "gen_empty": "生成空回复：hy4偶发reasoning吃光token",
    "judge_low": "judge低分：回复质量确实不达标",
    "other": "其他（待归类）",
}


def collect():
    fails = []
    # 共情/策略批：judge<4 或空
    for f in ["01_中文共情_CPsyCoun", "02_支持策略_ESConv"]:
        p = RESULTS / f"{f}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for r in d.get("results", []):
            j = r.get("judge") or {}
            s = j.get("score")
            if s is None or s < 4:
                auto_cat = "gen_empty" if not r.get("reply") else "judge_low"
                fails.append({
                    "id": r["id"], "batch": f, "scene": r["scene"], "type": "judge",
                    "text": r.get("text", ""), "reply": r.get("reply", ""),
                    "score": s, "judge_reason": j.get("reason", ""),
                    "auto_category": auto_cat, "verdict": "unannotated", "note": "",
                })
    # 情绪批：hit=false
    p = RESULTS / "03_情绪识别_GoEmotions.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        for r in d.get("results", []):
            if not r.get("hit"):
                fails.append({
                    "id": r["id"], "batch": "03_情绪识别_GoEmotions", "scene": r["scene"], "type": "emotion",
                    "text": r.get("text", ""), "pred": r.get("pred"), "gold": r.get("gold"),
                    "auto_category": "emotion_boundary", "verdict": "unannotated", "note": "",
                })
    # 危机批：漏检
    p = RESULTS / "04_危机安全_CueSupport.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        for r in d.get("results", []):
            if not r.get("crisis") and "high" in str(r.get("expected_risk", "")):
                fails.append({
                    "id": r["id"], "batch": "04_危机安全_CueSupport", "scene": r["scene"], "type": "crisis",
                    "text": r.get("text", ""), "level": r.get("level"),
                    "auto_category": "crisis_mismatch", "verdict": "unannotated", "note": "",
                })
    # 保留已有人工标注
    existing = {}
    if CASES.exists():
        for c in json.loads(CASES.read_text(encoding="utf-8")).get("cases", []):
            existing[c["id"]] = c
    for c in fails:
        if c["id"] in existing:
            c["category"] = existing[c["id"]].get("category")
            c["verdict"] = existing[c["id"]].get("verdict", "unannotated")
            c["note"] = existing[c["id"]].get("note", "")
    obj = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "taxonomy": TAXONOMY,
        "note": "失败模式 taxonomy 为 open coding 初始归纳，人工标注 verdict=real 才算真产品失败",
        "n_cases": len(fails),
        "cases": fails,
    }
    CASES.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[collect] {len(fails)} failure cases -> {CASES}")
    print("  by auto_category:", dict(Counter(c["auto_category"] for c in fails)))


def report():
    if not CASES.exists():
        print("[report] 先跑 collect")
        return
    d = json.loads(CASES.read_text(encoding="utf-8"))
    cases = d["cases"]
    print(f"\n{'='*70}\n误差分析报告：{len(cases)} 个失败 case\n{'='*70}")
    print("\n--- 按自动归类 ---")
    for cat, n in Counter(c["auto_category"] for c in cases).most_common():
        print(f"  {cat:22} {n:3}  {TAXONOMY.get(cat,'')[:50]}")
    print("\n--- 按人工裁定（verdict）---")
    for v, n in Counter(c.get("verdict", "unannotated") for c in cases).most_common():
        print(f"  {v:20} {n:3}")
    print("\n--- 真产品失败（verdict=real）---")
    reals = [c for c in cases if c.get("verdict") == "real"]
    if not reals:
        print("  （暂无人工标注为 real 的 case）")
    for c in reals:
        print(f"  [{c['category']}] {c['id']}: {c.get('note','')[:60]}")
    print("\n--- 评测集错配（verdict=mismatch）---")
    mm = [c for c in cases if c.get("verdict") == "mismatch"]
    for c in mm[:8]:
        print(f"  [{c['category']}] {c['id']}: {c['text'][:60]}")
    if len(mm) > 8:
        print(f"  ... 另 {len(mm)-8} 条")


def annotate(case_id, category, verdict, note=""):
    if not CASES.exists():
        print("[annotate] 先跑 collect")
        return
    d = json.loads(CASES.read_text(encoding="utf-8"))
    found = False
    for c in d["cases"]:
        if c["id"] == case_id:
            c["category"] = category
            c["verdict"] = verdict
            c["note"] = note
            found = True
            break
    if not found:
        print(f"[annotate] 未找到 case: {case_id}")
        return
    CASES.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[annotate] {case_id} -> {category} / {verdict}")


def to_regression():
    """产物回流：标注为 real 的失败 case → 回归用例（下次 compare 必跑，防复发）。"""
    if not CASES.exists():
        print("[to-regression] 先跑 collect")
        return
    d = json.loads(CASES.read_text(encoding="utf-8"))
    reals = [c for c in d["cases"] if c.get("verdict") == "real"]
    reg = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "误差分析回流的真实失败用例；标注 real 才算，mismatch 不进（评测集错配不是产品失败）",
        "n_cases": len(reals),
        "cases": reals,
    }
    REGRESSION.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[to-regression] {len(reals)} real failures -> {REGRESSION}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("collect")
    sub.add_parser("report")
    a = sub.add_parser("annotate")
    a.add_argument("id"); a.add_argument("category"); a.add_argument("verdict", choices=["real", "mismatch"])
    a.add_argument("--note", default="")
    sub.add_parser("to-regression")
    args = ap.parse_args()
    if args.cmd == "collect":
        collect()
    elif args.cmd == "report":
        report()
    elif args.cmd == "annotate":
        annotate(args.id, args.category, args.verdict, args.note)
    elif args.cmd == "to-regression":
        to_regression()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
