#!/usr/bin/env python3
"""V1+V2 三场景真 hy4 验证：分手 / 亲友离世 / 宠物离世 访谈→报告完整链路。

先读码/看实物（本文件每一处判定都能追到实物，不拍脑袋）：
- eval/e2_interview_chain.py：本脚本**复用**它的 validate_report_fields / check_personalization /
  check_turns / mock_trace / prepare_real_env / real_trace，不自造指标。
- eval/walkthrough_journey.py：PERSONAS 三场景来源，scenarios.json 沿用其 answers。
- outputs/e2/e2_real_after_u017_fix.json：分手真跑实物（11轮/6filler、报告9字段、exit=0，U-017 修复后），
  本次按计划不重跑分手。
- docs/THREE_SCENARIO_VALIDATION_PLAN.md：本工作包定义。

与 E2 的唯一口径差异（按本任务验收标准，不是新指标）：
  个性化锚点阈值 E2=2 → 本脚本 =3（复用 e2.check_personalization 的匹配逻辑，只提高门槛）；
  其余字段齐全（validate_report_fields）、轮数（check_turns）、报告生成判定完全复用 E2。
  “报告9字段” = e2.REQUIRED_TOP 的 8 个顶层字段 + e2.ANALYSIS_BLOCK 按 loss_type 的专属分析块
  （breakup→relationship_analysis / loved_one→grief_analysis / pet→pet_grief_analysis）。

分层（用户关心成本）：
  --mock  构造 trace 验评测结构，秒级、不烧 key（V12-V1 验收主力）。
  默认     真实 hy4 跑访谈→报告（约 6-10 分钟/场景）；key 只从 work/local-test-key.txt 读入内存，
           不打印明文、不入库。端点挂起由编排层（后台 job）超 20 分钟标 ENV_FAIL，不硬等。

用法（在 MakeAgain-backend 目录）：
  python eval/scenario_run.py --mock                          # 三场景结构验收（不烧 key）
  python eval/scenario_run.py --mock --scenario pet           # 单场景结构验收
  python eval/scenario_run.py --mock --fault generic_report    # 构造性守卫：锚点不足 → exit 1
  python eval/scenario_run.py --scenario loved_one --out ../outputs/scenarios/loved_one.json
  python eval/scenario_run.py --scenario pet --out ../outputs/scenarios/pet.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(HERE))

import e2_interview_chain as e2  # noqa: E402
from e2_interview_chain import (  # noqa: E402  复用，不自造
    ANALYSIS_BLOCK,
    CHECKPOINT_NAMES,
    REASONABLE_MAX_TURNS,
    REQUIRED_TOP,
    check_turns,
    load_batch as _e2_load_batch,
    mock_trace,
    prepare_real_env as _e2_prepare_real_env,
    real_trace,
    validate_report_fields,
)

DEFAULT_BATCH = HERE / "batches" / "scenarios.json"
DEFAULT_OUT = BACKEND.parent / "outputs" / "scenarios" / "scenario_run_result.json"

# 唯一口径差异：个性化锚点门槛从 E2 的 2 提高到本任务的 3。
MIN_ANCHORS = 3
MAX_TURNS = REASONABLE_MAX_TURNS  # 16，复用 E2 从产品常量推导的阈值
SCENARIOS = ("breakup", "loved_one", "pet", "all")


def load_batch(path: Path = DEFAULT_BATCH) -> list[dict]:
    """复用 e2.load_batch 的校验逻辑，只把默认批次换成 scenarios.json。"""
    return _e2_load_batch(path)


def _personalization(item: dict, report: dict) -> dict:
    """复用 e2.check_personalization 的匹配逻辑，只把门槛换成 MIN_ANCHORS=3。"""
    anchors = (item.get("persona") or {}).get("anchors") or []
    _e2_ok, matched, missed, _e2_min = e2.check_personalization(report, anchors)
    return {
        "ok": len(matched) >= MIN_ANCHORS,
        "matched": matched,
        "missed": missed,
        "min_required": MIN_ANCHORS,
        "e2_min_required": _e2_min,  # 留痕：证明只改门槛、没改匹配算法
        "total_anchors": len(anchors),
    }


def report_field_count(report: dict, loss_type: str | None) -> int:
    """9 字段命中计数：8 顶层 + 1 按 loss_type 的专属分析块（与 validate_report_fields 同口径）。"""
    if not isinstance(report, dict):
        return 0
    n = sum(1 for k in REQUIRED_TOP if k in report)
    block = ANALYSIS_BLOCK.get(loss_type or "")
    if block and isinstance(report.get(block), dict) and report[block]:
        n += 1
    return n


def evaluate_scenario(item: dict, trace: dict) -> dict:
    """纯函数：把一条 trace 打成 5 个检查点。mock / 真 hy4 共用同一套判定。"""
    persona = item.get("persona") or {}
    loss_type = persona.get("loss_type")
    report = trace.get("report")
    fields_ok, missing = validate_report_fields(report, loss_type)
    pers = _personalization(item, report) if report else {
        "ok": False, "matched": [], "missed": persona.get("anchors") or [],
        "min_required": MIN_ANCHORS, "e2_min_required": e2.PERSONALIZATION_MIN_ANCHORS,
        "total_anchors": len(persona.get("anchors") or []),
    }
    turns = check_turns(trace.get("n_turns", 0), trace.get("gen_filler_turns", 0))

    cps = {
        "CP1_reached_complete": bool(trace.get("reached_complete") and trace.get("done")),
        "CP2_report_generated": bool(isinstance(report, dict) and bool(report)),
        "CP3_report_fields_complete": bool(fields_ok),
        "CP4_personalization_match": bool(pers["ok"]),
        "CP5_followup_turns_bounded": bool(turns["ok"]),
    }
    return {
        "checkpoints": cps,
        "overall_pass": all(cps.values()),
        "evidence": {
            "missing_fields": missing,
            "report_field_count": report_field_count(report, loss_type),
            "personalization": pers,
            "turns": turns,
            "soft_warnings": ["FILLER_TURNS_HIGH"] if turns["gen_filler_turns"] > e2.FILLER_SOFT_LIMIT else [],
            "done": bool(trace.get("done")),
            "report_ready": bool(trace.get("report_ready")),
            "elapsed_s": trace.get("elapsed_s"),
        },
    }


def prepare_real_env() -> None:
    """复用 E2 的隔离 DB + hy4 + extra headers 装配；再把 session id 换成 scenario 标识便于追踪。"""
    _e2_prepare_real_env()
    os.environ["LLM_EXTRA_HEADERS"] = json.dumps({
        "x-opencode-session": "scenario-validate-" + uuid.uuid4().hex[:8],
        "User-Agent": "make-again-scenario-validate/1.0",
    })


def run(items: list[dict], mode: str = "mock", fault: str | None = None,
        scenario: str = "all") -> dict:
    """跑三场景验证，返回结构化结果。mode=mock 不碰 key/网络；mode=real 现场生成报告。"""
    if mode not in ("mock", "real"):
        raise ValueError(f"unknown mode: {mode}")
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    selected = [it for it in items if scenario == "all" or it.get("scenario") == scenario]
    if not selected:
        raise ValueError(f"no scenario matched: {scenario}")

    results: list[dict] = []
    client = db = current_db = token = None
    try:
        if mode == "real":
            prepare_real_env()
            from gateway.client import LLMClient
            from memory.db import _RawSessionLocal
            from memory.scope import current_db as _current_db
            client = LLMClient()
            db = _RawSessionLocal()
            current_db = _current_db
            db.info.update(managed=True, borrowed=True)
            token = current_db.set(db)
        for item in selected:
            persona = item["persona"]
            print(f"\n=== scenario {mode}: {item.get('id')} / {persona.get('tag')} ===")
            trace = mock_trace(item, fault) if mode == "mock" else real_trace(item, db, client)
            scored = evaluate_scenario(item, trace)
            ev = scored["evidence"]
            results.append({
                "id": item.get("id"), "source_id": item.get("source_id"),
                "scenario": item.get("scenario"), "tag": persona.get("tag"),
                "loss_type": persona.get("loss_type"),
                "trace": {k: v for k, v in trace.items() if k != "report"},
                "checkpoints": scored["checkpoints"],
                "overall_pass": scored["overall_pass"],
                "evidence": ev,
                "report": trace.get("report"),  # mock 为构造值；real 为 hy4 真实输出
            })
            print(f"  检查点: {scored['checkpoints']} -> {'PASS' if scored['overall_pass'] else 'FAIL'}")
            print(f"  字段数: {ev['report_field_count']}/9  "
                  f"锚点: {len(ev['personalization']['matched'])}/{ev['personalization']['total_anchors']} "
                  f"(门槛{ev['personalization']['min_required']})  "
                  f"轮数: {ev['turns']['n_turns']} filler={ev['turns']['gen_filler_turns']} "
                  f"上限={ev['turns']['max_allowed_turns']}  耗时: {ev['elapsed_s']}s")
    finally:
        if current_db is not None and token is not None:
            current_db.reset(token)
        if db is not None:
            db.close()

    pass_counts = {name: sum(1 for r in results if r["checkpoints"].get(name)) for name in CHECKPOINT_NAMES}
    overall = bool(results) and all(r["overall_pass"] for r in results)
    return {
        "tool": "scenario_run", "schema_version": "1.0", "mode": mode,
        "fault": fault, "scenario": scenario,
        "口径": ("mock=构造trace验评测结构，非产品行为证据；persona=synthetic-handwritten；不烧key"
                 if mode == "mock" else "真实 hy4 同步访谈→报告；persona=synthetic-handwritten；key 不入库"),
        "reuse": "判定函数复用 e2_interview_chain；唯一差异=个性化锚点门槛 E2 的2 → 本任务3",
        "timeout_policy": "真跑端点挂起由编排层后台 job 超20分钟标 ENV_FAIL，不硬等",
        "threshold": {
            "reasonable_max_turns": MAX_TURNS,
            "personalization_min_anchors": MIN_ANCHORS,
            "required_report_fields": 9,
        },
        "results": results,
        "summary": {
            "users": len(results),
            "checkpoint_pass_counts": pass_counts,
            "soft_warning_users": sum(1 for r in results if r["evidence"]["soft_warnings"]),
            "overall_pass": overall,
            "exit_code": 0 if overall else 1,
            "scenarios": [{
                "scenario": r["scenario"], "loss_type": r["loss_type"],
                "report_field_count": r["evidence"]["report_field_count"],
                "anchors_matched": len(r["evidence"]["personalization"]["matched"]),
                "anchors_total": r["evidence"]["personalization"]["total_anchors"],
                "n_turns": r["evidence"]["turns"]["n_turns"],
                "gen_filler_turns": r["evidence"]["turns"]["gen_filler_turns"],
                "elapsed_s": r["evidence"]["elapsed_s"],
                "pass": r["overall_pass"],
            } for r in results],
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="三场景（分手/亲友离世/宠物离世）访谈→报告真 hy4 验证")
    ap.add_argument("--mock", action="store_true", help="构造 trace 验结构（不烧 key）")
    ap.add_argument("--scenario", choices=SCENARIOS, default="all", help="选跑场景，默认 all")
    ap.add_argument("--batch", type=Path, default=DEFAULT_BATCH, help="批次 JSON")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="结果落盘路径")
    ap.add_argument("--fault", choices=(None, "invalid_report", "runaway_followup", "generic_report"),
                    default=None, help="mock 模式注入故障，构造性验证守卫")
    args = ap.parse_args(argv)

    items = load_batch(args.batch)
    result = run(items, mode="mock" if args.mock else "real", fault=args.fault, scenario=args.scenario)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(f"\n[scenario_run] {result['summary']} -> {args.out}")
    return int(result["summary"]["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
