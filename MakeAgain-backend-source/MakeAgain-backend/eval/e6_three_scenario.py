#!/usr/bin/env python3
"""F1：A/B 改动后三场景（分手/亲友离世/宠物离世）完整五段旅程真 hy4 验证。

背景（先读码/看实物，不拍脑袋）：
- A 场景切换接线 d059174a：agent.companion.chat() L229 路由前调 _maybe_transition_scene，
  命中则归档旧 Journey + 新建新关系 Journey；普通陪伴对话不应切换（memory/state.py
  detect_scene_transition L394，USE_PERSONA 默认关，正则保守+第三人称排除）。
- B 画像隔离 55589d20：store.get_portrait/upsert_portrait 加 companion_id 尾参（不传完全兼容），
  agent/interview.py::_persist_report L744-750 报告落库时写当前 companion.id 归属；
  三条场景引擎 breakup/loved_one/pet 仍用 3 参 get_portrait 注入对话上下文（向后兼容）。
- 五段完整链路与 5 检查点的唯一实现 = eval/e6_full_journey.py（real_journey + evaluate_journey）。
  本脚本**复用**它，不自造链路/不自造检查点；persona 沿用 E6_完整旅程.json。

复用清单（只读，不改它们的所有权文件）：
- e6_full_journey: evaluate_journey（5CP 判定）、real_journey（真 hy4 完整五段）、
  mock_trace、load_batch、prepare_real_env、CHECKPOINT_NAMES、SIMULATED_DAYS。
- e2_interview_chain: CP1 的报告判定（e6.evaluate_journey 内部已复用）。
- scenario_run.report_field_count: 「报告 9 字段」计数口径（8 顶层 + loss_type 专属分析块）。

A/B 专项探针（确定性产品函数，零 LLM，不烧 key），针对本任务重点：
- F1-V3 场景切换不误触发：对三条真实 dialogue_query 在 USE_PERSONA=1（功能开）下跑
  detect_scene_transition → should_switch 必须 False；正对照"我开始新恋情了"必须 True；
  第三人称"我前任开始新恋情了"必须 False；再经真实 _maybe_transition_scene 验证
  companion 段数/loss_type 前后不变（数字）。
- F1-V4 画像注入正确：用真跑报告经真实 _persist_report 落画像（带当前 companion 归属），
  验证 companion_id 正确读到画像、错误 companion_id 读不到（隔离生效）、3 参兼容仍可读，
  且真实 _context_block 注入拿到该画像（[思念对象画像] + 锚点命中，数字）。
  真跑 journey 自身的 context_has_portrait/context_anchor_hits/portrait_key_count 即主证据。

分层（用户关心成本）：
  --mock  构造 trace 验三场景全链路结构（不烧 key，F1-V1 主力）；A/B 探针标 NOT_RUN。
  默认     真 hy4 三场景各 1 次完整五段链路（每段约 8-10 分钟）；端点挂起超 20 分钟
          由编排层后台 job 标 ENV_FAIL 不硬等；key 只从 work/local-test-key.txt 读入内存，
          get_settings.cache_clear()+改 settings 字段（.env 会覆盖），不打印、不入库。

用法（在 MakeAgain-backend 目录）：
  python eval/e6_three_scenario.py --mock --out ../outputs/scenarios/full_journey_mock.json
  python eval/e6_three_scenario.py --scenario breakup --out ../outputs/scenarios/full_journey_breakup.json
  python eval/e6_three_scenario.py --scenario loved_one --out ../outputs/scenarios/full_journey_loved_one.json
  python eval/e6_three_scenario.py --scenario pet --out ../outputs/scenarios/full_journey_pet.json
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
REPO = BACKEND.parent
for _p in (str(BACKEND), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e6_full_journey as e6  # noqa: E402 - 复用完整链路与 5CP 判定，不自造
import scenario_run as sr  # noqa: E402 - 复用报告 9 字段计数口径

DEFAULT_BATCH = HERE / "batches" / "E6_完整旅程.json"
DEFAULT_OUT = REPO / "outputs" / "scenarios" / "full_journey_all.json"
SCENARIOS = ("breakup", "loved_one", "pet", "all")
COMPANION_TYPE = {"breakup": "breakup", "loved_one": "relative", "pet": "pet"}


# ---------------------------------------------------------------- A/B 确定性探针（零 LLM）
def _persona_on(original: str | None) -> None:
    os.environ["USE_PERSONA"] = "1"


def _restore_persona(original: str | None) -> None:
    if original is None:
        os.environ.pop("USE_PERSONA", None)
    else:
        os.environ["USE_PERSONA"] = original


def ab_probes(item: dict, trace: dict, db) -> dict:
    """A/B 改动专项验证：全部为确定性产品函数，零 LLM、不烧 key。

    db 为已装配的评测会话（真 hy4 模式由 run 传入；mock 模式不调用本函数）。
    """
    from agent import companion as companion_mod
    from agent import interview as interview_mod
    from memory import state as scene_state
    from memory import store

    p = item["persona"]
    loss = p["loss_type"]
    anchors = p["anchors"]
    query = item["journey"]["dialogue_query"]
    probes: dict = {"mode": "real", "query": query, "loss_type": loss}

    # ---- F1-V3-A：场景切换检测器（USE_PERSONA=1，功能真开） ----
    original = os.environ.get("USE_PERSONA")
    _persona_on(original)
    try:
        det_real = scene_state.detect_scene_transition(db, "ab-probe", query, loss)
        det_pos = scene_state.detect_scene_transition(db, "ab-probe", "我开始新恋情了，想重新开始。", loss)
        det_third = scene_state.detect_scene_transition(db, "ab-probe", "我前任开始新恋情了。", loss)

        # F1-V3-B：真实接线路径 _maybe_transition_scene（零 LLM），普通对话不改 companion/loss_type
        uid_sw = "ab-switch-" + uuid.uuid4().hex[:8]
        store.get_or_create_user(db, uid_sw, loss_type=loss)
        c0 = store.create_companion(db, uid_sw, relationship_type=COMPANION_TYPE[loss],
                                    subject_name=anchors[0], make_current=True)
        comp_before = len(store.list_companions(db, uid_sw))
        lt_before = store.effective_loss_type(db, uid_sw)
        companion_mod._maybe_transition_scene(db, uid_sw, query, session_id=None)
        comp_after = len(store.list_companions(db, uid_sw))
        lt_after = store.effective_loss_type(db, uid_sw)
    finally:
        _restore_persona(original)

    probes["scene"] = {
        "detector_on": True,
        "dialogue_should_switch": bool(det_real["should_switch"]),
        "dialogue_matched": det_real.get("matched"),
        "positive_control_should_switch": bool(det_pos["should_switch"]),
        "positive_target_scene": det_pos.get("target_scene"),
        "third_person_should_switch": bool(det_third["should_switch"]),
        "companion_count_before": comp_before,
        "companion_count_after": comp_after,
        "loss_type_before": lt_before,
        "loss_type_after": lt_after,
        # 通过判据：普通对话不切 + 正对照能切 + 第三人称不切 + 接线后段数/loss 不变
        "pass": (
            det_real["should_switch"] is False
            and det_pos["should_switch"] is True
            and det_third["should_switch"] is False
            and comp_before == comp_after
            and lt_before == lt_after
        ),
    }

    # ---- F1-V4：画像隔离后对话注入仍拿到正确画像 ----
    uid_p = "ab-portrait-" + uuid.uuid4().hex[:8]
    store.get_or_create_user(db, uid_p, loss_type=loss)
    comp = store.create_companion(db, uid_p, relationship_type=COMPANION_TYPE[loss],
                                  subject_name=anchors[0], make_current=True)
    report = trace.get("report")
    if isinstance(report, dict) and report:
        interview_mod._persist_report(db, uid_p, loss, report)  # 真实落库：带当前 companion.id 归属
    right = store.get_portrait(db, uid_p, "object", companion_id=comp.id)
    wrong = store.get_portrait(db, uid_p, "object", companion_id="nonexistent-companion")
    compat = store.get_portrait(db, uid_p, "object")  # 3 参向后兼容
    ctx = companion_mod._context_block(
        db, uid_p, query,
        {"user": store.get_portrait(db, uid_p, "user", companion_id=comp.id), "object": right},
        [],
    )
    subject_ok = bool(right) and anchors[0] in str(right.get("称呼", ""))
    probes["portrait"] = {
        "written_with_companion_id": bool(comp.id),
        "read_right_cid_nonempty": bool(right),
        "read_wrong_cid_empty": not bool(wrong),
        "read_3arg_backcompat_nonempty": bool(compat),
        "subject_name": (right or {}).get("称呼"),
        "subject_matches_anchor": subject_ok,
        "context_has_portrait": ("[用户画像]" in ctx or "[思念对象画像]" in ctx),
        "context_anchor_hits": sum(1 for a in anchors if a in ctx),
        # 真跑 journey 自身注入证据（主证据，来自 e6.real_journey 的 memory_context）
        "journey_context_has_portrait": bool(trace.get("context_has_portrait")),
        "journey_context_anchor_hits": int(trace.get("context_anchor_hits", 0)),
        "journey_portrait_key_count": int(trace.get("portrait_key_count", 0)),
        "pass": (
            bool(right)
            and not bool(wrong)
            and bool(compat)
            and subject_ok
            and ("[思念对象画像]" in ctx or "[用户画像]" in ctx)
        ),
    }
    probes["pass"] = probes["scene"]["pass"] and probes["portrait"]["pass"]
    return probes


def ab_probes_not_run() -> dict:
    return {
        "status": "NOT_RUN",
        "reason": "mock 模式只验链路结构；A/B 专项为确定性产品函数探针，在真 hy4 模式执行",
        "pass": None,
    }


# ---------------------------------------------------------------- 编排
def run(items: list[dict], mode: str = "mock", fault: str | None = None,
        scenario: str = "all") -> dict:
    """跑三场景完整旅程；5CP 判定完全交给 e6.evaluate_journey。"""
    if mode not in ("mock", "real"):
        raise ValueError(f"unknown mode: {mode}")
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    selected = [it for it in items if scenario == "all" or it["persona"]["loss_type"] == scenario]
    if not selected:
        raise ValueError(f"no persona matched scenario: {scenario}")

    results: list[dict] = []
    client = db = current_db = token = None
    try:
        if mode == "real":
            e6.prepare_real_env()
            from gateway.client import LLMClient
            from memory.db import init_db, _RawSessionLocal
            from memory.scope import current_db as _current_db
            init_db()
            client = LLMClient()
            db = _RawSessionLocal()
            current_db = _current_db
            token = current_db.set(db)
        for item in selected:
            p = item["persona"]
            loss = p["loss_type"]
            print(f"\n=== F1 E6 {mode}: {item.get('id')} / {p.get('tag')} ===")
            trace = e6.mock_trace(item, fault) if mode == "mock" else e6.real_journey(item, db, client)
            scored = e6.evaluate_journey(item, trace)  # 复用，不自造 5CP
            report_field_count = sr.report_field_count(trace.get("report"), loss)
            abp = ab_probes_not_run() if mode == "mock" else ab_probes(item, trace, db)
            results.append({
                "id": item.get("id"), "source_id": item.get("source_id"),
                "tag": p.get("tag"), "loss_type": loss,
                "trace": {k: v for k, v in trace.items() if k != "report"},
                "checkpoints": scored["checkpoints"],
                "e2_checkpoints": scored["e2_checkpoints"],
                "overall_pass": scored["overall_pass"],
                "evidence": scored["evidence"],
                "thresholds": scored["thresholds"],
                "report_field_count": report_field_count,
                "ab_probes": abp,
                "journey_pass": scored["overall_pass"],
                "ab_pass": abp.get("pass"),
                "report": trace.get("report"),
            })
            ev = scored["evidence"]
            print(f"  5CP: {scored['checkpoints']} -> {'PASS' if scored['overall_pass'] else 'FAIL'}")
            print(f"  报告字段={report_field_count}/9 锚点={ev['report_anchors_matched']} "
                  f"轮数={ev['turns']['n_turns']}/filler={ev['turns']['gen_filler_turns']} "
                  f"注入锚点={ev['context_anchor_hits']} 画像字段={ev['portrait_key_count']}")
            if mode == "real":
                print(f"  A/B探针: 切换不误触发={abp['scene']['pass']} "
                      f"(对话switch={abp['scene']['dialogue_should_switch']}, "
                      f"正对照={abp['scene']['positive_control_should_switch']}, "
                      f"段数{abp['scene']['companion_count_before']}->{abp['scene']['companion_count_after']})；"
                      f"画像注入={abp['portrait']['pass']} "
                      f"(右cid读={abp['portrait']['read_right_cid_nonempty']}, "
                      f"错cid空={abp['portrait']['read_wrong_cid_empty']}, "
                      f"journey锚点={abp['portrait']['journey_context_anchor_hits']})")
    finally:
        if current_db is not None and token is not None:
            current_db.reset(token)
        if db is not None:
            db.close()

    pass_counts = {name: sum(1 for r in results if r["checkpoints"].get(name)) for name in e6.CHECKPOINT_NAMES}
    journey_ok = bool(results) and all(r["journey_pass"] for r in results)
    ab_results = [r["ab_pass"] for r in results if r["ab_pass"] is not None]
    ab_ok = bool(ab_results) and all(ab_results)
    overall = journey_ok and (mode == "mock" or ab_ok)
    return {
        "tool": "e6_three_scenario", "schema_version": "1.0",
        "mode": mode, "fault": fault, "scenario": scenario,
        "口径": (
            "mock=构造trace验三场景全链路结构，非产品行为证据；A/B探针NOT_RUN；persona=synthetic-handwritten"
            if mode == "mock" else
            "真实 hy4 三场景各1次完整五段链路（访谈→报告→对话→回写→跨天召回→触发）；"
            "5CP判定复用 e6_full_journey.evaluate_journey；跨天用 ts 回拨3天+新session夹具；"
            "A/B探针=确定性产品函数零LLM（场景切换不误触发/画像隔离注入）；key 不入库"
        ),
        "reuse": "e6_full_journey.{evaluate_journey,real_journey,mock_trace,prepare_real_env,load_batch}; "
                 "scenario_run.report_field_count (9字段口径)",
        "ab_context": "A=场景切换接线 d059174a; B=画像隔离 55589d20; 基线 main af2821ae (544 passed)",
        "results": results,
        "summary": {
            "users": len(results),
            "checkpoint_pass_counts": pass_counts,
            "journey_pass": journey_ok,
            "ab_pass": ab_ok if mode == "real" else None,
            "scene_no_false_trigger": sum(1 for r in results if (r["ab_probes"].get("scene") or {}).get("pass")) or None,
            "portrait_isolation_ok": sum(1 for r in results if (r["ab_probes"].get("portrait") or {}).get("pass")) or None,
            "overall_pass": overall,
            "exit_code": 0 if overall else 1,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="F1 A/B后三场景完整旅程真hy4验证（复用E6 5CP）")
    ap.add_argument("--mock", action="store_true", help="构造 trace 验结构（不烧 key）")
    ap.add_argument("--batch", type=Path, default=DEFAULT_BATCH, help="批次 JSON")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="结果落盘路径")
    ap.add_argument("--scenario", choices=SCENARIOS, default="all", help="选跑场景，默认 all")
    ap.add_argument("--fault", choices=e6.FAULTS, default=None, help="mock 模式注入故障（守卫有效性）")
    args = ap.parse_args(argv)
    if args.fault and not args.mock:
        ap.error("--fault 只能在 --mock 模式使用")

    items = e6.load_batch(args.batch)
    result = run(items, mode="mock" if args.mock else "real", fault=args.fault, scenario=args.scenario)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[F1-E6三场景] {result['summary']} -> {args.out}")
    return int(result["summary"]["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
