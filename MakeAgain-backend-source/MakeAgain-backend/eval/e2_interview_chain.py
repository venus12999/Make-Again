#!/usr/bin/env python3
"""E2 理解段链路评测：访谈 → 报告 的可重复链路检查点 + 两个已知 bug 的回归守卫。

先读码/看实物再设计（本文件的每一条检查点都能追到本地实物）：
- eval/walkthrough_journey.py：真实走查脚本。PERSONAS 三场景（分手/亲友离世/宠物），
  answer(async_report=False) 逐轮推进，action=complete 时拿同步报告。
- outputs/walkthrough/journey_walkthrough.json：真实 hy4 同步报告样本。
  报告字段结构（title/keywords/summary/quote/user_portrait/object_portrait/goal/
  heal_plan/relationship_analysis）取自该实物；失控证据 n_turns=20、gen_filler_turns=15。
- agent/interview.py：
  * _build_report_parsed L656-679：max_tokens=8000（原 4000 时推理模型 reasoning 吃满，
    raw 为空 → ReportGenerationError("INVALID_REPORT_OUTPUT")；修复见 D-042 / 走查分支 5b248c4）。
  * answer() L490-494「覆盖度保护」+ L26 MAX_TURNS=20：修复"答完 5 题后空转 15 轮"失控。
- docs/EVAL_REBUILD_PLAN.md §2 E2；AGENTS.md：链路 bug 只能靠链路评测抓。

分层（用户关心成本）：
  --mock   构造 trace 验评测结构，秒级、不烧 key（验收主力）；
  默认      真实 hy4 跑访谈→报告（约 10 分钟/人），key 只从 work/local-test-key.txt 读，
            不打印明文、不入库。

方法学参考（带日期+距今权重，遵守 AGENTS.md 时间衰减）：
  Anthropic《Demystifying evals for AI agents》2026-01-08，
  https://www.anthropic.com/research/demystifying-evals-for-ai-agents
  距今 2026-11-23 = 319 天 / 30.44 ≈ 10.5 个月 → 权重 0.7（6-18 月档，参考：须查更新版）。
  采纳其"code-based graders 优先（fast/cheap/objective/reproducible）"与
  "把真实失败样本做成回归用例"两条；本评测 5 个检查点全部是确定性代码判定，
  不引入新的 LLM judge，故权重衰减对本设计的实现无实质影响。

用法（在 MakeAgain-backend 目录）：
  python eval/e2_interview_chain.py --mock                       # 全 3 persona 结构验收
  python eval/e2_interview_chain.py --mock --fault invalid_report # 注入 bug1 → exit 1
  python eval/e2_interview_chain.py --mock --fault runaway_followup # 注入 bug2 → exit 1
  python eval/e2_interview_chain.py --limit 1 --out ../outputs/e2/e2_real.json  # 真实 hy4 跑 1 人
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))
DEFAULT_BATCH = HERE / "batches" / "E2_理解段链路.json"
DEFAULT_OUT = BACKEND.parent / "outputs" / "e2" / "e2_chain_result.json"
KEY_FILE = BACKEND.parent / "work" / "local-test-key.txt"

# 阈值口径（从产品代码常量推导，不是拍脑袋）：
# 访谈 5 个维度（agent/interview.py DIMENSIONS），每个维度允许 1 主问 + MAX_FOLLOWUP_PER_DIM=2
# 次追问 → 5×3=15 轮是产品契约内的合法上界；+1 冗余 → 16。
# 实测失控样本 20 轮（gen_filler=15）仍被拦截（4 轮区分度）。
# MAX_TURNS=20 是硬熔断，不能当"合理上限"——否则守卫只能抓到熔断后的 20 轮。
# 软信号：真实 hy4 复跑出现 13 轮/8 filler（模型自收尾，覆盖度保护未提前触发），
# 16 轮门槛不会误杀这种"轻度空转"，但 FILLER_SOFT_LIMIT 会把它作为告警如实暴露。
REASONABLE_MAX_TURNS = 16
FILLER_SOFT_LIMIT = 5
# 个性化判定：报告全文（含画像字段）至少命中 2 个该 persona 的专属锚点。
# 真实报告会改写措辞（如"跨年承诺"），故不要求全中；2/5 足以区分"个性化"与"通用模板"。
PERSONALIZATION_MIN_ANCHORS = 2

REQUIRED_TOP = {
    "title": "str", "summary": "str", "quote": "str", "keywords": "list",
    "user_portrait": "dict", "object_portrait": "dict", "goal": "dict",
    "heal_plan": "dict",
}
# 从真实 hy4 报告实物（journey_walkthrough.json）抽出的关键子字段。
EXPECTED_SUBKEYS = {
    "user_portrait": ("失去类型", "关系与背景", "当前情绪状态", "困惑点"),
    "object_portrait": ("称呼", "关系", "共同记忆"),
    "goal": ("type", "label", "reason"),
    "heal_plan": ("summary", "stages"),
}
# 按失去类型要求的专属分析块（与 interview._persist_report 的消费口径一致）。
ANALYSIS_BLOCK = {"breakup": "relationship_analysis", "loved_one": "grief_analysis", "pet": "pet_grief_analysis"}

CHECKPOINT_NAMES = (
    "CP1_reached_complete",
    "CP2_report_generated",
    "CP3_report_fields_complete",
    "CP4_personalization_match",
    "CP5_followup_turns_bounded",
)


# ---------------------------------------------------------------- 数据 / 工具
def load_batch(path: Path = DEFAULT_BATCH) -> list[dict]:
    """读批次。每条 item 的 persona 段才是链路输入；source_id 必须可追溯。"""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    items = obj.get("items") if isinstance(obj, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError(f"batch has no items: {path}")
    for it in items:
        p = it.get("persona") or {}
        if not it.get("source_id") or not p.get("answers") or not p.get("anchors"):
            raise ValueError(f"item missing source_id/persona.answers/anchors: {it.get('id')}")
    return items


def report_text(report: dict) -> str:
    """报告全文（含嵌套画像）用于锚点匹配；中文不转义。"""
    return json.dumps(report, ensure_ascii=False) if isinstance(report, dict) else ""


def validate_report_fields(report: dict, loss_type: str | None = None) -> tuple[bool, list[str]]:
    """bug1 守卫：报告非空 + 关键字段齐全。返回 (是否齐全, 缺失字段清单)。

    对齐 agent.interview._valid_report 的最小合法性 + 真实报告实物的关键子字段。
    """
    missing: list[str] = []
    if not isinstance(report, dict) or not report:
        return False, ["<report is empty or not a dict>"]

    for key, kind in REQUIRED_TOP.items():
        value = report.get(key)
        if kind == "str":
            if not isinstance(value, str) or (key in ("title", "summary") and not value.strip()):
                missing.append(f"{key}({kind},nonempty)")
        elif kind == "list":
            if not isinstance(value, list) or not value or not all(isinstance(x, str) for x in value):
                missing.append(f"{key}(list[str],nonempty)")
        elif kind == "dict":
            if not isinstance(value, dict) or (key in ("user_portrait", "object_portrait", "goal") and not value):
                missing.append(f"{key}(dict,nonempty)")

    for key, subkeys in EXPECTED_SUBKEYS.items():
        block = report.get(key)
        if not isinstance(block, dict):
            continue  # 顶层已记缺失，不重复
        for sub in subkeys:
            if sub not in block:
                missing.append(f"{key}.{sub}")

    plan = report.get("heal_plan")
    if isinstance(plan, dict) and (not isinstance(plan.get("stages"), list) or not plan["stages"]):
        missing.append("heal_plan.stages(nonempty list)")

    if loss_type:
        block_key = ANALYSIS_BLOCK.get(loss_type)
        if block_key and (not isinstance(report.get(block_key), dict) or not report[block_key]):
            missing.append(f"{block_key}(dict,nonempty,loss_type={loss_type})")

    return not missing, missing


def check_personalization(report: dict, anchors: list[str]) -> tuple[bool, list[str], list[str], int]:
    """bug1 的质量侧检查：报告是否与输入 persona 匹配（防"通用模板"假报告）。"""
    text = report_text(report)
    matched = [a for a in anchors if a and a in text]
    missed = [a for a in anchors if a and a not in text]
    return len(matched) >= PERSONALIZATION_MIN_ANCHORS, matched, missed, PERSONALIZATION_MIN_ANCHORS


def check_turns(n_turns: int, gen_filler_turns: int) -> dict:
    """bug2 守卫：追问轮数上限检查。所有数字都是真实 trace 里的整数。"""
    return {
        "n_turns": int(n_turns),
        "gen_filler_turns": int(gen_filler_turns),
        "max_allowed_turns": REASONABLE_MAX_TURNS,
        "ok": int(n_turns) <= REASONABLE_MAX_TURNS,
    }


def evaluate_trace(item: dict, trace: dict) -> dict:
    """纯函数：把一条 trace 打成 5 个检查点。mock / 真实模式共用同一套判定。"""
    persona = item.get("persona") or {}
    loss_type = persona.get("loss_type")
    report = trace.get("report")
    turns = check_turns(trace.get("n_turns", 0), trace.get("gen_filler_turns", 0))

    fields_ok, missing = validate_report_fields(report, loss_type)
    if report:
        pers_ok, matched, missed, min_a = check_personalization(report, persona.get("anchors") or [])
    else:
        pers_ok, matched, missed, min_a = False, [], persona.get("anchors") or [], PERSONALIZATION_MIN_ANCHORS

    cps = {
        "CP1_reached_complete": bool(trace.get("reached_complete") and trace.get("done")),
        "CP2_report_generated": bool(isinstance(report, dict) and bool(report)),
        "CP3_report_fields_complete": bool(fields_ok),
        "CP4_personalization_match": bool(pers_ok),
        "CP5_followup_turns_bounded": bool(turns["ok"]),
    }
    return {
        "checkpoints": cps,
        "overall_pass": all(cps.values()),
        "evidence": {
            "missing_fields": missing,
            "personalization": {
                "matched_anchors": matched, "missed_anchors": missed,
                "min_required": min_a, "total_anchors": len(persona.get("anchors") or []),
            },
            "turns": turns,
            "soft_warnings": (["FILLER_TURNS_HIGH"] if turns["gen_filler_turns"] > FILLER_SOFT_LIMIT else []),
            "done": bool(trace.get("done")),
            "report_ready": bool(trace.get("report_ready")),
            "elapsed_s": trace.get("elapsed_s"),
        },
    }


# ---------------------------------------------------------------- mock 后端
def _mock_report(persona: dict, include_anchors: bool = True) -> dict:
    """用 persona 锚点拼一份结构合法的假报告（仅验评测结构，非产品输出）。"""
    a = list(persona.get("anchors") or [])
    hook = "；".join(a[:3]) if include_anchors else "那段经历"
    name = a[0] if include_anchors else "TA"
    return {
        "title": f"给还在路上的你（mock · {persona.get('loss_type')}）",
        "keywords": (a[:5] if include_anchors else ["失去", "怀念", "慢慢走"]),
        "summary": f"mock 报告：{hook}。你说起{name}，这些都还在；带着记忆继续生活。" if include_anchors else "mock 报告：这是一段关于失去与继续生活的通用描述。",
        "quote": "mock quote: 慢慢来。",
        "user_portrait": {"失去类型": persona.get("loss_type"), "关系与背景": hook,
                          "当前情绪状态": "难过但愿意讲述", "困惑点": "放不下"},
        "object_portrait": {"称呼": name, "关系": "重要的人", "共同记忆": hook},
        "goal": {"type": "carry_on", "label": "带着记忆走下去", "reason": hook},
        "heal_plan": {"summary": "先稳住，再慢慢走。",
                      "stages": [{"title": "先稳住", "desc": "吃饭睡觉", "time": "第1-2周"},
                                 {"title": "慢慢走", "desc": "允许想念", "time": "第3-4周"}]},
        ANALYSIS_BLOCK.get(persona.get("loss_type"), "relationship_analysis"): {
            "summary": hook, "patterns": "mock", "narrative": hook,
        },
    }


def mock_trace(item: dict, fault: str | None = None) -> dict:
    """构造 trace。fault 只模拟 trace 层症状，不改判定逻辑——用于证明守卫有效。"""
    p = item["persona"]
    answers = list(p["answers"])
    turns: list[dict] = []
    gen_filler = 0

    if fault == "runaway_followup":
        # 复刻 walkthrough 实测：5 题答完后模型空转，直到 MAX_TURNS=20 才熔断。
        hard_cap_turns = 20
        for i in range(hard_cap_turns):
            if answers:
                ans = answers.pop(0)
            else:
                gen_filler += 1
                ans = "嗯，我补充一点：还在慢慢消化。"
            turns.append({"turn": i + 1, "question": f"还能再说说吗？（mock 空转 {i+1}）",
                          "answer": ans[:60], "action": "complete" if i == hard_cap_turns - 1 else "followup"})
        n_turns = 20
    else:
        # 健康路径：维度答完即 complete（与修复后 interview.answer 的覆盖度保护一致）。
        for i, ans in enumerate(answers):
            turns.append({"turn": i + 1, "question": f"维度{i+1}的问题？",
                          "answer": ans[:60], "action": "complete" if i == len(answers) - 1 else "next"})
        n_turns = len(answers)

    if fault == "invalid_report":
        report = {}  # 模拟 max_tokens 不足 → raw 空 → INVALID_REPORT_OUTPUT 的可见症状
    elif fault == "generic_report":
        report = _mock_report(p, include_anchors=False)
    else:
        report = _mock_report(p, include_anchors=True)

    return {
        "tag": p.get("tag"), "loss_type": p.get("loss_type"), "n_turns": n_turns,
        "gen_filler_turns": gen_filler, "done": True, "report_ready": bool(report),
        "reached_complete": bool(turns and turns[-1]["action"] == "complete"),
        "elapsed_s": 0.0, "report": report, "turns": turns,
    }


# ---------------------------------------------------------------- 真实 hy4 后端
def load_conf() -> dict:
    """只读 key 文件的键名/值，不打印、不入库。"""
    conf: dict[str, str] = {}
    for line in KEY_FILE.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            conf[k.strip()] = v.strip()
    return conf


def prepare_real_env() -> None:
    """复刻 walkthrough_journey.main 的环境装配（隔离 DB + hy4 + extra headers）。"""
    conf = load_conf()
    os.environ.update(
        LLM_BASE_URL=conf["BASE_URL"], LLM_API_KEY=conf["KEY"],
        LLM_FAST_MODEL="hy4-flash-free", LLM_MODEL="hy4",
        MOCK_LLM="false", ENVIRONMENT="development",
        LLM_EXTRA_HEADERS=json.dumps({
            "x-opencode-session": "e2-chain-" + uuid.uuid4().hex[:8],
            "User-Agent": "make-again-e2-chain/1.0",
        }),
        DATABASE_URL="sqlite:///" + tempfile.mkdtemp(prefix="e2chain-") + "/t.sqlite3",
        MEDIA_SIGNING_KEY="x" * 40,
        STORAGE_LOCAL_DIR=tempfile.mkdtemp(prefix="e2chain-media-"),
        PUBLIC_BASE_URL="http://testserver",
    )
    from config import get_settings
    get_settings.cache_clear()
    s = get_settings()
    s.llm_base_url = conf["BASE_URL"]
    s.llm_api_key = conf["KEY"]
    s.llm_fast_model = "hy4-flash-free"
    s.llm_model = "hy4"
    s.llm_report_model = ""
    s.mock_llm = False


def real_trace(item: dict, db, client) -> dict:
    """真实走查一个 persona：start → answer 逐轮 → 同步报告。逻辑对齐 walkthrough_journey.walk_one。"""
    from agent import interview
    from memory import store
    from memory.db import init_db
    from memory.scope import current_db

    init_db()
    p = item["persona"]
    uid = "e2-" + uuid.uuid4().hex[:8]
    store.get_or_create_user(db, uid)
    t0 = time.time()
    st = interview.start(db, uid, p["loss_type"])
    sid = st["session_id"]
    answers = list(p["answers"])
    turns: list[dict] = []
    gen_filler = 0
    reached_complete = False

    for turn in range(REASONABLE_MAX_TURNS + 10):  # 允许越过阈值，才能测出失控
        sess = store.get_interview(db, sid)
        if not sess or (sess.state or {}).get("done"):
            break
        state = sess.state or {}
        qs = [h.get("content", "") for h in state.get("history", []) if h.get("role") == "assistant"]
        question = qs[-1] if qs else ""
        if answers:
            ans = answers.pop(0)
        else:
            gen_filler += 1
            ans = "嗯，我补充一点：这件事我一直没细说，现在跟你讲了心里松一点，还在慢慢消化。"
        resp = interview.answer(db, sid, ans, client, async_report=False)
        turns.append({"turn": turn + 1, "question": question[:120], "answer": ans[:80],
                      "action": resp.get("action"), "done": resp.get("done")})
        print(f"  turn{turn+1}: action={resp.get('action')} q={question[:36]}")
        if resp.get("action") == "complete":
            reached_complete = True
            break

    sess = store.get_interview(db, sid)
    state = (sess.state or {}) if sess else {}
    report = state.get("report")
    return {
        "tag": p.get("tag"), "loss_type": p.get("loss_type"),
        "session_id": sid, "n_turns": len(turns), "gen_filler_turns": gen_filler,
        "done": bool(state.get("done")), "report_ready": bool(state.get("report_ready")),
        "reached_complete": reached_complete, "elapsed_s": round(time.time() - t0, 1),
        "report": report, "turns": turns,
    }


# ---------------------------------------------------------------- 编排
def run(items: list[dict], mode: str = "mock", fault: str | None = None,
        limit: int = 0) -> dict:
    """跑 E2，返回结构化结果。mode=mock 不碰 key/网络；mode=real 现场生成报告。"""
    if mode not in ("mock", "real"):
        raise ValueError(f"unknown mode: {mode}")
    selected = items[:limit] if limit else items
    results: list[dict] = []
    client = None
    db = None
    current_db = None
    token = None
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
            p = item["persona"]
            print(f"\n=== E2 {mode}: {item.get('id')} / {p.get('tag')} ===")
            if mode == "mock":
                trace = mock_trace(item, fault)
            else:
                trace = real_trace(item, db, client)
            scored = evaluate_trace(item, trace)
            results.append({
                "id": item.get("id"), "source_id": item.get("source_id"),
                "tag": p.get("tag"), "loss_type": p.get("loss_type"),
                "trace": {k: v for k, v in trace.items() if k != "report"},
                "checkpoints": scored["checkpoints"], "overall_pass": scored["overall_pass"],
                "evidence": scored["evidence"],
                "report": trace.get("report"),  # mock 为构造值；real 为 hy4 真实输出
            })
            print(f"  检查点: {scored['checkpoints']} -> {'PASS' if scored['overall_pass'] else 'FAIL'}")
            print(f"  轮数: n_turns={scored['evidence']['turns']['n_turns']} "
                  f"filler={scored['evidence']['turns']['gen_filler_turns']} "
                  f"上限={scored['evidence']['turns']['max_allowed_turns']}")
    finally:
        if current_db is not None and token is not None:
            current_db.reset(token)
        if db is not None:
            db.close()

    pass_counts = {name: sum(1 for r in results if r["checkpoints"].get(name)) for name in CHECKPOINT_NAMES}
    overall = bool(results) and all(r["overall_pass"] for r in results)
    return {
        "tool": "e2_interview_chain", "schema_version": "1.0", "mode": mode, "fault": fault,
        "口径": ("mock=构造trace验评测结构，非产品行为证据；persona=synthetic-handwritten"
                 if mode == "mock" else "真实 hy4 同步访谈→报告；persona=synthetic-handwritten；key 不入库"),
        "threshold": {"reasonable_max_turns": REASONABLE_MAX_TURNS,
                      "filler_soft_limit": FILLER_SOFT_LIMIT,
                      "personalization_min_anchors": PERSONALIZATION_MIN_ANCHORS},
        "results": results,
        "summary": {"users": len(results), "checkpoint_pass_counts": pass_counts,
                    "soft_warning_users": sum(1 for r in results if r["evidence"]["soft_warnings"]),
                    "overall_pass": overall, "exit_code": 0 if overall else 1},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E2 访谈→报告链路评测")
    ap.add_argument("--mock", action="store_true", help="构造 trace 验结构（不烧 key）")
    ap.add_argument("--batch", type=Path, default=DEFAULT_BATCH, help="批次 JSON")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="结果落盘路径")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个 persona（真实模式调试用）")
    ap.add_argument("--fault", choices=(None, "invalid_report", "runaway_followup", "generic_report"),
                    default=None, help="mock 模式注入故障，构造性验证守卫")
    args = ap.parse_args(argv)

    items = load_batch(args.batch)
    result = run(items, mode="mock" if args.mock else "real", fault=args.fault, limit=args.limit)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(f"\n[E2] {result['summary']} -> {args.out}")
    return int(result["summary"]["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
