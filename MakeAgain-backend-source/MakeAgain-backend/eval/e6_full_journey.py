#!/usr/bin/env python3
"""E6 跨段链路评测：3 场景 persona 各走一遍五段闭环，在段间衔接点设检查。

先读码/看实物再设计（每个检查点都能追到本地实物）：
- eval/walkthrough_journey.py：三场景 PERSONAS + 同步报告走查脚本；真实输出见
  outputs/walkthrough/journey_walkthrough.json（n_turns=20 / gen_filler=15 / 报告字段齐全）。
- eval/e2_interview_chain.py：访谈→报告 5 检查点；E6 的 CP1 直接复用它的 evaluate_trace，
  真跑数字见 outputs/e2/e2_real_breakup_rescored.json（13 轮 / 8 filler / 415.3s / 锚点 5/5）。
- agent/interview.py::_persist_report（L682-739）：报告落画像（user/objectPortrait）+ type=report
  记忆（importance=9）+ 分析报告记忆（importance=8）→ 这是「报告→对话」衔接的写入侧。
- agent/companion.py::_context_block（L86-119）/ breakup_chat（L45-117）：对话时注入
  [用户画像]/[思念对象画像]/[相关记忆] 并回传 memory_context → 这是衔接的读取侧。
- agent/breakup.py L103：schedule_memory_write(extractor=breakup_extract...) → durable Job →
  memory/async_write.py::_do_write（L69-150）落 Fact + reflect + 索引 → 「对话→回写」衔接。
- memory/recall.py（recall L133-153 / _recall_dual L556-663）：query-intent 门 + 双路召回 →
  「记忆→跨天召回」衔接；agent/nudge.py::_build_candidates（L80-127）与
  agent/daily.py::generate_theme_guide（L140-189）→ 「召回→触发时机」衔接。

分层（用户关心成本，详见 docs/E6_FULL_JOURNEY_RESULT.md）：
  日常快评：E1 单点 / E2 --mock / E6 --mock（本工具，秒级、不烧 key）；
  每周/大改深评：E2 真 hy4 + E6 默认真 hy4 只跑 1 场景（--limit 1，约 10 分钟），
  3 场景全真跑不是日常门禁。key 只从 work/local-test-key.txt 读，不打印、不入库。

方法学参考（带 URL + 日期 + 按天精确算的衰减权重，遵守 AGENTS.md）：
  Anthropic《Demystifying evals for AI agents》2026-01-08，
  https://www.anthropic.com/research/demystifying-evals-for-ai-agents
  基准日 2026-11-23：(2026-11-23 − 2026-01-08).days = 319 天，319/30.44 ≈ 10.5 个月
  → 权重 0.7（6-18 月档，参考：须查更新版）。采纳其两条：
  ①code-based graders 优先（本工具全部检查点为确定性代码判定，不新增 LLM judge）；
  ②把真实失败做成回归用例（--fault 四种故障注入即构造性守卫）。
  本轮无新增外部调研；其余依据均为本地一手代码/走查实物（权重 1.0）。
  中文原生集缺口沿用 D-031 结论：宠物离世中文权威集 NOT_FOUND，故自造并标 synthetic-handwritten。

用法（在 MakeAgain-backend 目录）：
  python eval/e6_full_journey.py --mock                       # 3 场景全链路结构验收（不烧 key）
  python eval/e6_full_journey.py --mock --fault recall_miss   # 注入故障 → exit 1（守卫有效）
  python eval/e6_full_journey.py --limit 1 --out ../outputs/e6/e6_real_breakup.json  # 真 hy4 1 场景
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
REPO = BACKEND.parent
for _p in (str(BACKEND), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e2_interview_chain as e2  # noqa: E402 - 复用 E2 的报告判定（只读，不改其所有权文件）

DEFAULT_BATCH = HERE / "batches" / "E6_完整旅程.json"
DEFAULT_OUT = REPO / "outputs" / "e6" / "e6_journey_result.json"
KEY_FILE = REPO / "work" / "local-test-key.txt"

CHECKPOINT_NAMES = (
    "CP1_report_ready",
    "CP2_report_to_dialogue",
    "CP3_dialogue_to_writeback",
    "CP4_memory_to_crossday_recall",
    "CP5_recall_to_trigger",
)
# 口径（自造，标 synthetic-handwritten 批次；阈值来自产品常量/实测，不拍脑袋）：
# - 跨天模拟：把记忆 ts 回拨 3 天 + 用新 session_id 模拟重开 App（dual 索引 meta 不回拨，
#   本检查只验"跨天后仍可召回"，不验时间衰减曲线）。
# - 负例 query 用 U-013 已验证的"今天晚饭吃什么"（query-intent 门应返回 0 条）。
# - 触发时机：本地 23:30 应出 nudge 候选；14:30 不应出 late_night 规则。
SIMULATED_DAYS = 3
MIN_CONTEXT_ANCHORS = 1
MIN_RECALL_HITS = 1
MAX_NEGATIVE_RECALL = 0
FAULTS = (None, "missing_report_injection", "missing_writeback", "recall_miss", "trigger_miss")


# ---------------------------------------------------------------- 数据 / 纯判定
def load_batch(path: Path = DEFAULT_BATCH) -> list[dict]:
    """读批次。每条 item 需 persona（E2 报告输入）+ journey（E6 衔接输入），来源可追溯。"""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    items = obj.get("items") if isinstance(obj, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError(f"batch has no items: {path}")
    for it in items:
        p = it.get("persona") or {}
        j = it.get("journey") or {}
        if not it.get("source_id") or not (it.get("source") or {}).get("dataset"):
            raise ValueError(f"item not traceable: {it.get('id')}")
        if not p.get("answers") or not p.get("anchors") or not p.get("loss_type"):
            raise ValueError(f"item missing persona: {it.get('id')}")
        for key in ("dialogue_query", "recall_query", "recall_negative_query",
                    "trigger_local_now", "trigger_daytime_now"):
            if not j.get(key):
                raise ValueError(f"item missing journey.{key}: {it.get('id')}")
    return items


def evaluate_journey(item: dict, trace: dict) -> dict:
    """纯函数：把一个完整旅程 trace 打成 5 个衔接检查点。mock / real 共用同一判定。

    每个证据字段都是整数/布尔，不用平均分糊过去；失败能定位到具体环节。
    """
    persona = item["persona"]
    anchors = persona.get("anchors") or []
    e2_scored = e2.evaluate_trace(item, trace)  # CP1 复用 E2 的 5 检查点判定

    cps = {
        "CP1_report_ready": bool(e2_scored["overall_pass"]),
        "CP2_report_to_dialogue": bool(
            int(trace.get("report_memory_count", 0)) >= 1
            and int(trace.get("portrait_key_count", 0)) >= 1
            and trace.get("context_has_portrait")
            and int(trace.get("context_anchor_hits", 0)) >= MIN_CONTEXT_ANCHORS
        ),
        "CP3_dialogue_to_writeback": bool(
            trace.get("writeback_memory_id")
            and int(trace.get("memories_after", 0)) > int(trace.get("memories_before", 0))
            and int(trace.get("writeback_anchor_hits", 0)) >= MIN_CONTEXT_ANCHORS
        ),
        "CP4_memory_to_crossday_recall": bool(
            int(trace.get("days_shift", 0)) >= 1
            and int(trace.get("recall_hits", 0)) >= MIN_RECALL_HITS
            and trace.get("recall_top_has_anchor")
            and int(trace.get("negative_recall_hits", 0)) <= MAX_NEGATIVE_RECALL
        ),
        "CP5_recall_to_trigger": bool(
            trace.get("late_nudge_rule")
            and not trace.get("daytime_late_nudge_rule")
            and trace.get("guide_personalized")
        ),
    }
    return {
        "checkpoints": cps,
        "overall_pass": all(cps.values()),
        "e2_checkpoints": e2_scored["checkpoints"],
        "evidence": {
            # 段1：访谈→报告（真实数字来自 E2 判定）
            "turns": e2_scored["evidence"]["turns"],
            "missing_fields": e2_scored["evidence"]["missing_fields"],
            "report_anchors_matched": e2_scored["evidence"]["personalization"]["matched_anchors"],
            # 段2：报告→对话记忆注入
            "report_memory_count": int(trace.get("report_memory_count", 0)),
            "portrait_key_count": int(trace.get("portrait_key_count", 0)),
            "context_has_portrait": bool(trace.get("context_has_portrait")),
            "context_anchor_hits": int(trace.get("context_anchor_hits", 0)),
            # 段3：对话→回写
            "job_status": trace.get("job_status"),
            "memories_before": int(trace.get("memories_before", 0)),
            "memories_after": int(trace.get("memories_after", 0)),
            "writeback_anchor_hits": int(trace.get("writeback_anchor_hits", 0)),
            # 段4：记忆→跨天召回
            "days_shift": int(trace.get("days_shift", 0)),
            "recall_hits": int(trace.get("recall_hits", 0)),
            "recall_top_has_anchor": bool(trace.get("recall_top_has_anchor")),
            "recall_top_score": float(trace.get("recall_top_score", 0.0)),
            "negative_recall_hits": int(trace.get("negative_recall_hits", 0)),
            # 段5：召回→触发
            "late_nudge_rule": trace.get("late_nudge_rule"),
            "daytime_late_nudge_rule": trace.get("daytime_late_nudge_rule"),
            "guide_personalized": bool(trace.get("guide_personalized")),
            "guide_anchor_hits": int(trace.get("guide_anchor_hits", 0)),
            "total_anchors": len(anchors),
        },
        "thresholds": {
            "days_shift_min": SIMULATED_DAYS,
            "context_anchor_hits_min": MIN_CONTEXT_ANCHORS,
            "recall_hits_min": MIN_RECALL_HITS,
            "negative_recall_hits_max": MAX_NEGATIVE_RECALL,
        },
    }


# ---------------------------------------------------------------- mock 后端
def mock_trace(item: dict, fault: str | None = None) -> dict:
    """构造 trace（仅验评测结构，非产品行为证据）。数字按 persona 锚点数确定性生成。"""
    p = item["persona"]
    anchors = p["anchors"]
    report = e2._mock_report(p, include_anchors=True)
    ctx_hits = min(3, len(anchors))
    trace = {
        "tag": p["tag"], "loss_type": p["loss_type"],
        "n_turns": len(p["answers"]), "gen_filler_turns": 0,
        "done": True, "report_ready": True, "reached_complete": True,
        "elapsed_s": 0.0, "report": report,
        "report_memory_count": 2,           # 报告 summary + 场景分析各一条（见 _persist_report）
        "portrait_key_count": 11 + len(anchors) % 3,
        "context_has_portrait": True, "context_anchor_hits": ctx_hits,
        "job_status": "completed",
        "memories_before": 2, "memories_after": 3,
        "writeback_memory_id": "mock-mem-" + uuid.uuid4().hex[:8],
        "writeback_anchor_hits": min(2, len(anchors)),
        "days_shift": SIMULATED_DAYS,
        "recall_hits": 2, "recall_top_has_anchor": True, "recall_top_score": 0.8333,
        "negative_recall_hits": 0,
        "late_nudge_rule": "emotion:晚上", "daytime_late_nudge_rule": None,
        "guide_personalized": True, "guide_anchor_hits": min(2, len(anchors)),
    }
    # 故障只改 trace 症状，不改判定逻辑——证明每个衔接检查点不是摆设。
    if fault == "missing_report_injection":
        trace.update(context_has_portrait=False, context_anchor_hits=0)
    elif fault == "missing_writeback":
        trace.update(memories_after=trace["memories_before"], writeback_memory_id=None,
                     writeback_anchor_hits=0, job_status="failed")
    elif fault == "recall_miss":
        trace.update(recall_hits=0, recall_top_has_anchor=False, recall_top_score=0.0)
    elif fault == "trigger_miss":
        trace.update(late_nudge_rule=None, guide_personalized=False, guide_anchor_hits=0)
    return trace


# ---------------------------------------------------------------- 真实 hy4 后端
def prepare_real_env() -> None:
    """复刻 walkthrough/e2 的环境装配：隔离 DB + hy4 + extra headers。key 不打印。"""
    conf: dict[str, str] = {}
    for line in KEY_FILE.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            conf[k.strip()] = v.strip()
    os.environ.update(
        LLM_BASE_URL=conf["BASE_URL"], LLM_API_KEY=conf["KEY"],
        LLM_FAST_MODEL="hy4-flash-free", LLM_MODEL="hy4",
        MOCK_LLM="false", ENVIRONMENT="development",
        LLM_EXTRA_HEADERS=json.dumps({
            "x-opencode-session": "e6-journey-" + uuid.uuid4().hex[:8],
            "User-Agent": "make-again-e6-journey/1.0",
        }),
        DATABASE_URL="sqlite:///" + tempfile.mkdtemp(prefix="e6journey-") + "/t.sqlite3",
        MEDIA_SIGNING_KEY="x" * 40,
        STORAGE_LOCAL_DIR=tempfile.mkdtemp(prefix="e6journey-media-"),
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


def _real_interview(item: dict, db, client, uid: str) -> dict:
    """E2 真跑的访谈段（逻辑对齐 e2.real_trace），uid 由调用方建好陪伴对象后传入。"""
    from agent import interview
    from memory import store

    p = item["persona"]
    t0 = time.time()
    st = interview.start(db, uid, p["loss_type"])
    sid = st["session_id"]
    answers = list(p["answers"])
    turns: list[dict] = []
    gen_filler = 0
    reached_complete = False
    for turn in range(e2.REASONABLE_MAX_TURNS + 10):
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
        turns.append({"turn": turn + 1, "question": question[:100], "answer": ans[:60],
                      "action": resp.get("action"), "done": resp.get("done")})
        print(f"  [E6访谈] turn{turn+1}: action={resp.get('action')} q={question[:30]}")
        if resp.get("action") == "complete":
            reached_complete = True
            break
    sess = store.get_interview(db, sid)
    state = (sess.state or {}) if sess else {}
    return {
        "tag": p["tag"], "loss_type": p["loss_type"], "session_id": sid,
        "n_turns": len(turns), "gen_filler_turns": gen_filler,
        "done": bool(state.get("done")), "report_ready": bool(state.get("report_ready")),
        "reached_complete": reached_complete, "report": state.get("report"),
        "elapsed_s": round(time.time() - t0, 1), "turns": turns,
    }


def _run_memory_worker(uid: str, max_jobs: int = 6) -> str:
    """跑 durable job worker 直到本 user 的 memory job 完成，返回终态（pv: 评测驱动真实 worker）。"""
    from sqlalchemy import select
    from memory.b07_models import Job
    from memory.db import _RawSessionLocal
    import admin_cli

    status = "NOT_ENQUEUED"
    for _ in range(max_jobs):
        with _RawSessionLocal() as chk:
            jobs = chk.scalars(select(Job).where(Job.user_id == uid).order_by(Job.id)).all()
            mem = [j for j in jobs if j.kind == "memory"]
            status = mem[-1].status if mem else "NOT_ENQUEUED"
            pending = any(j.status in ("pending", "retry") for j in jobs)
        if not pending:
            break
        if not admin_cli.run_one():
            break
    return status


def real_journey(item: dict, db, client) -> dict:
    """真 hy4 走 1 个 persona 的五段闭环（评测用隔离 DB；eval 侧只做时间/scope 夹具）。"""
    from datetime import timedelta

    from agent import companion, daily, nudge
    from agent import interview  # noqa: F401 - 与 _real_interview 同源，确保 interview 模块已装载
    from memory import recall as recall_mod
    from memory import store
    from memory.db import _RawSessionLocal
    from memory.models import utcnow
    from sqlalchemy import select
    from memory.b07_models import Job

    p = item["persona"]
    j = item["journey"]
    anchors = p["anchors"]

    # --- 夹具（口径：eval 侧模拟真实用户已完成 onboarding consent，不改产品代码）---
    # 注意：不在评测会话上设 db.info['companion_id']。scope filter 会按 companion_id
    # 过滤 Portrait/MemoryEntry，而 store.add_memory/upsert_portrait 写的是 NULL-scope
    # 行（这是产品现状），一旦设置就会把刚落的报告记忆和画像全部藏掉，链路假失败。
    # 只在投递给 worker 的 Job 行上补 companion_id（见下方 flush 之后），让 worker 的
    # UC-4 scope guard 有真实可校验的归属，同时评测会话能读全链路数据。
    uid = "e6-" + uuid.uuid4().hex[:8]
    store.get_or_create_user(db, uid, loss_type=p["loss_type"])
    companion_id_map = {"breakup": "breakup", "loved_one": "relative", "pet": "pet"}
    comp = store.create_companion(db, uid, relationship_type=companion_id_map[p["loss_type"]],
                                  subject_name=anchors[0], make_current=True)
    state = store.get_or_create_user_state(db, uid)
    state.consent_accepted_at = utcnow()
    db.commit()
    trace = _real_interview(item, db, client, uid)

    # --- 段2：报告落库 → 对话注入（真实产品函数） ---
    report_mems = [m for m in store.list_memories(db, uid) if m.type == "report"]
    portraits = {"user": store.get_portrait(db, uid, "user"),
                 "object": store.get_portrait(db, uid, "object")}
    portrait_key_count = sum(len(v) for v in portraits.values() if isinstance(v, dict))
    chat_resp = companion.chat(db, uid, j["dialogue_query"], client, session_id=None, no_tools=True)
    ctx = chat_resp.get("memory_context") or ""
    context_anchor_hits = sum(1 for a in anchors if a in ctx)
    trace.update(
        report_memory_count=len(report_mems),
        portrait_key_count=portrait_key_count,
        context_has_portrait=("[用户画像]" in ctx or "[思念对象画像]" in ctx),
        context_anchor_hits=context_anchor_hits,
    )
    print(f"  [E6段2] 报告记忆={len(report_mems)} 画像字段={portrait_key_count} "
          f"上下文锚点={context_anchor_hits}/{len(anchors)}")

    # --- 段3：对话 → durable job → worker 回写（真实产品函数） ---
    before_ids = {str(m.id) for m in store.list_memories(db, uid)}
    from memory.jobs import flush_intents
    flush_intents(db)
    db.commit()  # 评测会话非 managed：真提交，worker 的独立 session 才看得见 Job
    # eval 夹具：给本 user 的待处理 Job 补 companion 归属，使 worker 的 scope guard
    # 走"有归属 + 已 consent"分支（产品主链路代码零改动）。
    with _RawSessionLocal() as chk:
        pending_jobs = chk.scalars(select(Job).where(
            Job.user_id == uid, Job.status.in_(("pending", "retry")))).all()
        for job_row in pending_jobs:
            job_row.companion_id = comp.id
        chk.commit()
    job_status = _run_memory_worker(uid)
    with _RawSessionLocal() as chk:
        job_row = chk.scalars(select(Job).where(Job.user_id == uid, Job.kind == "memory")
                              .order_by(Job.id.desc())).first()
        job_error = (job_row.error_code if job_row else None)
    after = store.list_memories(db, uid)
    new_entries = [m for m in after if str(m.id) not in before_ids]
    newest = max(new_entries, key=lambda m: m.ts) if new_entries else None
    writeback_text = ((newest.summary or "") + " " + (newest.content or "")) if newest else ""
    trace.update(
        job_status=f"{job_status}" + (f":{job_error}" if job_error else ""),
        memories_before=len(before_ids), memories_after=len(after),
        writeback_memory_id=str(newest.id) if newest else None,
        writeback_anchor_hits=sum(1 for a in anchors if a in writeback_text),
    )
    print(f"  [E6段3] job={trace['job_status']} 记忆 {len(before_ids)}→{len(after)} "
          f"锚点命中={trace['writeback_anchor_hits']}")

    # --- 段4：跨天召回（夹具：ts 回拨 3 天 + 新 session_id 模拟重开 App）---
    for m in store.list_memories(db, uid):
        m.ts = m.ts - timedelta(days=SIMULATED_DAYS)
    db.commit()
    hits = recall_mod.recall(db, uid, j["recall_query"], top_k=5, recent_text="")
    neg_hits = recall_mod.recall(db, uid, j["recall_negative_query"], top_k=5, recent_text="")
    top = hits[0]["entry"] if hits else None
    top_text = ((top.summary or "") + " " + (top.content or "")) if top else ""
    trace.update(
        days_shift=SIMULATED_DAYS, recall_hits=len(hits),
        recall_top_has_anchor=any(a in top_text for a in anchors),
        recall_top_score=float(hits[0]["score"]) if hits else 0.0,
        negative_recall_hits=len(neg_hits),
    )
    print(f"  [E6段4] 回拨{SIMULATED_DAYS}天 recall_hits={len(hits)} "
          f"top_score={trace['recall_top_score']:.4f} 负例={len(neg_hits)}")

    # --- 段5：召回 → 触发时机（nudge 规则 + daily 个性化开场，真实产品函数）---
    late_now = dt.datetime.fromisoformat(j["trigger_local_now"])
    day_now = dt.datetime.fromisoformat(j["trigger_daytime_now"])
    late_cands = nudge._build_candidates(db, uid, late_now, breakup=p["loss_type"] == "breakup")
    day_cands = nudge._build_candidates(db, uid, day_now, breakup=p["loss_type"] == "breakup")
    theme_desc = daily.get_themes(db, uid, now=day_now)["themes"][0]["desc"]
    guide = daily.generate_theme_guide(db, uid, client, now=day_now)
    guide_text = guide.get("guide") or ""
    trace.update(
        late_nudge_rule=late_cands[0]["rule_key"] if late_cands else None,
        daytime_late_nudge_rule=("late_night" if any(c["rule_key"] == "late_night" for c in day_cands) else None),
        guide_personalized=bool(guide_text and guide_text != theme_desc),
        guide_anchor_hits=sum(1 for a in anchors if a in guide_text),
    )
    print(f"  [E6段5] 深夜触发={trace['late_nudge_rule']} 白天误触发="
          f"{trace['daytime_late_nudge_rule']} 引导个性化={trace['guide_personalized']}")
    # 对话/引导原文不入库到结果 JSON 之外的任何地方；key 从不出现。
    trace["dialogue_reply_preview"] = (chat_resp.get("reply") or "")[:120]
    trace["guide_preview"] = guide_text[:120]
    return trace


# ---------------------------------------------------------------- 编排
def run(items: list[dict], mode: str = "mock", fault: str | None = None,
        limit: int = 0) -> dict:
    """跑 E6，返回结构化结果。mode=mock 不碰 key/网络；mode=real 现场走完整旅程。"""
    if mode not in ("mock", "real"):
        raise ValueError(f"unknown mode: {mode}")
    selected = items[:limit] if limit else items
    results: list[dict] = []
    db = None
    token = None
    current_db = None
    try:
        if mode == "real":
            prepare_real_env()
            from gateway.client import LLMClient
            from memory.db import init_db, _RawSessionLocal
            from memory.scope import current_db as _current_db
            init_db()
            client = LLMClient()
            db = _RawSessionLocal()
            current_db = _current_db
            token = current_db.set(db)
        else:
            client = None
        for item in selected:
            p = item["persona"]
            print(f"\n=== E6 {mode}: {item.get('id')} / {p.get('tag')} ===")
            trace = mock_trace(item, fault) if mode == "mock" else real_journey(item, db, client)
            scored = evaluate_journey(item, trace)
            results.append({
                "id": item.get("id"), "source_id": item.get("source_id"),
                "tag": p.get("tag"), "loss_type": p.get("loss_type"),
                "trace": {k: v for k, v in trace.items() if k != "report"},
                "e2_checkpoints": scored["e2_checkpoints"],
                "checkpoints": scored["checkpoints"],
                "overall_pass": scored["overall_pass"],
                "evidence": scored["evidence"], "thresholds": scored["thresholds"],
                "report": trace.get("report"),  # mock 为构造值；real 为 hy4 真实输出
            })
            print(f"  检查点: {scored['checkpoints']} -> {'PASS' if scored['overall_pass'] else 'FAIL'}")
            ev = scored["evidence"]
            print(f"  段间数字: 报告记忆={ev['report_memory_count']} 注入锚点={ev['context_anchor_hits']} "
                  f"记忆{ev['memories_before']}→{ev['memories_after']} 召回={ev['recall_hits']} "
                  f"负例={ev['negative_recall_hits']} 触发={ev['late_nudge_rule']}")
    finally:
        if current_db is not None and token is not None:
            current_db.reset(token)
        if db is not None:
            db.close()

    pass_counts = {name: sum(1 for r in results if r["checkpoints"].get(name)) for name in CHECKPOINT_NAMES}
    overall = bool(results) and all(r["overall_pass"] for r in results)
    return {
        "tool": "e6_full_journey", "schema_version": "1.0", "mode": mode, "fault": fault,
        "口径": ("mock=构造trace验链路结构，非产品行为证据；persona=synthetic-handwritten"
                 if mode == "mock" else
                 "真实 hy4 跑 1 场景（访谈→报告→对话→回写→跨天召回→触发）；persona=synthetic-handwritten；"
                 "跨天用 ts 回拨3天+新session夹具；consent/companion 为 eval 夹具；key 不入库"),
        "thresholds": {"simulated_days": SIMULATED_DAYS,
                       "context_anchor_hits_min": MIN_CONTEXT_ANCHORS,
                       "recall_hits_min": MIN_RECALL_HITS,
                       "negative_recall_hits_max": MAX_NEGATIVE_RECALL},
        "results": results,
        "summary": {"users": len(results), "checkpoint_pass_counts": pass_counts,
                    "overall_pass": overall, "exit_code": 0 if overall else 1},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E6 完整旅程跨段链路评测")
    ap.add_argument("--mock", action="store_true", help="构造 trace 验结构（不烧 key）")
    ap.add_argument("--batch", type=Path, default=DEFAULT_BATCH, help="批次 JSON")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="结果落盘路径")
    ap.add_argument("--limit", type=int, default=0, help="真hy4模式只跑前 N 个 persona（慢评默认1）")
    ap.add_argument("--fault", choices=FAULTS, default=None, help="mock 模式注入故障，构造性验证守卫")
    args = ap.parse_args(argv)
    if args.fault and not args.mock:
        ap.error("--fault 只能在 --mock 模式使用")

    items = load_batch(args.batch)
    result = run(items, mode="mock" if args.mock else "real", fault=args.fault, limit=args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[E6] {result['summary']} -> {args.out}")
    return int(result["summary"]["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
