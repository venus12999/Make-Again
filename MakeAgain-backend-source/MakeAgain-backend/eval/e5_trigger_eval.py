#!/usr/bin/env python3
"""E5 行动段触发评测：时间×情绪条件矩阵 + 打扰控制 + 内容具体性（确定性规则判定）。

先读码/看实物再设计（每条期望都能追到本地产品实物，不靠 git show）：
- agent/nudge.py
  * _now_tags L35-60（小时→时间标签，含周末/周六/周日）、_is_late_night L63-64（23:00-05:00）
  * _build_candidates L80-127（情绪节点 time_tag 命中 + freq>=2 聚合；有夜间节点时不再出 late_night）
  * _generate_texts L159-191（mock 走 _fallback_text；真实走 fast 模型 + 记忆证据）
  * _quote_fallback L194-205（无 due 时每天最多一条语录陪伴）
  * _in_quiet_hours L208-222、get_nudges L225-260（enabled/静默/去重 mark_nudge）
  * _fallback_text L130-146（具体动作模板：走十分钟/泡热茶/写两行/洗澡/听歌/发消息）
- agent/weekly.py：week_key L30-35（ISO 周）、due L38-75（归档/引导/已看/数据门槛）、mark_seen L98-101
- agent/daily.py：_today_key L58-62、get_themes L79-101、generate_opening L116-137（当日 DailyPick 缓存）
- agent/prompts_registry.py nudge.system L465-487：具体小事、建议口吻、≤50字、不空泛安慰、不命令。
- 批次：eval/batches/E5_行动段触发.json（全部 synthetic-handwritten，source 指回上述代码行）。

分层（用户关心成本）：
  --mock（默认，推荐）  本地隔离 SQLite + MOCK_LLM=true，不读 key、不联网、秒级；
                        文案走产品 _fallback_text，内容质量判定对模板兜底口径成立。
  --real（可选）        真 hy4-flash-free 生成 nudge 文案；key 只从 work/local-test-key.txt 读，
                        不打印明文、不入库。pytest 不跑此模式。

方法学（带 URL + 日期 + 按天精确算的衰减权重，遵守 AGENTS.md）：
  Anthropic《Demystifying evals for AI agents》2026-01-08，
  https://www.anthropic.com/research/demystifying-evals-for-ai-agents
  基准日 2026-11-23：距离 319 天 / 30.44 ≈ 10.5 个月 → 权重 0.7（6-18 月档，参考，须查更新版）。
  采纳其“code-based graders 优先（fast/cheap/objective/reproducible）”与
  “把真实失败样本做成回归用例”；本评测触发/去重/内容质量全部是确定性代码判定，
  不引入未校准的 LLM judge，故权重衰减不改变本设计的实现。该引用已在 E2 评测同一来源复核，
  非本轮新联网调研；如后续官方发布更新版须重新评估（本仓 AGENTS.md 时间衰减规则）。

用法（在 MakeAgain-backend 目录）：
  python eval/e5_trigger_eval.py --mock                       # 默认，全 19 case，不烧 key
  python eval/e5_trigger_eval.py --mock --out ../outputs/e5/e5_mock.json
  python eval/e5_trigger_eval.py --real --limit 3             # 真模型验证（可选，需本地 key 文件）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))
DEFAULT_BATCH = HERE / "batches" / "E5_行动段触发.json"
DEFAULT_OUT = BACKEND.parent / "outputs" / "e5" / "e5_trigger_result.json"
KEY_FILE = BACKEND.parent / "work" / "local-test-key.txt"

# ---- 内容质量口径（从产品实物推导，不是拍脑袋）----
# 具体动作线索：直接取 nudge._fallback_text 的五个动作 + nudge.system 的示例动作；
# 真实模型若换用其它具体动作，用下面的通用动作动词兜底识别（避免误杀合法改写）。
CONCRETE_ACTION_CUES = (
    "走十分钟", "透透气", "泡杯热茶", "早点躺下", "写两行", "交给纸笔",
    "洗个热水澡", "放松下来", "听首", "给朋友发条消息", "早点休息", "出去走",
)
ACTION_VERBS = (
    "走", "泡", "写", "洗", "听", "发", "躺", "休息", "喝", "吃", "坐",
    "出去", "照顾", "记", "做", "深呼吸", "放下", "打开", "看看", "歇",
)
# 空话/命令红线：nudge.system 原则 1/4/5（不空喊加油振作、不命令快去睡、不抽象评价）
VAGUE_OR_COMMAND = (
    "加油", "振作", "想开点", "要坚强", "一切都会好的", "你已经走出来了",
    "别想了", "快去睡", "必须", "应该振作",
)
SUGGESTION_MARKERS = ("好吗", "要不要", "好不好", "好不好呀")
MAX_NUDGE_CHARS = 50          # nudge.system 原则 4：一句话不超过 50 字
MIN_NUDGE_CHARS = 6           # 短于 6 字基本不可能是“一件具体的小事”
MAX_DAILY_OPENING_CHARS = 60  # daily._opening_text 是单句短文案，留冗余上限


# ---------------------------------------------------------------- 数据 / 环境
def load_batch(path: Path = DEFAULT_BATCH) -> list[dict]:
    """读批次；每条 item 必须 source_id 可追溯且带 local_time/setup/expect。"""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    items = obj.get("items") if isinstance(obj, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError(f"batch has no items: {path}")
    for it in items:
        for key in ("id", "source_id", "scene", "language", "user_input",
                    "module", "local_time", "setup", "expect"):
            if key not in it:
                raise ValueError(f"item missing {key}: {it.get('id')}")
        if it["module"] not in ("nudge", "weekly", "daily"):
            raise ValueError(f"unknown module: {it['module']}")
    return items


def prepare_env(mock: bool) -> None:
    """装配隔离环境。mock 模式绝不读 key 文件、绝不联网。"""
    root = tempfile.mkdtemp(prefix="e5trigger-")
    env = dict(
        DATABASE_URL="sqlite:///" + root + "/t.sqlite3",
        ENVIRONMENT="test" if mock else "development",
        MOCK_LLM="true" if mock else "false",
        MEDIA_SIGNING_KEY="x" * 40,
        STORAGE_LOCAL_DIR=root + "/media",
        PUBLIC_BASE_URL="http://testserver",
    )
    if not mock:
        conf: dict[str, str] = {}
        key_path = Path(os.environ.get("E5_KEY_FILE", str(KEY_FILE)))
        for line in key_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                conf[k.strip()] = v.strip()
        env.update(
            LLM_BASE_URL=conf["BASE_URL"], LLM_API_KEY=conf["KEY"],
            LLM_FAST_MODEL="hy4-flash-free", LLM_MODEL="hy4",
            LLM_EXTRA_HEADERS=json.dumps({
                "x-opencode-session": "e5-trigger-" + uuid.uuid4().hex[:8],
                "User-Agent": "make-again-e5-trigger/1.0",
            }),
        )
    os.environ.update(env)
    from config import get_settings
    get_settings.cache_clear()
    s = get_settings()
    if not mock:
        s.llm_base_url = env["LLM_BASE_URL"]
        s.llm_api_key = env["LLM_API_KEY"]
        s.llm_fast_model = "hy4-flash-free"
        s.llm_model = "hy4"
        s.llm_report_model = ""
        s.mock_llm = False


def _offset_hours() -> int:
    from config import get_settings
    return int(get_settings().timezone_offset_hours)


def _as_input_time(local_dt: datetime) -> datetime:
    """批次写的是本地墙钟时间；产品函数内部会 +timezone_offset_hours，故这里减回去。"""
    return local_dt - timedelta(hours=_offset_hours())


# ---------------------------------------------------------------- 判定器（确定性）
def check_text_quality(text: str) -> dict:
    """规则检查一条 nudge 文案：非空、长度、具体动作、建议口吻、无空话/命令。

    口径来自 nudge.system（principles 1/2/4/5）与 nudge._fallback_text 的动作模板。
    """
    t = (text or "").strip()
    failures: list[str] = []
    cue: list[str] = []
    verb: list[str] = []
    if not t:
        failures.append("EMPTY")
    else:
        if len(t) < MIN_NUDGE_CHARS:
            failures.append("TOO_SHORT")
        if len(t) > MAX_NUDGE_CHARS:
            failures.append("TOO_LONG")
        vague = [w for w in VAGUE_OR_COMMAND if w in t]
        if vague:
            failures.append("VAGUE_OR_COMMAND:" + "/".join(vague))
        if not any(m in t for m in SUGGESTION_MARKERS):
            failures.append("NO_SUGGESTION_MARKER")
        cue = [c for c in CONCRETE_ACTION_CUES if c in t]
        verb = [v for v in ACTION_VERBS if v in t]
        if not cue and not verb:
            failures.append("NO_CONCRETE_ACTION")
    return {
        "ok": not failures, "char_len": len(t), "failures": failures,
        "matched_cues": cue, "matched_verbs": verb, "text": t,
    }


def check_daily_opening(text: str) -> dict:
    """每日开场文案规则：非空、含「」两主题槽、有“还是”、长度合理（daily._opening_text 口径）。"""
    t = (text or "").strip()
    failures: list[str] = []
    if not t:
        failures.append("EMPTY")
    else:
        if t.count("「") < 2 or t.count("」") < 2:
            failures.append("NOT_TWO_THEME_SLOTS")
        if "还是" not in t:
            failures.append("NO_CHOICE_MARKER")
        if len(t) > MAX_DAILY_OPENING_CHARS:
            failures.append("TOO_LONG")
    return {"ok": not failures, "char_len": len(t), "failures": failures, "text": t}


# ---------------------------------------------------------------- 场景构造（纯写入本地隔离 DB）
def _seed_user(db, item: dict) -> str:
    from memory import store
    from memory.models import EmotionNode, MemoryEntry

    uid = "e5-" + uuid.uuid4().hex[:10]
    setup = item.get("setup") or {}
    store.get_or_create_user(db, uid, loss_type=setup.get("loss_type"))

    for n in setup.get("emotion_nodes") or []:
        db.add(EmotionNode(
            user_id=uid, trigger=n.get("trigger") or n.get("time_tag") or "synthetic",
            emotion=n.get("emotion") or "难过", frequency=int(n.get("frequency", 1)),
            time_tag=n.get("time_tag"),
        ))
    for m in setup.get("memories") or []:
        db.add(MemoryEntry(user_id=uid, content=m.get("content") or m.get("summary") or "",
                           summary=m.get("summary") or "", time_tag=m.get("time_tag")))
    prefs = setup.get("nudge_prefs") or {}
    if prefs:
        store.update_nudge_prefs(
            db, uid, enabled=prefs.get("enabled", True),
            quiet_start=prefs.get("quiet_start"), quiet_end=prefs.get("quiet_end"),
        )
    db.commit()
    return uid


def _seed_weekly(db, item: dict, wk: str) -> str:
    from agent import onboarding
    from memory import store

    uid = "e5-" + uuid.uuid4().hex[:10]
    setup = item.get("setup") or {}
    store.get_or_create_user(db, uid, loss_type="breakup")
    phase = setup.get("phase")
    if phase and phase != "new":
        onboarding.set_phase(db, uid, phase)
    if setup.get("companion_archived"):
        comp = store.create_companion(db, uid, relationship_type="breakup", make_current=True)
        comp.archived_at = datetime.now()  # 只判非空（已归档），具体时刻不参与判定
        db.commit()
    wr = setup.get("weekly_report")
    if isinstance(wr, dict):
        row = store.save_weekly_report(db, uid, wk, wr.get("content") or {})
        if wr.get("seen"):
            store.mark_weekly_report_seen(db, uid, wk)
        assert row is not None
    return uid


# ---------------------------------------------------------------- 三类评测执行
def evaluate_nudge(db, client, item: dict) -> dict:
    from agent import nudge
    from config import get_settings
    from memory import store

    local = datetime.fromisoformat(item["local_time"])
    now = _as_input_time(local)
    uid = _seed_user(db, item)

    # 触发规则证明：先看纯本地候选（_build_candidates 不改动 DB），再看实际返回。
    tags = sorted(nudge._now_tags(local))
    late_night = nudge._is_late_night(local)
    candidates = nudge._build_candidates(db, uid, local,
                                         breakup=store.effective_loss_type(db, uid) == "breakup")

    first = nudge.get_nudges(db, uid, client, now=now)
    rule_keys = [n["rule_key"] for n in first["nudges"]]
    types = [n.get("type") for n in first["nudges"]]
    emotion_push = sum(1 for t in types if t == "emotion")

    # 内容质量只对“真轻推”（emotion/time）判定；quote 是产品设计的陪伴语录，非行动建议。
    qualities = [check_text_quality(n.get("text", "")) for n in first["nudges"]
                 if n.get("type") in ("emotion", "time")]

    # 打扰控制：同一 rule_key 当天不重复推；语录每天最多一条。再拉两次看真实状态。
    second = nudge.get_nudges(db, uid, client, now=now)
    third = nudge.get_nudges(db, uid, client, now=now)
    second_keys = [n["rule_key"] for n in second["nudges"]]
    third_keys = [n["rule_key"] for n in third["nudges"]]
    primary = [k for k in rule_keys if k != "quote"]
    repeated = [k for k in primary if k in second_keys or k in third_keys]
    quotes_total = sum(1 for keys in (rule_keys, second_keys, third_keys) if "quote" in keys)
    anti_ok = (not repeated) and quotes_total <= 1 and not third_keys

    exp = item["expect"]
    checks = {
        "trigger_rules_match": rule_keys == list(exp["trigger_rule_keys"]),
        "nudge_count_match": len(first["nudges"]) == int(exp["nudge_count"]),
        "emotion_push_match": emotion_push == int(exp["emotion_push_count"]),
        "text_quality_match": (all(q["ok"] for q in qualities) if qualities else True)
                              == bool(exp.get("text_quality", True)),
        "anti_nag_ok": anti_ok,
    }
    return {
        "id": item["id"], "source_id": item["source_id"], "module": "nudge",
        "kind": item.get("kind"), "scene": item["scene"],
        "local_time": item["local_time"], "tags": tags, "late_night": late_night,
        "candidates": [{"rule_key": c["rule_key"], "type": c["type"], "trigger": c["trigger"],
                        "emotion": c["emotion"], "frequency": c["frequency"],
                        "is_night": c["is_night"]} for c in candidates],
        "expected": exp,
        "actual": {
            "rule_keys": rule_keys, "nudge_count": len(first["nudges"]),
            "emotion_push_count": emotion_push,
            "types": types, "texts": [n.get("text", "") for n in first["nudges"]],
            "pull_sequence": [rule_keys, second_keys, third_keys],
            "repeated_primary_rules": repeated, "quotes_in_3_pulls": quotes_total,
        },
        "content_quality": {"applicable": bool(qualities), "checked": len(qualities),
                            "passed": sum(1 for q in qualities if q["ok"]), "items": qualities},
        "checks": checks, "pass": all(checks.values()),
        "offset_hours": _offset_hours(), "model_mode": "mock" if client.mock else "real",
        "tz_note": "local_time 是本地墙钟；传给产品函数前减 timezone_offset_hours",
        "cfg_model": get_settings().llm_fast_model,
    }


def evaluate_weekly(db, item: dict) -> dict:
    from agent import weekly

    local = datetime.fromisoformat(item["local_time"])
    now = _as_input_time(local)
    wk = weekly.week_key(now)
    uid = _seed_weekly(db, item, wk)
    exp = item["expect"]
    res = weekly.due(db, uid, now=now)
    checks: dict[str, bool] = {}
    if exp.get("week_key"):
        checks["week_key_match"] = wk == exp["week_key"]
    if "due" in exp:
        checks["due_match"] = bool(res["due"]) == bool(exp["due"])
    after = None
    if item.get("kind") == "anti_nag" and exp.get("after_seen_due") is False:
        weekly.mark_seen(db, uid, wk)
        after = weekly.due(db, uid, now=now)
        checks["after_seen_blocked"] = (
            after["due"] is False
            and exp.get("after_seen_reason_contains", "") in (after.get("reason") or "")
        )
    if exp.get("reason_contains"):
        checks["reason_match"] = exp["reason_contains"] in (res.get("reason") or "")
    if exp.get("compare_week_key"):
        cmp_local = datetime.fromisoformat(item["setup"]["compare_local_time"])
        cmp_wk = weekly.week_key(_as_input_time(cmp_local))
        checks["compare_week_key_match"] = cmp_wk == exp["compare_week_key"]
    else:
        cmp_wk = None
    return {
        "id": item["id"], "source_id": item["source_id"], "module": "weekly",
        "kind": item.get("kind"), "scene": item["scene"],
        "local_time": item["local_time"], "expected": exp,
        "actual": {"week_key": wk, "compare_week_key": cmp_wk,
                   "due": bool(res["due"]), "reason": res.get("reason"),
                   "generating": res.get("generating"),
                   "after_seen": ({"due": bool(after["due"]), "reason": after.get("reason")}
                                  if after else None)},
        "checks": checks, "pass": all(checks.values()),
    }


def evaluate_daily(db, item: dict) -> dict:
    from agent import daily

    local = datetime.fromisoformat(item["local_time"])
    now = _as_input_time(local)
    uid = _seed_user(db, item)

    first = daily.generate_opening(db, uid, now=now)
    second = daily.generate_opening(db, uid, now=now)  # 同一天二次拉
    third = None
    second_local = (item.get("setup") or {}).get("second_local_time")
    if second_local:
        third = daily.generate_opening(db, uid, now=_as_input_time(datetime.fromisoformat(second_local)))

    titles = [t["title"] for t in (first.get("themes") or [])]
    opening_ok = len(titles) >= 2 and all(t in first.get("opening", "") for t in titles[:2])
    quality = check_daily_opening(first.get("opening", ""))
    exp = item["expect"]
    checks = {
        "first_cache_match": bool(first.get("cached")) == bool(exp.get("first_cached", False)),
        "second_cache_match": bool(second.get("cached")) == bool(exp.get("same_day_cached", True)),
        "opening_two_themes": opening_ok == bool(exp.get("opening_has_two_themes", True)),
        "text_quality": quality["ok"] == bool(exp.get("text_quality", True)),
        "same_day_stable": first.get("opening") == second.get("opening"),
    }
    if third is not None:
        checks["next_day_refresh"] = (
            bool(third.get("cached")) == bool(exp.get("next_day_cached", False))
            and bool(third.get("opening"))
        )
    return {
        "id": item["id"], "source_id": item["source_id"], "module": "daily",
        "kind": item.get("kind"), "scene": item["scene"],
        "local_time": item["local_time"], "expected": exp,
        "actual": {"themes": titles, "mood": first.get("mood"), "tone": first.get("tone"),
                   "opening": first.get("opening"), "first_cached": bool(first.get("cached")),
                   "second_cached": bool(second.get("cached")),
                   "next_day_opening": third.get("opening") if third else None,
                   "next_day_cached": bool(third.get("cached")) if third else None},
        "content_quality": {"applicable": True, "checked": 1,
                            "passed": 1 if quality["ok"] else 0, "items": [quality]},
        "checks": checks, "pass": all(checks.values()),
    }


# ---------------------------------------------------------------- 编排
def run(items: list[dict], mode: str = "mock") -> dict:
    if mode not in ("mock", "real"):
        raise ValueError(f"unknown mode: {mode}")
    prepare_env(mock=(mode == "mock"))
    from gateway.client import LLMClient
    from memory.db import _RawSessionLocal, init_db

    init_db()
    client = LLMClient()
    results: list[dict] = []
    db = _RawSessionLocal()
    try:
        for item in items:
            module = item["module"]
            print(f"\n=== E5 {mode}: {item['id']} / {module} / {item['scene']} ===")
            if module == "nudge":
                r = evaluate_nudge(db, client, item)
            elif module == "weekly":
                r = evaluate_weekly(db, item)
            else:
                r = evaluate_daily(db, item)
            results.append(r)
            print(f"  checks={r['checks']} -> {'PASS' if r['pass'] else 'FAIL'}")
    finally:
        db.close()

    nudge_results = [r for r in results if r["module"] == "nudge"]
    weekly_results = [r for r in results if r["module"] == "weekly"]
    daily_results = [r for r in results if r["module"] == "daily"]

    # E5-V1：触发矩阵真实数字（该触发 vs 不该触发），按 expect 的 nudge/due 分侧统计。
    should_trigger = [r for r in nudge_results if r["actual"]["nudge_count"] > 0
                      and r["actual"]["rule_keys"] != ["quote"]]
    should_not_trigger = [r for r in nudge_results if r["actual"]["rule_keys"] == ["quote"]
                          or r["actual"]["nudge_count"] == 0]
    weekly_due = [r for r in weekly_results if r["actual"]["due"]]
    weekly_suppressed = [r for r in weekly_results if not r["actual"]["due"]]
    daily_refreshed = [r for r in daily_results if r["checks"].get("next_day_refresh") is True]

    # E5-V2：打扰控制数字
    anti = {
        "nudge_cases_checked": len(nudge_results),
        "repeat_push_blocked": sum(1 for r in nudge_results
                                   if not r["actual"]["repeated_primary_rules"]),
        "quote_at_most_once_per_day": sum(1 for r in nudge_results
                                          if r["actual"]["quotes_in_3_pulls"] <= 1),
        "third_pull_empty": sum(1 for r in nudge_results if not r["actual"]["pull_sequence"][2]),
        "weekly_seen_blocked": sum(1 for r in weekly_results
                                   if r["checks"].get("after_seen_blocked") is True),
        "daily_same_day_cached": sum(1 for r in daily_results
                                     if r["actual"].get("second_cached") is True),
        "daily_next_day_refreshed": len(daily_refreshed),
    }

    # E5-V3：内容质量数字（只统计 applicable 的判定）
    q_items = [q for r in results for q in (r.get("content_quality") or {}).get("items", [])
               if (r.get("content_quality") or {}).get("applicable")]
    quality = {
        "checked": len(q_items), "passed": sum(1 for q in q_items if q["ok"]),
        "failed": sum(1 for q in q_items if not q["ok"]),
        "char_len_min": min((q["char_len"] for q in q_items), default=0),
        "char_len_max": max((q["char_len"] for q in q_items), default=0),
        "rules": {"max_chars": MAX_NUDGE_CHARS, "min_chars": MIN_NUDGE_CHARS,
                  "suggestion_markers": list(SUGGESTION_MARKERS),
                  "vague_or_command_blacklist": list(VAGUE_OR_COMMAND),
                  "concrete_action_cues": list(CONCRETE_ACTION_CUES)},
    }

    overall = bool(results) and all(r["pass"] for r in results)
    return {
        "tool": "e5_trigger_eval", "schema_version": "1.0", "mode": mode,
        "口径": ("mock=本地隔离SQLite+MOCK_LLM=true，文案走产品_fallback_text，"
                 "条件矩阵/去重/内容质量为确定性规则判定；case=synthetic-handwritten；不读key不联网"
                 if mode == "mock" else
                 "真 hy4-flash-free 生成 nudge 文案；case=synthetic-handwritten；key 只从本地文件读、不打印不入库"),
        "summary": {
            "cases": len(results),
            "by_module": {"nudge": len(nudge_results), "weekly": len(weekly_results),
                          "daily": len(daily_results)},
            "passed": sum(1 for r in results if r["pass"]),
            "failed": sum(1 for r in results if not r["pass"]),
            "V1_trigger_matrix": {
                "nudge_should_trigger": len(should_trigger),
                "nudge_should_not_trigger": len(should_not_trigger),
                "weekly_due": len(weekly_due), "weekly_suppressed": len(weekly_suppressed),
                "daily_next_day_refresh": len(daily_refreshed),
                "emotion_trigger_cases": sum(1 for r in nudge_results
                                             if any(c["type"] == "emotion" for c in r["candidates"])),
                "time_trigger_cases": sum(1 for r in nudge_results
                                          if any(c["type"] == "time" for c in r["candidates"])
                                          and not any(c["type"] == "emotion" for c in r["candidates"])),
            },
            "V2_anti_nag": anti,
            "V3_content_quality": quality,
            "overall_pass": overall, "exit_code": 0 if overall else 1,
        },
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E5 行动段触发评测（时间×情绪矩阵 + 打扰控制 + 内容质量）")
    ap.add_argument("--mock", action="store_true", help="mock 模式（默认，不烧 key）")
    ap.add_argument("--real", action="store_true", help="真 hy4 验证（可选，需 work/local-test-key.txt）")
    ap.add_argument("--batch", type=Path, default=DEFAULT_BATCH, help="批次 JSON")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="结果落盘")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（真模型调试用）")
    ap.add_argument("--key-file", type=Path, default=None,
                    help="真模型 key 文件路径（仅 --real 使用；mock 模式绝不读取）")
    args = ap.parse_args(argv)
    if args.key_file is not None:
        os.environ["E5_KEY_FILE"] = str(args.key_file)
    if args.mock and args.real:
        ap.error("--mock 与 --real 互斥")
    mode = "real" if args.real else "mock"  # 默认 mock：不烧 key 是安全默认

    items = load_batch(args.batch)
    if args.limit:
        items = items[:args.limit]
    result = run(items, mode=mode)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(f"\n[E5] {result['summary']} -> {args.out}")
    return int(result["summary"]["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
