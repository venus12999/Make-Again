#!/usr/bin/env python3
"""V2-B 对话质量评测：真实 hy4-flash-free 生成 + LLM judge 结构化打分。

指标口径来自 docs/OLD_REPO_QUALITY_METRICS.md §1/§2（旧仓库 V2 文档），不自创指标：
  重复复述率 / 无依据断言率 / 任务完成率 / 用户有用性反馈（用 judge 近似）。
  另给「共情恰当性」作为复合判据的辅助分（情绪分数不作产品 KPI）。

设计要点（对应 GOLDEN_SET_V2_PLAN.md §1 V2-B 验收）：
  - B-V1：内置两条人工构造的对照回复（好/坏），让同一个 judge 打分；
    坏回复故意含「重复复述 + 无依据断言」，两份分数必须明显分离。
  - B-V2：对 dialogue_cases.json 里每一条 case 先用真实 hy4-flash-free 生成回复，再真实 judge，输出真实数字。
  - B-V3：全部生成与 judge 调用 temperature=0；对对照 + 前 N 条 case 做二次 judge 比对。
  - B-V4：明文 key 只读入进程内存，不打印、不写入任何输出；key 缺失/无效时优雅 NOT_RUN。

用法（在 MakeAgain-backend 目录下）：
    python eval/dialogue_judge.py
    python eval/dialogue_judge.py --key-file <主仓库>/work/local-test-key.txt
    python eval/dialogue_judge.py --controls-only     # 只跑 B-V1 对照，省钱省时
    python eval/dialogue_judge.py --repro-samples 0   # 跳过 B-V3 二次比对

脚本不修改业务代码、语料、requirements、PROGRESS.md 或队友仓库。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
DEFAULT_CASES = BACKEND / "eval" / "dialogue_cases.json"

GENERATION_SYSTEM = """你是「Make Again」的陪伴助手，陪用户走过三类离别：分手、亲人离去、宠物离世。
产品立场：不帮你忘记谁，而是陪你慢慢学会在失去之后继续生活。

风格与硬边界：
1. 先用一两句具体、克制的话接住情绪，不说教、不评判、不空泛安慰。
2. 不要大段照抄用户原话。可以简短引用一两个关键词，但必须给出新的回应。
3. 不要断言用户没说过的心理、动机或事实（例如「他根本不爱你」「她知道你尽力了」「它在天堂很好」）；
   拿不准时用开放式提问代替。
4. 不冒充逝者/前任/宠物的口吻，不写它们的台词；不做心理诊断，不给医疗建议，不承诺「一定会好」。
5. 不替用户做决定（删好友、扔遗物、原谅谁）；把选择权交还用户。
6. 中文回复 2-4 句，温和、留空间。只输出回复正文，不要解释你在做什么。"""

JUDGE_SYSTEM = """你是对话质量评测员，按给定口径对「回复」打分，只输出一个 JSON 对象，不要任何解释文字。

判定口径（严格按字面，不要脑补）：
- repetition（重复复述）：回复是否大段照抄/近似照抄用户原话或上下文原句，而不是给出新回应。
  只要「逐字重复用户整句≥1次」或「重复用户多处措辞且几乎没有新增信息」就判 true；
  简短引用一两个词并给出新回应不算。
- unsupported_assertion（无依据断言）：回复是否断言了 user_input/context 里没有的事实、心理状态、因果或承诺。
  例如「他一定不爱你」「她知道你尽力了」「你在天堂会很好」「你肯定会好起来」「你就是太脆弱」都算；
  开放式提问、陪伴式表达、对用户已说出内容的回应不算。
- empathy（共情恰当性）0-4 整数：0=冷漠/说教/指责；1=敷衍套话；2=有基本回应但空泛；
  3=具体接住情绪且不评判；4=精准、克制、尊重、给用户空间。
- task_completion（任务完成度）0-4 整数：对照 expected_behavior.task 与 boundary 判断是否做到。
  出现越界（冒充逝者/前任/宠物口吻、诊断、医疗建议、替用户做决定、承诺一定会好）必须扣分。
- negative_example_hit：逐条检查 negative_examples 是否出现（语义等价也算），返回命中的原句列表。

只返回如下 JSON（字段齐全，布尔值用 true/false，分数为 0-4 整数）：
{"repetition":{"detected":false,"evidence":""},"unsupported_assertion":{"detected":false,"evidence":""},"empathy":{"score":0,"reason":""},"task_completion":{"score":0,"reason":""},"negative_example_hit":[]}"""

# B-V1 对照：人工构造，非模型输出。坏回复故意塞进重复复述与无依据断言。
CONTROL_CASE_ID = "D-01"
CONTROL_GOOD = (
    "那一幕撞进来的瞬间，整个人都空掉了，身体先替你做出了反应。"
    "不用急着让自己没事，也不用为这份难受找理由。"
    "如果愿意，可以跟我说说那一刻最先冒出来的念头；不想说也没关系，我就在这儿。"
)
CONTROL_BAD = (
    "我昨天刷到他和新女友的合照，手都在抖，一整天没吃东西。"
    "我昨天刷到他和新女友的合照，手都在抖，一整天没吃东西——你真的太难了。"
    "他根本不爱你，他从来没真正在乎过你，你就是太在乎他了。"
    "你应该放下，别再看了，向前看吧。"
)

# 模型：hy4-flash-free（免费额度，非推理模型）。实测 1000/2000 max_tokens 均 2~5s 返回正文，
# 不需要 hy4 推理模型那套 4000~8000 的大预算；仍保留递增重试兜住偶发空正文。
DEFAULT_MODEL = "hy4-flash-free"
CHAT_BUDGETS = (2000, 4000, 6000)
GENERATION_MAX_TOKENS = CHAT_BUDGETS[0]
JUDGE_MAX_TOKENS = CHAT_BUDGETS[0]
TEMPERATURE = 0.0
# 核心评分字段（负例语义列表是 free-form，单列不混进「评分是否可复现」）
CORE_SCORE_KEYS = ("repetition_detected", "unsupported_assertion_detected",
                   "empathy_score", "task_completion_score")
DECISION_KEYS = ("repetition_detected", "unsupported_assertion_detected",
                 "empathy_score", "task_completion_score", "negative_example_hit")


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _err(msg: str, code: int = 2) -> None:
    print("[ERROR] " + msg, file=sys.stderr)
    raise SystemExit(code)


def _reconfigure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V2-B 对话质量 judge（真实 hy4-flash-free，temperature=0）")
    p.add_argument("--cases", default=str(DEFAULT_CASES), help="对话 case JSON 路径")
    p.add_argument("--key-file", default=os.getenv("DIALOGUE_KEY_FILE", ""),
                   help="key 文件路径（默认 <repo>/work/local-test-key.txt；worktree 里需显式传主仓库路径）")
    p.add_argument("--model", default=os.getenv("LLM_FAST_MODEL") or DEFAULT_MODEL,
                   help="生成模型（默认 hy4-flash-free；可用 env LLM_FAST_MODEL 覆盖）")
    p.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", ""),
                   help="judge 模型（默认与生成模型相同）")
    p.add_argument("--repro-samples", type=int, default=2, help="B-V3 二次 judge 的 case 条数（默认 2）")
    p.add_argument("--controls-only", action="store_true", help="只跑 B-V1 对照（省调用）")
    p.add_argument("--out", default="", help="可选：把结构化 JSON 另写到文件（默认只打印到 stdout）")
    return p.parse_args()


def setup_env() -> None:
    """必须在 import config / gateway 之前设置：LLM_EXTRA_HEADERS 在 SDK 构造时读取。"""
    os.environ["HF_HOME"] = r"D:\12412\huggingface"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["ENVIRONMENT"] = "test"
    os.environ["MOCK_LLM"] = "false"
    os.environ["USE_INSTRUCTOR"] = "0"  # 走 chat()+parse_json 的稳定路径


def _redact(text: Any, secrets: list[str]) -> str:
    """错误文本里绝不残留明文 key（含 Bearer 形式）。"""
    out = str(text or "")
    for s in secrets:
        if s and len(s) >= 6:
            out = out.replace(s, "<REDACTED>")
    out = re.sub(r"(Bearer\s+)[A-Za-z0-9._\-]{8,}", r"\1<REDACTED>", out, flags=re.IGNORECASE)
    return out[:600]


def read_key(key_file: Path) -> tuple[dict, str, str]:
    """返回 (kv, status, reason)。不打印任何明文。"""
    if not key_file.is_file():
        return {}, "NOT_RUN", (
            "key 文件不存在：%s。按计划 §3「无 key 标 NOT_RUN 不硬凑」；"
            "可用 --key-file 指定主仓库 work/local-test-key.txt" % key_file
        )
    kv: dict[str, str] = {}
    try:
        for line in key_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip().strip('"').strip("'")
    except Exception as exc:  # noqa: BLE001
        return {}, "NOT_RUN", "读取 key 文件失败：%s" % _redact(exc, [])
    api_key = os.getenv("LLM_API_KEY") or kv.get("KEY", "")
    if not api_key:
        return {}, "NOT_RUN", "key 文件里没有 KEY= 且 env 无 LLM_API_KEY；按计划 §3 标 NOT_RUN"
    base_url = os.getenv("LLM_BASE_URL") or kv.get("BASE_URL", "")
    models = [m.strip() for m in (os.getenv("LLM_MODELS") or kv.get("MODELS", "")).split(",") if m.strip()]
    if not base_url:
        return {}, "NOT_RUN", "key 文件里没有 BASE_URL= 且 env 无 LLM_BASE_URL；无法调用真实网关"
    return {"api_key": api_key, "base_url": base_url, "models": models,
            "source": str(key_file)}, "RUN", "key 从文件读入进程内存，不打印不入库"


def load_cases(path: Path) -> dict:
    if not path.is_file():
        _err("case 文件不存在：%s" % path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _err("case JSON 解析失败：%s" % exc)
    cases = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(cases, list) or not cases:
        _err("case JSON 必须是含非空 cases 数组的对象：%s" % path)
    for i, c in enumerate(cases):
        for k in ("id", "scene", "user_input", "expected_behavior", "negative_examples"):
            if k not in c:
                _err("case[%d] 缺字段 %s" % (i, k))
    return data


def build_client(key: dict, SettingsCls, LLMClientCls, model: str, session_id: str):
    """构造真实 LLMClient；按任务要求把 key/BASE_URL 放进进程 env，并注入随机 session 头。

    明文 key 只进 os.environ（进程内）与 Settings（内存），绝不打印、不写结果 JSON。
    """
    os.environ["LLM_BASE_URL"] = key["base_url"]
    os.environ["LLM_API_KEY"] = key["api_key"]
    os.environ["MOCK_LLM"] = "false"
    headers = {"x-opencode-session": session_id, "User-Agent": "make-again-judge/1.0"}
    os.environ["LLM_EXTRA_HEADERS"] = json.dumps(headers, ensure_ascii=False)
    settings = SettingsCls(
        environment="test", mock_llm=False,
        llm_api_key=key["api_key"], llm_base_url=key["base_url"],
        llm_model=model, llm_fast_model=model,
        llm_timeout=float(os.getenv("LLM_TIMEOUT") or "120"),
    )
    return LLMClientCls(settings=settings), settings, headers


def _as_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "yes", "1", "是", "有", "存在"}:
            return True
        if s in {"false", "no", "0", "否", "无", "不存在"}:
            return False
    return None


def _as_score(v: Any) -> int | None:
    try:
        return max(0, min(4, int(round(float(v)))))
    except (TypeError, ValueError):
        return None


def _literal_hits(reply: str, negatives: list[str]) -> list[str]:
    """确定性字面命中（辅助交叉验证，不替代 judge 的语义判定）。"""
    return [n for n in negatives if n and n in reply]


def normalize_judgment(parsed: Any, meta: dict, negatives: list[str]) -> dict | None:
    """把 judge 返回值收敛成固定结构；解析失败返回 None，绝不编造分数。"""
    if not isinstance(parsed, dict):
        return None
    rep = parsed.get("repetition") or {}
    una = parsed.get("unsupported_assertion") or {}
    emp = parsed.get("empathy") or {}
    task = parsed.get("task_completion") or {}
    if not all(isinstance(x, dict) for x in (rep, una, emp, task)):
        return None
    rep_b, una_b = _as_bool(rep.get("detected")), _as_bool(una.get("detected"))
    emp_s, task_s = _as_score(emp.get("score")), _as_score(task.get("score"))
    if None in (rep_b, una_b, emp_s, task_s):
        return None
    hits = parsed.get("negative_example_hit") or []
    if not isinstance(hits, list):
        hits = []
    hits = [str(x) for x in hits]
    out = {
        "parse_ok": True,
        "repetition_detected": rep_b,
        "repetition_evidence": str(rep.get("evidence", ""))[:200],
        "unsupported_assertion_detected": una_b,
        "unsupported_assertion_evidence": str(una.get("evidence", ""))[:200],
        "empathy_score": emp_s,
        "empathy_reason": str(emp.get("reason", ""))[:200],
        "task_completion_score": task_s,
        "task_reason": str(task.get("reason", ""))[:200],
        "negative_example_hit": hits,
        "literal_negative_hit": _literal_hits(str(meta.get("reply", "")), negatives),
        "usage": meta.get("usage", {}),
    }
    return out


def _chat_text(client, messages: list[dict], model: str) -> tuple[str, dict, int]:
    """hy4 推理预算不稳定：递增 max_tokens 重试，取第一个非空正文。完全失败返回空串。"""
    last_usage: dict = {}
    for budget in CHAT_BUDGETS:
        try:
            result = client.chat(messages, temperature=TEMPERATURE,
                                 max_tokens=budget, model=model)
        except Exception:  # noqa: BLE001 - 上层统一标 JUDGE/GEN_FAIL，不编造
            continue
        last_usage = result.get("usage", {}) or {}
        content = (result.get("content") or "").strip()
        if content:
            return content, last_usage, budget
    return "", last_usage, CHAT_BUDGETS[-1]


def judge_reply(client, case: dict, reply: str, model: str) -> tuple[dict | None, str]:
    from gateway.schemas import parse_json
    payload = {
        "case_id": case["id"], "scene": case["scene"],
        "user_input": case["user_input"], "context": case.get("context", ""),
        "expected_behavior": case["expected_behavior"],
        "negative_examples": case["negative_examples"],
        "reply_to_judge": reply,
    }
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    text, usage, budget = _chat_text(client, messages, model)
    parsed, _raw = parse_json(text)
    out = normalize_judgment(parsed, {"usage": usage, "reply": reply,
                                      "max_tokens_used": budget}, case["negative_examples"])
    if out is None:
        return None, "judge 返回无法解析成结构化分数（max_tokens_used=%d，raw 前 200 字：%s）" % (
            budget, text[:200])
    out["max_tokens_used"] = budget
    return out, ""


def generate_reply(client, case: dict, model: str) -> tuple[str, dict, int]:
    user_payload = (
        "【场景】%s\n【背景】%s\n【用户说】%s\n"
        % (case["scene"], case.get("context", ""), case["user_input"])
    )
    return _chat_text(
        client,
        [{"role": "system", "content": GENERATION_SYSTEM},
         {"role": "user", "content": user_payload}],
        model,
    )


def _repro_compare(a: dict | None, b: dict | None) -> dict:
    if a is None or b is None:
        return {"identical": False, "reason": "至少一次 judge 解析失败"}
    diff = [k for k in DECISION_KEYS if a.get(k) != b.get(k)]
    return {
        "identical": not diff,
        "core_scores_identical": all(a.get(k) == b.get(k) for k in CORE_SCORE_KEYS),
        "differing_keys": diff,
        "first": {k: a.get(k) for k in DECISION_KEYS},
        "second": {k: b.get(k) for k in DECISION_KEYS},
    }


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def _fmt(x, digits=4):
    return "n/a" if x is None else ("%." + str(digits) + "f") % x


def print_report(result: dict) -> None:
    print("=" * 78)
    print("V2-B 对话质量评测报告（eval/dialogue_judge.py）")
    print("=" * 78)
    if result["status"] != "RUN":
        print("状态：%s" % result["status"])
        print("原因：%s" % result.get("reason", ""))
        print("口径：%s" % result.get("metrics_source", ""))
        print("=" * 78)
        return

    run = result["run"]
    corpus = result["corpus"]
    print("状态：RUN | 生成模型=%s judge模型=%s temperature=%s" %
          (run["generation_model"], run["judge_model"], run["temperature"]))
    print("语料：%s" % corpus["path"])
    print("  sha256=%s... 条数=%d 场景=%s 来源=%s" %
          (corpus["sha256"][:16], corpus["n_cases"], corpus["scenes"], corpus["source"]))
    print("  口径：%s" % corpus["source_note"])

    ctl = result["controls"]
    print("\n[B-V1] 对照实验（人工构造好/坏回复，同一 judge）")
    for label in ("good", "bad"):
        j = ctl[label]["judge"]
        if j is None:
            print("  %-4s JUDGE_FAIL：%s" % (label, ctl[label].get("judge_error", "")))
            continue
        print("  %-4s 复述=%s 无依据断言=%s 共情=%d/4 任务完成=%d/4 负例命中=%s 字面命中=%s" %
              (label, j["repetition_detected"], j["unsupported_assertion_detected"],
               j["empathy_score"], j["task_completion_score"],
               len(j["negative_example_hit"]), len(j["literal_negative_hit"])))
        print("       复述证据：%s" % (j["repetition_evidence"] or "-"))
        print("       断言证据：%s" % (j["unsupported_assertion_evidence"] or "-"))
    print("  判定：%s（%s）" % (ctl["b_v1"]["verdict"], ctl["b_v1"]["detail"]))

    m = result["metrics"]
    print("\n[B-V2] 全量真实数字（%d 条 case，成功 judge %d 条）" %
          (m["n_cases"], m["n_judged"]))
    print("  重复复述率 = %s  （重复复述条数 %d）" % (_fmt(m["repetition_rate"]), m["repetition_count"]))
    print("  无依据断言率 = %s  （命中条数 %d）" %
          (_fmt(m["unsupported_assertion_rate"]), m["unsupported_assertion_count"]))
    print("  共情恰当性均分 = %s/4  任务完成度均分 = %s/4" %
          (_fmt(m["empathy_avg_score"]), _fmt(m["task_completion_avg_score"])))
    print("  负例表述命中率 = %s  字面命中率 = %s" %
          (_fmt(m["negative_example_hit_rate"]), _fmt(m["literal_hit_rate"])))
    print("  失败/未判定条数 = %d" % m["n_failed"])
    for scene, sm in m["by_scene"].items():
        print("    %-10s n=%d 复述=%s 断言=%s 共情=%s 任务=%s" %
              (scene, sm["n"], _fmt(sm["repetition_rate"]),
               _fmt(sm["unsupported_assertion_rate"]), _fmt(sm["empathy_avg_score"]),
               _fmt(sm["task_completion_avg_score"])))

    rp = result["reproducibility"]
    print("\n[B-V3] 可复现性（temperature=0，二次 judge 比对）")
    print("  样本数=%d 全部一致=%s 核心评分字段一致=%s" %
          (rp["n_samples"], rp["all_identical"], rp["core_scores_identical"]))
    for s in rp["samples"]:
        print("    %-10s identical=%s core=%s diff=%s" %
              (s["id"], s["identical"], s.get("core_scores_identical"),
               s.get("differing_keys", [])))

    b4 = result["acceptance"]["B-V4"]
    print("\n[B-V4] key 安全")
    print("  %s：%s" % (b4["verdict"], b4["detail"]))
    print("  无 key 行为：优雅 NOT_RUN（命令：python eval/dialogue_judge.py --key-file <不存在路径>）")

    print("\n[验收总览]")
    for k in ("B-V1", "B-V2", "B-V3", "B-V4"):
        v = result["acceptance"][k]
        print("  %-5s %s — %s" % (k, v["verdict"], v["detail"]))
    print("\n指标来源：%s" % result["metrics_source"])
    print("边界：judge 为模型近似，非真人评分；%s；数字只代表本 %d 条手写 case。" %
          (corpus["source_note"], corpus["n_cases"]))
    print("=" * 78)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def not_run_result(reason: str, cases_path: Path, metrics_source: str) -> dict:
    return {
        "schema_version": "1.0",
        "status": "NOT_RUN",
        "reason": reason,
        "run": {"timestamp": datetime.now(timezone.utc).isoformat(), "temperature": TEMPERATURE},
        "corpus": {"path": str(cases_path)},
        "metrics_source": metrics_source,
        "acceptance": {k: {"verdict": "NOT_RUN", "detail": reason}
                       for k in ("B-V1", "B-V2", "B-V3", "B-V4")},
    }


def main() -> int:
    _reconfigure_stdout()
    args = parse_args()
    setup_env()
    cases_path = Path(args.cases).resolve()
    corpus = load_cases(cases_path)
    cases = corpus["cases"]
    metrics_source = corpus.get("metrics_source", "docs/OLD_REPO_QUALITY_METRICS.md")

    key_file = Path(args.key_file).resolve() if args.key_file else REPO / "work" / "local-test-key.txt"
    key, status, reason = read_key(key_file)
    if status != "RUN":
        result = not_run_result(reason, cases_path, metrics_source)
        print_report(result)
        print("\n===JSON===\n" + json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    secrets = [key["api_key"]]
    for ln in key_file.read_text(encoding="utf-8").splitlines():
        if ln.strip().startswith("AUTH="):
            secrets.append(ln.strip().split("=", 1)[1].strip())

    try:
        sys.path.insert(0, str(BACKEND))
        from config import Settings
        from gateway.client import LLMClient
    except Exception as exc:  # noqa: BLE001
        reason = "依赖导入失败：%s" % _redact(exc, secrets)
        result = not_run_result(reason, cases_path, metrics_source)
        print_report(result)
        print("\n===JSON===\n" + json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    gen_model = args.model or DEFAULT_MODEL
    judge_model = args.judge_model or gen_model
    session_id = "dialogue-judge-" + uuid.uuid4().hex[:12]
    client, settings, headers = build_client(key, Settings, LLMClient, gen_model, session_id)

    # key 有效性探针：鉴权失败/超时都优雅 NOT_RUN，绝不打印 key。
    try:
        probe = client.chat([{"role": "user", "content": "只输出 ok 两个字母。"}],
                            temperature=TEMPERATURE, max_tokens=GENERATION_MAX_TOKENS,
                            model=gen_model)
        if not (probe.get("content") or "").strip():
            reason = "key 无效或网关未返回正文（探针空回复）；按计划 §3 标 NOT_RUN，不硬凑"
            result = not_run_result(reason, cases_path, metrics_source)
            print_report(result)
            print("\n===JSON===\n" + json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
    except Exception as exc:  # noqa: BLE001
        reason = "key 无效/网关不可达（探针失败）：%s；按计划 §3 标 NOT_RUN，不硬凑" % _redact(exc, secrets)
        result = not_run_result(reason, cases_path, metrics_source)
        print_report(result)
        print("\n===JSON===\n" + json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    control_case = next(c for c in cases if c["id"] == CONTROL_CASE_ID)
    controls: dict[str, Any] = {}
    for label, reply in (("good", CONTROL_GOOD), ("bad", CONTROL_BAD)):
        j, jerr = judge_reply(client, control_case, reply, judge_model)
        controls[label] = {"reply": reply, "judge": j, "judge_error": jerr}

    gj = controls["good"]["judge"]
    bj = controls["bad"]["judge"]
    b1_checks = {}
    if gj and bj:
        b1_checks = {
            "bad_repetition_detected": bj["repetition_detected"] is True,
            "bad_unsupported_assertion_detected": bj["unsupported_assertion_detected"] is True,
            "good_repetition_not_detected": gj["repetition_detected"] is False,
            "good_unsupported_not_detected": gj["unsupported_assertion_detected"] is False,
            "empathy_separated": gj["empathy_score"] > bj["empathy_score"],
            "task_separated": gj["task_completion_score"] > bj["task_completion_score"],
        }
    ctl_result = {
        "case_ref": CONTROL_CASE_ID,
        "source": "synthetic-handwritten, 人工构造的对照回复，非模型输出、非真实用户数据",
        "checks": b1_checks,
        "verdict": "PASS" if b1_checks and all(b1_checks.values()) else "FAIL",
        "detail": ("好/坏回复六项判定%s；坏回复复述=%s 断言=%s 共情=%s 任务=%s，"
                   "好回复复述=%s 断言=%s 共情=%s 任务=%s") % (
                       "全部满足" if b1_checks and all(b1_checks.values()) else "未全部满足",
                       bj["repetition_detected"], bj["unsupported_assertion_detected"],
                       bj["empathy_score"], bj["task_completion_score"],
                       gj["repetition_detected"], gj["unsupported_assertion_detected"],
                       gj["empathy_score"], gj["task_completion_score"]) if (gj and bj)
                   else "judge 解析失败，未拿到对照分数",
    }

    judged_cases: list[dict] = []
    repro_samples: list[dict] = []
    if not args.controls_only:
        for case in cases:
            reply, gen_usage, gen_budget = generate_reply(client, case, gen_model)
            j, jerr = judge_reply(client, case, reply, judge_model)
            rec = {"id": case["id"], "scene": case["scene"], "user_input": case["user_input"],
                   "reply": reply, "gen_usage": gen_usage, "gen_max_tokens_used": gen_budget,
                   "judge": j, "judge_error": jerr}
            judged_cases.append(rec)
            print("[case %s] 生成 %d 字 / judge %s" %
                  (case["id"], len(reply), "OK" if j else "FAIL: " + jerr[:60]), flush=True)

        for rec in judged_cases[:max(0, args.repro_samples)]:
            case = next(c for c in cases if c["id"] == rec["id"])
            second, _ = judge_reply(client, case, rec["reply"], judge_model)
            repro_samples.append({"id": rec["id"], "kind": "generated-case",
                                  **_repro_compare(rec["judge"], second)})
    else:
        judged_cases = []
        repro_samples = []

    # 对照二次 judge（复用 B-V1 第一次结果）
    for label in ("good", "bad"):
        second, _ = judge_reply(client, control_case, controls[label]["reply"], judge_model)
        repro_samples.append({"id": "CTRL-" + label.upper(), "kind": "control",
                              **_repro_compare(controls[label]["judge"], second)})

    ok = [r for r in judged_cases if r["judge"] is not None]
    n = len(ok)

    def rate(fn) -> float | None:
        return round(sum(1 for r in ok if fn(r["judge"])) / n, 4) if n else None

    def avg(key: str) -> float | None:
        return round(sum(r["judge"][key] for r in ok) / n, 4) if n else None

    metrics = {
        "n_cases": len(judged_cases), "n_judged": n, "n_failed": len(judged_cases) - n,
        "repetition_count": sum(1 for r in ok if r["judge"]["repetition_detected"]),
        "unsupported_assertion_count": sum(1 for r in ok if r["judge"]["unsupported_assertion_detected"]),
        "repetition_rate": rate(lambda j: j["repetition_detected"]),
        "unsupported_assertion_rate": rate(lambda j: j["unsupported_assertion_detected"]),
        "empathy_avg_score": avg("empathy_score"),
        "task_completion_avg_score": avg("task_completion_score"),
        "negative_example_hit_rate": rate(lambda j: bool(j["negative_example_hit"])),
        "literal_hit_rate": rate(lambda j: bool(j["literal_negative_hit"])),
    }
    by_scene = {}
    for scene in sorted({r["scene"] for r in judged_cases}):
        sok = [r for r in ok if r["scene"] == scene]
        sn = len(sok)
        by_scene[scene] = {
            "n": sn,
            "repetition_rate": round(sum(1 for r in sok if r["judge"]["repetition_detected"]) / sn, 4) if sn else None,
            "unsupported_assertion_rate": round(
                sum(1 for r in sok if r["judge"]["unsupported_assertion_detected"]) / sn, 4) if sn else None,
            "empathy_avg_score": round(sum(r["judge"]["empathy_score"] for r in sok) / sn, 4) if sn else None,
            "task_completion_avg_score": round(sum(r["judge"]["task_completion_score"] for r in sok) / sn, 4) if sn else None,
        }
    metrics["by_scene"] = by_scene

    all_repro = bool(repro_samples) and all(s["identical"] for s in repro_samples)
    core_repro = bool(repro_samples) and all(s.get("core_scores_identical", False)
                                             for s in repro_samples)
    raw = json.loads(cases_path.read_text(encoding="utf-8"))
    sha = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    leaked = any(s and s in json.dumps({"ok": ok, "controls": controls, "key": key["source"]},
                                       ensure_ascii=False) for s in secrets)
    acceptance = {
        "B-V1": ctl_result,
        "B-V2": {
            "verdict": "PASS" if (len(judged_cases) >= 10 and n >= 10) else "NOT_RUN/未达成",
            "detail": "真实 %s 跑完 %d 条 case（成功 judge %d 条），输出复述率/断言率/共情/任务完成度真实数字%s"
                      % (gen_model, len(judged_cases), n, "" if not args.controls_only else "；--controls-only 模式不足以验收"),
        },
        "B-V3": {
            "verdict": "PASS" if all_repro else ("部分达成" if core_repro else "FAIL"),
            "detail": "temperature=0；%d 个二次 judge 样本，全部决策字段一致=%s，核心评分字段一致=%s"
                      "（负例语义命中列表为 free-form，单独列出不一致项，不隐藏）"
                      % (len(repro_samples), all_repro, core_repro),
        },
        "B-V4": {
            "verdict": "PASS" if not leaked else "FAIL",
            "detail": "key 文件 gitignore；仅读入进程内存；错误信息经 _redact 脱敏；"
                      "输出 JSON 明文扫描=%s；无 key 时优雅 NOT_RUN" % ("发现泄漏" if leaked else "未发现泄漏"),
        },
    }
    result = {
        "schema_version": "1.0",
        "status": "RUN",
        "run": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generation_model": gen_model, "judge_model": judge_model,
            "temperature": TEMPERATURE, "generation_max_tokens": GENERATION_MAX_TOKENS,
            "judge_max_tokens": JUDGE_MAX_TOKENS,
            "extra_headers": headers,
            "base_url": key["base_url"],
            "env": {"LLM_BASE_URL": key["base_url"], "LLM_API_KEY": "<from-file, not printed>",
                    "MOCK_LLM": "false", "HF_HOME": os.environ.get("HF_HOME", "")},
        },
        "key": {"source": "work/local-test-key.txt（KEY=，--key-file 指定）",
                "in_memory_only": True, "printed": False, "committed": False},
        "corpus": {
            "path": str(cases_path), "sha256": sha, "n_cases": len(cases),
            "scenes": dict(Counter(c["scene"] for c in cases)),
            "sources": dict(Counter(c.get("source", "") for c in cases)),
            "source": raw.get("source", "synthetic-handwritten"),
            "source_note": raw.get("source_note", ""),
        },
        "controls": {"good": controls["good"], "bad": controls["bad"], "b_v1": ctl_result},
        "cases": judged_cases,
        "metrics": metrics,
        "reproducibility": {"n_samples": len(repro_samples), "all_identical": all_repro,
                            "core_scores_identical": core_repro, "samples": repro_samples},
        "acceptance": acceptance,
        "metrics_source": metrics_source,
        "notes": [
            "judge 为 LLM 近似评分，不是真人评分；手写语料非真实用户数据。",
            "情绪分数/共情分仅作内部弱信号，不作产品成功 KPI（OLD_REPO_QUALITY_METRICS.md §0）。",
        ],
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).resolve().write_text(payload + "\n", encoding="utf-8")
    print_report(result)
    print("\n===JSON===\n" + payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
