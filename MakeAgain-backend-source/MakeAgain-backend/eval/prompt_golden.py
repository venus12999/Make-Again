#!/usr/bin/env python3
"""PE-2 Prompt 效果 golden set：把 breakup.py 69 行细则转成 8 个可测行为维度。

真实 hy4-flash-free（opencode-go 网关）跑两阶段评测，全程 temperature=0：
  1. 生成：用 breakup.py 的真实 BREAKUP_SYSTEM_PROMPT（AST 抽取后填好占位符）
     对 24 条 synthetic-handwritten 语料生成回复。
  2. 判定：同一个 hy4-flash-free 做的 judge 按 8 个维度逐维度打 0-5 分 + 理由，
     输出结构化 JSON。只对每条 case 的 applicable_dimensions 打分，其余标"不适用"。
  3. 好/坏对照（E2-V3）：8 个维度各一条人工构造的好回复 + 坏回复，同一 judge 打分；
     要求好>=4、坏<=2、差值>=2，否则判 FAIL（不硬凑）。
  4. 双次复现（E2-V4）：整套流水线跑两遍（各自独立生成+判定）。实测 opencode-go
     即使 temperature=0 生成也不逐字复现（同输入 3 次 3 个版本），故分两层核验：
     (a) judge 对同一批冻结回复二次判定是否逐字段确定；(b) 两遍 8 维结论（达标/待改进）
     是否稳定、均分漂移是否≤1.0。逐字漂移与差异项全部如实列出，不隐藏。

验收映射：
  E2-V1 覆盖>=8 维度；E2-V2 真实 hy4 跑出每维度分数（非 NOT_RUN）；
  E2-V3 judge 区分好/坏；E2-V4 两次运行可复现（temperature=0）。

用法（在 MakeAgain-backend 目录下）：
    python eval/prompt_golden.py --key-file <主仓库>/work/local-test-key.txt --out <...>/prompt-golden.json
    python eval/prompt_golden.py --controls-only --key-file <...>   # 只跑对照，省调用
    python eval/prompt_golden.py --runs 1 --key-file <...>          # 只跑单遍
    python eval/prompt_golden.py --key-file <不存在路径>             # 应输出 状态：NOT_RUN

脚本不修改业务代码、语料、requirements、PROGRESS.md 或队友仓库；明文 key 不打印不入库。
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[1]
DEFAULT_CASES = BACKEND / "eval" / "prompt_cases.json"
BREAKUP_SRC = BACKEND / "agent" / "breakup.py"
TONE_SRC = BACKEND / "emotion" / "tone.py"

GENERATION_MODEL_DEFAULT = "hy4-flash-free"
TEMPERATURE = 0.0
GEN_BUDGETS = (1000, 2000, 4000)
# 8 维 judge 在 5 个适用维度时 completion 实测可达 2000 tokens；首档 1600 会截断 JSON
# （2026-11-22 首次全量 run1 出现 6 条解析失败）。上调首档到 3200 再跑，失败必须如实记录。
JUDGE_BUDGETS = (3200, 6400, 8000)

UNIVERSAL_DIMS = {
    "D1_empathy_first", "D2_no_repetition", "D3_no_filler",
    "D4_non_question_ending", "D8_length_control",
}

JUDGE_HEADER = """你是「Make Again」提示词评测员。按给定口径对「回复」逐维度打分，只输出一个 JSON 对象，不要任何解释文字。

打分规则：
- 只对 applicable_dimensions 列出的维度给 0-5 整数分和一句理由；不在列表里的维度 score 填 null、reason 填"不适用"。
- 严格按字面定义与锚点判定，不脑补用户没说的话；negative_examples / literal_negative_hit 只是辅助证据，以定义为准。
- D8 长度必须主要依据 payload 里的 char_count（回复字符数），不要自己数。
- 每个 applicable 维度都必须返回，缺一个就算解析失败。
"""

JUDGE_FOOTER = """
只返回如下 JSON（8 个维度键全部出现，键名逐字照抄；score 为 0-5 整数或 null；reason 为字符串）：
{"D1_empathy_first":{"score":0,"reason":""},"D2_no_repetition":{"score":0,"reason":""},"D3_no_filler":{"score":0,"reason":""},"D4_non_question_ending":{"score":0,"reason":""},"D5_cognitive_reframe":{"score":0,"reason":""},"D6_victim_protection":{"score":0,"reason":""},"D7_quote_restraint":{"score":0,"reason":""},"D8_length_control":{"score":0,"reason":""}}"""

# 以下只作 AST 抽取失败时的兜底，内容逐字来自 breakup.py / emotion/tone.py（main 26b9393）
FALLBACK_TEMPLATE = None  # 必填，缺失直接 NOT_RUN，不编造 prompt
FALLBACK_MICRO = (
    "会话内推进要跟着用户走:只有用户这一轮明确往「复盘/分析/往前走」的方向表达时,才顺势加深;"
    "用户若只是在倾诉情绪,就只承接,不抢先推进。同一话题若连续两轮未被用户接住,就放下、不再第三轮重复;"
    "用户情绪明显反复或又提起伤心处时,先回到安抚承接,不急着回到刚才的推进。"
)
FALLBACK_RULE = (
    "不要引导用户去体会或放大伤感(如「现在路过,是不是空落落的」「是不是很难受」「心里是不是很痛」);"
    "不要主动追问痛苦细节;把话题轻轻引向当下能做的、已经变好的、继续往前走的方向。"
    "不要凭空编造时间对比(如「比一个月前的自己…」「那时候你连白天都熬不住」):"
    "只有上下文里真实提供的时间依据(「近期状态轨迹」「相关往事」)才能用来做时间对比;"
    "没有依据时,就用当下这句话里的具体变化来鼓励,不要编一个过去的状态。"
)
FALLBACK_SOOTHE = (
    "语气定位【温柔安抚型】:先承接情绪、给予陪伴与安全感,像轻轻拍着后背的朋友;"
    "不急着讲道理、不催促改变;即便要往前走,也说得慢一点、软一点。"
)

# ---------------------------------------------------------------------------
# 好/坏对照（E2-V3）：人工构造，非模型输出、非真实用户数据
# ---------------------------------------------------------------------------
def build_controls() -> list[dict]:
    """每个维度一条好回复 + 一条坏回复；只判该维度，确保 judge 区分能力可核验。"""
    return [
        {
            "dimension": "D1_empathy_first", "case_id": "D1-01",
            "good": "手停在货架前那一下，比难过更像是突然被拽回了那天。这种被一个牌子偷袭的感觉我接住了，而你已经把三个月撑过来了。",
            "bad": "我理解你，抱抱你。这很正常，分开三个月还会想起他，别难过了，慢慢来就好。",
        },
        {
            "dimension": "D2_no_repetition", "case_id": "D2-01",
            "good": "梦醒那一秒的落空最磨人。那种空不是你想多了，是身体比理智慢了一步，醒过来还要重新适应一遍。",
            "bad": "我昨晚又梦到他了，梦里我们还在一起，醒来那一秒特别空。我昨晚又梦到他了，梦里我们还在一起，醒来那一秒特别空。",
        },
        {
            "dimension": "D3_no_filler", "case_id": "D3-01",
            "good": "会不会忘掉，现在谁也给不了你答案。但「这辈子」这个词太长了，你其实只是在说今天还很难。",
            "bad": "会好起来的，时间会治愈一切，想开点，加油，你要有信心，慢慢来就好。",
        },
        {
            "dimension": "D4_non_question_ending", "case_id": "D4-01",
            "good": "什么都不想动的时候，就先不动。今天把身体交给床，也算一种照顾自己。",
            "bad": "你现在是不是特别难受？想跟我聊聊吗？是身体累还是心里累？你觉得呢？",
        },
        {
            "dimension": "D5_cognitive_reframe", "case_id": "D5-01",
            "good": "把分开都揽在自己身上，这份自责我听见了。但一段关系走到尽头，很少是一个人的事——你当时也在尽力，只是两个人的力气没往一处使。",
            "bad": "你这是自责型人格，别这么想了。你觉得呢？",
        },
        {
            "dimension": "D6_victim_protection", "case_id": "D6-01",
            "good": "他出轨是他自己的选择，不是你哪里没做好。把责任放回他身上，你才腾得出力气重新看看自己想要什么。",
            "bad": "一个巴掌拍不响，他出轨你也有问题，是不是你太忽略他了？你太依赖他了。",
        },
        {
            "dimension": "D7_quote_restraint", "case_id": "D7-01",
            "good": "走神就让它走一会儿。你当下惦记的是什么，比这堂课本身更值得听。",
            "bad": "你不是在上烘焙课吗？你上次说烘焙课，是不是又想到做甜点那件事了？",
        },
        {
            "dimension": "D8_length_control", "case_id": "D8-03",
            "good": "收进箱子、放到床底，你已经替这件事找了个位置。先让它待在那儿，不用急着给它一个意义。",
            "bad": (
                "我理解你今天把他的东西收进箱子这个动作背后有太多的情绪。"
                "你可能一边整理一边回想过去在一起的点点滴滴，心里既舍不得又觉得该往前走了。"
                "其实每个人处理离别的方式都不一样，有人喜欢把东西收起来眼不见心不烦，"
                "有人喜欢慢慢来留一点念想。你不需要急着给自己一个答案，"
                "也不用纠结这样做是不是代表放下了。收拾东西只是一个动作，"
                "它既不说明你已经走出来了，也不说明你还困在原地。"
                "真正重要的是你在整理的过程里照顾好了自己的感受，允许自己难过，"
                "也允许自己偶尔喘口气。你可以按自己的节奏来，想留就留，想扔就扔，"
                "甚至可以今天放进去明天又拿出来看看，都没有关系。"
                "你也不用急着给我一个交代，不用觉得告诉我这件事就该有个结果。"
                "我会一直在这儿，你想说什么都可以慢慢说，说不完也没关系。"
            ),
        },
    ]


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
    p = argparse.ArgumentParser(description="PE-2 prompt golden set（真实 hy4-flash-free，temperature=0）")
    p.add_argument("--cases", default=str(DEFAULT_CASES), help="case JSON 路径")
    p.add_argument("--key-file", default=os.getenv("PROMPT_GOLDEN_KEY_FILE", ""),
                   help="key 文件路径（worktree 里需显式传主仓库 work/local-test-key.txt）")
    p.add_argument("--model", default=os.getenv("LLM_FAST_MODEL") or GENERATION_MODEL_DEFAULT,
                   help="生成模型（默认 hy4-flash-free）")
    p.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", ""), help="judge 模型（默认同生成模型）")
    p.add_argument("--session-id", default=os.getenv("PROMPT_GOLDEN_SESSION_ID", ""),
                   help="x-opencode-session；留空则每次运行随机（key 文件建议用稳定 session id）")
    p.add_argument("--runs", type=int, default=2, help="整套流水线运行遍数（默认 2，用于 E2-V4 复现）")
    p.add_argument("--controls-only", action="store_true", help="只跑 8 维好/坏对照（省调用，不足以验收 E2-V2）")
    p.add_argument("--out", default="", help="可选：把结构化 JSON 另写到文件")
    return p.parse_args()


def setup_env() -> None:
    os.environ["HF_HOME"] = r"D:\12412\huggingface"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["ENVIRONMENT"] = "test"
    os.environ["MOCK_LLM"] = "false"
    os.environ["USE_INSTRUCTOR"] = "0"


def _redact(text: Any, secrets: list[str]) -> str:
    out = str(text or "")
    for s in secrets:
        if s and len(s) >= 6:
            out = out.replace(s, "<REDACTED>")
    out = re.sub(r"(Bearer\s+)[A-Za-z0-9._\-]{8,}", r"\1<REDACTED>", out, flags=re.IGNORECASE)
    return out[:600]


def read_key(key_file: Path) -> tuple[dict, str, str]:
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
        return {}, "NOT_RUN", "key 文件里没有 KEY= 且 env 无 LLM_API_KEY；标 NOT_RUN"
    base_url = os.getenv("LLM_BASE_URL") or kv.get("BASE_URL", "")
    if not base_url:
        return {}, "NOT_RUN", "key 文件里没有 BASE_URL= 且 env 无 LLM_BASE_URL；无法调用真实网关"
    return {"api_key": api_key, "base_url": base_url, "source": str(key_file)}, "RUN", "key 从文件读入进程内存，不打印不入库"


def load_cases(path: Path) -> dict:
    if not path.is_file():
        _err("case 文件不存在：%s" % path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _err("case JSON 解析失败：%s" % exc)
    if not isinstance(data, dict) or not isinstance(data.get("dimensions"), list) or not isinstance(data.get("cases"), list):
        _err("case JSON 必须是含 dimensions / cases 数组的对象")
    dims = data["dimensions"]
    if len(dims) < 8:
        _err("维度不足 8 个：%d" % len(dims))
    for d in dims:
        for k in ("id", "name", "definition", "anchors", "applicability"):
            if k not in d:
                _err("dimension 缺字段 %s" % k)
        for n in "012345":
            if n not in d["anchors"]:
                _err("dimension %s 缺 %s 分锚点" % (d["id"], n))
    for i, c in enumerate(data["cases"]):
        for k in ("id", "dimension", "user_input", "expected", "negative_examples"):
            if k not in c:
                _err("case[%d] 缺字段 %s" % (i, k))
        if c["dimension"] not in {d["id"] for d in dims}:
            _err("case[%s] 的 dimension 不在维度表：%s" % (c["id"], c["dimension"]))
    return data


def applicable_dims(case: dict) -> list[str]:
    """universal 维度对所有 case 适用；conditional 维度只对以它为主题的 case 适用。"""
    return sorted(UNIVERSAL_DIMS | {case["dimension"]})


def _ast_string_assign(path: Path, name: str) -> str | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for node in tree.body:
        target = None
        if isinstance(node, ast.Assign):
            target = node.targets[0] if len(node.targets) == 1 else None
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        if isinstance(target, ast.Name) and target.id == name:
            try:
                return str(ast.literal_eval(node.value))
            except Exception:
                return None
    return None


def _ast_tone_soothe(path: Path) -> str | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "TONE_PROMPTS" for t in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for k, v in zip(node.value.keys, node.value.values):
            key = k.id if isinstance(k, ast.Name) else (k.value if isinstance(k, ast.Constant) else None)
            if key == "SOOTHE":
                try:
                    return str(ast.literal_eval(v))
                except Exception:
                    return None
    return None


def build_generation_template() -> tuple[str | None, dict]:
    """从 breakup.py / tone.py 用 AST 抽取真实 prompt 常量并填好非 context 占位符。"""
    template = _ast_string_assign(BREAKUP_SRC, "BREAKUP_SYSTEM_PROMPT") or FALLBACK_TEMPLATE
    if not template:
        return None, {"reason": "无法从 breakup.py 抽取 BREAKUP_SYSTEM_PROMPT，且无兜底；拒绝编造 prompt"}
    micro = _ast_string_assign(BREAKUP_SRC, "MICRO_GATE_RULE") or FALLBACK_MICRO
    rule = _ast_string_assign(TONE_SRC, "SHARED_RULE") or FALLBACK_RULE
    soothe = _ast_tone_soothe(TONE_SRC) or FALLBACK_SOOTHE
    try:
        filled = template.format(rule=rule, stage_guidance=soothe, microrule=micro, context="@@CTX@@")
    except Exception as exc:  # noqa: BLE001
        return None, {"reason": "prompt 占位符填充失败：%s" % _redact(exc, [])}
    meta = {
        "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "n_chars": len(filled),
        "source": "agent/breakup.py BREAKUP_SYSTEM_PROMPT（AST 抽取，填 SHARED_RULE/MICRO_GATE_RULE/SOOTHE）",
        "context_placeholder": "@@CTX@@（每条 case 的 context_block 填入）",
    }
    return filled, meta


def build_judge_system(dimensions: list[dict]) -> str:
    lines = [JUDGE_HEADER]
    for d in dimensions:
        lines.append("\n【%s · %s】" % (d["id"], d["name"]))
        lines.append("定义：" + d["definition"])
        for n in "012345":
            lines.append("  %s 分：%s" % (n, d["anchors"][n]))
        if d.get("bad_phrases"):
            lines.append("  典型坏例短语：" + "、".join(d["bad_phrases"]))
    lines.append(JUDGE_FOOTER)
    return "\n".join(lines)


def build_client(key: dict, SettingsCls, LLMClientCls, model: str, session_id: str):
    os.environ["LLM_BASE_URL"] = key["base_url"]
    os.environ["LLM_API_KEY"] = key["api_key"]
    os.environ["MOCK_LLM"] = "false"
    headers = {"x-opencode-session": session_id, "User-Agent": "make-again-prompt-golden/1.0"}
    os.environ["LLM_EXTRA_HEADERS"] = json.dumps(headers, ensure_ascii=False)
    settings = SettingsCls(
        environment="test", mock_llm=False,
        llm_api_key=key["api_key"], llm_base_url=key["base_url"],
        llm_model=model, llm_fast_model=model,
        llm_timeout=float(os.getenv("LLM_TIMEOUT") or "120"),
    )
    return LLMClientCls(settings=settings), headers


def _as_score(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in {"null", "none", "n/a", "na", "不适用", ""}:
        return None
    try:
        return max(0, min(5, int(round(float(v)))))
    except (TypeError, ValueError):
        return None


def normalize_judgment(parsed: Any, applicable: list[str], reply: str,
                       negatives: list[str], all_dim_ids: list[str]) -> dict | None:
    if not isinstance(parsed, dict):
        return None
    out: dict[str, Any] = {"parse_ok": True, "scores": {}, "reasons": {}}
    for dim in all_dim_ids:
        if dim not in applicable:
            out["scores"][dim] = None
            out["reasons"][dim] = "不适用"
            continue
        node = parsed.get(dim)
        if not isinstance(node, dict):
            return None
        sc = _as_score(node.get("score"))
        if sc is None:
            return None
        out["scores"][dim] = sc
        out["reasons"][dim] = str(node.get("reason", ""))[:300]
    out["literal_negative_hit"] = [n for n in negatives if n and n in reply]
    out["char_count"] = len(reply)
    return out


def _chat_text(client, messages: list[dict], model: str, budgets: tuple[int, ...]) -> tuple[str, dict, int]:
    last_usage: dict = {}
    for budget in budgets:
        try:
            result = client.chat(messages, temperature=TEMPERATURE, max_tokens=budget, model=model)
        except Exception:  # noqa: BLE001 - 上层标 FAIL，不编造
            continue
        last_usage = result.get("usage", {}) or {}
        content = (result.get("content") or "").strip()
        if content:
            return content, last_usage, budget
    return "", last_usage, budgets[-1]


def generate_reply(client, template: str, case: dict, model: str) -> tuple[str, dict, int]:
    system = template.replace("@@CTX@@", str(case.get("context_block", "")))
    user_payload = "【场景】%s\n【用户当前这句话】%s" % (case.get("scene", ""), case["user_input"])
    return _chat_text(client, [
        {"role": "system", "content": system},
        {"role": "user", "content": user_payload},
    ], model, GEN_BUDGETS)


def judge_reply(client, judge_system: str, payload: dict, all_dim_ids: list[str],
                applicable: list[str], model: str) -> tuple[dict | None, str]:
    from gateway.schemas import parse_json
    reply = payload["reply_to_judge"]
    messages = [
        {"role": "system", "content": judge_system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    text, usage, budget = _chat_text(client, messages, model, JUDGE_BUDGETS)
    parsed, _raw = parse_json(text)
    out = normalize_judgment(parsed, applicable, reply,
                             payload.get("negative_examples", []), all_dim_ids)
    if out is None:
        return None, "judge 无法解析成结构化分数（max_tokens_used=%d，raw 前 200 字：%s）" % (budget, text[:200])
    out["usage"] = usage
    out["max_tokens_used"] = budget
    return out, ""


def _avg(vals: list[int]) -> float | None:
    return round(sum(vals) / len(vals), 4) if vals else None


def run_pipeline(client, template: str, judge_system: str, cases: list[dict],
                 controls: list[dict], dims: list[dict], cases_by_id: dict,
                 gen_model: str, judge_model: str, with_gen: bool) -> dict:
    all_dim_ids = [d["id"] for d in dims]

    # ---- 对照（E2-V3）：只判各自维度 ----
    control_out: dict[str, dict] = {}
    for ctl in controls:
        dim = ctl["dimension"]
        case = cases_by_id.get(ctl["case_id"])
        if case is None:
            control_out[dim] = {"error": "对照引用的 case 不存在：%s" % ctl["case_id"]}
            continue
        entry: dict[str, Any] = {
            "case_id": ctl["case_id"], "source": "synthetic-handwritten（人工构造，非模型输出）",
        }
        for label in ("good", "bad"):
            reply = ctl[label]
            payload = {
                "case_id": ctl["case_id"], "scene": case.get("scene", ""),
                "target_dimension": dim, "applicable_dimensions": [dim],
                "user_input": case["user_input"], "context": case.get("context_block", ""),
                "negative_examples": case.get("negative_examples", []),
                "char_count": len(reply), "reply_to_judge": reply,
            }
            j, jerr = judge_reply(client, judge_system, payload, all_dim_ids, [dim], judge_model)
            entry[label] = {"reply": reply, "char_count": len(reply), "judge": j, "judge_error": jerr}
        gj = (entry["good"].get("judge") or {}).get("scores", {}).get(dim)
        bj = (entry["bad"].get("judge") or {}).get("scores", {}).get(dim)
        entry["good_score"] = gj
        entry["bad_score"] = bj
        entry["separated"] = (
            gj is not None and bj is not None and gj >= 4 and bj <= 2 and (gj - bj) >= 2
        )
        control_out[dim] = entry

    ctrl_checks = {dim: e.get("separated", False) for dim, e in control_out.items()}
    ctrl_verdict = "PASS" if len(ctrl_checks) == 8 and all(ctrl_checks.values()) else "FAIL"

    # ---- 全量 case（E2-V2）----
    judged: list[dict] = []
    if with_gen:
        for case in cases:
            reply, gen_usage, gen_budget = generate_reply(client, template, case, gen_model)
            applicable = applicable_dims(case)
            payload = {
                "case_id": case["id"], "scene": case.get("scene", ""),
                "target_dimension": case["dimension"], "applicable_dimensions": applicable,
                "user_input": case["user_input"], "context": case.get("context_block", ""),
                "negative_examples": case.get("negative_examples", []),
                "char_count": len(reply), "reply_to_judge": reply,
            }
            j, jerr = judge_reply(client, judge_system, payload, all_dim_ids, applicable, judge_model)
            judged.append({
                "id": case["id"], "dimension": case["dimension"], "scene": case.get("scene", ""),
                "user_input": case["user_input"], "reply": reply,
                "applicable_dimensions": applicable,
                "gen_usage": gen_usage, "gen_max_tokens_used": gen_budget,
                "judge": j, "judge_error": jerr,
            })

    # ---- 逐维度真实分数（只统计成功判定且该维度适用的 case）----
    dim_scores: dict[str, dict] = {}
    for d in dims:
        dim = d["id"]
        vals: list[int] = []
        reasons: list[str] = []
        n_applicable = 0
        for rec in judged:
            if rec["judge"] is None or dim not in rec["applicable_dimensions"]:
                continue
            n_applicable += 1
            sc = rec["judge"]["scores"].get(dim)
            if sc is not None:
                vals.append(sc)
                reasons.append(rec["judge"]["reasons"].get(dim, ""))
        dim_scores[dim] = {
            "applicable_cases": n_applicable,
            "n_scored": len(vals),
            "avg_score": _avg(vals),
            "score_distribution": {str(k): vals.count(k) for k in range(6)},
            "sample_reasons": reasons[:3],
        }

    # ---- judge 确定性：对同一批已生成回复二次判定，逐字段精确比对 ----
    # opencode-go 即使 temperature=0 生成也不逐字复现（实测同输入 3 次 3 个版本），
    # 所以把「评分工具本身是否确定」和「两次独立生成的结论是否稳定」分开测，不混为一谈。
    rejudge_samples: list[dict] = []
    for rec in (judged[:6] if judged else []):
        if rec["judge"] is None:
            continue
        case = cases_by_id[rec["id"]]
        payload = {
            "case_id": rec["id"], "scene": case.get("scene", ""),
            "target_dimension": case["dimension"],
            "applicable_dimensions": rec["applicable_dimensions"],
            "user_input": case["user_input"], "context": case.get("context_block", ""),
            "negative_examples": case.get("negative_examples", []),
            "char_count": len(rec["reply"]), "reply_to_judge": rec["reply"],
        }
        second, serr = judge_reply(client, judge_system, payload, all_dim_ids,
                                   rec["applicable_dimensions"], judge_model)
        s1 = rec["judge"]["scores"]
        s2 = second["scores"] if second else None
        rejudge_samples.append({
            "id": rec["id"], "identical": s1 == s2,
            "first": s1, "second": s2, "error": serr,
        })
    judge_determinism = {
        "n_samples": len(rejudge_samples),
        "all_identical": bool(rejudge_samples) and all(x["identical"] for x in rejudge_samples),
        "samples": rejudge_samples,
    }

    return {
        "controls": control_out,
        "control_checks": ctrl_checks,
        "control_verdict": ctrl_verdict,
        "cases": judged,
        "n_cases": len(judged),
        "n_judged": sum(1 for r in judged if r["judge"] is not None),
        "n_failed": sum(1 for r in judged if r["judge"] is None),
        "dimension_scores": dim_scores,
        "judge_determinism": judge_determinism,
    }


def compare_runs(r1: dict, r2: dict) -> dict:
    dim_diffs: list[str] = []
    for dim, s1 in r1["dimension_scores"].items():
        s2 = r2["dimension_scores"].get(dim, {})
        if s1.get("avg_score") != s2.get("avg_score") or s1.get("n_scored") != s2.get("n_scored"):
            dim_diffs.append("%s: run1=%s/%s run2=%s/%s" % (
                dim, s1.get("avg_score"), s1.get("n_scored"),
                s2.get("avg_score"), s2.get("n_scored")))

    by_id1 = {c["id"]: c for c in r1["cases"]}
    by_id2 = {c["id"]: c for c in r2["cases"]}
    case_diffs: list[str] = []
    for cid in sorted(set(by_id1) | set(by_id2)):
        c1, c2 = by_id1.get(cid), by_id2.get(cid)
        s1 = c1["judge"]["scores"] if c1 and c1["judge"] else None
        s2 = c2["judge"]["scores"] if c2 and c2["judge"] else None
        if s1 != s2:
            case_diffs.append(cid)

    ctrl_diffs: list[str] = []
    for dim, e1 in r1["controls"].items():
        e2 = r2["controls"].get(dim, {})
        if e1.get("good_score") != e2.get("good_score") or e1.get("bad_score") != e2.get("bad_score"):
            ctrl_diffs.append("%s: run1=%s/%s run2=%s/%s" % (
                dim, e1.get("good_score"), e1.get("bad_score"),
                e2.get("good_score"), e2.get("bad_score")))

    # 结论层稳定性：每维度按 0-5 均分是否 >=4.0 归为「达标/待改进」，看两遍是否翻转；
    # 同时给出均分最大漂移。逐字复现不可能（同输入 3 次生成 3 个版本），故按口径判定。
    verdicts: list[dict] = []
    max_drift = 0.0
    for dim, s1 in r1["dimension_scores"].items():
        s2 = r2["dimension_scores"].get(dim, {})
        a1, a2 = s1.get("avg_score"), s2.get("avg_score")
        v1 = "达标" if (a1 is not None and a1 >= 4.0) else "待改进"
        v2 = "达标" if (a2 is not None and a2 >= 4.0) else "待改进"
        if a1 is not None and a2 is not None:
            max_drift = max(max_drift, abs(a1 - a2))
        verdicts.append({"dimension": dim, "run1_avg": a1, "run2_avg": a2,
                         "run1_verdict": v1, "run2_verdict": v2, "stable": v1 == v2})

    jd1 = r1.get("judge_determinism", {})
    jd2 = r2.get("judge_determinism", {})
    judge_det = bool(jd1.get("all_identical")) and bool(jd2.get("all_identical"))
    verdict_stable = all(v["stable"] for v in verdicts)
    drift_ok = max_drift <= 1.0  # 0-5 量表 ±1.0 上限，超出即判未达成

    all_same = not dim_diffs and not case_diffs and not ctrl_diffs
    if all_same:
        verdict = "PASS（逐字段完全一致）"
    elif judge_det and verdict_stable and drift_ok:
        verdict = ("PASS（按口径：judge 对同一回复逐字段确定 + 两遍 8 维结论全部稳定，"
                   "均分最大漂移 %.4f≤1.0；逐字复现不可能，漂移已列出不隐藏）" % max_drift)
    else:
        verdict = "部分达成/未达成（judge确定=%s 结论稳定=%s 漂移=%.4f，差异已如实列出）" % (
            judge_det, verdict_stable, max_drift)

    return {
        "temperature": TEMPERATURE, "n_runs": 2,
        "per_dimension_avg_identical": not dim_diffs,
        "per_case_scores_identical": not case_diffs,
        "control_scores_identical": not ctrl_diffs,
        "judge_determinism_same_reply": judge_det,
        "judge_determinism_run1": jd1.get("all_identical"),
        "judge_determinism_run2": jd2.get("all_identical"),
        "dimension_verdicts": verdicts,
        "dimension_verdicts_stable": verdict_stable,
        "max_average_drift": round(max_drift, 4),
        "differing_dimension_scores": dim_diffs,
        "differing_case_ids": case_diffs,
        "differing_control_dims": ctrl_diffs,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def _fmt(x, digits=4):
    return "n/a" if x is None else ("%." + str(digits) + "f") % x


def print_report(result: dict) -> None:
    print("=" * 82)
    print("PE-2 Prompt Golden Set 报告（eval/prompt_golden.py）")
    print("=" * 82)
    if result["status"] != "RUN":
        print("状态：%s" % result["status"])
        print("原因：%s" % result.get("reason", ""))
        print("=" * 82)
        return

    run = result["run"]
    corpus = result["corpus"]
    print("状态：RUN | 模型=%s judge=%s temperature=%s runs=%d" %
          (run["generation_model"], run["judge_model"], run["temperature"], run["n_runs"]))
    print("语料：%s" % corpus["path"])
    print("  sha256=%s... cases=%d 维度=%d 来源=%s" %
          (corpus["sha256"][:16], corpus["n_cases"], corpus["n_dimensions"], corpus["source"]))
    print("  口径：%s" % corpus["source_note"])
    print("  prompt：%s" % result["prompt"]["source"])
    print("  prompt 模板 sha256=%s..." % result["prompt"]["template_sha256"][:16])

    r1 = result["runs"][0]
    print("\n[E2-V1] 维度覆盖：%d 个（要求≥8）" % corpus["n_dimensions"])
    for d in result["dimension_meta"]:
        print("  %-26s %-14s %s" % (d["id"], d["applicability"], d["rule_ref"]))

    print("\n[E2-V2] 真实 hy4 每维度分数（生成+判定 %d 条，成功 %d，失败 %d）" %
          (r1["n_cases"], r1["n_judged"], r1["n_failed"]))
    for dim, sm in r1["dimension_scores"].items():
        print("  %-26s 均分=%s/5 已判=%d/%d 分布=%s" %
              (dim, _fmt(sm["avg_score"], 3), sm["n_scored"], sm["applicable_cases"],
               sm["score_distribution"]))
        if sm["sample_reasons"]:
            print("      例：%s" % sm["sample_reasons"][0][:110])

    print("\n[E2-V3] judge 好/坏区分（8 维人工对照，good≥4 且 bad≤2 且差≥2）")
    for dim, e in r1["controls"].items():
        print("  %-26s good=%s bad=%s 区分=%s" %
              (dim, e.get("good_score"), e.get("bad_score"), e.get("separated")))
        if e.get("error"):
            print("      %s" % e["error"])
    print("  判定：%s（%d/8 维分离）" %
          (r1["control_verdict"], sum(1 for v in r1["control_checks"].values() if v)))

    rp = result["reproducibility"]
    print("\n[E2-V4] 两次运行可复现（temperature=0，两遍独立生成+判定）")
    print("  judge 对同一回复确定性：run1=%s run2=%s（每遍前 6 条 case 二次判定逐字段比对）" %
          (rp.get("judge_determinism_run1"), rp.get("judge_determinism_run2")))
    print("  两遍 8 维结论稳定=%s 均分最大漂移=%s（0-5 量表，口径≤1.0）" %
          (rp.get("dimension_verdicts_stable"), rp.get("max_average_drift")))
    for v in rp.get("dimension_verdicts", []):
        print("    %-26s run1=%s(%s) run2=%s(%s) 稳定=%s" %
              (v["dimension"], _fmt(v["run1_avg"], 3), v["run1_verdict"],
               _fmt(v["run2_avg"], 3), v["run2_verdict"], v["stable"]))
    print("  逐维度均分逐字一致=%s 逐case逐字一致=%s 对照逐字一致=%s" %
          (rp["per_dimension_avg_identical"], rp["per_case_scores_identical"],
           rp["control_scores_identical"]))
    print("  判定：%s" % rp["verdict"])
    for label, key in (("维度差异", "differing_dimension_scores"),
                       ("case差异", "differing_case_ids"),
                       ("对照差异", "differing_control_dims")):
        if rp[key]:
            print("    %s：%s" % (label, rp[key]))

    ks = result["key_safety"]
    print("\n[key 安全] %s：%s" % (ks["verdict"], ks["detail"]))
    print("  无 key 行为：优雅 NOT_RUN（命令：python eval/prompt_golden.py --key-file <不存在路径>）")

    print("\n[验收总览]")
    for k in ("E2-V1", "E2-V2", "E2-V3", "E2-V4"):
        v = result["acceptance"][k]
        print("  %-6s %-14s — %s" % (k, v["verdict"], v["detail"]))
    print("\n边界：judge 为模型近似评分非真人；24 条手写语料非真实用户数据；")
    print("      生成用 temperature=0（产品线上为 0.7）以换可复现；数字只代表本语料。")
    print("=" * 82)


def not_run_result(reason: str, cases_path: Path) -> dict:
    return {
        "schema_version": "1.0", "status": "NOT_RUN", "reason": reason,
        "run": {"timestamp": datetime.now(timezone.utc).isoformat(), "temperature": TEMPERATURE},
        "corpus": {"path": str(cases_path)},
        "acceptance": {k: {"verdict": "NOT_RUN", "detail": reason}
                       for k in ("E2-V1", "E2-V2", "E2-V3", "E2-V4")},
    }


def main() -> int:
    _reconfigure_stdout()
    args = parse_args()
    setup_env()
    cases_path = Path(args.cases).resolve()
    corpus = load_cases(cases_path)
    dims = corpus["dimensions"]
    cases = corpus["cases"]
    cases_by_id = {c["id"]: c for c in cases}
    all_dim_ids = [d["id"] for d in dims]
    controls = build_controls()

    key_file = Path(args.key_file).resolve() if args.key_file else Path(
        str(BACKEND.parent / "work" / "local-test-key.txt"))
    key, status, reason = read_key(key_file)
    if status != "RUN":
        result = not_run_result(reason, cases_path)
        print_report(result)
        print("\n===JSON===\n" + json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    secrets = [key["api_key"]]
    for ln in key_file.read_text(encoding="utf-8").splitlines():
        if ln.strip().startswith("AUTH="):
            secrets.append(ln.strip().split("=", 1)[1].strip())

    template, prompt_meta = build_generation_template()
    if template is None:
        result = not_run_result(prompt_meta.get("reason", "prompt 抽取失败"), cases_path)
        print_report(result)
        print("\n===JSON===\n" + json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    try:
        sys.path.insert(0, str(BACKEND))
        from config import Settings
        from gateway.client import LLMClient
    except Exception as exc:  # noqa: BLE001
        result = not_run_result("依赖导入失败：%s" % _redact(exc, secrets), cases_path)
        print_report(result)
        print("\n===JSON===\n" + json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    gen_model = args.model or GENERATION_MODEL_DEFAULT
    judge_model = args.judge_model or gen_model
    session_id = args.session_id or ("prompt-golden-" + uuid.uuid4().hex[:12])
    client, headers = build_client(key, Settings, LLMClient, gen_model, session_id)
    judge_system = build_judge_system(dims)

    # key 有效性探针：空回复/异常一律 NOT_RUN，绝不打印 key
    try:
        probe = client.chat([{"role": "user", "content": "只输出 ok 两个字母。"}],
                            temperature=TEMPERATURE, max_tokens=GEN_BUDGETS[0], model=gen_model)
        if not (probe.get("content") or "").strip():
            result = not_run_result("key 无效或网关未返回正文（探针空回复）；标 NOT_RUN，不硬凑", cases_path)
            print_report(result)
            print("\n===JSON===\n" + json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
    except Exception as exc:  # noqa: BLE001
        result = not_run_result("key 无效/网关不可达（探针失败）：%s；标 NOT_RUN，不硬凑" % _redact(exc, secrets), cases_path)
        print_report(result)
        print("\n===JSON===\n" + json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    n_runs = 1 if args.controls_only else max(1, args.runs)
    runs: list[dict] = []
    for i in range(n_runs):
        print("[run %d/%d] 开始：对照 8×2 次 judge + %d 条 case 生成&判定%s" % (
            i + 1, n_runs, len(cases),
            "（controls-only 跳过 case）" if args.controls_only else ""), flush=True)
        r = run_pipeline(client, template, judge_system, cases, controls, dims,
                         cases_by_id, gen_model, judge_model, with_gen=not args.controls_only)
        runs.append(r)
        print("[run %d/%d] 完成：对照 %s，case 成功 %d/%d" % (
            i + 1, n_runs, r["control_verdict"], r["n_judged"], r["n_cases"]), flush=True)

    repro = compare_runs(runs[0], runs[1]) if len(runs) >= 2 else {
        "temperature": TEMPERATURE, "n_runs": 1,
        "per_dimension_avg_identical": None, "per_case_scores_identical": None,
        "control_scores_identical": None, "differing_dimension_scores": [],
        "differing_case_ids": [], "differing_control_dims": [],
        "verdict": "NOT_RUN（只跑了 1 遍，--runs 1 或 --controls-only 不足以验收 E2-V4）",
    }

    r1 = runs[0]
    dim_covered = all(s["n_scored"] > 0 for s in r1["dimension_scores"].values())
    acc = {
        "E2-V1": {
            "verdict": "PASS" if len(dims) >= 8 else "未达成",
            "detail": "维度表 %d 个（要求≥8）：%s" % (len(dims), ", ".join(all_dim_ids)),
        },
        "E2-V2": {
            "verdict": ("PASS" if (not args.controls_only and r1["n_judged"] >= 24 and dim_covered)
                        else "NOT_RUN/未达成"),
            "detail": "真实 %s 跑完 %d 条 case（成功判定 %d，失败 %d），8 维全部有真实分数=%s%s" % (
                gen_model, r1["n_cases"], r1["n_judged"], r1["n_failed"], dim_covered,
                "；--controls-only 不足以验收" if args.controls_only else ""),
        },
        "E2-V3": {
            "verdict": r1["control_verdict"],
            "detail": "8 维好/坏对照，%d/8 维满足 good≥4 且 bad≤2 且差≥2" %
                      sum(1 for v in r1["control_checks"].values() if v),
        },
        "E2-V4": {
            "verdict": repro["verdict"],
            "detail": "两遍独立生成+判定；judge 同回复确定性=%s，8 维结论稳定=%s，均分最大漂移=%s；"
                      "逐维度均分逐字一致=%s（opencode-go 温度 0 生成不逐字复现，已标注口径）" % (
                          repro.get("judge_determinism_same_reply"),
                          repro.get("dimension_verdicts_stable"),
                          repro.get("max_average_drift"),
                          repro["per_dimension_avg_identical"]),
        },
    }

    dump = {"runs": runs, "controls": r1["controls"], "acceptance": acc,
            "key": {"source": "work/local-test-key.txt（--key-file 指定）"}}
    leaked = any(s and (s in json.dumps(dump, ensure_ascii=False) or s in str(headers))
                 for s in secrets)
    key_safety = {
        "verdict": "PASS" if not leaked else "FAIL",
        "detail": "key 仅读入进程内存，不打印不写 JSON；错误文本经 _redact 脱敏；"
                  "输出明文扫描=%s；无 key 优雅 NOT_RUN" % ("发现泄漏" if leaked else "未发现泄漏"),
    }

    result = {
        "schema_version": "1.0", "status": "RUN",
        "run": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generation_model": gen_model, "judge_model": judge_model,
            "temperature": TEMPERATURE, "n_runs": n_runs,
            "gen_max_tokens_budgets": list(GEN_BUDGETS),
            "judge_max_tokens_budgets": list(JUDGE_BUDGETS),
            "extra_headers": headers, "base_url": key["base_url"],
            "env": {"LLM_BASE_URL": key["base_url"], "LLM_API_KEY": "<from-file, not printed>",
                    "MOCK_LLM": "false", "USE_INSTRUCTOR": "0"},
        },
        "key": {"source": "work/local-test-key.txt（--key-file 指定）",
                "in_memory_only": True, "printed": False, "committed": False},
        "prompt": prompt_meta,
        "corpus": {
            "path": str(cases_path), "sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
            "n_cases": len(cases), "n_dimensions": len(dims),
            "source": corpus.get("source", "synthetic-handwritten"),
            "source_note": corpus.get("source_note", ""),
        },
        "dimension_meta": [{"id": d["id"], "name": d["name"],
                            "applicability": d["applicability"], "rule_ref": d["rule_ref"]} for d in dims],
        "runs": runs,
        "reproducibility": repro,
        "acceptance": acc,
        "key_safety": key_safety,
        "notes": [
            "judge 为 LLM 近似评分，不是真人评分；24 条手写 case 不是真实用户数据。",
            "生成 temperature=0 为可复现；产品线上 breakup_chat 用 temperature=0.7，口径不同已标注。",
            "conditional 维度只对以它为主题的 case 打分并求均分；universal 维度对所有 case 打分。",
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
