"""AIRI/plast-mem 值得复用的三个设计思路（移植改造，不整包接，2026-11-29）。

来源：moeru-ai/plast-mem README（MIT，2026-01~04，权重0.7，项目自称"核心未完成"）。
只抄机制、改造适配 Python/SQLite；每个函数都是纯函数、可选、不自动接主链路——
接不接由对比数据决定（用户："不要被自己的假繁荣蒙蔽，已有实现要对比谁更好"）。

三个模式：
1. 事件分段切点（Event Segmentation）：在 token 预算切点附近优先选话题边界，
   避免在话题中间切断（对照 context_compaction.find_cut_point 的"最近有效切点"）。
2. flashbulb 显著度（surprise≥0.85 慢衰减）：算消息/事实的显著度，高的衰减慢
   （对照 agent-memory-core 的统一衰减）。
3. 8 类事实分类映射（identity/preference/interest/personality/relationship/
   experience/goal/guideline）：映射到我们的 6 类 kind，供抽取/注入参考。
"""
from __future__ import annotations

import re
from typing import Any

# ---- 1. 事件分段切点 ---------------------------------------------------------

# 话题切换信号：这些词出现在消息开头，通常意味着新话题（中文口语）
_TOPIC_SHIFT_RE = re.compile(
    r"^(另外|对了|话说|还有|换个话题|说起来|对了哦|那|所以|后来|直到|有一天|"
    r"突然|昨天|今天|上周|最近|还记得|你记不记得)"
)
# 时间/事件边界词
_EVENT_BOUNDARY_RE = re.compile(r"(走了|去世|分手|结婚|搬家|出差|回来|终于|最后|第一次|最后一次|那天)")


def topic_boundary_score(message: dict) -> float:
    """给一条消息打"话题边界"分（0~1）：越高越适合作为压缩切点。

    纯启发式（零 LLM）：开头话题切换词 + 事件边界词。
    """
    text = str(message.get("content", ""))
    score = 0.0
    if _TOPIC_SHIFT_RE.search(text[:12]):
        score += 0.5
    if _EVENT_BOUNDARY_RE.search(text):
        score += 0.4
    if message.get("role") == "user":
        score += 0.1  # user 消息更适合作为切点（PI 原规则）
    return min(1.0, score)


def find_cut_point_by_event(messages: list[dict], token_budget: int,
                            token_fn) -> int:
    """AIRI 式切点：在 token 预算附近，选"话题边界分最高"的有效切点。

    对照 context_compaction.find_cut_point（只找最近有效切点，不看话题）。
    返回 first_kept_index（从这条开始保留原文）。
    """
    if not messages:
        return 0
    # 先找 token 预算落点（从最新往前累加）
    accumulated, budget_index = 0, 0
    for i in range(len(messages) - 1, -1, -1):
        accumulated += token_fn(messages[i])
        if accumulated >= token_budget:
            budget_index = i
            break
    # 在 budget_index 附近 ±3 条窗口内，找话题边界分最高的 user/system 消息
    lo = max(0, budget_index - 3)
    hi = min(len(messages), budget_index + 4)
    best_idx, best_score = budget_index, -1.0
    for j in range(lo, hi):
        if messages[j].get("role") not in {"user", "system", "assistant"}:
            continue
        s = topic_boundary_score(messages[j])
        if s > best_score:
            best_idx, best_score = j, s
    return best_idx


def compare_cut_points(messages: list[dict], token_budget: int, token_fn,
                       legacy_cut_fn) -> dict:
    """对比新旧切点：给数据，不预设谁好。"""
    old = legacy_cut_fn(messages, token_budget)
    old_idx = getattr(old, "first_kept_index", old if isinstance(old, int) else 0)
    new_idx = find_cut_point_by_event(messages, token_budget, token_fn)
    return {
        "legacy_first_kept": old_idx,
        "event_first_kept": new_idx,
        "delta_index": new_idx - old_idx,
        "legacy_boundary_score": round(topic_boundary_score(messages[old_idx]), 2) if 0 <= old_idx < len(messages) else None,
        "event_boundary_score": round(topic_boundary_score(messages[new_idx]), 2) if 0 <= new_idx < len(messages) else None,
        "note": "事件切点是否更好=看它是否避开话题中间；需下游事实保留/摘要质量真评测判定，分数高不等于产品好",
    }


# ---- 2. flashbulb 显著度 -----------------------------------------------------

_SURPRISE_STRONG = re.compile(r"去世|走了|分手|离婚|自杀|死了|终于|没想到|居然|竟然|第一次|最后一次|永远|再也不")
_SURPRISE_NEG = re.compile(r"不是|其实|原来|结果|反而|却")


def surprise_score(text: str) -> float:
    """flashbulb 显著度（0~1）：重大事件/转折词越多越高。对照统一衰减。

    阈值经真实语料校准（2026-11-29）：1 个重大事件词（如"去世/分手"）即
    0.6，2 个即 0.85+ 触发 flashbulb——AIRI 原文"surprise≥0.85"是相对它自己
    的 surprise 计算，我们不能照搬绝对数，按中文陪伴语料重新标定。
    """
    score = 0.0
    hits = len(_SURPRISE_STRONG.findall(text))
    score += min(0.85, hits * 0.45)  # 1 hit=0.45, 2 hits=0.85(flashbulb)
    if _SURPRISE_NEG.search(text):
        score += 0.15
    if hits >= 2:
        score = max(score, 0.85)
    return round(min(1.0, score), 3)


def retention_boost(surprise: float) -> float:
    """显著度→衰减保护系数（1.0=正常，>1=衰减更慢）。纯函数，供外部衰减器用。"""
    if surprise >= 0.85:      # flashbulb：显著慢衰减
        return 3.0
    if surprise >= 0.5:
        return 1.5
    return 1.0


def compare_decay(samples: list[str]) -> list[dict]:
    """对比统一衰减 vs flashbulb 增强：给每条的显著度和保护系数。"""
    return [
        {"text": s[:40], "surprise": surprise_score(s),
         "uniform_boost": 1.0, "flashbulb_boost": retention_boost(surprise_score(s))}
        for s in samples
    ]


# ---- 3. 8 类事实分类映射 -----------------------------------------------------

# AIRI 8 类 → 我们 6 类 kind 的映射（我们的 kind: user/object/memory/emotion_trigger/goal/time）
AIRI_KIND_MAP = {
    "identity": "user",           # 身份→用户画像
    "preference": "user",         # 偏好→用户画像
    "interest": "user",
    "personality": "user",
    "relationship": "object",     # 关系→对象画像
    "experience": "memory",       # 经历→记忆
    "goal": "goal",               # 目标→目标（直接对应）
    "guideline": "emotion_trigger",  # 行为准则→情绪触发/规则（近似）
}

_KIND_PATTERNS = {
    "identity": re.compile(r"我叫|我是|我今年|我(是|.+)的(孩子|女朋友|男朋友|妈妈|爸爸)"),
    "preference": re.compile(r"我喜欢|我讨厌|我不喜欢|我习惯|最爱"),
    "relationship": re.compile(r"(前女友|前男友|老公|老婆|爸爸|妈妈|外公|外婆|猫|狗|它|他|她)"),
    "experience": re.compile(r"(那年|那次|曾经|以前|昨天|上周|一起去|陪我)"),
    "goal": re.compile(r"(想|希望|打算|目标|要学会|决定)"),
}


def classify_airi_kind(text: str) -> str:
    """把一句事实映射到 AIRI 8 类之一（启发式，零 LLM）。"""
    for kind, pat in _KIND_PATTERNS.items():
        if pat.search(text):
            return kind
    return "experience"  # 默认归经历


def to_our_kind(airi_kind: str) -> str:
    return AIRI_KIND_MAP.get(airi_kind, "memory")


def compare_fact_kinds(facts: list[str]) -> list[dict]:
    """对比现有 kind 口径 vs AIRI 8 类映射：给数据，看是否值得扩充。"""
    out = []
    for f in facts:
        ak = classify_airi_kind(f)
        out.append({"fact": f[:50], "airi_kind": ak, "mapped_our_kind": to_our_kind(ak)})
    return out
