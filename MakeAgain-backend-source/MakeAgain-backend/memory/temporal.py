"""时间依据的按需召回:摘要按状态触发、事件按时间参照,零 LLM、纯本地。

解决引擎说「你比一个月前的自己强多了——那时候你连白天都熬不住」这类**没有依据的时间对比**:
只在用户「回想过去状态 / 讲现在状态·变化 / 带了时间参照」时,才召回
- 最近的每日摘要(`DailyChatSummary`,作为「近期状态轨迹」),
- 或按时间匹配的事件记忆(`MemoryEntry` 的 `time` fact / 写入时刻 `ts`),
注入上下文;其余轮次不注入、不打扰、不灌水。

时间口径(现状):
- `MemoryEntry.ts` = 写入时刻(用户告诉我们的时刻),用它做相对时间窗口(如「两个月前」)的近似锚点。
- `MemoryEntry.facts` 里 `kind="time"` 的 fact = 事件发生的真实时间(自由文本,如「去年生日」「三年前」),用它做文本时间匹配。
"""
from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Optional

from config import get_settings
from memory import store
from memory.facts import facts_text
from memory.recall import _relevance, _tokens


# ============================================================================
# R4 · 时序替换总开关 USE_MATURE_TEMPORAL（os.getenv，默认关闭）
# 关闭时 parse_time_ref / _temporal_reflect 走原来的纯词典路径，行为与基线逐字节一致；
# 开启时用 dateparser 精确解析中文相对时间，失败/未覆盖再回退原词典路径。
# ============================================================================

def mature_temporal_enabled() -> bool:
    """总开关：默认关闭；只有 1/true/yes/on（忽略大小写与空白）才开启。"""
    return (os.getenv("USE_MATURE_TEMPORAL") or "").strip().lower() in {"1", "true", "yes", "on"}


# 资产来源（gh api 实读原文）：getzep/graphiti  graphiti_core/prompts/extract_edges.py
#   - L60-L71  EdgeTimestamps 的 valid_at / invalid_at Field(description)
#   - L270-L286 extract_timestamps() 的 Rules（未来计划用陈述时间、软 deadline 不作失效、
#     只有日期补 00:00:00、ISO8601+Z、不幻觉）——原文复制，作为 LLM 时序 prompt 增强素材。
# 本模块本身零 LLM，故这里只落地为常量；调用方拼 prompt 时可直接注入。
_EDGE_TIMESTAMP_FIELD_DESCRIPTIONS = {
    "valid_at": (
        "When the fact became true. Future-tense plans/promises use the reference time, "
        "not the future occurrence date. ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z)"
    ),
    "invalid_at": (
        "When the fact stopped being true. Soft deadlines are not invalid_at values. "
        "ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z)"
    ),
}

_GRAPHITI_TEMPORAL_RULES = (
    'Resolve relative expressions ("last week", "2 years ago", "yesterday") using REFERENCE TIME.',
    'If the fact is ongoing (present tense), set valid_at to REFERENCE TIME.',
    'If the fact is a future-tense plan, promise, scheduled event, or deadline, set valid_at to '
    'REFERENCE TIME. Do NOT set valid_at to the future occurrence date.',
    'Preserve future occurrence dates and deadlines in the fact text. Do NOT use a soft deadline '
    '("by March 2026", "by the end of Q3 2025") as invalid_at.',
    'Use a future date as invalid_at only when the fact describes an ongoing state that ends then '
    '(for example, "is unavailable until December 25, 2024" or "works at Acme until January 15, 2027").',
    'If a change or end is expressed, set invalid_at to the relevant time.',
    'Leave both null if no time is stated or resolvable.',
    'If only a date is mentioned (no time), assume 00:00:00.',
    'Use ISO 8601 with Z suffix (e.g., 2025-04-30T00:00:00Z).',
    'Do NOT hallucinate or infer dates from unrelated events.',
)


# ---- 触发信号:回想过去状态 / 讲现在状态·变化(宁缺毋滥,只为一个紧凑块) ----

_REFLECT_HINTS = (
    "那时候", "刚开始", "刚分手", "刚失去", "从前", "当初", "原来我", "那阵子", "前阵子",
    "好多了", "比以前", "比之前", "进步", "走出来", "好转", "恢复", "不一样了", "变了",
    "好一些", "越来越好", "没那么难受", "平静多了",
)


def _temporal_reflect(message: str) -> bool:
    """是否在回想过去状态 / 讲现在状态·变化(决定要不要召回「近期状态轨迹」)。"""
    text = message or ""
    if any(k in text for k in _REFLECT_HINTS):
        return True
    return parse_time_ref(text) is not None


# ---- 时间参照解析(零 LLM) ----

_CN_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9}

# 「X 天/周/个月/年 前」→ 相对时间(约数)
_TIME_REL_RE = re.compile(
    r"(?P<num>[\d一二两三四五六七八九十]{1,3})\s*个?\s*(?P<unit>星期|周|天|日|月|年)前"
)
_UNIT_DAYS = {"星期": 7, "周": 7, "天": 1, "日": 1, "月": 30, "年": 365}

# 文本时间词:有粗略时长的给窗口(用于 ts 兜底);纯事件词只做 time fact 文本匹配。
_TEXT_WORD_WINDOW = {
    "去年": (270, 460),
    "前年": (600, 820),
    "上个月": (15, 50),
    "上上个月": (40, 80),
    "上周": (4, 12),
    "上个星期": (4, 12),
    "上上周": (10, 21),
    "前几天": (1, 10),
    "前些天": (1, 15),
}
_TEXT_TIME_WORDS = tuple(_TEXT_WORD_WINDOW) + ("生日", "纪念日", "过年", "春节", "元旦")


def _cn_to_int(s: str) -> int | None:
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if s in _CN_DIGIT:
        return _CN_DIGIT[s]
    if s == "十":
        return 10
    if s.endswith("十") and s[:-1] in _CN_DIGIT:
        return _CN_DIGIT[s[:-1]] * 10
    if s.startswith("十") and s[1:] in _CN_DIGIT:
        return 10 + _CN_DIGIT[s[1:]]
    if "十" in s:
        h, t = s.split("十", 1)
        if h in _CN_DIGIT and t in _CN_DIGIT:
            return _CN_DIGIT[h] * 10 + _CN_DIGIT[t]
    return None


def _legacy_parse_time_ref(query: str) -> dict | None:
    """基线逐字节保留：纯词典时间参照解析（开关关闭 / 新路径失败时回退到这里）。

    返回 {lo, hi, text}:lo/hi 为相对天数窗口(ts 兜底,可 None);text 为要匹配 time fact 的词。
    无时间参照返回 None。
    """
    text = query or ""
    m = _TIME_REL_RE.search(text)
    if m:
        n = _cn_to_int(m.group("num"))
        if n is not None:
            center = n * _UNIT_DAYS[m.group("unit")]
            return {"lo": center * 0.7, "hi": center * 1.5, "text": []}
    words = [w for w in _TEXT_TIME_WORDS if w in text]
    if words:
        lo = hi = None
        for w in words:
            if w in _TEXT_WORD_WINDOW:
                a, b = _TEXT_WORD_WINDOW[w]
                lo = a if lo is None else min(lo, a)
                hi = b if hi is None else max(hi, b)
        return {"lo": lo, "hi": hi, "text": words}
    return None


# ---- R4 · dateparser 精确解析（DIRECT_INSTALL，开关开启才调用）----
# 资产来源（gh api 实读原文）：scrapinghub/dateparser
#   dateparser/data/date_translation_data/zh.py L146-L205：
#     "1 day ago": ["昨天"] / "1 month ago": ["上个月"] / "1 year ago": ["去年"] /
#     "2 day ago": ["前天"] / "3 day ago": ["大前天"] /
#     "\\1 year ago": ["(\\d++[.,]?\\d*+)年前"] / "\\1 month ago": ["(\\d++...)个月前"]
#   注意：中文数字（「三年前」）不在 dateparser 的 \d+ 词条内，故用基线 _cn_to_int 只做
#   数字归一化/校验，真正时间计算交给 dateparser.parse(text, languages=['zh'])。
_DP_STANDARD_WORDS = (
    "大前天", "前天", "昨天", "今天", "去年", "前年",
    "上上个月", "上个月", "上个星期", "上周",
)


def _dateparser_parse(text: str, anchor: datetime):
    """唯一的 dateparser 调用点，便于测试注入失败以验证回退。"""
    import dateparser  # 延迟导入：开关关闭时环境里没有 dateparser 也不影响基线
    return dateparser.parse(text, languages=["zh"], settings={"RELATIVE_BASE": anchor})


def _dp_age_days(parsed: datetime, anchor: datetime) -> float:
    """把 dateparser 的绝对日期换算成基线形态的相对天数（可 None 的窗口）。"""
    return float(max(0, (anchor.date() - parsed.date()).days))


def _mature_parse_time_ref(query: str) -> dict | None:
    """dateparser 路径。解析不出来返回 None，由 parse_time_ref 回退基线词典。"""
    text = query or ""
    anchor = datetime.now()

    # 1) 「X 天/周/个月/年 前」：中文数字归一化后交给 dateparser 精确计算。
    m = _TIME_REL_RE.search(text)
    if m:
        n = _cn_to_int(m.group("num"))
        if n is not None:
            phrase = f"{n}{m.group('unit')}前"
            try:
                parsed = _dateparser_parse(phrase, anchor)
            except Exception:
                parsed = None
            if parsed is not None:
                age = _dp_age_days(parsed, anchor)
                return {"lo": age, "hi": age, "text": []}
            return None

    # 2) 标准中文时间词：整词交给 dateparser；长词优先，避免「上个月」吃掉「上上个月」。
    hits = sorted({w for w in _DP_STANDARD_WORDS if w in text}, key=len, reverse=True)
    hits = [w for w in hits if not any(w != other and w in other for other in hits)]
    if hits:
        ages = []
        for w in hits:
            try:
                parsed = _dateparser_parse(w, anchor)
            except Exception:
                parsed = None
            if parsed is not None:
                ages.append(_dp_age_days(parsed, anchor))
        if ages:
            words = [w for w in _TEXT_TIME_WORDS if w in text]
            for w in hits:
                if w not in words:
                    words.append(w)
            lo = hi = min(ages)
            return {"lo": lo, "hi": hi, "text": words}
    return None


def parse_time_ref(query: str) -> dict | None:
    """解析用户话里的时间参照；签名与返回形态 {lo, hi, text} 与基线一致。

    开关关闭 → 基线纯词典路径（逐字节一致）；开启 → dateparser 精确解析，
    任何失败/未覆盖 → 回退基线词典路径（含标准词校验）。
    """
    if not mature_temporal_enabled():
        return _legacy_parse_time_ref(query)
    try:
        ref = _mature_parse_time_ref(query)
    except Exception:
        ref = None
    if ref is not None:
        return ref
    return _legacy_parse_time_ref(query)


# ---- mem0 过期判定资产移植（开关开启才生效；基线不认识 expiration_date）----
# 资产来源（gh api 实读原文）：mem0ai/mem0  mem0/memory/main.py L427-L451
# 逐字节复制 _normalize_expiration_date / _payload_is_expired，仅按本模块 import 风格调整类型标注。

def _normalize_expiration_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError as exc:
            raise ValueError("expiration_date must be a valid date in YYYY-MM-DD format.") from exc
    raise ValueError("expiration_date must be a date string in YYYY-MM-DD format.")


def _payload_is_expired(payload: Optional[Dict[str, Any]]) -> bool:
    if not payload:
        return False
    expiration_date = payload.get("expiration_date")
    if not expiration_date:
        return False
    try:
        return date.fromisoformat(str(expiration_date)) < datetime.now(timezone.utc).date()
    except ValueError:
        return False


def payload_is_expired(payload) -> bool:
    """过期判定入口。开关关闭时基线没有任何过期概念→恒 False；开启时走 mem0 移植判定。"""
    if not mature_temporal_enabled():
        return False
    try:
        return _payload_is_expired(payload)
    except Exception:
        return False


# ---- 召回 ----

def _today_key() -> str:
    offset = get_settings().timezone_offset_hours
    return (datetime.now() + timedelta(hours=offset)).strftime("%Y-%m-%d")


def _fmt_md(date_key: str) -> str:
    try:
        y, m, d = date_key.split("-")
        return f"{int(m)}月{int(d)}日"
    except (ValueError, AttributeError):
        return date_key or ""


def recall_daily_summaries(db, user_id: str, message: str, top_k: int = 5) -> str:
    """触发式召回最近每日摘要,作为「近期状态轨迹」。未触发或无摘要返回空串。"""
    if not _temporal_reflect(message):
        return ""
    today = _today_key()
    recent = [s for s in store.list_daily_chat_summaries(db, user_id) if s.date_key < today][:7]
    if not recent:
        return ""
    q = _tokens(message)
    # 按与当前消息的相关度(「情绪/状态」的零 LLM 代理)排序,全 0 时稳定保留最近几天
    ranked = sorted(recent, key=lambda s: _relevance(q, _tokens(s.summary or "")), reverse=True)
    top = ranked[:top_k]
    top.sort(key=lambda s: s.date_key)  # 最旧→最新,让模型看到进展
    lines = [f"- {_fmt_md(s.date_key)}: {s.summary}" for s in top]
    return "[近期状态轨迹(按日期,只作时间对比依据,不要编造轨迹之外的日期)]\n" + "\n".join(lines)


def recall_by_time(db, user_id: str, query: str, top_k: int = 2, skip_ids=None) -> str:
    """按时间参照召回事件记忆。用户说「两个月前的那件事」时,把当时那条记忆捞回来。"""
    ref = parse_time_ref(query)
    if not ref:
        return ""
    skip = skip_ids or set()
    entries = store.list_memories(db, user_id)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    scored: list[tuple[float, object]] = []
    for e in entries:
        if e.id in skip:
            continue
        s = 0.0
        if ref["lo"] is not None:
            age = (now - e.ts).total_seconds() / 86400.0
            if ref["lo"] <= age <= ref["hi"]:
                s += 3.0
        if ref["text"]:
            ft = facts_text(e.facts)
            for w in ref["text"]:
                if w in ft:
                    s += 2.0
        if s > 0:
            scored.append((s, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for _s, e in scored[:top_k]:
        text = (e.summary or e.content or "").strip()[:80]
        if text:
            out.append(f"- {e.ts.month}月{e.ts.day}日: {text}")
    if not out:
        return ""
    return "[相关往事(你当时说过,按时间召回)]\n" + "\n".join(out)
