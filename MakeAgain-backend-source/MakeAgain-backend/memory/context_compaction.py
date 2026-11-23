"""长对话上下文压缩（从 PI compaction.ts 移植，MIT，2026；docs/CONTEXT_REUSE_PLAN.md）。

核心算法逐字移植自 earendil-works/pi 的纯函数实现：
- token 估算 chars/4（保守高估）
- 从最新往前找切点，绝不在 toolResult 切
- 结构化摘要 + UPDATE 合并上次摘要
- CompactionEntry{summary, first_kept_entry_id, tokens_before}
- 失败保护（error/length 不落库）、摘要不调工具

改造点（复用而非照搬）：
- TS → Python
- 摘要格式中文化，适配情感陪伴（保留关系/情绪/事实，而非代码任务的 file paths）
- 存储用我们的 SQLite（context_compactions 表），不用 PI 的 session 文件格式

开关 USE_CONTEXT_COMPACTION 默认关闭，关闭时零影响。
"""
from __future__ import annotations

import json
import logging
import math
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

# 默认参数（PI: reserveTokens=16384, keepRecentTokens=20000；我们小模型按比例缩小）
DEFAULT_RESERVE_TOKENS = int(os.getenv("COMPACTION_RESERVE_TOKENS", "1200"))
DEFAULT_KEEP_RECENT_TOKENS = int(os.getenv("COMPACTION_KEEP_RECENT_TOKENS", "2000"))
DEFAULT_CONTEXT_WINDOW = int(os.getenv("COMPACTION_CONTEXT_WINDOW", "6000"))


def compaction_enabled() -> bool:
    return os.getenv("USE_CONTEXT_COMPACTION", "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class CompactionEntry:
    id: str
    session_id: str
    summary: str
    first_kept_id: str          # 从第几条消息开始保留原文（之前的压缩掉）
    tokens_before: int
    created_at: str
    usage: dict | None = None
    details: dict | None = None


@dataclass
class CutPointResult:
    first_kept_index: int       # 开始保留的消息 index
    turn_start_index: int = -1  # split turn 时的 turn 起点
    is_split_turn: bool = False


# ---------------- token 估算（PI: estimateTokens, chars/4）----------------

def estimate_tokens(message: dict) -> int:
    """PI 用 chars/4 保守高估。中文字符按1字符算（中文1字≈1token，chars/4 对中文偏小，故中文按每字1token）。"""
    role = message.get("role", "")
    content = message.get("content", "")
    chars = 0
    if isinstance(content, str):
        chars = len(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    chars += len(block.get("text", ""))
                elif block.get("type") == "thinking":
                    chars += len(block.get("thinking", ""))
                elif block.get("type") == "toolCall":
                    chars += len(block.get("name", "")) + len(json.dumps(block.get("arguments", {}), ensure_ascii=False))
    # 中文占比高时按每字1token估计（hy4 中文 tokenizer 大致如此），英文按 chars/4
    zh = sum(1 for c in str(content) if "\u4e00" <= c <= "\u9fff")
    if zh > 0:
        return max(1, zh + (chars - zh) // 4)
    return max(1, math.ceil(chars / 4))


def estimate_context_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(m) for m in messages)


def should_compact(context_tokens: int, context_window: int = DEFAULT_CONTEXT_WINDOW,
                   reserve_tokens: int = DEFAULT_RESERVE_TOKENS) -> bool:
    """PI: contextTokens > contextWindow - reserveTokens"""
    return context_tokens > context_window - reserve_tokens


# ---------------- 切点（PI: findCutPoint）----------------

def _is_cut_point(message: dict) -> bool:
    """有效切点：user/assistant；绝不在 toolResult 切（工具结果必须跟调用在一起）。"""
    return message.get("role") in {"user", "assistant", "system", "custom"}


def find_cut_point(messages: list[dict], keep_recent_tokens: int = DEFAULT_KEEP_RECENT_TOKENS) -> CutPointResult:
    """从最新往前累加 token，到预算后找最近有效切点。"""
    if not messages:
        return CutPointResult(first_kept_index=0)
    accumulated = 0
    cut_index = 0
    for i in range(len(messages) - 1, -1, -1):
        mt = estimate_tokens(messages[i])
        if mt == 0:
            continue
        accumulated += mt
        if accumulated >= keep_recent_tokens:
            # 找 >= i 的最近有效切点
            candidates = [j for j in range(i, len(messages)) if _is_cut_point(messages[j])]
            cut_index = candidates[0] if candidates else 0
            break
    # 是否在 turn 中间切（切点不是 user）
    is_split = cut_index > 0 and messages[cut_index].get("role") != "user"
    turn_start = -1
    if is_split:
        for j in range(cut_index - 1, -1, -1):
            if messages[j].get("role") == "user":
                turn_start = j
                break
    return CutPointResult(first_kept_index=cut_index, turn_start_index=turn_start, is_split_turn=is_split)


# ---------------- 摘要 prompt（PI 结构化格式，中文化适配情感陪伴）----------------

SUMMARY_SYSTEM = "你是对话记忆整理员。把对话压缩成一段供后续接续的结构化摘要。只输出摘要，不寒暄。"

SUMMARY_PROMPT = """下面的对话需要压缩成一段结构化摘要，供另一个 AI 接着陪伴用户时使用。

严格按此格式：

## 这段关系/话题
- 用户在经历什么（分手/丧亲/宠物离世/其他）、对象是 whom

## 用户的画像与偏好
- 已确认的称呼、身份、偏好、明确不想谈的话题

## 情绪脉络
- 用户情绪怎么变化的、现在大致在什么状态

## 已聊过的重要事实
- 具体发生的事、提到的人/物/时间（原话里的细节要保留）

## 陪伴者做过什么
- 给过的建议、承诺、下一步

## 接续要点
- 下次续聊该接住什么、避免什么

每节简明。保留原话里的具体细节（名字、日期、物品），不要泛化。"""

UPDATE_PROMPT = """下面是【新的对话片段】和【之前的摘要】。请把新信息合并进摘要，规则：
- 保留之前摘要的全部重要信息
- 加入新的事实、情绪变化、进展
- 更新情绪脉络与接续要点
- 若事实有更新/修正，用新信息覆盖旧的（注明"原本…现在…"）
- 不要编造片段里没有的内容

严格按这 6 节格式：## 这段关系/话题 / ## 用户的画像与偏好 / ## 情绪脉络 / ## 已聊过的重要事实 / ## 陪伴者做过什么 / ## 接续要点

<previous-summary>
{previous}
</previous-summary>

<new-conversation>
{conversation}
</new-conversation>
"""


def serialize_conversation(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = "用户" if m.get("role") == "user" else ("陪伴者" if m.get("role") == "assistant" else m.get("role", ""))
        content = m.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_summary_messages(messages_to_summarize: list[dict], previous_summary: str | None = None) -> list[dict]:
    conv = serialize_conversation(messages_to_summarize)
    if previous_summary:
        user_content = UPDATE_PROMPT.format(previous=previous_summary, conversation=conv)
    else:
        user_content = f"<conversation>\n{conv}\n</conversation>\n\n{SUMMARY_PROMPT}"
    return [{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": user_content}]


# ---------------- 压缩执行 ----------------

@dataclass
class CompactionResult:
    entry: CompactionEntry | None
    kept_messages: list[dict]
    error: str | None = None


def compact(messages: list[dict], client, session_id: str, previous_entry: CompactionEntry | None = None,
            keep_recent_tokens: int = DEFAULT_KEEP_RECENT_TOKENS) -> CompactionResult:
    """执行一次压缩：摘要旧消息 + 保留最近消息。失败时不压缩（返回 kept=全部原文）。"""
    cut = find_cut_point(messages, keep_recent_tokens)
    to_summarize = messages[:cut.first_kept_index]
    kept = messages[cut.first_kept_index:]
    if not to_summarize:
        return CompactionResult(entry=None, kept_messages=messages)

    msgs = build_summary_messages(to_summarize, previous_entry.summary if previous_entry else None)
    try:
        resp = client.chat(msgs, temperature=0.0, model=client.settings.llm_fast_model, max_tokens=1500)
        summary = (resp.get("content") or "").strip()
    except Exception as exc:
        return CompactionResult(entry=None, kept_messages=messages, error=f"SUMMARY_CALL_FAILED: {str(exc)[:100]}")

    # 失败保护（PI: getSummarizationFailure）
    if not summary:
        return CompactionResult(entry=None, kept_messages=messages, error="SUMMARY_EMPTY")
    if "```" in summary and "## " not in summary:
        return CompactionResult(entry=None, kept_messages=messages, error="SUMMARY_MALFORMED")
    if len(summary) < 30:
        return CompactionResult(entry=None, kept_messages=messages, error="SUMMARY_TOO_SHORT")

    entry = CompactionEntry(
        id=uuid.uuid4().hex, session_id=session_id, summary=summary,
        first_kept_id=str(cut.first_kept_index),  # 用 index 作 id（我们的消息没有独立 id）
        tokens_before=estimate_context_tokens(messages),
        created_at=datetime.now(timezone.utc).isoformat(),
        usage=resp.get("usage"),
        details={"n_summarized": len(to_summarize), "n_kept": len(kept), "is_split_turn": cut.is_split_turn},
    )
    return CompactionResult(entry=entry, kept_messages=kept)


# ---------------- 存储（SQLite，复用我们的 db）----------------

def save_entry(db, entry: CompactionEntry) -> None:
    """持久化压缩记录。

    用独立会话真提交——当前请求 session 在 managed 模式下 commit() 只 flush，
    且 api/security.py 的 get_session 不 finish，摘要若不独立提交会永远落不了库（P1 bug）。
    独立会话与原请求无事务耦合，失败 swallow（压缩记录丢失不影响主对话）。
    """
    try:
        from memory.db import _RawSessionLocal
        d = _RawSessionLocal()
        try:
            d.execute(
                text("INSERT INTO context_compactions (id, session_id, summary, first_kept_id, tokens_before, created_at, usage_json, details_json) "
                     "VALUES (:id, :session_id, :summary, :first_kept_id, :tokens_before, :created_at, :usage_json, :details_json)"),
                {"id": entry.id, "session_id": entry.session_id, "summary": entry.summary,
                 "first_kept_id": entry.first_kept_id, "tokens_before": entry.tokens_before,
                 "created_at": entry.created_at, "usage_json": json.dumps(entry.usage or {}, ensure_ascii=False),
                 "details_json": json.dumps(entry.details or {}, ensure_ascii=False)},
            )
            d.commit()
        finally:
            d.close()
    except Exception as exc:
        logger.warning("compaction save failed (non-blocking): %s", exc)


def load_latest_entry(db, session_id: str) -> CompactionEntry | None:
    row = db.execute(
        text("SELECT * FROM context_compactions WHERE session_id=:sid ORDER BY created_at DESC LIMIT 1"),
        {"sid": session_id},
    ).fetchone()
    if not row:
        return None
    return CompactionEntry(
        id=row["id"], session_id=row["session_id"], summary=row["summary"],
        first_kept_id=row["first_kept_id"], tokens_before=row["tokens_before"], created_at=row["created_at"],
        usage=json.loads(row["usage_json"] or "{}"), details=json.loads(row["details_json"] or "{}"),
    )
