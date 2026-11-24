"""Fact 校验 job（Anthropic dreaming 最小版，context engineering research P1 / D-063）。

思路：以 chat_messages 为用户原话 ground truth，离线、确定性、零 LLM 地扫出
「已被后续对话推翻 / 已经过时」的 Fact 候选，交给上层（lead）决定如何处理。

硬约束：
- **只读**：全模块只有 SELECT，不改任何 Fact / MemoryEntry / ChatMessage，不 commit/flush。
- **纯函数口径**：同一 (db, user_id) 输入返回同一结果；不读环境开关、不调网络、不调模型。
- **只报可疑，不自动修**：返回 conflict/stale 候选列表，接线（是否提示用户/触发 supersede）由 lead 决定。
  未接线 = 未验证，本模块的存在不等于产品链路已经具备 Fact 自净能力。
- 判据是启发式正则 + 词形签名重叠，**不是真值判定器**：命中代表「值得复核」，不代表一定错。

判据（三条，互不调用 LLM）：
1. correction（conflict）：后续用户消息含纠错语气（不是/不叫/记错了/改成…）且与 Fact 有词形重叠。
2. change_cue（stale）：后续用户消息含状态变化语气（现在/好多了/不再…）且与 Fact 有词形重叠。
3. temporal（stale）：Fact 自带时间锚（最近/目前/正在/当时…）且距离最新一条消息超过 STALE_DAYS 天。

来源：memory_stream.facts（JSON 列表，kind/fact/confidence）；该列为空时退回 content。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from memory.tokenize_zh import signature

STALE_DAYS = 30
MAX_RESULTS = 50

# 修正/否定语气：出现即视为「这条 Fact 可能被用户亲手推翻」的强线索。
_CORRECTION_RE = re.compile(
    r"(并不是|并非|不是|不叫|没叫|没有|记错|搞错|弄错|错了|不对|"
    r"改成|改叫|换成|其实是|其实不是|不再是|已经不|根本不|根本没)"
)
# 状态变化语气：旧事实可能仍「真过」，但已经不描述当前状态。
_CHANGE_RE = re.compile(
    r"(好多了|好转|改善了|好了很多|不再|重新开始|戒了|出院|走出来|"
    r"又开始了|现在|如今|后来|最近|这两[天周月])"
)
# 时间锚：Fact 本身带时效性词，放久了需要复核。
_TEMPORAL_RE = re.compile(r"(最近|目前|现在|正在|这阵子|那段时间|当时|之前|以前|上周|上个月|去年)")


@dataclass(frozen=True)
class FactIssue:
    """一条可疑 Fact。所有字段都是可 JSON 序列化的值，便于上层记录/取证。"""

    memory_id: str
    fact_index: int
    fact_id: str
    fact_text: str
    kind: str                    # conflict | stale
    rule: str                    # correction | change_cue | temporal
    reason: str
    confidence: float
    evidence_message_id: int | None
    evidence_text: str
    evidence_ts: str
    shared_signatures: tuple[str, ...]
    source: str                  # facts_json | content

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["shared_signatures"] = list(self.shared_signatures)
        return d


def _as_datetime(value: object) -> datetime | None:
    """把 SQLite 里可能是 str / datetime 的时间转成可比较的 naive UTC。"""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        raw = value.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None
    else:
        return None
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def _iso(value: object) -> str:
    dt = _as_datetime(value)
    return dt.isoformat() if dt is not None else str(value or "")


def _load_rows(db: Session, user_id: str, sql: str) -> list[dict[str, Any]]:
    """SELECT only；用 mappings() 而非 row['col']（SQLAlchemy 2.0 Row 不支持字符串下标，D-070 教训）。"""
    try:
        result = db.execute(text(sql), {"uid": str(user_id)})
        return [dict(row) for row in result.mappings().all()]
    except Exception as exc:
        # 仅当上游 models.py 的列集合变化时降级，绝不吞掉其它错误（保持只读契约）。
        if "no such column" not in str(exc).lower():
            raise
        fallback = "SELECT id, content, ts FROM memory_stream WHERE user_id = :uid ORDER BY id ASC"
        if "memory_stream" in sql and "chat_messages" not in sql:
            return [dict(row) for row in db.execute(text(fallback), {"uid": str(user_id)}).mappings().all()]
        raise


def _fact_texts(facts_value: object, content: object) -> list[tuple[str, str]]:
    """返回 [(fact_text, source), ...]；facts JSON 优先，空则退回 content。"""
    items: list[tuple[str, str]] = []
    if isinstance(facts_value, str) and facts_value.strip():
        try:
            parsed = json.loads(facts_value)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, list):
            for entry in parsed:
                if isinstance(entry, dict):
                    txt = str(entry.get("fact") or "").strip()
                else:
                    txt = str(entry or "").strip()
                if txt:
                    items.append((txt, "facts_json"))
    elif isinstance(facts_value, (list, tuple)):
        for entry in facts_value:
            txt = str(entry.get("fact") or "").strip() if isinstance(entry, dict) else str(entry or "").strip()
            if txt:
                items.append((txt, "facts_json"))
    if not items:
        txt = str(content or "").strip()
        if txt:
            items.append((txt, "content"))
    return items


def check_facts(db: Session, user_id: str, *, stale_days: int = STALE_DAYS,
                max_results: int = MAX_RESULTS) -> list[FactIssue]:
    """返回该用户可疑 conflict/stale Fact 列表（只读、零 LLM）。

    同一条 Fact 同类只保留置信度最高的一条证据；结果按 conflict 优先、置信度降序排列。
    空数据 / 无聊天记录返回空列表；不修改任何记录，也不提交事务。
    """
    if db is None:
        raise ValueError("check_facts 需要 SQLAlchemy Session")
    if user_id is None or str(user_id).strip() == "":
        return []

    messages = _load_rows(
        db, user_id,
        "SELECT id, role, content, ts FROM chat_messages "
        "WHERE user_id = :uid ORDER BY id ASC",
    )
    entries = _load_rows(
        db, user_id,
        "SELECT id, facts, content, ts FROM memory_stream "
        "WHERE user_id = :uid ORDER BY ts ASC, id ASC",
    )
    if not messages or not entries:
        return []

    latest_msg_dt = max((_as_datetime(m.get("ts")) for m in messages), default=None)
    # 预计算消息签名与原文，避免 O(facts × messages) 里重复 normalize。
    prepared = [
        (m, signature(m.get("content")))
        for m in messages
        if str(m.get("role") or "") == "user" and str(m.get("content") or "").strip()
    ]

    # 每个 (memory_id, fact_index, kind) 只留最佳候选。
    best: dict[tuple[str, int, str], FactIssue] = {}
    for entry in entries:
        memory_id = str(entry.get("id") or "")
        entry_ts = entry.get("ts")
        for idx, (fact_text, source) in enumerate(_fact_texts(entry.get("facts"), entry.get("content"))):
            fact_sig = signature(fact_text)
            candidates: list[FactIssue] = []

            for msg, msg_sig in prepared:
                shared = sorted(fact_sig & msg_sig)
                if not shared:
                    continue
                content = str(msg.get("content") or "")
                if _CORRECTION_RE.search(content):
                    candidates.append(FactIssue(
                        memory_id=memory_id, fact_index=idx,
                        fact_id=f"{memory_id}:{idx}", fact_text=fact_text,
                        kind="conflict", rule="correction",
                        reason="后续用户消息含纠错语气且与事实有词形重叠",
                        confidence=min(0.99, 0.55 + 0.08 * len(shared)),
                        evidence_message_id=int(msg.get("id")) if msg.get("id") is not None else None,
                        evidence_text=content, evidence_ts=_iso(msg.get("ts")),
                        shared_signatures=tuple(shared), source=source,
                    ))
                elif _CHANGE_RE.search(content):
                    candidates.append(FactIssue(
                        memory_id=memory_id, fact_index=idx,
                        fact_id=f"{memory_id}:{idx}", fact_text=fact_text,
                        kind="stale", rule="change_cue",
                        reason="后续用户消息含状态变化语气且与事实有词形重叠",
                        confidence=min(0.95, 0.45 + 0.08 * len(shared)),
                        evidence_message_id=int(msg.get("id")) if msg.get("id") is not None else None,
                        evidence_text=content, evidence_ts=_iso(msg.get("ts")),
                        shared_signatures=tuple(shared), source=source,
                    ))

            # 时间锚规则：不依赖消息重叠，只看事实是否带时效词且「老」。
            fact_dt = _as_datetime(entry_ts)
            if _TEMPORAL_RE.search(fact_text) and fact_dt and latest_msg_dt:
                age = latest_msg_dt - fact_dt
                if age >= timedelta(days=max(0, stale_days)):
                    candidates.append(FactIssue(
                        memory_id=memory_id, fact_index=idx,
                        fact_id=f"{memory_id}:{idx}", fact_text=fact_text,
                        kind="stale", rule="temporal",
                        reason=f"事实含时间锚且距最新消息 {age.days} 天（阈值 {stale_days}）",
                        confidence=0.5,
                        evidence_message_id=None, evidence_text="", evidence_ts=_iso(latest_msg_dt),
                        shared_signatures=(),  # 时间规则不依赖消息文本重叠
                        source=source,
                    ))

            for issue in candidates:
                key = (memory_id, idx, issue.kind)
                prev = best.get(key)
                # 置信度高者优先；同分取消息 id 小的（更早的推翻证据）。
                if prev is None or (issue.confidence, -(issue.evidence_message_id or 0)) > (
                        prev.confidence, -(prev.evidence_message_id or 0)):
                    best[key] = issue

    ordered = sorted(
        best.values(),
        key=lambda i: (0 if i.kind == "conflict" else 1, -i.confidence, i.memory_id, i.fact_index),
    )
    return ordered[:max(0, max_results)]


def group_by_kind(issues: list[FactIssue]) -> dict[str, list[FactIssue]]:
    """按 conflict/stale 分组（纯函数视图，方便上层分别展示或接线）。"""
    grouped: dict[str, list[FactIssue]] = {"conflict": [], "stale": []}
    for issue in issues:
        grouped.setdefault(issue.kind, []).append(issue)
    return grouped
