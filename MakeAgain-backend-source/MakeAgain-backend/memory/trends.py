"""R4 · hindsight 趋势资产移植（Trend / ObservationEvidence / compute_trend）。

资产来源（gh api 实读原文）：vectorize-io/hindsight
  hindsight-api-slim/hindsight_api/engine/reflect/observations.py
    L15-L30   Trend 枚举（STABLE/STRENGTHENING/WEAKENING/NEW/STALE）
    L33-L58   ObservationEvidence（memory_id / quote / relevance / timestamp + 时区校验）
    L115-L186 compute_trend（recent_days=30 / old_days=90，按证据密度判趋势）
以下三段为逐字节复制（仅去掉源文件里未用到的 import / 保持模块风格）；
文件下半部是本项目的适配层：把 MemoryEntry 流映射成证据、把趋势映射成数值增量。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Trend(str, Enum):
    """Computed trend for an observation based on evidence timestamps.

    Trends indicate how an observation's evidence is distributed over time:
    - STABLE: Evidence spread across time, continues to present
    - STRENGTHENING: More/denser evidence recently than before
    - WEAKENING: Evidence mostly old, sparse recently
    - NEW: All evidence within recent window
    - STALE: No evidence in recent window (may no longer apply)
    """

    STABLE = "stable"
    STRENGTHENING = "strengthening"
    WEAKENING = "weakening"
    NEW = "new"
    STALE = "stale"


class ObservationEvidence(BaseModel):
    """A single piece of evidence supporting an observation.

    Each evidence item must include an exact quote from the source memory
    to ensure observations are grounded and verifiable.
    """

    memory_id: str = Field(description="ID of the memory unit this evidence comes from")
    quote: str = Field(description="Exact quote from the memory supporting the observation")
    relevance: str = Field(default="", description="Brief explanation of how this quote supports the observation")
    timestamp: datetime = Field(description="When the source memory was created")

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_timezone_aware(cls, v: datetime | str | None) -> datetime:
        """Ensure timestamp is always timezone-aware UTC."""
        if v is None:
            return datetime.now(timezone.utc)
        if isinstance(v, str):
            # Parse ISO format string, handling 'Z' suffix
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v
        raise ValueError(f"Invalid timestamp type: {type(v)}")


def compute_trend(
    evidence: list[ObservationEvidence],
    now: datetime | None = None,
    recent_days: int = 30,
    old_days: int = 90,
) -> Trend:
    """Compute the trend for an observation based on evidence timestamps.

    The trend indicates how the evidence is distributed over time:
    - STABLE: Evidence spread across time, continues to present
    - STRENGTHENING: More evidence recently than historically
    - WEAKENING: Evidence mostly old, sparse recently
    - NEW: All evidence is recent (within recent_days)
    - STALE: No evidence in recent window

    Args:
        evidence: List of evidence items with timestamps
        now: Reference time for calculations (defaults to current UTC time)
        recent_days: Number of days to consider "recent" (default 30)
        old_days: Number of days to consider "old" (default 90)

    Returns:
        Computed Trend enum value
    """
    if now is None:
        now = datetime.now(timezone.utc)

    # Ensure now is timezone-aware
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    if not evidence:
        return Trend.STALE

    recent_cutoff = now - timedelta(days=recent_days)
    old_cutoff = now - timedelta(days=old_days)

    # Normalize timestamps to UTC for comparison
    def normalize_ts(ts: datetime) -> datetime:
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts

    recent = [e for e in evidence if normalize_ts(e.timestamp) > recent_cutoff]
    old = [e for e in evidence if normalize_ts(e.timestamp) < old_cutoff]
    middle = [e for e in evidence if old_cutoff <= normalize_ts(e.timestamp) <= recent_cutoff]

    # No recent evidence = stale
    if not recent:
        return Trend.STALE

    # All evidence is recent = new
    if not old and not middle:
        return Trend.NEW

    # Compare density (evidence per day)
    recent_density = len(recent) / recent_days if recent_days > 0 else 0
    older_period = old_days - recent_days
    older_density = (len(old) + len(middle)) / older_period if older_period > 0 else 0

    # Avoid division by zero
    if older_density == 0:
        return Trend.NEW

    ratio = recent_density / older_density

    if ratio > 1.5:
        return Trend.STRENGTHENING
    elif ratio < 0.5:
        return Trend.WEAKENING
    else:
        return Trend.STABLE


# ============================================================================
# 本项目适配层（非 hindsight 原文）：MemoryEntry 流 → 证据 → 数值增量。
# 只被 USE_MATURE_TEMPORAL 开启时的 state.compute_state 调用；关闭时零调用。
# ============================================================================

TREND_SCORE_DELTA = {
    Trend.STRENGTHENING: 2.0,
    Trend.WEAKENING: -2.0,
    Trend.NEW: 1.0,
    Trend.STALE: -1.0,
    Trend.STABLE: 0.0,
}


def evidence_from_memories(mems, now: datetime | None = None) -> list[ObservationEvidence]:
    """把记忆流映射成 hindsight 证据：quote 取摘要/正文，timestamp 取记忆写入时刻。"""
    out: list[ObservationEvidence] = []
    for m in mems:
        ts = getattr(m, "ts", None)
        if ts is None:
            continue
        quote = ((getattr(m, "summary", None) or getattr(m, "content", None) or "") or "").strip()[:200]
        out.append(ObservationEvidence(
            memory_id=str(getattr(m, "id", "") or ""),
            quote=quote,
            timestamp=ts,
        ))
    return out


def trend_from_memories(mems, now: datetime | None = None) -> Trend:
    """对记忆流算趋势；异常由调用方吞掉并回退基线聚合。"""
    return compute_trend(evidence_from_memories(mems, now=now), now=now)
