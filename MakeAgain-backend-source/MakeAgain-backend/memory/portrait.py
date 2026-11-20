"""画像增量合并:聊天积累的新事实定期合并进画像,让画像随对话演进。

触发策略(零新表、零迁移):
- 以最新 user 画像的 updated_at 为水位线,统计其后的记忆条数;
  达到 portrait_merge_every 才考虑合并。
- 只收集 confidence >= portrait_merge_min_confidence 且 kind∈{user,object} 的 fact;
  无可用 fact 或 mock 客户端时跳过(不触发 LLM、不破坏离线确定性)。
- 用一次 fast 模型调用把累积 fact 与当前画像合并:仅填「暂未提及」、补稳定事实,
  不覆盖已确认的值、不编造。

【WP-C 内核替换】开关 USE_ENTITY_PORTRAIT 默认关闭：
- 关闭时 merge_if_due 逐字节走 _baseline_merge_if_due（基线原文），可被
  `git show cde75f8:MakeAgain-backend/memory/portrait.py` 做 oracle 比对；
- 开启时改用 evidence-grounded Observation（hindsight observations.py 单文件复制）
  + graphiti extract_attributes() HARD RULES（抗幻觉），并接进实体图双向链接。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum

from pydantic import BaseModel, Field, computed_field, field_validator
from sqlalchemy import select

from config import get_settings
from memory import store
from memory.models import Portrait

_MERGE_SYSTEM = (
    "你是「重逢」的记忆整理助手。根据现有画像与聊天中累积的稳定事实,输出合并后的画像 JSON。"
)

# 视为「未提及」的占位,可被新事实填充
_PLACEHOLDERS = {"暂未提及", "", None}


# ==================== 基线原文（开关关闭时的 oracle 路径，逐字未改） ====================
def _portrait_text(d: dict) -> str:
    if not d:
        return "(空)"
    return "\n".join(f"- {k}: {v}" for k, v in d.items() if v)


def _user_updated_at(db, user_id: str) -> datetime | None:
    row = db.execute(
        select(Portrait)
        .where(Portrait.user_id == user_id, Portrait.kind == "user")
        .order_by(Portrait.updated_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    return row.updated_at if row else None


def _collect_facts(db, user_id: str, since: datetime, limit: int = 40) -> list[str]:
    """取 since 之后、带高置信 user/object 事实的记忆,返回去重后的 fact 文本列表。"""
    s = get_settings()
    min_conf = s.portrait_merge_min_confidence
    seen: set[str] = set()
    out: list[str] = []
    for m in store.list_memories(db, user_id, since=since):
        for f in (m.facts or []):
            if not isinstance(f, dict):
                continue
            if f.get("kind") not in ("user", "object"):
                continue
            conf = f.get("confidence")
            if conf is not None:
                try:
                    if float(conf) < min_conf:
                        continue
                except (TypeError, ValueError):
                    # 不可解析的置信度按 0 处理,不进入画像合并。
                    continue
            text = str(f.get("fact") or "").strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
                if len(out) >= limit:
                    return out
    return out


def _merge_prompt(current: dict, facts: list[str]) -> str:
    return (
        f"现有画像:\n{_portrait_text(current)}\n\n"
        "新累积的稳定事实(均为用户 / 思念对象相关信息):\n"
        + "\n".join(f"- {f}" for f in facts)
        + "\n\n请输出合并后的画像 JSON:保留现有画像里已确认的信息,把「暂未提及」的字段用新事实填上,"
          "补入新的稳定事实;不要删除或改写已确认的值,不要编造。只输出 JSON 对象本身。"
    )


def _baseline_merge_if_due(db, user_id: str, client) -> bool:
    """cde75f8 基线 merge_if_due 原文（开关关闭时逐字节执行此函数）。"""
    s = get_settings()
    if getattr(client, "mock", False):
        return False
    updated_at = _user_updated_at(db, user_id)
    if updated_at is None:
        return False  # 尚无画像(未做访谈),不自动建画像
    recent = store.list_memories(db, user_id, since=updated_at)
    if len(recent) < s.portrait_merge_every:
        return False
    facts = _collect_facts(db, user_id, since=updated_at)
    if not facts:
        return False

    current = store.get_portrait(db, user_id, "user")
    parsed, _ = client.chat_json(
        [{"role": "system", "content": _MERGE_SYSTEM},
         {"role": "user", "content": _merge_prompt(current, facts)}],
        temperature=0.2, model=client.settings.llm_fast_model,
    )
    if not parsed or not isinstance(parsed, dict):
        return False

    merged = dict(current or {})
    for k, v in parsed.items():
        if not v:
            continue
        # 已有非占位值且 LLM 想改写 → 不覆盖,保护已确认信息
        if k in merged and merged[k] not in _PLACEHOLDERS and merged[k] != v:
            continue
        merged[k] = v
    store.upsert_portrait(db, user_id, "user", merged, status="draft")
    return True


# ==================== hindsight observations.py 单文件复制（evidence-grounded） ====================
# 来源：hindsight observations.py（gh api 实读，见 work/r4_hindsight_observations_raw.txt）
# 只取 Observation / ObservationEvidence / Trend / compute_trend 四个结构，零服务依赖。
class Trend(str, Enum):
    STABLE = "stable"
    STRENGTHENING = "strengthening"
    WEAKENING = "weakening"
    NEW = "new"
    STALE = "stale"


class ObservationEvidence(BaseModel):
    """单条证据：必须含原消息的逐字 quote，保证画像可溯源。"""

    memory_id: str = Field(description="ID of the memory unit this evidence comes from")
    quote: str = Field(description="Exact quote from the memory supporting the observation")
    relevance: str = Field(default="", description="Brief explanation of how this quote supports the observation")
    timestamp: datetime = Field(description="When the source memory was created")

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_timezone_aware(cls, v: datetime | str | None) -> datetime:
        if v is None:
            return datetime.now(timezone.utc)
        if isinstance(v, str):
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v
        raise ValueError(f"Invalid timestamp type: {type(v)}")


class Observation(BaseModel):
    title: str = Field(description="Short summary title for the observation (5-10 words)")
    content: str = Field(description="The observation content - detailed explanation of what we believe to be true")
    evidence: list[ObservationEvidence] = Field(default_factory=list, description="Supporting evidence with quotes")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="When this observation was first created"
    )

    @field_validator("created_at", mode="before")
    @classmethod
    def ensure_created_at_timezone_aware(cls, v: datetime | str | None) -> datetime:
        if v is None:
            return datetime.now(timezone.utc)
        if isinstance(v, str):
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v
        raise ValueError(f"Invalid created_at type: {type(v)}")

    @computed_field
    @property
    def trend(self) -> Trend:
        return compute_trend(self.evidence)

    @computed_field
    @property
    def evidence_span(self) -> dict[str, str | None]:
        if not self.evidence:
            return {"from": None, "to": None}
        timestamps = [e.timestamp for e in self.evidence]
        return {"from": min(timestamps).isoformat(), "to": max(timestamps).isoformat()}

    @computed_field
    @property
    def evidence_count(self) -> int:
        return len(self.evidence)


def compute_trend(evidence: list[ObservationEvidence], now: datetime | None = None,
                  recent_days: int = 30, old_days: int = 90) -> Trend:
    """从证据时间分布算法计算趋势（无 LLM、无幻觉）。"""
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if not evidence:
        return Trend.STALE
    recent_cutoff = now - timedelta(days=recent_days)
    old_cutoff = now - timedelta(days=old_days)

    def normalize_ts(ts: datetime) -> datetime:
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)

    recent = [e for e in evidence if normalize_ts(e.timestamp) > recent_cutoff]
    old = [e for e in evidence if normalize_ts(e.timestamp) < old_cutoff]
    middle = [e for e in evidence if old_cutoff <= normalize_ts(e.timestamp) <= recent_cutoff]
    if not recent:
        return Trend.STALE
    if not old and not middle:
        return Trend.NEW
    recent_density = len(recent) / recent_days if recent_days > 0 else 0
    older_period = old_days - recent_days
    older_density = (len(old) + len(middle)) / older_period if older_period > 0 else 0
    if older_density == 0:
        return Trend.NEW
    ratio = recent_density / older_density
    if ratio > 1.5:
        return Trend.STRENGTHENING
    if ratio < 0.5:
        return Trend.WEAKENING
    return Trend.STABLE


def build_observation(title: str, content: str, evidence: list[dict],
                      created_at: datetime | None = None) -> dict:
    """构造 evidence-grounded 画像条目（返回 JSON 可序列化 dict）。"""
    obs = Observation(
        title=title,
        content=content,
        evidence=[ObservationEvidence(**e) for e in evidence],
        **({"created_at": created_at} if created_at is not None else {}),
    )
    return obs.model_dump(mode="json")


# ==================== graphiti extract_attributes HARD RULES（抗幻觉） ====================
# 来源：graphiti prompts extract_attributes():384-468（gh api 实读，见 work/r4_extract_edges_raw.txt L199-257）
_GRAPHITI_HARD_RULES = """\
HARD RULES — violating any of these is a failure:

1. Each attribute value MUST be one of:
   (a) a clean value copied or directly normalized from the FACT,
   (b) the existing value already in EXISTING ATTRIBUTES (preserved unchanged), or
   (c) null / omitted, when neither (a) nor (b) applies.

2. NEVER write reasoning, justification, or commentary into any field. Specifically:
   - NEVER include parenthetical explanations like "(implied by ...)", "(Context: ...)",
     "(not explicitly stated ...)", "(based on ...)".
   - NEVER include first-person or deliberative phrases like "I should...", "However...",
     "Sticking to...", "Since no...", "the instruction is to...", "must be kept...".
   - NEVER list alternatives or candidates inside one field ("X, or Y, or maybe Z").
   - NEVER explain why a value is null. If unknown, set the field to null and stop.

3. Each attribute schema description tells you the FORMAT a real value should take. The
   description text is NEVER itself a value. NEVER copy schema description text into the field.

4. The literal strings "null", "N/A", "Not specified", "unknown", "none", "not provided",
   or any sentence describing absence are NOT valid values. If no value is supported by
   the FACT, set the field to null (or omit it) — do not write a sentence.

5. Each attribute value must be a short, well-formed instance of the type the field
   describes. If you cannot produce a clean value of that type from the FACT, the field is null.

6. Preserve existing attribute values unless the FACT explicitly provides a new value.

不得编造：任何在 FACT 与 EXISTING ATTRIBUTES 中都找不到依据的值，必须置 null 或省略。
"""


def _merge_prompt_with_hard_rules(current: dict, facts: list[str]) -> str:
    """基线 _merge_prompt + graphiti HARD RULES（仅开关开启时走）。"""
    return _merge_prompt(current, facts) + "\n\n" + _GRAPHITI_HARD_RULES


# ==================== 开关与 R2 接口 ====================
_ENABLED_VALUES = {"1", "true", "yes", "on"}


def entity_portrait_enabled() -> bool:
    """USE_ENTITY_PORTRAIT 开关：os.getenv，默认关闭。"""
    import os
    return (os.getenv("USE_ENTITY_PORTRAIT") or "").strip().lower() in _ENABLED_VALUES


def link_facts(facts: list[dict], **kwargs) -> list[dict]:
    """R2 遗留接口：extract_turn 产出的 facts 里 linked_memory_ids 恒为空，
    lead 在 extract 路径调用本接口即可按实体图回填真实关联 id。
    实现在 memory.entity_graph（sqlite-vec 实体图），此处做稳定 re-export。"""
    from memory.entity_graph import link_facts as _link_facts
    return _link_facts(facts, **kwargs)


def _collect_fact_records(db, user_id: str, since: datetime, limit: int = 40) -> list[dict]:
    """带证据的 fact 收集：每条带原消息 id、逐字 quote、时间戳（供 Observation 溯源）。"""
    s = get_settings()
    min_conf = s.portrait_merge_min_confidence
    seen: set[str] = set()
    out: list[dict] = []
    for m in store.list_memories(db, user_id, since=since):
        content = str(m.content or "")
        ts = m.ts.replace(tzinfo=timezone.utc) if m.ts is not None and m.ts.tzinfo is None else m.ts
        for f in (m.facts or []):
            if not isinstance(f, dict):
                continue
            if f.get("kind") not in ("user", "object"):
                continue
            conf = f.get("confidence")
            if conf is not None:
                try:
                    if float(conf) < min_conf:
                        continue
                except (TypeError, ValueError):
                    continue
            text = str(f.get("fact") or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            idx = content.find(text)
            quote = content[max(0, idx - 20): idx + len(text) + 20] if idx >= 0 else content[:200]
            out.append({
                "fact": text,
                "memory_id": str(m.id),
                "quote": quote,
                "timestamp": ts,
                "linked_memory_ids": list(f.get("linked_memory_ids") or []),
            })
            if len(out) >= limit:
                return out
    return out


def _is_supported(value, fact_texts: list[str], current: dict) -> bool:
    """graphiti 规则 (a)/(b)/(c) 的确定性执行：只接受 fact 里逐字出现的、
    或画像里已有的值；推理/编造/null 文案一律拒绝。"""
    if value is None or value == "":
        return True
    s = str(value).strip()
    if not s or s.lower() in {"null", "n/a", "not specified", "unknown", "none", "not provided"}:
        return False
    for fact in fact_texts:
        if s in fact or (len(fact) >= 2 and fact in s):
            return True
    return any(s == str(v) for v in (current or {}).values())


def _entity_merge_if_due(db, user_id: str, client) -> bool:
    """开关开启路径：evidence-grounded 画像合并 + 抗幻觉 + 实体图回填。"""
    s = get_settings()
    if getattr(client, "mock", False):
        return False
    updated_at = _user_updated_at(db, user_id)
    if updated_at is None:
        return False
    recent = store.list_memories(db, user_id, since=updated_at)
    if len(recent) < s.portrait_merge_every:
        return False
    records = _collect_fact_records(db, user_id, since=updated_at)
    if not records:
        return False

    current = store.get_portrait(db, user_id, "user")
    facts = [r["fact"] for r in records]
    parsed, _ = client.chat_json(
        [{"role": "system", "content": _MERGE_SYSTEM},
         {"role": "user", "content": _merge_prompt_with_hard_rules(current, facts)}],
        temperature=0.2, model=client.settings.llm_fast_model,
    )
    if not parsed or not isinstance(parsed, dict):
        return False

    merged = dict(current or {})
    for k, v in parsed.items():
        if not v:
            continue
        if not _is_supported(v, facts, current):
            continue  # graphiti HARD RULES：无依据即不写入（抗幻觉）
        if k in merged and merged[k] not in _PLACEHOLDERS and merged[k] != v:
            continue
        merged[k] = v

    # hindsight Observation：画像条目带 evidence quote + timestamp + 计算出的 trend
    observations = []
    for r in records:
        observations.append(build_observation(
            title=r["fact"][:12],
            content=r["fact"],
            evidence=[{
                "memory_id": r["memory_id"],
                "quote": r["quote"],
                "relevance": "逐字证据，可溯源回原消息",
                "timestamp": r["timestamp"],
            }],
        ))
    merged["observations"] = observations

    # 接实体图：把 fact 的 linked_memory_ids 从恒空回填为真实关联
    try:
        from memory.entity_graph import link_facts as _link
        for idx, r in enumerate(records):
            linked = _link([r], memory_id=r["memory_id"], user_id=user_id)
            if linked and linked[0].get("linked_memory_ids"):
                observations[idx]["linked_memory_ids"] = linked[0]["linked_memory_ids"]
        merged["observations"] = observations
    except Exception:
        pass  # 实体图失败不影响画像合并

    store.upsert_portrait(db, user_id, "user", merged, status="draft")
    return True


def merge_if_due(db, user_id: str, client) -> bool:
    """达到阈值则把新事实合并进 user 画像;返回是否执行了合并。

    开关 USE_ENTITY_PORTRAIT 默认关闭：关闭时逐字节走基线 _baseline_merge_if_due；
    开启时走 evidence-grounded + 抗幻觉 + 实体双向链接的 _entity_merge_if_due。
    """
    if not entity_portrait_enabled():
        return _baseline_merge_if_due(db, user_id, client)
    return _entity_merge_if_due(db, user_id, client)
