"""UC-4 · 跨 Journey 隔离（用户控制权地基，默认拒绝，宁严勿松）。

设计红线
--------
1. **先过滤后召回**：先用 four-dimension scope（user_id + companion_id + consent +
   lifecycle）在业务库里算出「允许进入上下文的 memory_id 集合」，再在这批行上排序。
   绝不先 list_memories / vec0 全量召回再丢弃越权项——那已经把隐私数据读进内存了。
2. **归档 Journey 只读**：companion.archived_at 置位（或 ending_stage=complete）的
   记忆不进入新 Journey 的检索；对其写入一律 ScopeDenied(ARCHIVED_READ_ONLY)。
3. **不继承**：新 Journey 的默认 scope 只含自己的 companion_id；上一段的对象画像 /
   场景记忆 / 记忆流都不继承。只有显式 CrossJourneyGrant 才能放宽 companion 维度。
4. **所有记忆读写经过 assert_scope()**：读取、写入、画像查询都走本模块守卫；调用方
   只需把 memory_id + 当前 user_id/companion_id 交给 assert_scope，守卫负责拒绝越权。
   注意：本模块是库层守卫，路由层需把 db.info['companion_id'] 传进来才形成端到端闭环
   （api/security.py 已有 companion 归属/归档校验，二者叠加，不重复造轮子）。

显式授权
--------
CrossJourneyGrant 是一份**内存中的、必须显式构造**的授权凭证（frozen dataclass），
由 grant_cross_journey() 校验「两个 companion 都归同一 user」后签发。没有凭证 = 没有跨
Journey；凭证里带 reason 便于审计。路由层若要持久化授权，应自行落审计表后重建该对象
（本工作包不拥有 models.py / 表结构，刻意不改数据库 schema）。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional, Sequence

from sqlalchemy import select

from config import get_settings
from memory import store as mstore
from memory.facts import facts_text
from memory.models import Companion, MemoryEntry, Portrait, UserState


# ---------------------------------------------------------------------------
# 拒绝语义
# ---------------------------------------------------------------------------
class ScopeDenied(PermissionError):
    """scope 守卫拒绝。code 是稳定机器码，message 给人看；不回显他人数据内容。"""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")

    def as_dict(self) -> dict:
        return {"code": self.code, "message": str(self)}


@dataclass(frozen=True)
class CrossJourneyGrant:
    """显式跨 Journey 授权凭证（from 归档旅程 → to 当前旅程）。"""

    user_id: str
    from_companion_id: str
    to_companion_id: str
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "from_companion_id": self.from_companion_id,
            "to_companion_id": self.to_companion_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Scope:
    """一次过滤的结果集合（frozen，可在一次请求里复用且不被下游篡改）。"""

    user_id: str
    companion_id: str
    allowed_companion_ids: frozenset[str]
    allowed_ids: frozenset[str]
    consent: bool
    lifecycle: str                       # active | archived
    grants: tuple[CrossJourneyGrant, ...] = ()

    def as_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "companion_id": self.companion_id,
            "allowed_companion_ids": sorted(self.allowed_companion_ids),
            "allowed_ids": sorted(self.allowed_ids),
            "consent": self.consent,
            "lifecycle": self.lifecycle,
            "grants": [g.as_dict() for g in self.grants],
        }


# ---------------------------------------------------------------------------
# 基础谓词
# ---------------------------------------------------------------------------
def _companion_for(db, user_id: str, companion_id: str) -> Companion:
    """取 companion 并校验归属；不存在/不属于 user 都按同一理由拒绝，不做存在性 Oracle。"""
    companion = db.get(Companion, companion_id)
    if companion is None or companion.user_id != user_id:
        raise ScopeDenied("COMPANION_SCOPE_DENIED", "陪伴对象不存在或不属于此用户")
    return companion


def is_archived(companion: Companion) -> bool:
    """只读档案态判据：archived_at 置位或结束仪式已完成，二者任一即只读。"""
    return companion.archived_at is not None or (companion.ending_stage or "active") == "complete"


def consent_granted(db, user_id: str) -> bool:
    """用户是否已明确接受记忆使用（consent）。默认未接受 = 没有任何记忆在 scope 内。"""
    state = db.get(UserState, user_id)
    return bool(state is not None and state.consent_accepted_at is not None)


def grant_cross_journey(db, user_id: str, from_companion_id: str,
                        to_companion_id: str, reason: str = "") -> CrossJourneyGrant:
    """签发跨 Journey 授权：必须两个 companion 都归同一 user，且不是同一段旅程。

    这是显式的、可审计的产品动作；缺任一条即抛 ScopeDenied(GRANT_SCOPE_DENIED)。
    """
    if not from_companion_id or not to_companion_id:
        raise ScopeDenied("GRANT_SCOPE_DENIED", "授权必须指定来源与目标 Journey")
    if from_companion_id == to_companion_id:
        raise ScopeDenied("GRANT_SCOPE_DENIED", "同一 Journey 不需要跨 Journey 授权")
    try:
        _companion_for(db, user_id, from_companion_id)
        _companion_for(db, user_id, to_companion_id)
    except ScopeDenied as exc:
        # 授权路径统一返回授权专用码，避免把 companion 的存在性差异暴露给调用方。
        raise ScopeDenied("GRANT_SCOPE_DENIED", "授权涉及的 Journey 不存在或不属于此用户") from exc
    return CrossJourneyGrant(user_id, from_companion_id, to_companion_id, reason)


def _valid_grants(user_id: str, companion_id: str,
                  grants: Optional[Iterable[CrossJourneyGrant]]) -> tuple[CrossJourneyGrant, ...]:
    """只接受目标 Journey 匹配、用户匹配的凭证；来源 Journey 由调用方 DB 校验。"""
    if not grants:
        return ()
    out: list[CrossJourneyGrant] = []
    for grant in grants:
        if (not isinstance(grant, CrossJourneyGrant) or grant.user_id != user_id
                or grant.to_companion_id != companion_id):
            raise ScopeDenied("GRANT_SCOPE_DENIED", "跨 Journey 授权与当前用户/旅程不匹配")
        out.append(grant)
    return tuple(out)


# ---------------------------------------------------------------------------
# 第一步：过滤（在召回之前完成）
# ---------------------------------------------------------------------------
def build_scope(db, user_id: str, companion_id: str, *,
                grants: Optional[Iterable[CrossJourneyGrant]] = None) -> Scope:
    """按 user_id + companion_id + consent + lifecycle 计算允许召回的 memory_id 集合。

    - 未 consent：返回空集合（默认拒绝，不读任何记忆体）；
    - 归档 Journey：只放行它自己的记忆（只读档案），不自动放行到任何新 Journey；
    - 活跃 Journey：只放行自己 companion_id 下的记忆，除非带合法 CrossJourneyGrant；
    - 授权来源 Journey 若仍归档，也仅放宽「读」维度，写仍受 assert_scope 的归档只读限制。
    """
    companion = _companion_for(db, user_id, companion_id)
    signed = _valid_grants(user_id, companion_id, grants)
    consent = consent_granted(db, user_id)
    archived = is_archived(companion)
    lifecycle = "archived" if archived else "active"

    allowed_cids = {companion_id}
    if not archived:
        # 只有当前 Journey 活跃时才可能继承旧 Journey；归档 Journey 自身不接受继承。
        for grant in signed:
            source = db.get(Companion, grant.from_companion_id)
            if source is not None and source.user_id == user_id:
                allowed_cids.add(grant.from_companion_id)

    if not consent:
        return Scope(user_id, companion_id, frozenset(allowed_cids), frozenset(),
                     False, lifecycle, signed)

    # 关键：SQL 层面就把 user_id / companion_id 限定死；unscoped=True 绕过 memory.scope
    # 的 ORM 层面 companion 过滤——scope 在这里由本模块显式决定，避免两层规则互相打架。
    rows = db.execute(
        select(MemoryEntry.id).where(
            MemoryEntry.user_id == user_id,
            MemoryEntry.companion_id.in_(allowed_cids),
        ).execution_options(unscoped=True)
    ).scalars().all()
    return Scope(user_id, companion_id, frozenset(allowed_cids), frozenset(str(r) for r in rows),
                 True, lifecycle, signed)


def _get_memory_unscoped(db, memory_id: str) -> MemoryEntry | None:
    """按 id 点查记忆（不经过 ORM companion 过滤），供守卫判定真实归属。"""
    return db.execute(
        select(MemoryEntry).where(MemoryEntry.id == memory_id).execution_options(unscoped=True)
    ).scalars().first()


def _confirm_in_scope(scope: Scope, rows) -> list:
    """集合级入口的二次复核（纯内存，不读库）：与 assert_scope 同一套 user/companion 判据。

    SQL 已按 scope 过滤；这里再核一次，防止 SQL 变更/ORM 过滤失效导致越权泄漏。
    """
    out = []
    for row in rows:
        if (getattr(row, "user_id", None) == scope.user_id
                and str(getattr(row, "companion_id", None)) in scope.allowed_companion_ids):
            out.append(row)
    return out


# ---------------------------------------------------------------------------
# 第二步：守卫（所有记忆读写都过这里）
# ---------------------------------------------------------------------------
def assert_scope(db, user_id: str, companion_id: str, memory_id: str, *,
                 grants: Optional[Iterable[CrossJourneyGrant]] = None,
                 operation: str = "read") -> MemoryEntry:
    """守卫：确认 memory_id 属于 user_id 且落在 companion_id 的合法 scope 内。

    - operation='read'：允许跨 Journey 授权；归档 Journey 自身可读（只读档案）。
    - operation='write'：归档 Journey 一律拒绝；跨 Journey 授权只放宽读，写只能在目标 Journey。
    越权一律 ScopeDenied，不返回任何越权数据。
    """
    if operation not in ("read", "write"):
        raise ScopeDenied("SCOPE_DENIED", "未知操作类型")
    scope = build_scope(db, user_id, companion_id, grants=grants)
    if operation == "write":
        if scope.lifecycle == "archived":
            raise ScopeDenied("ARCHIVED_READ_ONLY", "归档 Journey 只读，不能写入记忆")
        if not scope.consent:
            raise ScopeDenied("CONSENT_REQUIRED", "用户未授权记忆使用，不能写入")

    entry = _get_memory_unscoped(db, str(memory_id))
    if entry is None:
        raise ScopeDenied("SCOPE_DENIED", "记忆不存在或不在当前 scope")
    if entry.user_id != user_id or str(entry.companion_id) not in scope.allowed_companion_ids:
        raise ScopeDenied("SCOPE_DENIED", "记忆不属于当前用户/Journey 的合法 scope")
    if operation == "write" and str(entry.companion_id) != companion_id:
        raise ScopeDenied("SCOPE_DENIED", "跨 Journey 授权只允许读取，不允许改写旧记忆")
    return entry


# ---------------------------------------------------------------------------
# 受限读写入口（不触碰 store.py / models.py / 路由）
# ---------------------------------------------------------------------------
def scoped_get_memory(db, user_id: str, companion_id: str, memory_id: str, *,
                      grants: Optional[Iterable[CrossJourneyGrant]] = None) -> MemoryEntry:
    return assert_scope(db, user_id, companion_id, memory_id, grants=grants, operation="read")


def scoped_list_memories(db, user_id: str, companion_id: str, *,
                         grants: Optional[Iterable[CrossJourneyGrant]] = None,
                         limit: Optional[int] = None) -> list[MemoryEntry]:
    """只列 scope 内的记忆；不调用 store.list_memories（那会全量读用户记忆）。"""
    scope = build_scope(db, user_id, companion_id, grants=grants)
    if not scope.allowed_ids:
        return []
    q = (select(MemoryEntry)
         .where(MemoryEntry.id.in_(scope.allowed_ids))
         .order_by(MemoryEntry.ts.desc())
         .execution_options(unscoped=True))
    if limit is not None:
        q = q.limit(limit)
    return _confirm_in_scope(scope, db.execute(q).scalars().all())


def scoped_add_memory(db, user_id: str, companion_id: str, *, content: str,
                      type: str = "chat", summary: str | None = None,
                      facts: list | None = None, emotion: dict | None = None,
                      importance: float = 5.0, place_tag: str | None = None,
                      time_tag: str | None = None, cbt: dict | None = None) -> MemoryEntry:
    """写入前先过守卫：consent 必须存在、Journey 必须活跃、companion 必须归 user。"""
    companion = _companion_for(db, user_id, companion_id)
    if is_archived(companion):
        raise ScopeDenied("ARCHIVED_READ_ONLY", "归档 Journey 只读，不能新增记忆")
    if not consent_granted(db, user_id):
        raise ScopeDenied("CONSENT_REQUIRED", "用户未授权记忆使用，不能新增记忆")
    entry = mstore.add_memory(db, user_id, type=type, content=content, summary=summary,
                              facts=facts, emotion=emotion, importance=importance,
                              place_tag=place_tag, time_tag=time_tag, cbt=cbt)
    # store.add_memory 不写 companion_id（依赖 session scope 注入）；这里显式落归属，
    # 否则新 Journey 的记忆会变成无归属行，后续隔离无从判断。
    entry.companion_id = companion_id
    db.commit()
    db.refresh(entry)
    return entry


def scoped_portraits(db, user_id: str, companion_id: str, *,
                     grants: Optional[Iterable[CrossJourneyGrant]] = None) -> list[Portrait]:
    """画像（对象画像/场景记忆）同样按 Journey 隔离：默认不继承上一段的画像行。"""
    scope = build_scope(db, user_id, companion_id, grants=grants)
    if not scope.consent:
        return []
    return _confirm_in_scope(scope, db.execute(
        select(Portrait).where(
            Portrait.user_id == user_id,
            Portrait.companion_id.in_(scope.allowed_companion_ids),
        ).order_by(Portrait.kind, Portrait.version.desc()).execution_options(unscoped=True)
    ).scalars().all())


# ---------------------------------------------------------------------------
# 第二步：在已过滤集合上打分（与基线 recall 同一套公式，不读未过滤库）
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[一-鿿]|[a-zA-Z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text or ""))


def _relevance(query_tokens: set[str], entry_tokens: set[str]) -> float:
    if not query_tokens or not entry_tokens:
        return 0.0
    inter = query_tokens & entry_tokens
    return len(inter) / math.sqrt(len(query_tokens) * len(entry_tokens)) if inter else 0.0


def _overlap_fraction(entry_tokens: set[str], recent_tokens: set[str]) -> float:
    return len(entry_tokens & recent_tokens) / len(entry_tokens) if entry_tokens else 0.0


def score_scoped_entries(entries: Sequence[MemoryEntry], query: str, *,
                         top_k: int | None = None, recent_text: str | None = None,
                         now: Optional[datetime] = None) -> list[dict]:
    """与 memory.recall._recall_legacy 一致的 recency/importance/relevance 混合打分。

    输入只允许是 build_scope() 过滤后的行；这里不查库、不做任何额外加载。
    """
    s = get_settings()
    top_k = top_k or s.recall_top_k
    moment = now or datetime.now(timezone.utc).replace(tzinfo=None)
    q_tokens = _tokens(query)
    recent_tokens = _tokens(recent_text) if recent_text else set()
    scored: list[dict] = []
    for entry in entries:
        entry_tokens = _tokens(f"{facts_text(entry.facts)} {entry.summary or ''} {entry.content}")
        if recent_tokens and _overlap_fraction(entry_tokens, recent_tokens) >= s.recall_recent_overlap_threshold:
            continue
        days = max(0.0, (moment - entry.ts).total_seconds() / 86400.0)
        recency = math.exp(-days / 30.0)
        importance = (entry.importance or 5.0) / 10.0
        relevance = _relevance(q_tokens, entry_tokens)
        score = s.recall_alpha * recency + s.recall_beta * importance + s.recall_gamma * relevance
        scored.append({"entry": entry, "score": score, "recency": recency, "relevance": relevance})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def scoped_recall(db, user_id: str, companion_id: str, query: str, *,
                  top_k: int | None = None, recent_text: str | None = None,
                  grants: Optional[Iterable[CrossJourneyGrant]] = None,
                  recall_fn: Optional[Callable[[Sequence[MemoryEntry], str], list[dict]]] = None) -> list[dict]:
    """跨 Journey 安全的召回入口：**先 build_scope 过滤，再打分**（顺序不可颠倒）。

    recall_fn 是可注入的打分层（默认 score_scoped_entries），签名 (entries, query) -> hits；
    lead 若要把双路召回接进来，必须让它只吃已过滤的 entries，不能自行全量加载。
    """
    scope = build_scope(db, user_id, companion_id, grants=grants)
    if not scope.allowed_ids:
        return []
    entries = _confirm_in_scope(scope, db.execute(
        select(MemoryEntry).where(MemoryEntry.id.in_(scope.allowed_ids))
        .execution_options(unscoped=True)
    ).scalars().all())
    if not entries:
        return []
    scorer = recall_fn or (lambda batch, q: score_scoped_entries(batch, q, top_k=top_k,
                                                                   recent_text=recent_text))
    return scorer(entries, query)
