"""UC-2 记忆纠错闭环（用户说"记错了"）——产品地基，默认可用。

用户提交 correction(memory_id + 正确内容 + 理由)，本模块：
  1. 校验归属（only owner can correct；越权不泄漏存在性，按 404 拒绝）；
  2. 调既有五层机制 five_layer.supersede_content()（G1 已有）：旧事实 invalid_at +
     superseded_by 指向新版本，current 只返新版，as_of 历史时点仍见旧版，trace 两版都在；
  3. 同步业务库 MemoryEntry 内容（聊天上下文不再注入旧事实）与可选双路索引；
  4. 提供沿 supersede 指针的完整变更链查询（memory_history）。

不修改任何既有业务主链路文件；只新增本文件与 api/memory_correct.py。
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from memory import store as mstore
from memory.models import MemoryEntry

logger = logging.getLogger(__name__)

REASON_PREFIX = "纠错理由："
_REASON_RE = re.compile(re.escape(REASON_PREFIX) + r"([^\r\n]+)")
_MAX_CONTENT = 4000
_MAX_REASON = 500


class CorrectionError(Exception):
    """纠错用例的可映射错误（API 层转成 HTTP）。"""

    def __init__(self, code: str, message: str, status: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class MemoryNotFound(CorrectionError):
    """不存在或不属于当前用户。故意用 404，避免越权探测记忆是否存在的侧信道。"""

    def __init__(self):
        super().__init__("MEMORY_NOT_FOUND", "记忆不存在", 404)


def _now() -> datetime:
    """可注入的时钟；生产取业务库同一 UTC 口径，测试可冻结以断言双时间。"""
    return mstore.utcnow()


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def current_user_id(db: Session) -> str | None:
    """请求边界 SecureRoute 写入的认证用户；不信任请求体里的 user_id。"""
    return db.info.get("user_id")


def get_owned_entry(db: Session, memory_id: str, user_id: str | None) -> MemoryEntry:
    """取当前用户拥有的记忆；越权/不存在统一按 404 拒绝（宁严勿松）。"""
    entry = db.get(MemoryEntry, str(memory_id))
    if entry is None or str(entry.user_id) != str(user_id or ""):
        raise MemoryNotFound()
    return entry


def _scrub(text: Any, old: str, new: str) -> Any:
    """把旧事实文本从摘要/facts 中替换掉，避免它仍被聊天上下文召回。"""
    if not isinstance(text, str) or not old or old not in text:
        return text
    return text.replace(old, new)


def correct_memory(db: Session, memory_id: str, corrected_content: str, reason: str = "",
                   *, user_id: str | None = None) -> dict:
    """纠错主路径：业务库内容更新 + 五层库 supersede 留痕。

    返回旧记录（invalid_at / superseded_by）+ 新内容 + 理由，供 API 与验收取证。
    """
    uid = user_id if user_id is not None else current_user_id(db)
    entry = get_owned_entry(db, memory_id, uid)

    content = str(corrected_content or "").strip()
    if not content:
        raise CorrectionError("INVALID_CORRECTION", "纠错内容不能为空")
    if len(content) > _MAX_CONTENT:
        raise CorrectionError("CONTENT_TOO_LONG", f"纠错内容超过 {_MAX_CONTENT} 字")
    why = str(reason or "").strip()
    if len(why) > _MAX_REASON:
        raise CorrectionError("REASON_TOO_LONG", f"纠错理由超过 {_MAX_REASON} 字")

    old_content = entry.content or ""
    old_id = str(entry.id)
    moment = _now()

    adapter = mstore._get_five_layer_adapter()
    # 若记忆早于五层库启用（或五层写入曾失败），先按原内容补写前驱，
    # 保证 supersede 链完整——不制造没有前驱的孤儿后继。
    if not adapter.record(old_id):
        adapter.add(old_id, old_content or content, user_id=str(entry.user_id),
                    type=str(entry.type), provenance=[old_content or content],
                    valid_from=entry.ts)

    provenance = ([f"{REASON_PREFIX}{why}"] if why else []) + [content]
    old = adapter.supersede_content(
        old_id, content, user_id=str(entry.user_id), type=str(entry.type),
        subject=entry.summary or None, provenance=provenance, valid_from=moment,
    )
    successor = old.get("superseded_by") or old_id

    # 业务库同步：聊天上下文（recall 读 content/summary/facts）不再含旧事实。
    entry.content = content
    if old_content:
        entry.summary = _scrub(entry.summary, old_content, content)
        if isinstance(entry.facts, list):
            entry.facts = [_scrub(f, old_content, content) for f in entry.facts]
    db.commit()  # managed session 下只 flush；最终由请求边界统一 commit
    db.refresh(entry)

    # 双路召回开启时幂等重建该条索引，避免向量/关键词索引仍指向旧事实。
    if os.getenv("USE_DUAL_RECALL", "").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            from memory.recall import index_memory
            index_memory(entry, db=db)
        except Exception:
            logger.warning("correction dual-reindex failed; kept business update", exc_info=True)

    # 溯源审计事件（可选开关，失败不影响纠错本身；理由同时写在五层后继 provenance 里）。
    mstore.add_provenance_event(
        old_id, old_content, content, "CORRECTION",
        created_at=_iso(entry.ts), updated_at=_iso(moment),
        actor_id=uid, role=why or None,
    )

    return {
        "ok": True,
        "memory_id": old_id,
        "old_content": old_content,
        "content": content,
        "reason": why,
        "superseded_by": successor,
        "old_invalid_at": old.get("invalid_at"),
        "old_valid_from": old.get("valid_from"),
        "version": 2 if successor != old_id else 1,
    }


def _reason_from_trace(text: str) -> str:
    match = _REASON_RE.search(text or "")
    return match.group(1).strip() if match else ""


def memory_history(db: Session, memory_id: str, *, user_id: str | None = None) -> dict:
    """沿 supersede 指针返回完整变更链：每版的正文、有效期、后继、理由与 trace 在否。"""
    uid = user_id if user_id is not None else current_user_id(db)
    entry = get_owned_entry(db, memory_id, uid)
    adapter = mstore._get_five_layer_adapter()

    versions: list[dict] = []
    seen: set[str] = set()
    name = str(entry.id)
    while name and name not in seen:
        seen.add(name)
        rec = adapter.record(name)
        if not rec:
            break
        try:
            raw = adapter._store.find(name)
            body = getattr(raw, "body", "") or rec["abstract"]
        except Exception:
            body = rec["abstract"]
        trace = adapter.trace(name)
        versions.append({
            "name": name,
            "content": body,
            "valid_from": rec["valid_from"],
            "invalid_at": rec["invalid_at"],
            "superseded_by": rec["superseded_by"],
            "status": rec["status"],
            "reason": _reason_from_trace(trace.get("text", "")),
            "provenance_present": bool(rec["provenance"]),
            "trace_present": bool(trace.get("text")),
        })
        name = rec["superseded_by"] or ""

    return {
        "ok": True,
        "memory_id": str(entry.id),
        "current_content": entry.content,
        "chain": [v["name"] for v in versions],
        "count": len(versions),
        "versions": versions,
    }
