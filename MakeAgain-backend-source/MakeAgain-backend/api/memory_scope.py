"""UC-4 隔离守卫接主链路：scoped 记忆只读端点（由 lead 在 main.py 注册，本文件不改路由注册）。

USE_SCOPE_GUARD 默认开：所有端点走 memory/isolation.py 的 scoped_* 守卫，
「先按 user_id + companion_id + consent + lifecycle 过滤，再召回」。越权、不存在、
未授权一律返回同一个 403 SCOPE_DENIED / CONSENT_REQUIRED，错误体不回显记忆内容、
不区分"记忆不存在"与"记忆存在但不属于你"，杜绝存在性 Oracle。

owner 双层校验：
  1) SecureRoute 验 token、body/query 的 user_id、companion 归属/归档；
  2) 端点再按 db.info 的 user_id/companion_id 调 isolation.build_scope 复核，
     companion.user_id != user 直接 ScopeDenied（见 _scope_or_deny）。

开关关（USE_SCOPE_GUARD=0）时逐字节回退到 UC-4 之前的基线库函数
（memory.store.list_memories / memory.recall.recall / db.get），守卫代码一段不执行。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.security import SecureRoute, fail
from memory.db import get_session
from memory.jobs import scope_guard_enabled

router = APIRouter(prefix="/api/memory", route_class=SecureRoute)


def _uid(db) -> str:
    return str(db.info["user_id"])


def _cid(db) -> str:
    return str(db.info["companion_id"])


def _deny_scope():
    """统一拒绝：存在/不存在/越权同码同文案，不泄漏存在性、不回显对方内容。"""
    fail(403, "SCOPE_DENIED", "记忆不存在或无权访问")


def _scope_or_deny(db):
    """owner 复核 + 默认拒绝：companion 不属于当前 user / 未 consent 都拒绝。"""
    from memory import isolation as iso
    try:
        scope = iso.build_scope(db, _uid(db), _cid(db))
    except iso.ScopeDenied:
        _deny_scope()
    if not scope.consent:
        fail(403, "CONSENT_REQUIRED", "用户未授权记忆使用")
    return scope


def _entry_view(hit):
    entry = hit["entry"] if isinstance(hit, dict) and "entry" in hit else hit
    return {
        "id": str(entry.id),
        "type": entry.type,
        "content": entry.content,
        "summary": entry.summary,
        "importance": entry.importance,
        "companion_id": entry.companion_id,
        "ts": entry.ts.isoformat() if entry.ts else None,
    }


# ---------------------------------------------------------------------------
# 静态路径必须先于 /scoped/{memory_id}，否则会被路径参数吞掉
# ---------------------------------------------------------------------------
@router.get("/scoped/scope")
def scoped_scope(db=Depends(get_session)):
    """当前用户/旅程的 scope 摘要（验收证据用，只含本人数据）。"""
    if not scope_guard_enabled():
        fail(404, "NOT_FOUND", "scope 内省仅在隔离守卫开启时可用")
    return {"ok": True, "guard": True, "scope": _scope_or_deny(db).as_dict()}


@router.get("/scoped/recall")
def scoped_recall(query: str, top_k: int | None = None, db=Depends(get_session)):
    """先过滤后召回：B 用 scoped 端点查不到 A 的记忆。"""
    uid, cid = _uid(db), _cid(db)
    if scope_guard_enabled():
        from memory import isolation as iso
        _scope_or_deny(db)
        try:
            hits = iso.scoped_recall(db, uid, cid, query, top_k=top_k)
        except iso.ScopeDenied:
            _deny_scope()
        return {"ok": True, "guard": True, "path": "isolation.scoped_recall",
                "hits": [_entry_view(h) for h in hits], "count": len(hits)}
    # 开关关：基线召回（UC-4 之前的行为），守卫代码零执行。
    from memory import recall as recall_mod
    hits = recall_mod.recall(db, uid, query, top_k=top_k)
    return {"ok": True, "guard": False, "path": "recall.recall",
            "hits": [_entry_view(h) for h in hits], "count": len(hits)}


@router.get("/scoped/list")
def scoped_list(limit: int = 50, db=Depends(get_session)):
    """只列 scope 内的记忆；不调用 store.list_memories（那会全量读用户记忆）。"""
    uid, cid = _uid(db), _cid(db)
    if scope_guard_enabled():
        from memory import isolation as iso
        _scope_or_deny(db)
        entries = iso.scoped_list_memories(db, uid, cid, limit=limit)
        return {"ok": True, "guard": True, "path": "isolation.scoped_list_memories",
                "memories": [_entry_view(e) for e in entries], "count": len(entries)}
    from memory import store as mstore
    entries = mstore.list_memories(db, uid)[:limit]
    return {"ok": True, "guard": False, "path": "store.list_memories",
            "memories": [_entry_view(e) for e in entries], "count": len(entries)}


@router.get("/scoped/{memory_id}")
def scoped_get(memory_id: str, db=Depends(get_session)):
    """按 id 读单条：越权/不存在统一 403 SCOPE_DENIED，不泄漏存在性。"""
    uid, cid = _uid(db), _cid(db)
    if scope_guard_enabled():
        from memory import isolation as iso
        _scope_or_deny(db)
        try:
            entry = iso.scoped_get_memory(db, uid, cid, memory_id)
        except iso.ScopeDenied:
            _deny_scope()
        return {"ok": True, "guard": True, "path": "isolation.scoped_get_memory",
                "memory": _entry_view(entry)}
    from memory import store as mstore
    entry = db.get(mstore.MemoryEntry, memory_id)
    if entry is None:
        fail(404, "NOT_FOUND", "记忆不存在")
    return {"ok": True, "guard": False, "path": "db.get",
            "memory": _entry_view(entry)}
