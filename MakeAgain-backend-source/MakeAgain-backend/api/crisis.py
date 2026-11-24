"""U-004 · 危机事件只读端点。

鉴权完全复用 api/security.py 的 SecureRoute（token → user_id、owner 校验），
不自造鉴权；用户身份一律取 `db.info["user_id"]`，不信任 query/body 里的 user_id。
事件数据来自 memory/crisis_store.py 的 SQLite 持久化，不再是 WP4 的进程内内存表。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.security import SecureRoute
from memory.crisis_store import list_crisis_events
from memory.db import get_session

router = APIRouter(prefix="/api/crisis", route_class=SecureRoute)


@router.get("/events")
def crisis_events(limit: int = 100, db=Depends(get_session)):
    """列出当前登录用户的危机事件（新→旧）。只回本人数据。"""
    uid = str(db.info["user_id"])
    events = list_crisis_events(db, uid, limit=limit)
    return {"ok": True, "events": events, "count": len(events)}
