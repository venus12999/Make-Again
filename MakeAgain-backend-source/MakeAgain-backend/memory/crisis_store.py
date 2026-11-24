"""危机事件持久化（U-004）。

把 WP4 `emotion/crisis_flow.py` 的进程内事件落进 SQLite 的 `crisis_events` 表。
本模块**只做存储**：不改 emotion/crisis.py 的硬门槛检测，不改 crisis_flow.py 的
分级/确认/冷却逻辑；W0 若要接线，单独在调用处调 save_crisis_event 即可。

SQLAlchemy 2.0 注意（D-070 踩过的坑）：
  `Row` 不支持 `row["col"]` 字符串索引，会抛 "tuple indices must be integers"；
  读列必须走 `row._mapping`。本模块所有取列处都遵守。
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select, text

from memory.db import _RawSessionLocal

logger = logging.getLogger(__name__)


def _coerce_ts(value) -> datetime:
    """crisis_flow 的 ts 是 epoch float；兼容 datetime / ISO 字符串 / 缺失。"""
    from memory.models import utcnow

    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc).replace(tzinfo=None)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
        except ValueError:
            pass
    return utcnow()


def _coerce_keywords(value) -> list:
    """keywords 来自 detect() 的 hit（list[str]）；容忍 JSON 字符串/None。"""
    if isinstance(value, list):
        return [str(k) for k in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(k) for k in parsed]
        except ValueError:
            return [value] if value else []
    return []


def _iso(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat() + "Z"
    return str(value)


def save_crisis_event(event: dict, db=None) -> str:
    """落一条危机事件，返回事件 id。

    `event` 直接吃 crisis_flow.handle_crisis 返回的 `event` dict（也接受同结构的
    普通 dict），字段名映射：id/event_id、hit/keywords、session_id、severity、
    level、state、review_required、ts。companion_id 只认事件显式携带的值，
    不从 session_id 猜测（沿用 B07 迁移"不猜归属"的原则）。

    用独立会话真提交：managed 请求会话的 commit() 只 flush（见
    context_compaction.save_entry），事件若不独立提交会永远落不了库。
    失败不阻断主对话（危机话术仍需照常返回），只记 warning。
    """
    from memory.models import CrisisEvent

    event = dict(event or {})
    event_id = str(event.get("id") or event.get("event_id") or uuid.uuid4().hex)
    user_id = str(event.get("user_id") or "")
    if not user_id:
        raise ValueError("save_crisis_event: user_id is required")

    row = CrisisEvent(
        id=event_id,
        user_id=user_id,
        companion_id=event.get("companion_id"),
        severity=str(event.get("severity") or (f"level{event.get('level')}" if event.get("level") else "")),
        keywords=_coerce_keywords(event.get("keywords", event.get("hit"))),
        level=int(event.get("level") or 0),
        session_id=event.get("session_id"),
        text=str(event.get("text") or ""),
        state=str(event.get("state") or ""),
        review_required=bool(event.get("review_required")),
        ts=_coerce_ts(event.get("ts")),
    )
    d = _RawSessionLocal()
    try:
        d.add(row)
        d.commit()
    except Exception as exc:  # 持久化是增强，不能挡住危机话术返回
        logger.warning("crisis event save failed (non-blocking): %s", exc)
    finally:
        d.close()
    return event_id


def list_crisis_events(db, user_id: str, limit: int = 100) -> list[dict]:
    """按用户列出危机事件（新→旧），只读本人数据，调用方需已完成鉴权。

    `select(table)` + `row._mapping`：SQLAlchemy 2.0 不能用 `row["col"]`。
    """
    from memory.models import CrisisEvent

    table = CrisisEvent.__table__
    stmt = (
        select(table)
        .where(table.c.user_id == str(user_id))
        .order_by(table.c.ts.desc(), table.c.id.desc())
        .limit(max(1, min(int(limit), 200)))
    )
    out: list[dict] = []
    for row in db.execute(stmt):
        m = row._mapping  # D-070: 不能用 row["col"] 字符串索引
        out.append({
            "id": m["id"],
            "user_id": m["user_id"],
            "companion_id": m["companion_id"],
            "severity": m["severity"],
            "keywords": _coerce_keywords(m["keywords"]),
            "level": m["level"],
            "session_id": m["session_id"],
            "text": m["text"],
            "state": m["state"],
            "review_required": bool(m["review_required"]),
            "ts": _iso(m["ts"]),
        })
    return out


def count_crisis_events(db, user_id: str) -> int:
    """用户危机事件总数（text() 计数，供门禁/审计用）。"""
    return int(db.execute(
        text("SELECT COUNT(*) FROM crisis_events WHERE user_id = :uid"),
        {"uid": str(user_id)},
    ).scalar_one())
