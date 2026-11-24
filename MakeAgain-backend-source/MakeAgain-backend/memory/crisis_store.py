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

from sqlalchemy import select, text, update

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


def update_event_state(user_id: str, event_ref: str, state: str, extra: dict | None = None) -> int:
    """确认轮回写：把 crisis_flow 的确认结果落到 crisis_events，返回受影响行数。

    定位口径（R3）：
      主定位 = ``user_id + event_ref`` 直接命中 DB 主键 ``id``。依据：
      ``save_crisis_event`` 用事件 dict 的 ``id``（crisis_flow 里的 ``crisis-N``）
      作为 CrisisEvent.id 落库，所以内存 id 与 DB 主键是同一个值，不需要猜。
      兜底定位 = 若 event_ref 在 DB 中不存在（例如历史事件/异常路径），退化为
      ``user_id + 最近一条 state='pending_confirm' 事件``（按 ts、id 倒序第一条），
      **永远带 user_id 约束**，A 的确认绝不改 B 的行。
      两种定位都只读 ``row._mapping`` 取列（D-070 的 SQLAlchemy 2.0 Row 坑）。

    可落库字段：``state`` 与 ``extra['review_required']``（CrisisEvent 有对应列）。
    ``extra['review_entry']`` / ``extra['confirmed_at']`` 在当前 CrisisEvent schema
    无对应列（models.py 不在 R3 写作用域），不写库；它们仍保留在 crisis_flow 内存
    事件与 confirm_crisis 返回 dict 中——这是诚实边界，不伪造 DB 列。

    独立会话真提交（同 save_crisis_event，绕开 managed session 只 flush 的坑）；
    失败 swallow 并返回 0，调用方 confirm_crisis 的确认话术不受影响。
    """
    from memory.models import CrisisEvent

    table = CrisisEvent.__table__
    uid = str(user_id)
    ref = str(event_ref or "")
    values: dict = {"state": str(state)}
    if extra and "review_required" in extra:
        values["review_required"] = bool(extra["review_required"])

    d = _RawSessionLocal()
    try:
        target_id: str | None = None
        if ref:
            row = d.execute(
                select(table.c.id).where(table.c.user_id == uid, table.c.id == ref)
            ).first()
            if row is not None:
                target_id = row._mapping["id"]  # D-070: 不能用 row["id"]
        if target_id is None:
            row = d.execute(
                select(table.c.id)
                .where(table.c.user_id == uid, table.c.state == "pending_confirm")
                .order_by(table.c.ts.desc(), table.c.id.desc())
                .limit(1)
            ).first()
            if row is not None:
                target_id = row._mapping["id"]
        if target_id is None:
            return 0  # 无对应 DB 行（如纯内存事件），幂等 no-op

        result = d.execute(
            update(table)
            .where(table.c.user_id == uid, table.c.id == target_id)
            .values(**values)
        )
        d.commit()  # 独立会话真提交，不用调用方的 managed session
        return int(result.rowcount or 0)
    except Exception as exc:  # 回写是增强，不能挡确认轮话术
        d.rollback()
        logger.warning("crisis event state writeback failed (non-blocking): %s", exc)
        return 0
    finally:
        d.close()


def list_crisis_events(db, user_id: str, limit: int = 100, state: str | None = None) -> list[dict]:
    """按用户列出危机事件（新→旧），只读本人数据，调用方需已完成鉴权。

    `state` 为可选的精确过滤（confirmed_safe / escalated / pending_confirm），
    在 SQL 层先过滤再 limit，避免"limit 截断后才过滤"的语义坑；不传时行为逐字节不变。
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
    if state:
        stmt = stmt.where(table.c.state == str(state))
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
