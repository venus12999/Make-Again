"""UC-3 · 记忆用户控制权：数据导出与级联删除（产品地基，不是实验开关）。

设计红线
--------
1. 只操作调用者自己的数据：user_id 由 SecureRoute 从会话解析后传入；任何外部传入的
   资源 id 都必须先做 owner 校验，不属于本人一律当作「不存在」。
2. 删除不可逆：业务库行真删（SQL DELETE）；vec0 向量 + bm25 元数据真删；实体反链真删；
   五层库沿 provenance/supersede 图软删（invalid_at 终止区间，trace 保留作审计）。
3. 用户控制权覆盖「全部旅程」：导出/全量删除故意绕过 companion loader 过滤
   （with_loader_criteria），否则归档 Journey 的记忆会漏删漏导。这不是越权——owner
   边界是 user_id，companion 只是展示/召回范围的过滤器。
4. 宁严勿松：每一层都记录真实执行结果（回执/审计只含元数据，绝不写记忆正文/PII）。
   索引等外部存储失败不阻断业务删除，但会如实反映在 cascade 结果里。
5. 不改任何其它模块：复用 recall.purge_memory / entity_graph.remove_memory /
   store.prov_delete_memory / five_layer.invalidate 这些既有底层机制。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import inspect, select, text

from memory import entity_graph as _entity_graph
from memory import models as m
from memory import store as mstore
from memory.b07_models import Audit
from memory.models import utcnow
from memory.vector_index import get_index_for

logger = logging.getLogger(__name__)

EXPORT_SCHEMA_VERSION = "1.0"

# 导出/全量删除覆盖的记忆载体（category, table, user 列）。
# 不含 users/user_auth/companions 等账号与陪伴对象本身：那是账号注销，不是记忆删除。
_MEMORY_TABLES: tuple[tuple[str, str, str], ...] = (
    ("portrait", "portraits", "user_id"),
    ("emotion", "emotion_nodes", "user_id"),
    ("mood", "daily_moods", "user_id"),
    ("diary", "diary_notes", "user_id"),
    ("echo", "echo_entries", "user_id"),
    ("daily_chat_summary", "daily_chat_summaries", "user_id"),
    ("chat_message", "chat_messages", "user_id"),
)


# --------------------------------------------------------------------------- helpers
def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat() + "Z" if value.tzinfo is None else value.isoformat()
    return str(value)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return _iso(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _columns(db, table: str) -> set[str]:
    """取表列名（SQLAlchemy inspect 可移植，失败回退 SQLite PRAGMA）。"""
    try:
        return {c["name"] for c in inspect(db.get_bind()).get_columns(table)}
    except Exception:  # pragma: no cover - 非 SQLite/临时表兜底
        try:
            return {row[1] for row in db.execute(text(f"PRAGMA table_info({table})")).fetchall()}
        except Exception:
            return set()


def _candidate_specs(db) -> list[tuple[str, str]]:
    """发现 Candidate 载体：UC-1 可能建独立候选表，也可能在 memory_stream 上加 review_status。"""
    specs: list[tuple[str, str]] = []
    try:
        tables = inspect(db.get_bind()).get_table_names()
    except Exception:  # pragma: no cover
        tables = []
    for table in tables:
        if "candidate" not in table.lower():
            continue
        cols = _columns(db, table)
        if "user_id" in cols:
            specs.append((table, "user_id"))
        elif "author_user_id" in cols:
            specs.append((table, "author_user_id"))
    return specs


def _candidate_rows(db, user_id: str) -> list[dict]:
    """导出 Candidate：优先独立候选表；其次 memory_stream 里 review_status=pending 的候选。"""
    out: list[dict] = []
    for table, col in _candidate_specs(db):
        try:
            rows = db.execute(
                text(f"SELECT * FROM {table} WHERE {col} = :uid ORDER BY rowid"), {"uid": user_id}
            ).mappings().all()
            out.extend({"category": "candidate", "table": table, **_jsonable(dict(row))} for row in rows)
        except Exception as exc:  # pragma: no cover
            logger.warning("candidate table %s export failed: %s", table, exc)
    if "review_status" in _columns(db, "memory_stream"):
        try:
            rows = db.execute(
                text("SELECT * FROM memory_stream WHERE user_id = :uid AND review_status = 'pending'"),
                {"uid": user_id},
            ).mappings().all()
            out.extend({"category": "candidate", "table": "memory_stream", **_jsonable(dict(row))} for row in rows)
        except Exception as exc:  # pragma: no cover
            logger.warning("pending memory export failed: %s", exc)
    return out


def _unscoped_memories(db, user_id: str) -> list[m.MemoryEntry]:
    """跨 companion 读取该用户全部记忆（用户控制权不能被当前旅程范围截断）。"""
    return list(db.execute(
        select(m.MemoryEntry)
        .where(m.MemoryEntry.user_id == str(user_id))
        .order_by(m.MemoryEntry.ts.desc())
        .execution_options(unscoped=True)
    ).scalars().all())


def _find_memory(db, user_id: str, memory_id: str) -> m.MemoryEntry | None:
    return db.execute(
        select(m.MemoryEntry)
        .where(m.MemoryEntry.user_id == str(user_id), m.MemoryEntry.id == str(memory_id))
        .execution_options(unscoped=True)
    ).scalars().first()


def _table_rows(db, table: str, col: str, user_id: str) -> list[dict]:
    rows = db.execute(
        text(f"SELECT * FROM {table} WHERE {col} = :uid ORDER BY rowid"), {"uid": user_id}
    ).mappings().all()
    return [_jsonable(dict(row)) for row in rows]


def _count_rows(db, table: str, col: str, user_id: str) -> int:
    try:
        return int(db.execute(
            text(f"SELECT COUNT(*) FROM {table} WHERE {col} = :uid"), {"uid": user_id}
        ).scalar_one())
    except Exception:  # pragma: no cover
        return 0


def _count_facts(db, user_id: str) -> int:
    total = 0
    try:
        rows = db.execute(
            text("SELECT facts FROM memory_stream WHERE user_id = :uid"), {"uid": user_id}
        ).fetchall()
    except Exception:  # pragma: no cover
        return 0
    for (raw,) in rows:
        if isinstance(raw, (list, tuple)):
            total += len(raw)
        elif isinstance(raw, dict):
            total += 1
        elif isinstance(raw, str):
            try:
                parsed = json.loads(raw or "[]")
                total += len(parsed) if isinstance(parsed, (list, dict)) else 0
            except Exception:
                pass
    return total


def _memory_provenance(db, memory_id: str) -> dict:
    """单条记忆的 provenance：溯源事件链 + 五层记录 + trace 指针（全部如实读真实存储）。"""
    provenance: dict[str, Any] = {"history": mstore.get_history(str(memory_id))}
    if mstore.five_layer_enabled():
        try:
            adapter = mstore._get_five_layer_adapter()
            record = adapter.record(str(memory_id))
            provenance["five_layer"] = record
            if record:
                provenance["trace"] = _jsonable(adapter.trace(str(memory_id)))
        except Exception as exc:  # pragma: no cover
            logger.warning("five-layer provenance read failed for %s: %s", memory_id, exc)
    return provenance


def _serialize_memory(db, entry: m.MemoryEntry) -> dict:
    return {
        "id": str(entry.id),
        "user_id": str(entry.user_id),
        "companion_id": entry.companion_id,
        "type": entry.type,
        "content": entry.content,
        "summary": entry.summary,
        "facts": _jsonable(entry.facts or []),
        "emotion": _jsonable(entry.emotion),
        "cbt": _jsonable(entry.cbt),
        "importance": entry.importance,
        "place_tag": entry.place_tag,
        "time_tag": entry.time_tag,
        "ts": _iso(entry.ts),
        "provenance": _memory_provenance(db, str(entry.id)),
    }


def _write_audit(db, user_id: str, action: str, target_id: str | None, metadata: dict) -> str | None:
    """审计只写元数据（类别/数量/时间），绝不写记忆正文。失败 swallow，不破坏删除。"""
    try:
        row = Audit(
            actor_id=str(user_id),
            action=action[:256],
            target_id=str(target_id)[:128] if target_id is not None else None,
            metadata_safe=_jsonable(metadata),
        )
        db.add(row)
        db.flush()
        return row.id
    except Exception as exc:  # pragma: no cover
        logger.warning("memory deletion audit failed: %s", exc)
        return None


# --------------------------------------------------------------------------- 导出
def count_user_memory_data(db, user_id: str) -> dict[str, int]:
    """统计某用户各记忆类别的真实行数（raw SQL，绕开 companion loader 过滤）。"""
    user_id = str(user_id or "")
    counts: dict[str, int] = {
        "memory": int(db.execute(
            text("SELECT COUNT(*) FROM memory_stream WHERE user_id = :uid"), {"uid": user_id}
        ).scalar_one()),
        "fact": _count_facts(db, user_id),
        "candidate": 0,
    }
    for table, col in _candidate_specs(db):
        counts["candidate"] += _count_rows(db, table, col, user_id)
    if "review_status" in _columns(db, "memory_stream"):
        counts["candidate"] += int(db.execute(
            text("SELECT COUNT(*) FROM memory_stream WHERE user_id = :uid AND review_status = 'pending'"),
            {"uid": user_id},
        ).scalar_one())
    for category, table, col in _MEMORY_TABLES:
        counts[category] = _count_rows(db, table, col, user_id)
    return counts


def export_user_memory(db, user_id: str, *, include_provenance: bool = True) -> dict:
    """GET /api/memory/export —— 用户全部记忆 JSON（含 provenance，可携带）。"""
    user_id = str(user_id or "")
    if not user_id:
        raise ValueError("user_id required")

    memories = _unscoped_memories(db, user_id)
    serialized = [_serialize_memory(db, e) for e in memories]
    candidates = _candidate_rows(db, user_id)

    facts: list[dict] = []
    for entry in memories:
        for idx, fact in enumerate(entry.facts or []):
            facts.append({
                "id": f"{entry.id}:fact-{idx}",
                "memory_id": str(entry.id),
                "fact": _jsonable(fact),
                "provenance": {"history": mstore.get_history(str(entry.id))} if include_provenance else {},
            })

    source = serialized if include_provenance else [
        {k: v for k, v in item.items() if k != "provenance"} for item in serialized
    ]
    result: dict[str, Any] = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "exported_at": _iso(utcnow()),
        "user_id": user_id,
        "categories": count_user_memory_data(db, user_id),
        "source": source,
        "candidate": candidates,
        "fact": facts,
        "provenance_graph": {
            str(e.id): serialized[i]["provenance"] for i, e in enumerate(memories)
        } if include_provenance else {},
    }
    for category, table, col in _MEMORY_TABLES:
        result[category] = _table_rows(db, table, col, user_id)
    return result


# --------------------------------------------------------------------------- 级联删除
def _invalidate_five_layer_graph(adapter, memory_id: str) -> tuple[list[str], list[str]]:
    """沿 provenance/supersede 图软删：主记录 + 同名 -2/-3... 后继。

    返回 (本次新软删的记录名, 图中全部相关记录名)。主记录可能已被
    prov_delete_memory 先软删，此时它不会重复 invalidate，但仍算「图中受影响记录」。
    """
    from memory.five_layer import _slug

    base = _slug(memory_id)
    try:
        from agent_memory.core.database import Database

        with Database(adapter._store.layout).connect() as connection:
            rows = connection.execute(
                "SELECT name FROM records WHERE name = ? OR name LIKE ?",
                (base, base + "-%"),
            ).fetchall()
        all_names = [str(r["name"]) for r in rows]
    except Exception as exc:  # pragma: no cover
        logger.warning("five-layer graph walk failed for %s: %s", memory_id, exc)
        all_names = [base] if adapter.record(memory_id) else []

    to_invalidate = [
        name for name in all_names
        if adapter.record(name) and not adapter.record(name).get("invalid_at")
    ]
    newly: list[str] = []
    for name in to_invalidate:
        try:
            adapter.invalidate(name)
            newly.append(name)
        except Exception as exc:  # pragma: no cover
            logger.debug("five-layer invalidate %s failed: %s", name, exc)
    return newly, all_names


def _cascade_delete_one(
    db, user_id: str, memory_id: str, *, actor_id: str | None = None
) -> dict[str, Any]:
    """单条记忆的级联清理；返回每层真实执行结果。调用方已做 owner 校验。"""
    user_id, mid = str(user_id), str(memory_id)
    cascade: dict[str, Any] = {
        "business_row": False,
        "vector_index_clean": False,
        "bm25_index_clean": False,
        "entity_backlinks_cleaned": 0,
        "entity_backlinks_clean": False,
        "five_layer_soft_deleted": [],
        "five_layer_newly_invalidated": [],
        "five_layer_current_clean": False,
        "provenance_delete_event": False,
    }

    # 1) 业务库真删 + provenance DELETE 事件 + 五层主记录 invalidate。
    #    临时摘掉 companion loader 过滤：owner 已由 user_id 校验，跨旅程也必须能删。
    saved_cid = db.info.pop("companion_id", None)
    try:
        cascade["business_row"] = bool(mstore.prov_delete_memory(
            db, mid, actor_id=actor_id or user_id, role="user"
        ))
    finally:
        if saved_cid is not None:
            db.info["companion_id"] = saved_cid

    # 2) vec0 + bm25 真删（底层 purge 幂等；即使开关关闭也调用一次保证无残留）。
    try:
        from memory.recall import purge_memory

        purge_memory(mid, db=db)
    except Exception as exc:  # pragma: no cover
        logger.warning("index purge failed for %s: %s", mid, exc)
    try:
        diag = get_index_for(db).diagnostics(mid)
        cascade["vector_index_clean"] = diag["vec_rows"] == 0
        cascade["bm25_index_clean"] = diag["bm25_rows"] == 0
    except Exception as exc:  # pragma: no cover
        logger.warning("index diagnostics failed for %s: %s", mid, exc)

    # 3) 实体反链真删，并用真实查询确认无该 memory_id 残留。
    try:
        cascade["entity_backlinks_cleaned"] = _entity_graph.remove_memory(mid, user_id=user_id)
        leftover = [
            e for e in _entity_graph.list_entities(user_id)
            if mid in e.get("linked_memory_ids", [])
        ]
        cascade["entity_backlinks_clean"] = not leftover
    except Exception as exc:  # pragma: no cover
        logger.warning("entity backlink cleanup failed for %s: %s", mid, exc)

    # 4) 五层软删沿图展开：主记录 + supersede 后继全部退出 current，trace 保留可审计。
    if mstore.five_layer_enabled():
        try:
            adapter = mstore._get_five_layer_adapter()
            newly, names = _invalidate_five_layer_graph(adapter, mid)
            cascade["five_layer_newly_invalidated"] = newly
            cascade["five_layer_soft_deleted"] = names
            cascade["five_layer_current_clean"] = (
                all(adapter.present_in_current(name) is False for name in names) if names else True
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("five-layer graph delete failed for %s: %s", mid, exc)

    # 5) provenance 删除事件（审计）——真实回读 history 确认，不靠「调用成功」自我声明。
    history = mstore.get_history(mid)
    cascade["provenance_delete_event"] = any(
        h.get("event") == "DELETE" and h.get("is_deleted") for h in history
    )
    return cascade


def delete_memory(
    db, user_id: str, memory_id: str, *, actor_id: str | None = None
) -> dict[str, Any]:
    """DELETE /api/memory/{id} —— 级联删除单条；越权 id 返回 deleted=False。"""
    user_id, mid = str(user_id or ""), str(memory_id or "")
    entry = _find_memory(db, user_id, mid) if user_id and mid else None
    if entry is None:
        return {"deleted": False, "id": mid, "reason": "NOT_FOUND"}

    cascade = _cascade_delete_one(db, user_id, mid, actor_id=actor_id)
    audit_id = _write_audit(
        db, user_id, "DELETE /api/memory/{id}", mid,
        {"scope": "single", "memory_id": mid, "cascade": cascade, "irreversible": True},
    )
    return {
        "deleted": True,
        "id": mid,
        "cascade": cascade,
        "audit_id": audit_id,
        "irreversible": True,
        "deleted_at": _iso(utcnow()),
    }


def delete_all_memories(
    db, user_id: str, *, actor_id: str | None = None
) -> dict[str, Any]:
    """DELETE /api/memory/all —— 全量删除并返回「删了哪些类别、各多少」的执行回执。"""
    user_id = str(user_id or "")
    if not user_id:
        raise ValueError("user_id required")

    planned = count_user_memory_data(db, user_id)
    entries = _unscoped_memories(db, user_id)
    mids = [str(e.id) for e in entries]

    # 索引按用户一次性清空（返回的是真实清掉的 memory id 数）。
    try:
        index_cleared = int(get_index_for(db).clear_user(user_id))
    except Exception as exc:  # pragma: no cover
        logger.warning("clear_user index failed for %s: %s", user_id, exc)
        index_cleared = 0

    five_layer_soft_deleted: list[str] = []
    entity_backlinks_cleaned = 0
    for mid in mids:
        cascade = _cascade_delete_one(db, user_id, mid, actor_id=actor_id)
        five_layer_soft_deleted.extend(cascade["five_layer_soft_deleted"])
        entity_backlinks_cleaned += int(cascade["entity_backlinks_cleaned"])

    # 其它记忆载体真删（raw SQL DELETE，绕开 loader 过滤；越权 user_id 天然过滤）。
    deleted: dict[str, int] = {"memory": len(mids), "fact": planned["fact"], "candidate": planned["candidate"]}
    for category, table, col in _MEMORY_TABLES:
        try:
            deleted[category] = int(db.execute(
                text(f"DELETE FROM {table} WHERE {col} = :uid"), {"uid": user_id}
            ).rowcount)
        except Exception as exc:  # pragma: no cover
            logger.warning("delete %s for %s failed: %s", table, user_id, exc)
            deleted[category] = 0
    for table, col in _candidate_specs(db):
        try:
            n = int(db.execute(
                text(f"DELETE FROM {table} WHERE {col} = :uid"), {"uid": user_id}
            ).rowcount)
            deleted["candidate"] += n
        except Exception as exc:  # pragma: no cover
            logger.warning("delete candidate table %s failed: %s", table, exc)

    # 真查询回读：删后每个类别必须归零（U3-V3「数字准确」的验证面）。
    # 必须用同一个 db 会话：SecureRoute 在响应返回后才 finish() 提交，另开连接
    # 读不到本事务里的未提交 DELETE，会误报「没删掉」。
    verification = count_user_memory_data(db, user_id)

    audit_id = _write_audit(
        db, user_id, "DELETE /api/memory/all", None,
        {
            "scope": "all", "deleted": deleted, "planned": planned,
            "five_layer_soft_deleted": five_layer_soft_deleted,
            "index_cleared": index_cleared,
            "entity_backlinks_cleaned": entity_backlinks_cleaned,
            "irreversible": True,
        },
    )
    return {
        "ok": True,
        "user_id": user_id,
        "deleted_at": _iso(utcnow()),
        "planned": planned,
        "deleted": deleted,
        "verification": verification,
        "index": {"vector_and_bm25_memories_cleared": index_cleared},
        "entity_backlinks_cleaned": entity_backlinks_cleaned,
        "five_layer_soft_deleted": five_layer_soft_deleted,
        "audit_id": audit_id,
        "irreversible": True,
    }
