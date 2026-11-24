"""Session checkpoint 快照（context engineering research P2 / D-063，未接线=未验证）。

用途：长对话/跨 session 时把「当前进度 + 结构化状态」落一份快照，供下次 resume 读取。
与 context_compaction.py 的摘要不同：checkpoint 是显式、可加载的结构化 payload，
 compaction 是被动上下文压缩。二者互不依赖，改本模块不会碰到 D-070 的事实保留链。

红线：
- **不改 models.py**：本模块用原生 SQL 操作一张独立表 `session_checkpoints`，
  表结构在 CHECKPOINT_TABLE_SQL 明写。是否把表纳入 models.py / Alembic / 主链路，
  由 lead 接线决定；接线前本模块 = 未验证。
- 不 commit / 不 rollback：事务归属调用方（与项目 ManagedSession 约定一致）。
- 零 LLM、零网络、纯本地 SQLite。

建表 SQL（SQLite；Postgres 需把 TEXT JSON 换成 JSONB，ON CONFLICT 语法一致）：
    CREATE TABLE IF NOT EXISTS session_checkpoints (
        id          TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL,
        session_id  TEXT NOT NULL,
        payload     TEXT NOT NULL,   -- json.dumps(..., ensure_ascii=False, sort_keys=True)
        summary     TEXT NOT NULL DEFAULT '',
        created_at  TEXT NOT NULL,   -- ISO8601 UTC
        updated_at  TEXT NOT NULL,
        UNIQUE(user_id, session_id)
    );
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

TABLE_NAME = "session_checkpoints"

CHECKPOINT_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    payload     TEXT NOT NULL,
    summary     TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    UNIQUE(user_id, session_id)
)
"""


class CheckpointError(RuntimeError):
    """checkpoint 入参/存储错误（表缺失时给出明确接线指引）。"""


class CheckpointTableMissing(CheckpointError):
    """表未建：调用 ensure_checkpoint_table(db) 或由 lead 在 models.py 建表。"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_table(db: Session, exc: Exception) -> None:
    """把 SQLite 'no such table' 转成可执行的接线提示，其它异常原样抛出。"""
    if "no such table" in str(exc).lower():
        raise CheckpointTableMissing(
            f"表 {TABLE_NAME} 不存在：先调用 memory.checkpoint.ensure_checkpoint_table(db) "
            "或由 lead 在 models.py 定义后迁移（未接线=未验证）"
        ) from exc
    raise exc


def ensure_checkpoint_table(db: Session) -> None:
    """幂等建表。显式调用，绝不隐式改 schema；调用方自行 commit。"""
    db.execute(text(CHECKPOINT_TABLE_SQL))


def _stable_id(user_id: str, session_id: str) -> str:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"make-again:checkpoint:{user_id}:{session_id}").hex


def save_checkpoint(db: Session, user_id: str, session_id: str, payload: dict[str, Any],
                    *, summary: str = "") -> str:
    """写入/覆盖一个 (user_id, session_id) 的快照，返回快照 id。

    幂等 UPSERT（同一 user+session 只有一行）；不 commit。payload 必须是 dict。
    """
    uid, sid = str(user_id or "").strip(), str(session_id or "").strip()
    if not uid or not sid:
        raise CheckpointError("user_id / session_id 不能为空")
    if not isinstance(payload, dict):
        raise CheckpointError("payload 必须是 dict")
    moment = _now_iso()
    ckpt_id = _stable_id(uid, sid)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    sql = text(
        f"INSERT INTO {TABLE_NAME} (id, user_id, session_id, payload, summary, created_at, updated_at) "
        "VALUES (:id, :uid, :sid, :payload, :summary, :created, :updated) "
        "ON CONFLICT(user_id, session_id) DO UPDATE SET "
        "payload = excluded.payload, summary = excluded.summary, updated_at = excluded.updated_at"
    )
    try:
        db.execute(sql, {
            "id": ckpt_id, "uid": uid, "sid": sid, "payload": encoded,
            "summary": str(summary or ""), "created": moment, "updated": moment,
        })
    except OperationalError as exc:
        _check_table(db, exc)
    return ckpt_id


def load_checkpoint(db: Session, user_id: str, session_id: str) -> dict[str, Any] | None:
    """读取快照；不存在返回 None。payload 从 JSON 还原为 dict；不 commit。"""
    uid, sid = str(user_id or "").strip(), str(session_id or "").strip()
    if not uid or not sid:
        return None
    sql = text(
        f"SELECT id, user_id, session_id, payload, summary, created_at, updated_at "
        f"FROM {TABLE_NAME} WHERE user_id = :uid AND session_id = :sid"
    )
    try:
        row = db.execute(sql, {"uid": uid, "sid": sid}).mappings().first()
    except OperationalError as exc:
        _check_table(db, exc)
        return None
    if row is None:
        return None
    try:
        payload = json.loads(row["payload"])
    except (ValueError, TypeError) as exc:
        raise CheckpointError("checkpoint payload 不是合法 JSON") from exc
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint payload 必须是 JSON object")
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "session_id": row["session_id"],
        "payload": payload,
        "summary": row["summary"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def delete_checkpoint(db: Session, user_id: str, session_id: str) -> bool:
    """删除一个快照，返回是否删到了行；不 commit。"""
    uid, sid = str(user_id or "").strip(), str(session_id or "").strip()
    if not uid or not sid:
        return False
    sql = text(f"DELETE FROM {TABLE_NAME} WHERE user_id = :uid AND session_id = :sid")
    try:
        result = db.execute(sql, {"uid": uid, "sid": sid})
    except OperationalError as exc:
        _check_table(db, exc)
        return False
    return int(result.rowcount or 0) > 0
