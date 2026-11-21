"""UC-1 记忆确认流：抽取 → Candidate(pending) → 用户 confirm/reject/edit → Fact。

产品地基：USE_MEMORY_CONFIRM 默认开（只有显式 0/false/no/off 才关闭）。
本模块是薄编排层，不重造五层机制：
  - Fact 写入复用 memory.store.add_memory（业务库 + 五层并存写 + 双路索引）
  - 纠错复用 memory.five_layer.FiveLayerAdapter.supersede_content（旧事实失效 + 新版本）
  - 召回判据复用 memory.store.five_layer_current / list_memories

Candidate 本身没有 ORM 表（不动业务库 migration），用独立的轻量 SQLite 登记册，
口径与 provenance_store.py 一致：每次调用新开连接、失败 swallow、开关关时零副作用。
extract 侧接入点就是 submit_candidate()：开关开时它只落 Candidate，不写 Fact；
开关关时它逐字节走基线 store.add_memory()。
"""
from __future__ import annotations

import datetime as dt
import os
from contextlib import contextmanager
import pathlib
import sqlite3
import tempfile
import threading
import uuid

from memory import store as _store

_DISABLED_VALUES = frozenset({"0", "false", "no", "off"})

PENDING = "pending"
CONFIRMED = "confirmed"
REJECTED = "rejected"
EDITED = "edited"
_VALID_STATUSES = (PENDING, CONFIRMED, REJECTED, EDITED)

_candidate_lock = threading.Lock()
_initialised_paths: set[str] = set()


class CandidateError(Exception):
    """Candidate 状态/归属错误；API 层映射成 4xx。"""


class PermissionDenied(CandidateError):
    """越权访问他人 Candidate。"""


def confirmation_enabled() -> bool:
    """USE_MEMORY_CONFIRM 默认开：只有显式关闭值才回退基线。"""
    raw = (os.getenv("USE_MEMORY_CONFIRM") or "").strip().lower()
    if raw in _DISABLED_VALUES:
        return False
    # 未知值按开处理（产品地基，不能因为拼写错误悄悄关掉）。
    return True


def default_candidate_db_path() -> str:
    """候选登记册落点：显式 env > 业务库同目录旁 > 系统临时目录。"""
    explicit = (os.getenv("MEMORY_CONFIRM_DB_PATH") or "").strip()
    if explicit:
        return explicit
    url = (os.getenv("DATABASE_URL") or "").strip()
    if url.startswith("sqlite:///"):
        db_path = pathlib.Path(url[len("sqlite:///") :])
        return str(db_path.with_name(db_path.stem + "-candidates.sqlite3"))
    return str(pathlib.Path(tempfile.gettempdir()) / "make-again-candidates.sqlite3")


def _connect() -> sqlite3.Connection:
    path = default_candidate_db_path()
    parent = pathlib.Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5.0)
    connection.row_factory = sqlite3.Row
    with _candidate_lock:
        if path not in _initialised_paths:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS candidates (
                       id TEXT PRIMARY KEY,
                       user_id TEXT NOT NULL,
                       content TEXT NOT NULL,
                       edited_content TEXT,
                       review_status TEXT NOT NULL,
                       memory_id TEXT,
                       record_id TEXT,
                       successor_id TEXT,
                       reason TEXT,
                       created_at TEXT NOT NULL,
                       reviewed_at TEXT
                   )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_candidates_user_status "
                "ON candidates(user_id, review_status)"
            )
            connection.commit()
            _initialised_paths.add(path)
    return connection


@contextmanager
def _candidate_db():
    """每次调用独占连接；异常回滚、正常提交、退出必关（不留悬挂 sqlite 句柄）。"""
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def reset_confirm_cache() -> None:
    """只清登记册的建表缓存（测试/切库用，不碰业务库、不删数据）。"""
    with _candidate_lock:
        _initialised_paths.clear()


def _now() -> str:
    return dt.datetime.now(dt.UTC).replace(tzinfo=None).isoformat()


def _row_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def _assert_owner(row: sqlite3.Row, user_id: str) -> None:
    """归属校验在记忆层再做一次（API 的 SecureRoute 是外层防线，纵深防御）。"""
    if str(row["user_id"]) != str(user_id):
        # 不暴露 Candidate 是否存在：越权统一报"不存在/无权限"。
        raise PermissionDenied("无权访问此记忆候选")


def get_candidate(candidate_id: str, user_id: str | None = None) -> dict:
    with _candidate_db() as connection:
        row = connection.execute(
            "SELECT * FROM candidates WHERE id = ?", (str(candidate_id),)
        ).fetchone()
    if row is None:
        raise CandidateError("记忆候选不存在")
    if user_id is not None:
        _assert_owner(row, user_id)
    return _row_dict(row)


def list_candidates(user_id: str, status: str | None = PENDING, limit: int = 50) -> list[dict]:
    """按 owner 列出候选；status=None = 全部。开关关时登记册本就为空。"""
    if status is not None and status not in _VALID_STATUSES:
        raise CandidateError("无效的审核状态")
    sql = "SELECT * FROM candidates WHERE user_id = ?"
    params: list = [str(user_id)]
    if status is not None:
        sql += " AND review_status = ?"
        params.append(status)
    sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
    params.append(max(1, min(int(limit), 200)))
    with _candidate_db() as connection:
        rows = connection.execute(sql, params).fetchall()
    return [_row_dict(row) for row in rows]


def _insert_candidate(user_id: str, content: str, *, memory_type: str = "chat") -> dict:
    candidate = {
        "id": uuid.uuid4().hex,
        "user_id": str(user_id),
        "content": str(content),
        "edited_content": None,
        "review_status": PENDING,
        "memory_id": None,
        "record_id": None,
        "successor_id": None,
        "reason": None,
        "created_at": _now(),
        "reviewed_at": None,
        "type": str(memory_type or "chat"),
    }
    with _candidate_db() as connection:
        connection.execute(
            "INSERT INTO candidates (id, user_id, content, edited_content, review_status, "
            "memory_id, record_id, successor_id, reason, created_at, reviewed_at) "
            "VALUES (:id, :user_id, :content, :edited_content, :review_status, "
            ":memory_id, :record_id, :successor_id, :reason, :created_at, :reviewed_at)",
            candidate,
        )
        connection.commit()
    return candidate


def submit_candidate(db, user_id: str, *, type: str = "chat", content: str,
                     summary: str | None = None, facts: list | None = None,
                     emotion: dict | None = None, importance: float = 5.0,
                     place_tag: str | None = None, time_tag: str | None = None,
                     cbt: dict | None = None) -> dict:
    """抽取落点。

    开：只写 Candidate(review_status=pending)，不写业务库/五层/索引 → 不可召回。
    关：逐字节调用基线 store.add_memory（同样的参数、同样的副作用），不产生 Candidate。
    """
    if not confirmation_enabled():
        entry = _store.add_memory(
            db, user_id, type=type, content=content, summary=summary, facts=facts,
            emotion=emotion, importance=importance, place_tag=place_tag,
            time_tag=time_tag, cbt=cbt,
        )
        return {
            "candidate_id": None, "review_status": "auto", "mode": "baseline",
            "content": content, "memory_id": str(entry.id), "user_id": str(user_id),
        }
    candidate = _insert_candidate(user_id, content, memory_type=type)
    return {
        "candidate_id": candidate["id"], "review_status": PENDING, "mode": "confirm",
        "content": content, "memory_id": None, "user_id": str(user_id),
        "type": candidate["type"],
    }


def _effective_content(candidate: dict) -> str:
    return candidate.get("edited_content") or candidate["content"]


def confirm_candidate(db, candidate_id: str, user_id: str) -> dict:
    """Confirm → Fact：复用 store.add_memory 落业务库 + 五层并存写，之后才召回可见。"""
    candidate = get_candidate(candidate_id, user_id)
    if candidate["review_status"] in (CONFIRMED, EDITED):
        return candidate  # 幂等
    if candidate["review_status"] != PENDING:
        raise CandidateError("已拒绝的候选不能确认")
    content = _effective_content(candidate)
    entry = _store.add_memory(db, user_id, type=candidate.get("type") or "chat", content=content)
    # store.add_memory 在 USE_FIVE_LAYER=1 时已并存写过五层（record name = entry.id），
    # 这里只补缺失的情况，避免对同一 slug 重复 add。
    record_id = str(entry.id)
    if _store.five_layer_enabled():
        try:
            adapter = _store._get_five_layer_adapter()
            if not adapter.record(record_id):
                adapter.add(
                    record_id, content, user_id=str(user_id),
                    type=candidate.get("type") or "chat",
                    provenance=[content], valid_from=entry.ts,
                )
        except Exception:  # 五层是第二存储，写失败 swallow 回退基线
            record_id = None
    with _candidate_db() as connection:
        connection.execute(
            "UPDATE candidates SET review_status = ?, memory_id = ?, record_id = ?, "
            "reviewed_at = ? WHERE id = ? AND user_id = ?",
            (CONFIRMED, str(entry.id), record_id, _now(), str(candidate_id), str(user_id)),
        )
        connection.commit()
    return get_candidate(candidate_id, user_id)


def reject_candidate(candidate_id: str, user_id: str, *, reason: str = "") -> dict:
    """Reject → 不召回：只改候选状态，绝不写 Fact/五层/索引。"""
    candidate = get_candidate(candidate_id, user_id)
    if candidate["review_status"] in (REJECTED,):
        return candidate  # 幂等
    if candidate["review_status"] in (CONFIRMED, EDITED):
        raise CandidateError("已确认的事实请用删除/纠错流程，不能简单拒绝")
    with _candidate_db() as connection:
        connection.execute(
            "UPDATE candidates SET review_status = ?, reason = ?, reviewed_at = ? "
            "WHERE id = ? AND user_id = ?",
            (REJECTED, str(reason or "")[:500], _now(), str(candidate_id), str(user_id)),
        )
        connection.commit()
    return get_candidate(candidate_id, user_id)


def edit_candidate(db, candidate_id: str, user_id: str, *, content: str,
                   reason: str = "") -> dict:
    """Edit → 纠正候选。

    - pending/rejected：只改候选内容，仍是 pending（未确认不能先成 Fact）。
    - confirmed/edited：底层走 five_layer.supersede_content——旧事实 invalid_at 且
      superseded_by 指向新版本；current 不再返旧事实，as_of 历史仍可溯源。
    """
    candidate = get_candidate(candidate_id, user_id)
    new_content = str(content or "").strip()
    if not new_content:
        raise CandidateError("纠正内容不能为空")
    if candidate["review_status"] in (PENDING, REJECTED):
        with _candidate_db() as connection:
            connection.execute(
                "UPDATE candidates SET edited_content = ?, review_status = ?, reason = ?, "
                "reviewed_at = ? WHERE id = ? AND user_id = ?",
                (new_content, PENDING, str(reason or "")[:500], _now(),
                 str(candidate_id), str(user_id)),
            )
            connection.commit()
        return get_candidate(candidate_id, user_id)

    successor_id = candidate.get("successor_id")
    record_id = candidate.get("record_id")
    if record_id:
        # 已存在的 Fact 必须 supersede 掉旧版本；这里不再看 USE_FIVE_LAYER 开关——
        # 记录既然在库里，纠错就必须让它失效，不能因为开关翻转留下"改了内容但旧事实仍召回"。
        adapter = _store._get_five_layer_adapter()
        if adapter.record(record_id):  # root 变化时避免制造没有前驱的孤儿
            previous = adapter.supersede_content(
                record_id, new_content, user_id=str(user_id),
                type=candidate.get("type") or "chat", provenance=[new_content],
                valid_from=dt.datetime.now(dt.UTC).replace(tzinfo=None),
            )
            successor_id = previous.get("superseded_by") or successor_id
    # 业务库正文同步更新（不走 prov_update_memory，避免与上面的 supersede 重复）。
    memory_id = candidate.get("memory_id")
    if memory_id:
        entry = db.get(_store.MemoryEntry, str(memory_id))
        if entry is not None and entry.user_id == str(user_id):
            entry.content = new_content
            db.commit()
            db.refresh(entry)
    with _candidate_db() as connection:
        connection.execute(
            "UPDATE candidates SET edited_content = ?, review_status = ?, successor_id = ?, "
            "reason = ?, reviewed_at = ? WHERE id = ? AND user_id = ?",
            (new_content, EDITED, successor_id, str(reason or "")[:500], _now(),
             str(candidate_id), str(user_id)),
        )
        connection.commit()
    return get_candidate(candidate_id, user_id)


def recallable(candidate_id: str, user_id: str) -> bool:
    """召回可见性判据：确认后的正文出现在业务库或五层 current 里才算可召回。"""
    candidate = get_candidate(candidate_id, user_id)
    if candidate["review_status"] not in (CONFIRMED, EDITED):
        return False
    text = _effective_content(candidate)
    with _store.SessionLocal() as db:
        if any(row.content == text for row in _store.list_memories(db, user_id)):
            return True
    return any(row["abstract"] == text for row in _store.five_layer_current(text))
