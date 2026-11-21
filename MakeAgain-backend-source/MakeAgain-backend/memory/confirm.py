"""UC-1 记忆确认流：抽取 → Candidate(pending) → 用户 confirm/reject/edit → Fact。

产品地基：USE_MEMORY_CONFIRM 默认开（只有显式 0/false/no/off 才关闭）。
本模块是薄编排层，不重造五层机制：
  - Fact 写入复用 memory.store.add_memory（业务库 + 五层并存写 + 双路索引）
  - 纠错复用 memory.five_layer.FiveLayerAdapter.supersede_content（旧事实失效 + 新版本）
  - 召回判据复用 memory.store.five_layer_current / list_memories

【F2 一体化】Candidate 不再用独立 SQLite 登记册。唯一真相是五层库的
index.db —— 与 agent-memory-core 的 records 表同一个 SQLite 文件。在该库里
建 candidates 扩展表（review_status 列，pending 即 Candidate 层），并给 records
表扩展 review_status/candidate_id 两列做「Candidate → Fact」关联。这样彻底消除
「业务库 + 五层库 + 独立候选登记册」三套存储里的第三套。

向后兼容（迁移）：老独立登记册（MEMORY_CONFIRM_DB_PATH / -candidates.sqlite3）只读。
get/list 读到老行时读时回填（INSERT OR IGNORE）到五层库候选表，此后读写全走新库；
因此老数据仍能 confirm/reject/edit。老库不存在/损坏一律当空，绝不新建 sqlite。

extract 侧接入点就是 submit_candidate()：开关开时它只落 Candidate(pending)，不写 Fact；
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

# 候选表列（与迁移前登记册逐列一致：老行可直接回填，字段语义不变）。
_CANDIDATE_COLUMNS = (
    "id TEXT PRIMARY KEY, user_id TEXT NOT NULL, content TEXT NOT NULL, "
    "edited_content TEXT, review_status TEXT NOT NULL, memory_id TEXT, "
    "record_id TEXT, successor_id TEXT, reason TEXT, created_at TEXT NOT NULL, "
    "reviewed_at TEXT"
)
_MIGRATE_KEYS = (
    "id", "user_id", "content", "edited_content", "review_status", "memory_id",
    "record_id", "successor_id", "reason", "created_at", "reviewed_at",
)


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
    """【迁移专用，只读】老独立登记册落点：显式 env > 业务库同目录旁 > 临时目录。

    新 Candidate 不再写这里；此函数只服务于向后兼容的历史数据读取。
    """
    explicit = (os.getenv("MEMORY_CONFIRM_DB_PATH") or "").strip()
    if explicit:
        return explicit
    url = (os.getenv("DATABASE_URL") or "").strip()
    if url.startswith("sqlite:///"):
        db_path = pathlib.Path(url[len("sqlite:///") :])
        return str(db_path.with_name(db_path.stem + "-candidates.sqlite3"))
    return str(pathlib.Path(tempfile.gettempdir()) / "make-again-candidates.sqlite3")


def _candidate_db_path() -> str:
    """候选登记册唯一真相：五层库的 index.db（与 records 表同一个 SQLite 文件）。"""
    return str(_store._get_five_layer_adapter()._store.layout.index_db)


def _connect() -> sqlite3.Connection:
    """连五层库 index.db，惰性建 candidates 扩展表 + records 扩展列。"""
    path = _candidate_db_path()
    parent = pathlib.Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=5.0)
    connection.row_factory = sqlite3.Row
    with _candidate_lock:
        if path not in _initialised_paths:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS candidates (" + _CANDIDATE_COLUMNS + ")"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_candidates_user_status "
                "ON candidates(user_id, review_status)"
            )
            # 五层库表扩展：把 Candidate(review_status) 与 Fact 记录关联起来。
            # agent-memory-core 的 records 表由库自己 CREATE；这里只加列，不重建。
            # 扩展列是增强，失败 swallow，不影响 Candidate 主流程。
            try:
                columns = {row[1] for row in connection.execute("PRAGMA table_info(records)")}
                if "review_status" not in columns:
                    connection.execute("ALTER TABLE records ADD COLUMN review_status TEXT")
                if "candidate_id" not in columns:
                    connection.execute("ALTER TABLE records ADD COLUMN candidate_id TEXT")
            except sqlite3.Error:
                pass
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
    """只清建表缓存（测试/切库用，不碰业务库、不删数据）。"""
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


# ------------------------------------------------------------------ 旧登记册迁移（只读 → 回填新库）
def _legacy_rows(sql: str, params: tuple) -> list[dict]:
    """【迁移】从老独立登记册读取；文件不存在/无表/损坏一律当空，绝不新建 sqlite。"""
    path = pathlib.Path(default_candidate_db_path())
    if not path.exists():
        return []
    connection = None
    try:
        connection = sqlite3.connect(str(path), timeout=5.0)
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(sql, params).fetchall()]
    except sqlite3.Error:
        return []
    finally:
        if connection is not None:
            connection.close()


def _migrate_legacy_rows(rows: list[dict]) -> None:
    """读时迁移：把老行回填到五层库候选表（幂等），后续读写全走新库。"""
    if not rows:
        return
    with _candidate_db() as connection:
        for row in rows:
            payload = {key: row.get(key) for key in _MIGRATE_KEYS}
            connection.execute(
                "INSERT OR IGNORE INTO candidates ("
                + ", ".join(_MIGRATE_KEYS)
                + ") VALUES ("
                + ", ".join(":" + key for key in _MIGRATE_KEYS)
                + ")",
                payload,
            )


def _new_row(candidate_id: str) -> sqlite3.Row | None:
    with _candidate_db() as connection:
        return connection.execute(
            "SELECT * FROM candidates WHERE id = ?", (str(candidate_id),)
        ).fetchone()


# ------------------------------------------------------------------ 读
def get_candidate(candidate_id: str, user_id: str | None = None) -> dict:
    row = _new_row(candidate_id)
    if row is None:
        legacy = _legacy_rows("SELECT * FROM candidates WHERE id = ?", (str(candidate_id),))
        if legacy:
            _migrate_legacy_rows(legacy)
            row = _new_row(candidate_id)
    if row is None:
        raise CandidateError("记忆候选不存在")
    if user_id is not None:
        _assert_owner(row, user_id)
    return _row_dict(row)


def list_candidates(user_id: str, status: str | None = PENDING, limit: int = 50) -> list[dict]:
    """按 owner 列出候选；status=None = 全部。先回填该用户的老登记册行再统一查新库。"""
    if status is not None and status not in _VALID_STATUSES:
        raise CandidateError("无效的审核状态")
    _migrate_legacy_rows(
        _legacy_rows("SELECT * FROM candidates WHERE user_id = ?", (str(user_id),))
    )
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

    开：只写 Candidate(review_status=pending) 到五层库候选表，不写业务库/五层 records/索引
        → 不可召回。
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


def _tag_fact_record(record_id: str | None, candidate_id: str, status: str) -> None:
    """【表扩展列】把已落 Fact 的五层 records 行标记 review_status/candidate_id。

    纯增强（可观测的 Candidate → Fact 关联），写失败 swallow，不影响确认主流程。
    """
    if not record_id:
        return
    try:
        from memory.five_layer import _slug

        with _candidate_db() as connection:
            connection.execute(
                "UPDATE records SET review_status = ?, candidate_id = ? WHERE name = ?",
                (status, str(candidate_id), _slug(record_id)),
            )
    except Exception:
        pass


def confirm_candidate(db, candidate_id: str, user_id: str) -> dict:
    """Confirm → Fact：复用 store.add_memory 落业务库 + 五层并存写，之后才召回可见。"""
    candidate = get_candidate(candidate_id, user_id)
    if candidate["review_status"] in (CONFIRMED, EDITED):
        return candidate  # 幂等
    if candidate["review_status"] != PENDING:
        raise CandidateError("已拒绝的候选不能确认")
    content = _effective_content(candidate)
    entry = _store.add_memory(db, user_id, type=candidate.get("type") or "chat", content=content)
    # WP-1 闭环：confirm 后从 content 派生 facts 并回填 linked_memory_ids（Candidate 不存 facts）。
    # 失败 swallow 不破坏确认主流程——链接是增强，不是确认成功的必要条件。
    try:
        from memory import async_write as _aw
        _derived_facts = [{"fact": content}]
        _aw.link_facts_for_entry(db, str(entry.id), str(user_id), _derived_facts)
    except Exception:
        pass
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
    # 五层 records 表扩展列：标记这个 Fact 来自哪条 Candidate。
    _tag_fact_record(record_id, candidate_id, CONFIRMED)
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
    # 扩展列：纠错后的 Fact 也标记 edited。
    _tag_fact_record(record_id, candidate_id, EDITED)
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
