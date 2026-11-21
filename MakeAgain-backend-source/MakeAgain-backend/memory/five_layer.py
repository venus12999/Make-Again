"""WP-B adapter: agent-memory-core 五层内核作为「并存第二存储」接入。

真实依赖（不是抄设计）：
  - agent_memory.core.store.Store        —— 唯一写入口，落 records 表 + 双 FTS5 surface
  - agent_memory.core.record.MemoryRecord—— valid_from/invalid_at/superseded_by/provenance/weight
  - agent_memory.core.record.invalidate() —— 软删（区间结束，文件/索引保留，可追溯）
  - agent_memory.core.recall.Recall._current_at() —— 双时态「当前/历史时点」判据
  - agent_memory.core.manage.Manage.sleep() —— sleep-time 权重衰减巩固
  - agent_memory.core.trace.read() —— provenance 指针回读原消息

设计红线：本模块是胶水层，只把 make-again 的 MemoryEntry 映射到库里的
Source→Candidate→Fact/Event→Summary→Index 五层；业务库（30+ ORM / 119 函数）
一行不改，开关 USE_FIVE_LAYER 默认关，写失败一律 swallow 回退旧路径。
"""
from __future__ import annotations

import datetime as dt
import pathlib
import re
import tempfile

from agent_memory.core import record as _record
from agent_memory.core import timestamp as _timestamp
from agent_memory.core.clock import Clock
from agent_memory.core.config import Config
from agent_memory.core.database import Database
from agent_memory.core.manage import Manage
from agent_memory.core.recall import Recall
from agent_memory.core.store import Store

DEFAULT_PROJECT = "make-again"
DEFAULT_TYPE = "fact"
DEFAULT_WEIGHT = 1.0
_ABSTRACT_MAX = 240
_SLUG_MAX = 80
# 标签切换归档：Journey 归属以自定义字段写进五层记录（库 records 表无 fields 列，
# 但 MemoryRecord.fields 会随 frontmatter 落盘并在 Store.records() 读回）。
_JOURNEY_FIELD = "companion_id"
# agent-memory-core 的 FACTORY schema 只有 profile/preference/entity/event/decision/
# procedure/fact/experience/reference；make-again 的业务 type 必须先映射成库认识的类型。
_TYPE_MAP = {
    "chat": "fact", "interview": "profile", "diary": "event",
    "reflect": "experience", "letter": "reference", "echo": "fact",
}
_VALID_TYPES = frozenset({
    "profile", "preference", "entity", "event",
    "decision", "procedure", "fact", "experience", "reference",
})


def _lib_type(value: str) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in _VALID_TYPES else _TYPE_MAP.get(candidate, DEFAULT_TYPE)


def _slug(value: str) -> str:
    """MemoryEntry.id 是 32 位 hex，天然是合法 slug；其它来源做保守归一化。"""
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")[:_SLUG_MAX]
    return slug or "memory"


def _one_line(text: str) -> str:
    """frontmatter 是逐行的：abstract 里不能有换行，否则下一行会被当成新字段。"""
    return " ".join(str(text or "").split())


def _aware(value: object) -> str:
    """库要求 ISO8601 带时区；业务库 ORM 存的是 naive UTC，补 Z 后交库 canonicalise。"""
    if isinstance(value, dt.datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=dt.UTC)
        return _timestamp.render(moment)
    text = str(value).strip()
    if text.endswith("Z") or re.search(r"[+-]\d\d:\d\d$", text):
        return _timestamp.canonical(text)
    return _timestamp.canonical(text + "Z")


def default_root() -> str:
    """第二存储的落点：显式 env > 业务库同目录旁 > 系统临时目录（惰性求值）。"""
    import os

    explicit = (os.getenv("FIVE_LAYER_STORE_DIR") or "").strip()
    if explicit:
        return explicit
    url = (os.getenv("DATABASE_URL") or "").strip()
    if url.startswith("sqlite:///"):
        db_path = pathlib.Path(url[len("sqlite:///") :])
        return str(db_path.with_suffix("")) + "-five-layer"
    return str(pathlib.Path(tempfile.gettempdir()) / "make-again-five-layer")


class FiveLayerAdapter:
    """把 agent-memory-core 的 Store/Database/Recall/Manage/trace 组合成两个查询 API。

    每个实例独占一个 root（文件真相）；.Database 里 records 表 + chunks/history
    两个 FTS5 surface 由库的 Indexer 在每次写入后自动投影。
    """

    def __init__(self, root: str | pathlib.Path, *, agent: str = "make-again",
                 clock: Clock | None = None, config: Config | None = None):
        self.root = pathlib.Path(root)
        self._config = config or Config()
        # 库的 Manage.sleep() 默认会在 store root 里 git commit；第二存储不能碰宿主仓库。
        self._config.manage.git_commit = False
        self._clock = clock or Clock()
        self._store = Store(root=self.root, config=self._config, clock=self._clock, agent=agent)
        self._store.init()

    # ------------------------------------------------------------------ 写入
    def add(self, memory_id: str, content: str, *, user_id: str = "",
            type: str = DEFAULT_TYPE, subject: str | None = None,
            provenance: list[str] | None = None, weight: float | None = None,
            valid_from: object = None, companion_id: str | None = None) -> str:
        """MemoryEntry → MemoryRecord（Source/Candidate → Fact），返回库里的 record name。

        companion_id 作为 Journey 归属写进自定义字段，供 archive_journey() 按 Journey
        批量归档；不传则记录不带归属，不会被任何 Journey 归档。
        """
        fields = {"project": DEFAULT_PROJECT,
                  "subject": _one_line(subject or user_id or "memory")[:_SLUG_MAX]}
        journey = str(companion_id or "").strip()
        if journey:
            fields[_JOURNEY_FIELD] = journey
        spec = {
            "type": _lib_type(type),
            "name": _slug(memory_id),
            "abstract": _one_line(content)[:_ABSTRACT_MAX],
            "body": str(content or ""),
            "fields": fields,
            "provenance": [str(p) for p in (provenance or []) if str(p).strip()],
            "weight": DEFAULT_WEIGHT if weight is None else float(weight),
        }
        if valid_from is not None:
            spec["valid_from"] = _aware(valid_from)
        record = self._store.record(**spec)
        return record.name

    def supersede(self, old_id: str, new_id: str) -> dict:
        """库自己的 supersede：写新事实并把旧事实区间终止（superseded_by 指向新记录）。"""
        old, new = _slug(old_id), _slug(new_id)
        self._store.supersede(old, new)
        return self.record(old)

    def _free_successor_name(self, old_id: str) -> str:
        """给 update 后继挑一个未被占用的 slug（沿用库 -2/-3 的后继命名惯例）。

        MemoryEntry 更新时 id 不变，五层库的旧记录名 = id；若对同名直接 add 会被库的
        `_enforce_update_only` 以「事实变了」拒绝，所以必须写一条新版本再 supersede。
        """
        base = _slug(old_id)
        ordinal = 2
        while True:
            candidate = f"{base}-{ordinal}"
            if self._store.find(candidate) is None:
                return candidate
            ordinal += 1

    def supersede_content(self, old_id: str, new_content: str, *,
                          user_id: str = "", type: str = DEFAULT_TYPE,
                          subject: str | None = None,
                          provenance: list[str] | None = None,
                          valid_from: object = None) -> dict:
        """update 路径：写一条新版本记录并把旧记录 supersede 到它。

        返回旧记录（含 superseded_by / invalid_at）；current 只返最新版本，
        as_of 历史时点仍可看到旧事实。若五层库此前没有该记录（开关此前关闭），
        按 add 语义补写，不制造没有前驱的孤儿后继。写失败由调用方 swallow 回退基线。
        """
        excerpts = provenance if provenance is not None else [str(new_content)]
        if not self.record(old_id):
            self.add(old_id, new_content, user_id=user_id, type=type,
                     subject=subject, provenance=excerpts, valid_from=valid_from)
            return self.record(old_id)
        successor = self._free_successor_name(old_id)
        self.add(successor, new_content, user_id=user_id, type=type,
                 subject=subject, provenance=excerpts, valid_from=valid_from)
        self.supersede(old_id, successor)
        return self.record(old_id)

    def invalidate(self, record_id: str, *, at: object = None,
                   successor: str | None = None) -> dict:
        """直接调用库的 record.invalidate()：软删，退出 active 集但记录仍在、仍可溯源。"""
        name = _slug(record_id)
        current = self._store.find(name)
        if current is None:
            raise KeyError(f"five_layer: no record {name}")
        when = _aware(at) if at is not None else self._clock.timestamp()
        _record.invalidate(current, when, _slug(successor) if successor else None)
        self._store.write(current)  # 重新投影 records 表 / FTS5 surface
        return self.record(name)

    def archive_journey(self, companion_id: str, *, at: object = None) -> int:
        """标签切换归档：把该 Journey（companion_id）下所有 active 记录批量软删。

        复用 invalidate() 的单条软删语义（区间结束、本体 + provenance 保留、可溯源），
        只挑 invalid_at 为空的 active 记录处理，因此天然幂等：重复归档第二次返回 0，
        不重复写、不改动已有的 invalid_at。current 不再返回归档项，as_of 历史时点
        仍可见。返回本次实际归档的记录数。
        """
        target = str(companion_id or "").strip()
        if not target:
            return 0
        active = [
            rec for rec in self._store.records(include_invalid=False)
            if str(rec.fields.get(_JOURNEY_FIELD, "")).strip() == target
        ]
        if not active:
            return 0
        when = _aware(at) if at is not None else self._clock.timestamp()
        for rec in active:
            self.invalidate(rec.name, at=when)
        return len(active)

    # ------------------------------------------------------------------ 查询
    def record(self, record_id: str) -> dict:
        found = self._store.find(_slug(record_id))
        if found is None:
            return {}
        return {
            "name": found.name, "abstract": found.abstract,
            "valid_from": found.valid_from, "invalid_at": found.invalid_at,
            "superseded_by": found.superseded_by, "weight": found.weight,
            "provenance": list(found.provenance), "status": found.status,
        }

    def current(self, query: str, as_of: str | None = None, limit: int = 8) -> list[dict]:
        """查当前事实：supersede 后只返回最新；as_of 历史时点能看到被取代的旧事实。

        判据复用 agent_memory.core.recall.Recall._current_at()（同一套双时态规则），
        文本匹配走 records 表的 abstract 子串——中文专名召回是 WP-A 的职责，这里
        只保证「当前/历史」语义正确。
        """
        needle = str(query or "").strip().lower()
        moment = _timestamp.parse(as_of) if as_of else None
        eligibility = Recall(self._store)  # 库的双时态判据，不自己发明
        found: list[dict] = []
        with Database(self._store.layout).connect() as connection:
            rows = connection.execute(
                "SELECT name, abstract, type, valid_from, invalid_at, superseded_by, "
                "weight, provenance FROM records ORDER BY valid_from DESC, name"
            ).fetchall()
            for row in rows:
                if needle and needle not in str(row["abstract"]).lower():
                    continue
                if moment is None:
                    if row["invalid_at"]:
                        continue  # 当前：区间已结束（被 supersede 或软删）的一律不返回
                elif not eligibility._current_at(row, moment):
                    continue  # 历史：交库的 _current_at 判 validity interval
                found.append({
                    "name": str(row["name"]), "abstract": str(row["abstract"]),
                    "type": str(row["type"]), "valid_from": str(row["valid_from"]),
                    "invalid_at": row["invalid_at"], "superseded_by": row["superseded_by"],
                    "weight": float(row["weight"]),
                    "provenance": [item for item in str(row["provenance"]).split("|") if item],
                    "status": "invalid" if row["invalid_at"] else "active",
                    "source": "five_layer",
                })
        return found[:limit]

    def present_in_current(self, record_id: str, as_of: str | None = None) -> bool:
        """按记录名（而非文本查询）确认它是否出现在 current / as_of 结果里。

        与 current() 共用 agent-memory-core 的双时态判据；删除同步后用它精确断言
        「current 不返已删项」，历史时点则返回 True（仍可溯源）。
        """
        name = _slug(record_id)
        moment = _timestamp.parse(as_of) if as_of else None
        eligibility = Recall(self._store)
        with Database(self._store.layout).connect() as connection:
            row = connection.execute(
                "SELECT name, valid_from, invalid_at FROM records WHERE name = ?", (name,)
            ).fetchone()
        if row is None:
            return False
        if moment is None:
            return not bool(row["invalid_at"])
        return eligibility._current_at(row, moment)

    def trace(self, record_id: str) -> dict:
        """provenance 指针 → 原始消息文本（库 trace.read 的 trace_evidence 门面）。"""
        result = self._store.trace_evidence(_slug(record_id))
        payload = result.as_dict()
        payload["text"] = "\n".join(
            evidence.text or "" for evidence in result.evidence if evidence.text
        )
        payload["message_texts"] = [
            message.get("content", "") or message.get("text", "")
            for message in payload["messages"]
        ] or [evidence.text or "" for evidence in result.evidence if evidence.text]
        return payload

    # ------------------------------------------------------------------ 巩固
    def settle(self) -> dict:
        """调用库的 Manage.sleep()：idle 记忆按 weight.decay_step 衰减，clamp 到 floor。"""
        report = Manage(self._store, self._clock).sleep()
        return report.as_dict()

    def schema_snapshot(self) -> dict:
        """V5 证据：库真实建出的 records 表 + 双 FTS5 surface。"""
        with Database(self._store.layout).connect() as connection:
            tables = [r[0] for r in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )]
            columns = [r[1] for r in connection.execute("PRAGMA table_info(records)")]
        return {
            "root": str(self.root),
            "index_db": str(self._store.layout.index_db),
            "tables": tables,
            "records_columns": columns,
            "fts_surfaces": [t for t in tables if t in ("chunks", "history")],
        }
