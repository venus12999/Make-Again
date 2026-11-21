"""F3 · 开关默认翻转验收（分支 memory-final/switch-defaults）。

覆盖 MEMORY_FINAL_CLOSURE_PLAN.md F3-V1..V3：
  V1 不设 env 时 dual + semantic + intent 默认开，且 dual/semantic 路径真实生效；
  V2 显式设 0/off/false/no 逐字节回退旧（legacy）路径；
  V3 用 eval/recall_corpus.json 跑默认配置 R@K>=0.55 且 neg_inject=0。
F3-V4（全量 pytest）由命令行单独执行并记录输出。

设计说明：默认值在 memory/recall.py 导入时以 os.environ.setdefault 注入，
故 V1/V3 用干净 env 的子进程验证「未设置=开」的真实导入行为；V2 在本进程内
用 monkeypatch 显式置 0，并与 _recall_legacy 直接输出做逐字节比对。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
FLAG_KEYS = ("USE_DUAL_RECALL", "USE_SEMANTIC_EMBEDDING", "USE_QUERY_INTENT", "USE_RECALL_FUSION")
HF_CACHE = r"D:\12412\huggingface"


def _clean_env() -> dict:
    env = os.environ.copy()
    for key in FLAG_KEYS:
        env.pop(key, None)
    env["HF_HOME"] = os.environ.get("F3_HF_HOME", HF_CACHE)
    env["HF_HUB_OFFLINE"] = "1"
    env["MOCK_LLM"] = "true"
    return env


def _freeze(result) -> bytes:
    """把召回结果序列化为稳定字节串（与 test_query_intent.py 的口径一致）。"""
    payload = [
        {
            "id": str(r["entry"].id),
            "content": r["entry"].content,
            "score": repr(r["score"]),
            "recency": repr(r["recency"]),
            "relevance": repr(r["relevance"]),
        }
        for r in result
    ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")


# ---------------------------------------------------------------------------
# V1 —— 干净子进程：未设置 env 时默认开，且 dual + semantic 真实产出
# ---------------------------------------------------------------------------
_V1_CODE = r'''
import os, sys, json, tempfile
from pathlib import Path
from datetime import datetime, timedelta
BACKEND = sys.argv[1]
sys.path.insert(0, BACKEND)
root = tempfile.mkdtemp(prefix="f3-v1-")
os.environ["DATABASE_URL"] = "sqlite:///" + str(Path(root) / "t.sqlite3")
os.environ["DUAL_RECALL_INDEX_DB"] = str(Path(root) / "idx.sqlite3")
os.environ.setdefault("MEDIA_SIGNING_KEY", "x" * 40)
os.environ.setdefault("STORAGE_LOCAL_DIR", str(Path(root) / "media"))
os.environ.setdefault("PUBLIC_BASE_URL", "http://testserver")

from memory.db import Base, SessionLocal
from memory.models import MemoryEntry
from memory import recall as r
from memory import vector_index as vi

Base.metadata.create_all(__import__("memory.db", fromlist=["engine"]).engine)
db = SessionLocal()
e = MemoryEntry(id="f3-v1-m1", user_id="f3-v1-u", type="chat",
                content="豆豆陪伴我度过难过的日子", summary=None, facts=[],
                importance=5.0, ts=datetime.now() - timedelta(days=1))
db.add(e)
db.commit()
indexed = r.index_memory(e, db=db)
out = r.recall(db, "f3-v1-u", "想起豆豆", top_k=3)
print(json.dumps({
    "dual": r._use_dual_recall_enabled(),
    "intent": r._use_query_intent_enabled(),
    "semantic": vi.semantic_enabled(),
    "kind": vi.embedding_kind(),
    "indexed": indexed,
    "n": len(out),
    "details": [dict(o.get("score_details") or {}) for o in out],
}, ensure_ascii=False))
db.close()
'''


def test_f3_v1_defaults_on_and_real_paths():
    proc = subprocess.run(
        [sys.executable, "-c", _V1_CODE, str(BACKEND)],
        env=_clean_env(), capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, "stdout:\n%s\nstderr:\n%s" % (proc.stdout, proc.stderr)
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    print("\n[F3-V1] %s" % json.dumps(data, ensure_ascii=False))

    # 三个开关默认开（不设 env）
    assert data["dual"] is True
    assert data["intent"] is True
    assert data["semantic"] is True
    # dual 路径真实生效：index_memory 成功 + 结果带 vec_score（仅 _recall_dual 产出）
    assert data["indexed"] is True
    assert data["n"] >= 1
    assert "vec_score" in data["details"][0], data["details"]
    # semantic 路径真实生效：端到端 embedding_kind 反映真语义模型
    assert data["details"][0].get("embedding_kind") == "semantic_bge_small_zh", data["details"]


# ---------------------------------------------------------------------------
# V2 —— 显式 0/off/false/no 逐字节回退 legacy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("off_value", ["0", "off", "false", "no"])
def test_f3_v2_explicit_off_byte_for_byte(monkeypatch, off_value):
    from memory.db import SessionLocal
    from memory.models import MemoryEntry
    from memory import recall as recall_mod

    monkeypatch.setenv("USE_DUAL_RECALL", off_value)
    monkeypatch.setenv("USE_QUERY_INTENT", off_value)
    monkeypatch.setenv("USE_RECALL_FUSION", off_value)
    monkeypatch.setenv("USE_SEMANTIC_EMBEDDING", off_value)

    # 逐字节比对必须消除时间抖动：冻结 recall 模块里的 datetime.now()
    fixed_now = datetime(2026, 11, 21, 12, 0, 0)

    class _FrozenDateTime:
        @classmethod
        def now(cls, tz=None):
            return fixed_now

        @staticmethod
        def fromisoformat(value):
            return datetime.fromisoformat(value)

    monkeypatch.setattr(recall_mod, "datetime", _FrozenDateTime)

    db = SessionLocal()
    try:
        e = MemoryEntry(id="f3-v2-m1", user_id="f3-v2-u", type="chat",
                        content="豆豆陪伴我度过难过的日子", summary=None, facts=[],
                        importance=5.0, ts=datetime(2026, 11, 20, 12, 0, 0))
        db.add(e)
        db.commit()
        query = "想起豆豆"
        legacy = _freeze(recall_mod._recall_legacy(db, "f3-v2-u", query, top_k=3))
        got = _freeze(recall_mod.recall(db, "f3-v2-u", query, top_k=3))
        print("\n[F3-V2:%s] legacy=%s got=%s equal=%s" % (
            off_value, legacy[:80], got[:80], legacy == got))
        assert got == legacy
    finally:
        db.close()


# ---------------------------------------------------------------------------
# V3 —— eval/recall_corpus.json 默认配置端到端（R@K / neg_inject）
# ---------------------------------------------------------------------------
_V3_CODE = r'''
import os, sys, json, tempfile
from pathlib import Path
BACKEND = sys.argv[1]
sys.path.insert(0, BACKEND)
sys.path.insert(0, str(Path(BACKEND) / "eval"))
root = tempfile.mkdtemp(prefix="f3-v3-")
os.environ["DATABASE_URL"] = "sqlite:///" + str(Path(root) / "golden.sqlite3")
os.environ["DUAL_RECALL_INDEX_DB"] = str(Path(root) / "idx.sqlite3")
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("MEDIA_SIGNING_KEY", "x" * 40)
os.environ.setdefault("STORAGE_LOCAL_DIR", str(Path(root) / "media"))
os.environ.setdefault("PUBLIC_BASE_URL", "http://testserver")

from datetime import datetime
from memory.db import Base, engine, _RawSessionLocal
from memory.models import MemoryEntry
from memory import recall as r
import run_recall_v2 as ev

memories, queries = ev.load()
Base.metadata.create_all(engine)
db = _RawSessionLocal()
for m in memories:
    db.add(MemoryEntry(
        id=m["id"], user_id=ev.EVAL_USER, type=m.get("type", "chat"),
        content=m["content"], summary=m.get("summary"), facts=m.get("facts", []),
        importance=float(m.get("importance", 5.0)),
        ts=datetime.fromisoformat(m["ts"].replace("Z", "+00:00")).replace(tzinfo=None),
    ))
db.commit()
indexed = sum(1 for m in memories if r.index_memory(db.get(MemoryEntry, m["id"]), db=db))
m = ev.metrics(queries, lambda q, top_k=3: r.recall(db, ev.EVAL_USER, q, top_k=top_k), 3)
print(json.dumps({
    "indexed": indexed,
    "n_queries": m["n_queries"],
    "n_negative": m["n_negative"],
    "Recall@K": m["Recall@K"],
    "MRR": m["MRR"],
    "neg_inject": m["negative_query_injection_rate"],
}, ensure_ascii=False))
db.close()
'''


def test_f3_v3_golden_corpus_default_config():
    proc = subprocess.run(
        [sys.executable, "-c", _V3_CODE, str(BACKEND)],
        env=_clean_env(), capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, "stdout:\n%s\nstderr:\n%s" % (proc.stdout, proc.stderr)
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    print("\n[F3-V3] %s" % json.dumps(data, ensure_ascii=False))

    assert data["indexed"] == 33
    assert data["Recall@K"] is not None and data["Recall@K"] >= 0.55
    assert data["neg_inject"] == 0.0
