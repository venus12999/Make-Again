#!/usr/bin/env python3
"""Recall Golden V2 评测：用 33 条记忆 + 18 条分级 query 跑双路召回 A/B。

与 V1 runner 的区别：V1 硬编码 3 条 fixture（天花板效应），本脚本读 recall_corpus.json
的完整语料，才有统计意义去决定 USE_DUAL_RECALL / USE_SEMANTIC_EMBEDDING。

指标口径来自 docs/OLD_REPO_QUALITY_METRICS.md：Recall@K / Precision@K / MRR / 无关注入率。
运行：python eval/run_recall_v2.py（在 MakeAgain-backend 目录下）
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
CORPUS = BACKEND / "eval" / "recall_corpus.json"
FIXED_NOW = datetime(2026, 11, 21, 12, 0, 0)
EVAL_USER = "golden-eval-user"


def load():
    d = json.loads(CORPUS.read_text(encoding="utf-8"))
    return d["memories"], d["queries"]


def metrics(queries, recall_fn, top_k):
    per_q = []
    for q in queries:
        expect = [str(x) for x in (q.get("memory_expect") or [])]
        if not expect:  # 负例 query：期望不召回任何记忆，单列
            res = recall_fn(q["memory_query"], top_k=top_k)
            per_q.append({"id": q["id"], "negative": True,
                          "retrieved": [str(r["entry"].id) for r in res]})
            continue
        res = recall_fn(q["memory_query"], top_k=top_k)
        retrieved = [str(r["entry"].id) for r in res]
        hits = [m for m in retrieved if m in expect]
        rank = next((retrieved.index(m) + 1 for m in expect if m in retrieved), None)
        per_q.append({
            "id": q["id"], "negative": False, "expected": expect, "retrieved": retrieved,
            "hits": hits, "first_relevant_rank": rank,
            "recall_at_k": round(len(hits) / len(expect), 4),
            "precision_at_k": round(len(hits) / top_k, 4),
            "reciprocal_rank": round(1.0 / rank, 4) if rank else 0.0,
            "expected_first": q.get("expected_first"),
            "first_is_expected": retrieved[0] == q.get("expected_first") if retrieved else False,
        })
    pos = [p for p in per_q if not p["negative"]]
    neg = [p for p in per_q if p["negative"]]
    n = len(pos)
    return {
        "n_queries": n, "n_negative": len(neg),
        "Recall@K": round(sum(p["recall_at_k"] for p in pos) / n, 4) if n else None,
        "Precision@K": round(sum(p["precision_at_k"] for p in pos) / n, 4) if n else None,
        "MRR": round(sum(p["reciprocal_rank"] for p in pos) / n, 4) if n else None,
        "expected_first_hit_rate": round(sum(p["first_is_expected"] for p in pos) / n, 4) if n else None,
        "negative_query_injection_rate": round(
            sum(1 for p in neg if p["retrieved"]) / len(neg), 4) if neg else None,
        "per_query": per_q,
    }


def main():
    # DATABASE_URL 必须在 import memory.db 之前设成独立临时库，避免污染主 app.db 与重复主键
    _tmp_root = tempfile.mkdtemp(prefix="ma-v2-run-")
    os.environ["DATABASE_URL"] = "sqlite:///" + str(Path(_tmp_root) / "golden.sqlite3")
    os.environ["HF_HOME"] = r"D:\12412\huggingface"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["MOCK_LLM"] = "true"  # 召回不依赖 LLM，避免网络
    os.environ.setdefault("ENVIRONMENT", "development")
    os.environ.setdefault("MEDIA_SIGNING_KEY", "x" * 40)
    os.environ.setdefault("STORAGE_LOCAL_DIR", str(Path(_tmp_root) / "media"))
    os.environ.setdefault("PUBLIC_BASE_URL", "http://testserver")

    memories, queries = load()
    print(f"[V2] corpus: {len(memories)} memories / {len(queries)} queries")

    from memory.db import Base, engine, _RawSessionLocal
    from memory.models import MemoryEntry
    from memory import recall as recall_mod
    Base.metadata.create_all(engine)

    db = _RawSessionLocal()
    for m in memories:
        db.add(MemoryEntry(
            id=m["id"], user_id=EVAL_USER, type=m.get("type", "chat"),
            content=m["content"], summary=m.get("summary"), facts=m.get("facts", []),
            importance=float(m.get("importance", 5.0)),
            ts=datetime.fromisoformat(m["ts"].replace("Z", "+00:00")).replace(tzinfo=None),
        ))
    db.commit()

    idx_root = Path(tempfile.mkdtemp(prefix="ma-v2-idx-"))

    results = {}
    for dual, sem in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        key = f"dual{dual}_sem{sem}"
        try:
            # 每组用独立索引文件，避免 Windows 下 unlink 被 sqlite 连接占用（WinError 32）
            idx_path = idx_root / f"golden_{key}.vecidx.sqlite3"
            os.environ["DUAL_RECALL_INDEX_DB"] = str(idx_path)
            os.environ["USE_DUAL_RECALL"] = str(dual)
            os.environ["USE_SEMANTIC_EMBEDDING"] = str(sem)
            recall_mod.reset_index() if hasattr(recall_mod, "reset_index") else None
            try:
                from memory.vector_index import reset_semantic_state
                reset_semantic_state()
            except Exception:
                pass
            indexed = sum(1 for m in memories
                          if recall_mod.index_memory(db.get(MemoryEntry, m["id"]), db=db))
            m = metrics(queries, lambda q, top_k=3: recall_mod.recall(db, EVAL_USER, q, top_k=top_k), 3)
            m.update({"status": "RUN", "dual": dual, "sem": sem, "indexed": indexed})
            results[key] = m
            print(f"[V2] {key}: R@K={m['Recall@K']} P@K={m['Precision@K']} MRR={m['MRR']} "
                  f"first_hit={m['expected_first_hit_rate']} neg_inject={m['negative_query_injection_rate']} idx={indexed}")
        except Exception as exc:
            results[key] = {"status": "ENV_FAIL", "error": str(exc)[:200]}
            print(f"[V2] {key}: ENV_FAIL {str(exc)[:100]}")

    db.close()
    out = BACKEND.parent / "outputs" / "golden-recall-v2.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[V2] report: {out}")


if __name__ == "__main__":
    main()
