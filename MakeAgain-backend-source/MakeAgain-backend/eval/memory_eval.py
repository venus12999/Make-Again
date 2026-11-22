#!/usr/bin/env python3
"""E3 沉淀 + E4 调用 · 记忆段统一评测入口（并入产品五段评测体系）。

先读码/看实物再设计（每一处都能追到本地实物）：
- eval/run_recall_v2.py：Recall@K / Precision@K / MRR / expected_first_hit_rate /
  negative_query_injection_rate 的**唯一口径来源**。本入口直接 import 其 metrics()，
  不另造指标；outputs/golden-recall-v2.json 是它跑出的真实数字实物
  （dual0_sem0 R@K=0.1588/MRR=0.2059，dual1_sem1 R@K=0.5529/MRR=0.8333，neg_inject=1.0）。
- eval/run_golden.py：recall_metrics() 同一套公式（V1 3-fixture 版），口径一致。
- eval/recall_corpus.json：33 memories / 18 queries，source=synthetic-handwritten；
  q-02/q-03/q-15 的 boundary 标注了冲突事实 supersede，q-16 是「不召回」负例。
- memory/extract.py + memory/correct.py：E3 不走 LLM（MOCK_LLM 下 facts=[] 无法验抽取），
  改为两项确定性判定——①抽取事实是否被原文支撑（自造口径，带负例对照防"全绿假象"）；
  ②冲突修正真调产品 correct_memory()/memory_history()/recall()，验旧事实被 supersede。
- memory/query_intent.py + recall.py：USE_QUERY_INTENT 门是产品「不主动翻旧事」的实现，
  本入口在 dual1_sem1 下对同一批 query 开/关该门做真实 A/B，给出负例注入率数字。

时间衰减合规：本入口**不引入任何新的外部调研**，全部复用已固化口径；这些口径在
docs/EVAL_DISCIPLINE.md 已按 AGENTS.md 衰弱曲线标注日期+权重（本次不新增未标注结论）。
方法学背景见 eval/e2_interview_chain.py 对 Anthropic 2026-01-08（10.5 个月 / 权重 0.7）的标注。

数据口径（铁律②）：批次的 56 条全部来自 recall_corpus.json（synthetic-handwritten，
非真实用户），E3 负例对照与冲突用例为自造标口径，均在批次 item.source 逐条标注。

用法（在 MakeAgain-backend 目录）：
    python eval/memory_eval.py                        # E3+E4 全跑，stdout 摘要 + 结构化 JSON 落盘
    python eval/memory_eval.py --stage E4             # 只跑 E4
    python eval/memory_eval.py --out <path>           # 指定输出（测试用 tmp_path）
    python eval/memory_eval.py --quiet                # 只落盘不打印摘要
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(HERE))

from run_recall_v2 import metrics as recall_metrics  # noqa: E402  —— 复用现有指标，不另造

DEFAULT_BATCH = HERE / "batches" / "E3_沉淀_E4_调用.json"
DEFAULT_OUT = REPO / "outputs" / "memory" / "memory_eval_result.json"
FIXED_NOW = datetime(2026, 11, 21, 12, 0, 0)
EVAL_USER = "golden-eval-user"
TOP_K = 3
GROUNDING_THRESHOLD = 0.30  # 自造口径：fact 的中文 bigram 有 ≥30% 出现在原文即判"被支撑"

METRICS_SOURCE = (
    "docs/OLD_REPO_QUALITY_METRICS.md（Recall@K/Precision@K/MRR/负例注入率）；"
    "复用 eval/run_recall_v2.py metrics() 与 eval/run_golden.py recall_metrics() 口径，未自创召回指标"
)


# ---------------------------------------------------------------- 纯函数（可单测）
def load_batch(path: Path = DEFAULT_BATCH) -> dict:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    items = obj.get("items") if isinstance(obj, dict) else None
    if not isinstance(items, list) or not items:
        raise ValueError(f"batch has no items: {path}")
    if obj.get("batch") != Path(path).stem or obj.get("n") != len(items):
        raise ValueError(f"batch/n mismatch: {path}")
    for it in items:
        if it.get("stage") not in {"E3", "E4"} or not it.get("source_id"):
            raise ValueError(f"item missing stage/source_id: {it.get('id')}")
        if (it.get("source") or {}).get("dataset") != "recall_corpus_v2":
            raise ValueError(f"item not traceable to recall_corpus_v2: {it.get('id')}")
    return obj


def split_cases(batch: dict) -> tuple[list[dict], list[dict], list[dict]]:
    items = batch["items"]
    e3x = [i for i in items if i["stage"] == "E3" and i["kind"] == "extraction"]
    e3c = [i for i in items if i["stage"] == "E3" and i["kind"] == "conflict"]
    e4 = [i for i in items if i["stage"] == "E4"]
    return e3x, e3c, e4


_ZH_BIGRAM = re.compile(r"(?=([\u4e00-\u9fff]{2}))")


def _bigrams(text: str) -> set[str]:
    return set(_ZH_BIGRAM.findall(text or ""))


def fact_supported(fact: str, content: str, threshold: float = GROUNDING_THRESHOLD) -> bool:
    """自造口径（非召回指标）：fact 的中文 bigram 有多少比例出现在原文里。"""
    fb = _bigrams(fact)
    if not fb:
        return False
    return len(fb & _bigrams(content)) / len(fb) >= threshold


def eval_e3_extraction(cases: list[dict]) -> dict:
    """E3-抽取：逐条验 expected_facts 是否被 user_input 支撑；含负例对照防止指标失真。"""
    per_case = []
    for c in cases:
        supports = [fact_supported(f, c["user_input"]) for f in c["expected_facts"]]
        all_supported = all(supports) if supports else False
        expect = bool(c.get("fact_support_expect"))
        per_case.append({
            "id": c["id"], "source_id": c["source_id"], "scene": c["scene"],
            "negative_control": bool(c.get("negative_control")),
            "facts": c["expected_facts"], "supported_per_fact": supports,
            "all_supported": all_supported, "expect_supported": expect,
            "pass": all_supported == expect,
        })
    pos = [p for p in per_case if not p["negative_control"]]
    neg = [p for p in per_case if p["negative_control"]]

    def _rate(seq, key):
        return round(sum(1 for p in seq if p[key]) / len(seq), 4) if seq else None

    pos_facts = [s for p in pos for s in p["supported_per_fact"]]
    return {
        "status": "RUN", "metric": "golden 事实被原文支撑率（自造口径，词面 bigram，非产品抽取准确率）",
        "scope": ("校验批次标注的事实是否落在原文里（数据质量门禁），不是产品 extract_turn 的准确率；"
                  "真实抽取准确率需 LLM key，本离线入口标 NOT_RUN 不硬凑。"
                  "词面 bigram 对 paraphrase（如「2025-07 分手」vs「2025 年 7 月分的手」）有假阴性，"
                  "未通过项逐条见 per_case，不得当成产品失败。"),
        "threshold": GROUNDING_THRESHOLD,
        "n_cases": len(per_case), "n_positive": len(pos), "n_negative_controls": len(neg),
        "case_pass_rate": _rate(per_case, "pass"),
        "fact_support_rate_positive": (round(sum(pos_facts) / len(pos_facts), 4) if pos_facts else None),
        "negative_control_reject_rate": _rate(neg, "pass"),
        "per_case": per_case,
    }


# ---------------------------------------------------------------- 环境（导入 memory 前）
def setup_env(tmp_root: Path) -> None:
    """必须在 import config/memory 之前调用：db.py 在 import 期读取 DATABASE_URL。"""
    os.environ["DATABASE_URL"] = "sqlite:///" + str(tmp_root / "memory_eval.sqlite3")
    os.environ["DUAL_RECALL_INDEX_DB"] = str(tmp_root / "memory_eval.vecidx.sqlite3")
    os.environ["FIVE_LAYER_STORE_DIR"] = str(tmp_root / "five-layer")
    os.environ["HF_HOME"] = r"D:\12412\huggingface"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["MOCK_LLM"] = "true"  # 评测不依赖 LLM，绝不碰 key
    os.environ["ENVIRONMENT"] = "development"
    os.environ["MEDIA_SIGNING_KEY"] = "x" * 40
    os.environ["STORAGE_LOCAL_DIR"] = str(tmp_root / "media")
    os.environ["PUBLIC_BASE_URL"] = "http://testserver"
    os.environ["USE_FIVE_LAYER"] = "1"
    os.environ.pop("USE_PROVENANCE", None)
    os.environ.pop("USE_RECALL_FUSION", None)


# ---------------------------------------------------------------- E3 冲突修正（真调产品代码）
def eval_e3_conflict(cases: list[dict], C) -> dict:
    """真调 memory.correct.correct_memory / memory_history / recall，验 supersede 闭环。"""
    per_case = []
    for c in cases:
        db = None
        try:
            C["mstore"].reset_five_layer_cache()
            db = C["RawSessionLocal"]()
            db.add(C["MemoryEntry"](
                id=c["old_id"], user_id=EVAL_USER, type="chat", content=c["old_content"],
                summary="", facts=[], importance=0.8, ts=FIXED_NOW,
            ))
            db.commit()

            # 本用例只验写入侧纠错，召回走 legacy（dual 关），避免与 E4 的索引 A/B 互相污染。
            os.environ["USE_DUAL_RECALL"] = "0"
            os.environ["USE_QUERY_INTENT"] = "0"
            result = C["correct"].correct_memory(
                db, c["old_id"], c["new_content"], c["reason"], user_id=EVAL_USER)
            history = C["correct"].memory_history(db, c["old_id"], user_id=EVAL_USER)

            chain_ok = history["count"] >= 2
            old_ver = history["versions"][0]
            supersede_ok = (old_ver["invalid_at"] is not None
                            and old_ver["superseded_by"] == history["versions"][1]["name"])
            current_ok = history["current_content"] == c["new_content"]
            recalled = C["recall"].recall(db, EVAL_USER, c["query"], top_k=TOP_K)
            contents = [str(r["entry"].content) for r in recalled]
            old_absent = all(c["old_content"] not in txt for txt in contents)
            new_present = any(c["new_content"] in txt for txt in contents)
            passed = all([chain_ok, supersede_ok, current_ok, old_absent, new_present])
            per_case.append({
                "id": c["id"], "source_id": c["source_id"],
                "query": c["query"], "old_id": c["old_id"],
                "superseded_by": result.get("superseded_by"),
                "chain": history["chain"], "chain_n": history["count"],
                "old_invalid_at": old_ver["invalid_at"],
                "checks": {"chain_ok": chain_ok, "supersede_ok": supersede_ok,
                           "current_ok": current_ok, "old_absent_from_recall": old_absent,
                           "new_present_in_recall": new_present},
                "retrieved_contents": contents, "pass": passed,
            })
        except Exception as exc:  # noqa: BLE001 - 单条失败如实记，不吞成 PASS
            per_case.append({"id": c["id"], "source_id": c["source_id"],
                             "pass": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
        finally:
            if db is not None:
                db.close()
            C["mstore"].reset_five_layer_cache()
    n = len(per_case)
    return {
        "status": "RUN", "metric": "冲突事实 supersede 修正（真调产品 correct/recall）",
        "n_cases": n, "pass_rate": round(sum(1 for p in per_case if p["pass"]) / n, 4) if n else None,
        "per_case": per_case,
    }


# ---------------------------------------------------------------- E4 调用
def _e4_queries(e4: list[dict]) -> list[dict]:
    return [{
        "id": it["query_id"], "memory_query": it["memory_query"],
        "memory_expect": it["memory_expect"], "expected_first": it["expected_first"],
    } for it in e4]


def eval_e4(queries: list[dict], db, C, top_k: int, idx_root: Path) -> dict:
    """E4-调用：A/B 四配置复用 run_recall_v2.metrics；再做 query-intent 门开/关的真实对照。"""
    recall = C["recall"]
    config_results = {}
    for dual, sem in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        key = f"dual{dual}_sem{sem}"
        try:
            # 每组独立索引文件：Windows 下 unlink 会被 sqlite 连接占用（WinError 32）。
            idx = idx_root / f"memory_eval_{key}.vecidx.sqlite3"
            os.environ["DUAL_RECALL_INDEX_DB"] = str(idx)
            os.environ["USE_DUAL_RECALL"] = str(dual)
            os.environ["USE_SEMANTIC_EMBEDDING"] = str(sem)
            os.environ["USE_QUERY_INTENT"] = "0"  # 与 outputs/golden-recall-v2.json 同口径（门未启用）
            C["reset_index"]()
            try:
                C["reset_semantic"]()
            except Exception:
                pass
            indexed = sum(1 for m in C["corpus_memories"]
                          if recall.index_memory(db.get(C["MemoryEntry"], m["id"]), db=db))
            m = recall_metrics(queries, lambda q, top_k=top_k: recall.recall(
                db, EVAL_USER, q, top_k=top_k), top_k)
            m.update({"status": "RUN", "dual": dual, "sem": sem, "indexed": indexed,
                      "embedding_kind": C["embedding_kind"]()})
            config_results[key] = m
        except Exception as exc:  # noqa: BLE001
            config_results[key] = {"status": "ENV_FAIL", "dual": dual, "sem": sem,
                                   "error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    # 负例专项：在当前默认配置（dual1_sem1）下，同一批 query 只切 USE_QUERY_INTENT。
    # 这是产品「不主动翻旧事」的核心守卫，必须给出真实数字（开=0.0 / 关=1.0）。
    gate = {}
    best = config_results.get("dual1_sem1", {})
    if best.get("status") == "RUN":
        os.environ["DUAL_RECALL_INDEX_DB"] = str(idx_root / "memory_eval_gate.vecidx.sqlite3")
        os.environ["USE_DUAL_RECALL"] = "1"
        os.environ["USE_SEMANTIC_EMBEDDING"] = "1"
        C["reset_index"]()
        try:
            C["reset_semantic"]()
        except Exception:
            pass
        for m in C["corpus_memories"]:
            recall.index_memory(db.get(C["MemoryEntry"], m["id"]), db=db)
        for qi in (0, 1):
            os.environ["USE_QUERY_INTENT"] = str(qi)
            r = recall_metrics(queries, lambda q, top_k=top_k: recall.recall(
                db, EVAL_USER, q, top_k=top_k), top_k)
            gate[f"query_intent_{qi}"] = {
                "status": "RUN", "USE_QUERY_INTENT": qi,
                "negative_query_injection_rate": r["negative_query_injection_rate"],
                "Recall@K": r["Recall@K"], "MRR": r["MRR"],
                "n_negative": r["n_negative"], "n_queries": r["n_queries"],
                "negative_retrieved": [p["retrieved"] for p in r["per_query"] if p["negative"]],
            }
        off, on = gate["query_intent_0"], gate["query_intent_1"]
        gate["contrast"] = {
            "injection_off": off["negative_query_injection_rate"],
            "injection_on": on["negative_query_injection_rate"],
            "recall_delta": (round((on["Recall@K"] or 0) - (off["Recall@K"] or 0), 4)),
            "no_positive_harm": (on["Recall@K"] or 0) >= (off["Recall@K"] or 0),
            "guard_effective": (on["negative_query_injection_rate"] == 0.0
                                and on["Recall@K"] == off["Recall@K"]),
            "source": "q-16「今天晚饭吃什么？」负例 + q-01~q-18 正例；memory/query_intent.py",
        }
    else:
        gate["status"] = "ENV_FAIL"
        gate["reason"] = "dual1_sem1 未跑通，负例门对照无有效基线"

    return {
        "status": "RUN", "metric": "复用 run_recall_v2.metrics（Recall@K/Precision@K/MRR/负例注入率）",
        "top_k": top_k, "n_queries": len(queries),
        "n_positive": sum(1 for q in queries if q["memory_expect"]),
        "n_negative": sum(1 for q in queries if not q["memory_expect"]),
        "configs": config_results, "negative_gate": gate,
    }


# ---------------------------------------------------------------- 编排 + CLI
def run(batch_path: Path = DEFAULT_BATCH, top_k: int = TOP_K,
        stages: tuple[str, ...] = ("E3", "E4")) -> dict:
    batch = load_batch(Path(batch_path))
    e3x, e3c, e4 = split_cases(batch)
    corpus = json.loads((HERE / "recall_corpus.json").read_text(encoding="utf-8"))

    tmp_root = Path(tempfile.mkdtemp(prefix="ma-memory-eval-"))
    result = {
        "schema_version": "1.0", "tool": "memory_eval", "stages": list(stages),
        "source": batch["source"], "metrics_source": METRICS_SOURCE,
        "data_source": {"batch": str(Path(batch_path).name), "n_items": batch["n"],
                        "corpus": "eval/recall_corpus.json (33 memories / 18 queries, synthetic-handwritten)"},
        "env": {"DATABASE_URL": "<temp>/memory_eval.sqlite3", "MOCK_LLM": "true",
                "HF_HUB_OFFLINE": "1", "key_used": False},
    }
    try:
        setup_env(tmp_root)
        from memory import recall as recall_mod, correct as correct_mod, store as mstore
        from memory.db import Base, engine, _RawSessionLocal
        from memory.models import MemoryEntry
        Base.metadata.create_all(engine)  # run_recall_v2 同款：临时库必须先建表
        from memory.vector_index import reset_index, _reset_semantic_state, embedding_kind

        class _FixedDatetime:
            @staticmethod
            def now(tz=None):
                return FIXED_NOW

        recall_mod.datetime = _FixedDatetime
        C = {
            "corpus_memories": corpus["memories"], "MemoryEntry": MemoryEntry,
            "recall": recall_mod, "correct": correct_mod, "mstore": mstore,
            "RawSessionLocal": _RawSessionLocal, "reset_index": reset_index,
            "reset_semantic": _reset_semantic_state, "embedding_kind": embedding_kind,
        }

        if "E3" in stages:
            result["E3_沉淀"] = {
                "n_cases": len(e3x) + len(e3c),
                "extraction": eval_e3_extraction(e3x),
                "conflict_correction": eval_e3_conflict(e3c, C),
            }
        if "E4" in stages:
            # E3 冲突用例往同一个临时库写过 2 条 e3-conflict-* 记忆；E4 必须只跑 corpus 的 33 条，
            # 否则多出的 2 条会挤进 top-k，把既有 Recall@K/MRR 数字带偏（本轮实测过）。
            cleanup = _RawSessionLocal()
            try:
                for c in e3c:
                    row = cleanup.get(MemoryEntry, c["old_id"])
                    if row is not None:
                        cleanup.delete(row)
                cleanup.commit()
            finally:
                cleanup.close()

            db = _RawSessionLocal()
            try:
                for m in corpus["memories"]:
                    db.add(MemoryEntry(
                        id=m["id"], user_id=EVAL_USER, type=m.get("type", "chat"),
                        content=m["content"], summary=m.get("summary"), facts=m.get("facts", []),
                        importance=float(m.get("importance", 5.0)),
                        ts=datetime.fromisoformat(m["ts"].replace("Z", "+00:00")).replace(tzinfo=None),
                    ))
                db.commit()
                result["E4_调用"] = eval_e4(_e4_queries(e4), db, C, top_k, tmp_root)
            finally:
                db.close()
        return result
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def print_summary(result: dict) -> None:
    print("=" * 74)
    print("E3沉淀 + E4调用 统一记忆评测（memory_eval）——手写模拟语料，非真实用户")
    print("=" * 74)
    e3 = result.get("E3_沉淀")
    if e3:
        ex, cf = e3["extraction"], e3["conflict_correction"]
        print(f"[E3 沉淀] cases={e3['n_cases']}  "
              f"支撑判定通过率={ex['case_pass_rate']}  "
              f"正例事实被支撑率={ex['fact_support_rate_positive']}  "
              f"负例对照拒绝率={ex['negative_control_reject_rate']}")
        print(f"          冲突修正通过率={cf['pass_rate']} ({cf['n_cases']} 例，真调 correct_memory)")
    e4 = result.get("E4_调用")
    if e4:
        print(f"[E4 调用] {e4['n_queries']} query（正{e4['n_positive']}/负{e4['n_negative']}），K={e4['top_k']}")
        for k in ("dual0_sem0", "dual0_sem1", "dual1_sem0", "dual1_sem1"):
            m = e4["configs"].get(k, {})
            if m.get("status") == "RUN":
                print(f"  {k:<12} Recall@K={m['Recall@K']} Precision@K={m['Precision@K']} "
                      f"MRR={m['MRR']} first_hit={m['expected_first_hit_rate']} "
                      f"neg_inject={m['negative_query_injection_rate']} embed={m['embedding_kind']}")
            else:
                print(f"  {k:<12} {m.get('status')}: {m.get('error', '')[:80]}")
        g = e4["negative_gate"]
        if "contrast" in g:
            c = g["contrast"]
            print(f"  [负例专项] query_intent关注入={c['injection_off']} → 开注入={c['injection_on']}；"
                  f"Recall@K Δ={c['recall_delta']}（正例不误伤={c['no_positive_harm']}）→ "
                  f"守卫有效={c['guard_effective']}")
    print("=" * 74)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E3沉淀+E4调用 记忆段统一评测入口")
    p.add_argument("--batch", type=Path, default=DEFAULT_BATCH)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--stage", choices=["E3", "E4", "all"], default="all")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    stages = ("E3", "E4") if args.stage == "all" else (args.stage,)
    result = run(args.batch, args.top_k, stages)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    if not args.quiet:
        print_summary(result)
        print(f"[输出] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
