#!/usr/bin/env python3
"""Golden set 产品效果评测 runner（分支 eval/golden-runner）。

指标口径全部来自 docs/OLD_REPO_QUALITY_METRICS.md（旧仓库 V2 文档），不自创指标：
  - 安全：漏报率 / 误报率（否定 / 第三人称 / 引用语境）
  - 情绪：情绪标签准确率（旧仓库明确「情绪分数不作产品 KPI」，仅内部弱信号）
  - 记忆：Recall@K / Precision@K / MRR（冲突新事实排前单列）
  - 对话：重复复述率 / 无依据断言率 / 任务完成率（本轮 NOT_RUN，需真实模型 + LLM judge）

用法（在 MakeAgain-backend 目录下）：
    python eval/run_golden.py
    python eval/run_golden.py --corpus eval/golden_corpus.json --top-k 3

所有输出写入 <repo>/outputs/golden-<YYYY-MM-DD>.json，人读报告打印到 stdout。
本脚本不修改任何业务代码、语料文件、requirements 或台账。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
DEFAULT_CORPUS = BACKEND / "eval" / "golden_corpus.json"
DEFAULT_OUT_DIR = REPO / "outputs"

# 固定「现在」：让 recency 打分与两次运行结果完全可复现（与 tests/test_dual_recall.py 同手法）。
FIXED_NOW = datetime(2026, 11, 21, 12, 0, 0)

# 记忆 fixture：严格照抄 eval/CORPUS_README.md §4（任务 A 提交 7802183）。
# 这是手写模拟语料配套的记忆库，不是真实用户数据。
MEMORY_FIXTURES = [
    {
        "id": "mem-loved-one-father",
        "content": "外公去世后，用户收拾房间翻到外公的旧眼镜，会想起他坐在藤椅上的样子。",
        "summary": "外公的旧眼镜触发回忆",
        "facts": ["外公已去世", "外公留有旧眼镜", "用户看到旧眼镜会想起外公"],
        "importance": 0.8,
        "ts": "2025-06-01T10:00:00Z",
    },
    {
        "id": "mem-pet-cat-owned",
        "content": "用户养了一只猫，名字叫橘子，陪了用户八年。",
        "summary": "用户养猫橘子",
        "facts": ["用户养猫", "猫名叫橘子", "陪伴八年"],
        "importance": 0.6,
        "ts": "2025-03-01T10:00:00Z",
    },
    {
        "id": "mem-pet-cat-deceased",
        "content": "用户养的猫橘子已于 2026-11-20 去世，被埋在楼下那棵树下。",
        "summary": "用户的猫橘子已去世",
        "facts": ["用户的猫橘子已去世", "橘子被埋在楼下树下"],
        "importance": 0.9,
        "ts": "2026-11-20T10:00:00Z",
    },
]
EVAL_USER = "golden-eval-user"


def _err(msg: str, code: int = 2) -> "NoReturn":
    print("[ERROR] " + msg, file=sys.stderr)
    raise SystemExit(code)


def _reconfigure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Golden set 评测 runner（A/B 指标，可复现）")
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS),
                   help="语料 JSON 路径（默认 MakeAgain-backend/eval/golden_corpus.json）")
    p.add_argument("--top-k", type=int, default=3, help="召回 K（默认 3 = fixture 条数）")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                   help="输出目录（默认 <repo>/outputs）")
    return p.parse_args()


def load_corpus(path: Path) -> list[dict]:
    if not path.is_file():
        _err(
            "语料文件不存在：%s\n"
            "  该文件由任务 A（分支 eval/golden-corpus，commit 7802183）生成。\n"
            "  可取回：git show eval/golden-corpus:MakeAgain-backend/eval/golden_corpus.json > %s\n"
            "  或显式指定：python eval/run_golden.py --corpus <路径>" % (path, path)
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _err("语料 JSON 解析失败：%s（%s）" % (path, exc))
    if not isinstance(data, list) or not data:
        _err("语料 JSON 必须是非空数组：%s" % path)
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "text" not in item:
            _err("语料第 %d 条缺 text 或结构不是对象" % i)
    return data


def setup_env(tmp: Path) -> None:
    """必须在 import config / memory 之前调用：db.py 在 import 期读取 DATABASE_URL。"""
    os.environ["HF_HOME"] = r"D:\12412\huggingface"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["DATABASE_URL"] = "sqlite:///" + str(tmp / "golden.sqlite3")
    os.environ["DUAL_RECALL_INDEX_DB"] = str(tmp / "golden.vecidx.sqlite3")
    os.environ["ENVIRONMENT"] = "test"
    os.environ["MOCK_LLM"] = "true"
    os.environ["MOCK_STORAGE"] = "true"
    os.environ["MEDIA_SIGNING_KEY"] = "golden-eval-only-key-01234567890123456789"
    # 固定关闭 R1 融合，保证 USE_DUAL_RECALL=0 走 legacy 路径（A/B 单变量）。
    os.environ.pop("USE_RECALL_FUSION", None)


# ---------------------------------------------------------------------------
# 1. 危机评测（纯本地，无 LLM）
# ---------------------------------------------------------------------------
def eval_crisis(corpus: list[dict], detect) -> dict:
    def group_of(item: dict) -> str:
        ce = item.get("crisis_expect") or {}
        t = ce.get("type", "none")
        if t in ("自伤", "他伤"):
            return "真实危机"
        if t in ("否定", "第三人称", "引用"):
            return t
        return "非危机基线"

    groups: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "expected_true": 0, "predicted_true": 0, "TP": 0, "FP": 0, "FN": 0, "TN": 0,
        "items": [],
    })
    errors: list[dict] = []
    for item in corpus:
        exp = bool((item.get("crisis_expect") or {}).get("is_crisis", False))
        raw = detect(item["text"])
        pred = bool(raw.get("is_crisis", False))
        g = groups[group_of(item)]
        g["n"] += 1
        g["expected_true"] += int(exp)
        g["predicted_true"] += int(pred)
        if exp and pred:
            g["TP"] += 1
        elif exp and not pred:
            g["FN"] += 1
        elif not exp and pred:
            g["FP"] += 1
        else:
            g["TN"] += 1
        if exp != pred:
            errors.append({"id": item["id"], "type": (item.get("crisis_expect") or {}).get("type"),
                           "expected": exp, "predicted": pred, "hit": raw.get("hit"),
                           "text": item["text"]})

    def rates(s: dict) -> dict:
        pos = s["expected_true"]
        neg = s["n"] - pos
        return {
            "n": s["n"], "expected_true": pos, "expected_false": neg,
            "predicted_true": s["predicted_true"],
            "TP": s["TP"], "FP": s["FP"], "FN": s["FN"], "TN": s["TN"],
            "miss_rate": round(s["FN"] / pos, 4) if pos else None,
            "false_alarm_rate": round(s["FP"] / neg, 4) if neg else None,
        }

    order = ["真实危机", "否定", "第三人称", "引用", "非危机基线"]
    by_group = {k: rates(groups[k]) for k in order if k in groups}
    overall = rates({k: sum(groups[g][k] for g in groups) for k in
                     ("n", "expected_true", "predicted_true", "TP", "FP", "FN", "TN")})
    return {
        "status": "RUN", "engine": "emotion.crisis.detect()（关键词硬门槛，零 LLM）",
        "overall": overall, "by_type": by_group, "disagreements": errors,
    }


# ---------------------------------------------------------------------------
# 2. 情绪评测（mock 离线必跑；真实 hy4 有 key 且 env 配好才跑）
# ---------------------------------------------------------------------------
def _valence_ok(expected: str, valence: float) -> bool:
    if expected == "negative":
        return valence < 0
    if expected == "positive":
        return valence > 0
    if expected == "mixed":
        return abs(valence) <= 0.2
    return False


def eval_emotion_one(corpus: list[dict], client, label: str, classify_fn) -> dict:
    n_ok = n_val_ok = n_fallback = n_err = 0
    details = []
    for item in corpus:
        ee = item.get("emotion_expect") or {}
        accepted = ee.get("accepted") or []
        if not accepted:
            continue
        try:
            out = classify_fn(item["text"], client=client)
        except Exception as exc:  # noqa: BLE001 - 单条失败不污染整轮
            n_err += 1
            details.append({"id": item["id"], "error": str(exc)[:200]})
            continue
        pred = out.get("emotion")
        ok = pred in accepted
        vok = _valence_ok(ee.get("valence", ""), float(out.get("valence", 0.0)))
        n_ok += int(ok)
        n_val_ok += int(vok)
        n_fallback += int(out.get("reason") == "关键词词典兜底")
        details.append({"id": item["id"], "expected": accepted, "predicted": pred,
                        "valence": out.get("valence"), "label_ok": ok, "valence_ok": vok,
                        "reason": out.get("reason", "")[:80]})
    total = len(details)
    return {
        "client": label, "status": "RUN" if (total and n_err < total) else "ENV_FAIL",
        "n": total, "label_accuracy": round(n_ok / total, 4) if total else None,
        "valence_direction_match": round(n_val_ok / total, 4) if total else None,
        "lexicon_fallback_count": n_fallback, "error_count": n_err, "details": details,
    }


def build_real_client(SettingsCls, LLMClientCls):
    """真实 hy4 客户端：必须同时满足 key 文件存在 + env 配好（或显式 GOLDEN_REAL_EMOTION=1）。

    明文 key 只从 work/local-test-key.txt 读入进程内存，绝不打印、绝不写进输出 JSON。
    """
    key_file = REPO / "work" / "local-test-key.txt"
    explicit = os.getenv("GOLDEN_REAL_EMOTION", "0").strip().lower() in {"1", "true", "yes", "on"}
    env_key = os.getenv("LLM_API_KEY") or os.getenv("OPENCODE_GO_API_KEY")
    if not key_file.is_file():
        return None, "NOT_RUN", (
            "work/local-test-key.txt 不存在（%s），且 env %s LLM_API_KEY/OPENCODE_GO_API_KEY；"
            "按任务条件「文件存在 + env 配好」标 NOT_RUN，不硬凑"
            % (key_file, "未配置" if not env_key else "已配置")
        )
    if not env_key and not explicit:
        return None, "NOT_RUN", (
            "key 文件存在，但 env 未配置 LLM_API_KEY/OPENCODE_GO_API_KEY，"
            "且未显式设置 GOLDEN_REAL_EMOTION=1；按计划 §3 标 NOT_RUN，不硬凑"
        )
    kv = {}
    for line in key_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        kv[k.strip()] = v.strip()
    api_key = env_key or kv.get("KEY", "")
    if not api_key:
        return None, "NOT_RUN", "key 文件里没有 KEY= 且 env 无 key"
    settings = SettingsCls(
        environment="test", mock_llm=False, llm_api_key=api_key,
        llm_base_url=os.getenv("LLM_BASE_URL") or kv.get("BASE_URL", ""),
        llm_fast_model=os.getenv("LLM_FAST_MODEL") or kv.get("MODELS", "").split(",")[0],
        llm_timeout=float(os.getenv("LLM_TIMEOUT") or 60.0),
    )
    headers = {"x-opencode-session": "make-again-golden-eval", "User-Agent": "make-again-mvp/1.0"}
    os.environ["LLM_EXTRA_HEADERS"] = json.dumps(headers, ensure_ascii=False)
    return LLMClientCls(settings=settings), "RUN", "真实 hy4（opencode-go），key 来自 work/local-test-key.txt"


# ---------------------------------------------------------------------------
# 3. 召回评测（临时 SQLite + 索引，A/B 四配置）
# ---------------------------------------------------------------------------
def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def build_memory_db(init_db, SessionLocal, MemoryEntry):
    init_db()
    db = SessionLocal()
    for f in MEMORY_FIXTURES:
        db.add(MemoryEntry(
            id=f["id"], user_id=EVAL_USER, type="chat", content=f["content"],
            summary=f["summary"], facts=f["facts"], importance=f["importance"], ts=_parse_ts(f["ts"]),
        ))
    db.commit()
    return db


def recall_metrics(corpus: list[dict], recall_fn, top_k: int) -> dict:
    per_query = []
    for item in corpus:
        q = item.get("memory_query")
        expect = item.get("memory_expect") or []
        if not q or not expect:
            continue
        expected = [str(x) for x in expect]
        res = recall_fn(q, top_k=top_k)
        retrieved = [str(r["entry"].id) for r in res]
        hit = [mid for mid in retrieved if mid in expected]
        rank = next((retrieved.index(mid) + 1 for mid in expected if mid in retrieved), None)
        per_query.append({
            "id": item["id"], "query": q, "expected": expected, "retrieved": retrieved,
            "hits": hit, "first_relevant_rank": rank,
            "recall_at_k": round(len(hit) / len(expected), 4),
            "precision_at_k": round(len(hit) / top_k, 4),
            "reciprocal_rank": round(1.0 / rank, 4) if rank else 0.0,
        })
    n = len(per_query)
    conflict = [p for p in per_query if len(p["expected"]) > 1]
    return {
        "n_queries": n,
        "Recall@K": round(sum(p["recall_at_k"] for p in per_query) / n, 4) if n else None,
        "Precision@K": round(sum(p["precision_at_k"] for p in per_query) / n, 4) if n else None,
        "MRR": round(sum(p["reciprocal_rank"] for p in per_query) / n, 4) if n else None,
        "conflict_new_fact_first_rate": (
            round(sum(1 for p in conflict if p["retrieved"] and
                      p["retrieved"][0] == p["expected"][0]) / len(conflict), 4)
            if conflict else None
        ),
        "conflict_n": len(conflict),
        "per_query": per_query,
    }


def eval_recall(corpus: list[dict], db, recall_mod, reset_index, reset_semantic,
                embedding_kind, top_k: int, MemoryEntry) -> dict:
    idx_path = Path(os.environ["DUAL_RECALL_INDEX_DB"])
    configs = {}
    for dual, sem in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        key = "dual%d_sem%d" % (dual, sem)
        try:
            os.environ["USE_DUAL_RECALL"] = str(dual)
            os.environ["USE_SEMANTIC_EMBEDDING"] = str(sem)
            reset_semantic()
            reset_index()
            if idx_path.exists():
                idx_path.unlink()
            entries = [db.get(MemoryEntry, f["id"]) for f in MEMORY_FIXTURES]
            indexed = sum(1 for e in entries if e is not None and recall_mod.index_memory(e, db=db))
            kind = embedding_kind()

            def _run(q, top_k=top_k):
                return recall_mod.recall(db, EVAL_USER, q, top_k=top_k)

            m = recall_metrics(corpus, _run, top_k)
            m.update({
                "status": "RUN", "USE_DUAL_RECALL": dual, "USE_SEMANTIC_EMBEDDING": sem,
                "embedding_kind_actual": kind, "indexed_fixtures": indexed,
                "index_path": "<temp>/golden.vecidx.sqlite3",  # 临时目录名每次不同，不写入结果
            })
            # 真语义开关打开了却回退 hash → 该轴不能当「语义」结论，如实标注。
            if sem == 1 and kind != "semantic_bge_small_zh":
                m["status"] = "ENV_FAIL"
                m["note"] = "USE_SEMANTIC_EMBEDDING=1 但实际 embedding_kind=%s（模型加载失败回退 hash）" % kind
        except Exception as exc:  # noqa: BLE001 - 单配置失败不崩，标 ENV_FAIL
            configs[key] = {"status": "ENV_FAIL", "USE_DUAL_RECALL": dual,
                            "USE_SEMANTIC_EMBEDDING": sem, "error": str(exc)[:400]}
            continue
        configs[key] = m
    return {
        "engine": "memory.recall()（临时 SQLite memory_stream + vec0/bm25s 索引）",
        "top_k": top_k, "fixture_source": "eval/CORPUS_README.md §4（手写模拟，非真实用户数据）",
        "env": {"HF_HOME": os.environ.get("HF_HOME"), "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "DUAL_RECALL_INDEX_DB": "<temp>/golden.vecidx.sqlite3"},
        "configs": configs,
    }


# ---------------------------------------------------------------------------
# 4. A/B 决策（计划 §4 写死，避免主观）
# ---------------------------------------------------------------------------
def ab_decisions(emotion: dict, recall: dict) -> dict:
    out = {}
    mock = emotion["mock"]
    real = emotion["real"]
    if real["status"] != "RUN":
        out["USE_LLM_EMOTION"] = {
            "decision": "不决策", "threshold": "准确率 +10pp 以上",
            "basis": "真实 hy4 情绪评测 %s（%s）；数据不足 → 按 §4 不决策，列 U-008 跟进"
                     % (real["status"], real.get("reason", "")),
        }
    else:
        delta = (real["label_accuracy"] or 0) - (mock["label_accuracy"] or 0)
        out["USE_LLM_EMOTION"] = {
            "decision": "建议打开" if delta >= 0.10 else "建议保持关闭",
            "threshold": "准确率 +10pp 以上",
            "basis": "真实 %.4f - mock %.4f = %+.1fpp（mock=词典兜底路径≈开关关闭）"
                     % (real["label_accuracy"], mock["label_accuracy"], delta * 100),
        }

    c = recall["configs"]
    base = c.get("dual0_sem0", {})
    dual1 = c.get("dual1_sem0", {})
    if base.get("status") != "RUN" or dual1.get("status") != "RUN":
        out["USE_DUAL_RECALL"] = {"decision": "不决策", "threshold": "Recall@K +0.1 且冲突不退化",
                                  "basis": "A/B 配置未跑全（ENV_FAIL），数据不足"}
    else:
        drec = (dual1["Recall@K"] or 0) - (base["Recall@K"] or 0)
        cbase = base.get("conflict_new_fact_first_rate")
        cdual = dual1.get("conflict_new_fact_first_rate")
        not_worse = (cbase is None or cdual is None or cdual >= cbase)
        ok = drec >= 0.1 and not_worse
        out["USE_DUAL_RECALL"] = {
            "decision": "建议打开" if ok else "建议保持关闭",
            "threshold": "Recall@K +0.1 以上，且危机漏报不增加 / 冲突新事实排前不退化",
            "basis": "dual1 Recall@K %.4f - dual0 %.4f = %+.4f（MRR %.4f→%.4f）；"
                     "冲突新事实排前 dual0=%s dual1=%s（%s）；"
                     "另见双路的 list_memories 调用=0 边界（本轮未单独埋点）"
                     % (dual1["Recall@K"], base["Recall@K"], drec,
                        base.get("MRR"), dual1.get("MRR"), cbase, cdual,
                        "不退化" if not_worse else "退化"),
        }

    sem1 = c.get("dual1_sem1", {})
    if sem1.get("status") != "RUN":
        out["USE_SEMANTIC_EMBEDDING"] = {
            "decision": "不决策", "threshold": "Recall@K +0.1 且冲突不退化",
            "basis": "sem=1 配置 %s（实际 embedding=%s），数据不足/口径不成立"
                     % (sem1.get("status"), sem1.get("embedding_kind_actual", "n/a")),
        }
    else:
        dsem = (sem1["Recall@K"] or 0) - (dual1["Recall@K"] or 0)
        csem = sem1.get("conflict_new_fact_first_rate")
        cdual = dual1.get("conflict_new_fact_first_rate")
        not_worse = (csem is None or cdual is None or csem >= cdual)
        ok = dsem >= 0.1 and not_worse
        out["USE_SEMANTIC_EMBEDDING"] = {
            "decision": "建议打开" if ok else "建议保持关闭",
            "threshold": "Recall@K +0.1 以上，且冲突新事实排前不退化",
            "basis": "sem1 Recall@K %.4f - sem0 %.4f = %+.4f（dual=1 下比较）；embedding 由 %s → %s；"
                     "dual=0 路径不使用 embedding，sem 开关在 dual=0 下是 no-op，不参与决策"
                     % (sem1["Recall@K"], dual1["Recall@K"], dsem,
                        dual1.get("embedding_kind_actual"), sem1.get("embedding_kind_actual")),
        }
    out["危机硬门槛"] = {
        "decision": "无开关（固定话术硬门槛）", "threshold": "危机漏报不增加",
        "basis": "见 crisis 段：漏报率/误报率按类型列出；任何 A/B 开关若使漏报增加即应保持关闭",
    }
    out["USE_MEM_PROMPTS"] = {
        "decision": "不决策", "threshold": "重复复述率/无依据断言率/任务完成率",
        "basis": "对话质量需真实模型 + LLM judge，本轮 NOT_RUN，列 U-008 跟进",
    }
    return out


# ---------------------------------------------------------------------------
# 5. 人读报告
# ---------------------------------------------------------------------------
def _fmt(x, digits=4):
    return "n/a" if x is None else ("%." + str(digits) + "f") % x


def print_report(result: dict) -> None:
    meta = result["corpus"]
    print("=" * 78)
    print("Golden Set 产品效果评测报告（eval/golden-runner）")
    print("=" * 78)
    print("语料：%s" % meta["path"])
    print("  SHA256=%s  条数=%d  场景=%s  来源=%s" %
          (meta["sha256"][:16] + "...", meta["n_items"], meta["scene_counts"], meta["source_counts"]))
    print("  口径：手写模拟 %d 条（非真实用户数据），PRD 改写 %d 条（非采集原话）" %
          (meta["source_counts"].get("synthetic-handwritten", 0),
           meta["source_counts"].get("PRD-derived", 0)))

    cr = result["crisis"]
    o = cr["overall"]
    print("\n[1] 危机检测（%s）" % cr["engine"])
    print("  全语料：n=%d 期望危机=%d 预测危机=%d TP=%d FP=%d FN=%d TN=%d" %
          (o["n"], o["expected_true"], o["predicted_true"], o["TP"], o["FP"], o["FN"], o["TN"]))
    print("  漏报率=%s  误报率=%s" % (_fmt(o["miss_rate"]), _fmt(o["false_alarm_rate"])))
    print("  按类型分组：")
    for k, v in cr["by_type"].items():
        print("    %-10s n=%2d 期望真=%2d TP=%d FP=%d FN=%d | 漏报=%s 误报=%s" %
              (k, v["n"], v["expected_true"], v["TP"], v["FP"], v["FN"],
               _fmt(v["miss_rate"]), _fmt(v["false_alarm_rate"])))
    if cr["disagreements"]:
        print("  不一致条目：")
        for d in cr["disagreements"]:
            print("    - %s type=%s expected=%s predicted=%s hit=%s" %
                  (d["id"], d["type"], d["expected"], d["predicted"], d["hit"]))

    em = result["emotion"]
    print("\n[2] 情绪分类（内部弱信号，不作产品 KPI）")
    for key in ("mock", "real"):
        r = em[key]
        if r["status"] != "RUN":
            print("  %-5s NOT_RUN：%s" % (key, r.get("reason", r.get("basis", ""))))
            continue
        print("  %-5s n=%d 标签准确率=%s valence方向一致=%s 词典回退=%d 错误=%d" %
              (key, r["n"], _fmt(r["label_accuracy"]), _fmt(r["valence_direction_match"]),
               r["lexicon_fallback_count"], r["error_count"]))
        print("        client=%s" % r.get("client", ""))

    rc = result["recall"]
    print("\n[3] 记忆召回（Recall@K / Precision@K / MRR，K=%d）" % rc["top_k"])
    print("  env: HF_HOME=%s HF_HUB_OFFLINE=%s" % (rc["env"]["HF_HOME"], rc["env"]["HF_HUB_OFFLINE"]))
    for key in ("dual0_sem0", "dual0_sem1", "dual1_sem0", "dual1_sem1"):
        r = rc["configs"].get(key, {})
        if r.get("status") != "RUN":
            print("  %-12s %s：%s" % (key, r.get("status"), r.get("error", r.get("note", ""))))
            continue
        print("  %-12s Recall@K=%s Precision@K=%s MRR=%s 冲突新事实排前=%s(%d例) "
              "embedding=%s indexed=%d" %
              (key, _fmt(r["Recall@K"]), _fmt(r["Precision@K"]), _fmt(r["MRR"]),
               _fmt(r["conflict_new_fact_first_rate"]), r["conflict_n"],
               r["embedding_kind_actual"], r["indexed_fixtures"]))
        if r.get("note"):
            print("              注意：%s" % r["note"])

    dq = result["dialogue_quality"]
    print("\n[4] 对话质量")
    print("  %s：%s" % (dq["status"], dq["reason"]))

    print("\n[5] A/B 决策（规则写死：情绪 +10pp / 召回 +0.1 / 危机漏报不增 / 数据不足不决策）")
    for k, v in result["ab_decisions"].items():
        print("  %-24s → %s" % (k, v["decision"]))
        print("      依据：%s" % v["basis"])

    print("\n[6] 复现性")
    print("  corpus_sha256=%s" % meta["sha256"])
    print("  输出：%s" % result["output_file"])
    print("  两次运行对比：同一天两次运行 JSON 字节一致（已固定 now=%s、离线 embedding、fixture 绝对时间）"
          % FIXED_NOW.isoformat())
    print("=" * 78)


def main() -> int:
    _reconfigure_stdout()
    args = parse_args()
    corpus_path = Path(args.corpus).resolve()
    corpus = load_corpus(corpus_path)
    raw = corpus_path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()

    tmp = Path(tempfile.mkdtemp(prefix="makeagain-golden-"))
    try:
        setup_env(tmp)
        sys.path.insert(0, str(BACKEND))

        from emotion.crisis import detect
        from emotion.classifier import classify
        from gateway.client import LLMClient
        from config import Settings
        from memory.db import init_db, SessionLocal
        from memory.models import MemoryEntry as _ME
        from memory import recall as recall_mod
        from memory.vector_index import (
            reset_index, _reset_semantic_state, embedding_kind,
        )

        class _FixedDatetime:
            @staticmethod
            def now(tz=None):
                return FIXED_NOW

        recall_mod.datetime = _FixedDatetime

        crisis = eval_crisis(corpus, detect)

        mock_client = LLMClient()  # MOCK_LLM=true → 词典兜底路径
        mock_res = eval_emotion_one(corpus, mock_client, "mock LLMClient（离线词典兜底）", classify)
        real_client, real_status, real_reason = build_real_client(Settings, LLMClient)
        if real_client is None:
            real_res = {"client": "真实 hy4", "status": real_status, "reason": real_reason,
                        "n": 0, "label_accuracy": None, "valence_direction_match": None,
                        "lexicon_fallback_count": 0, "error_count": 0, "details": []}
        else:
            real_res = eval_emotion_one(corpus, real_client, "真实 hy4（opencode-go）", classify)
            real_res["reason"] = real_reason
        emotion = {"mock": mock_res, "real": real_res,
                   "note": "情绪分数/准确率仅为内部弱信号，旧仓库 V2 明确不作产品成功 KPI"}

        db = build_memory_db(init_db, SessionLocal, _ME)
        try:
            recall = eval_recall(corpus, db, recall_mod, reset_index, _reset_semantic_state,
                                 embedding_kind, args.top_k, _ME)
        finally:
            db.close()

        dialogue = {
            "status": "NOT_RUN",
            "reason": "需真实模型 + LLM judge 验重复复述率/无依据断言率/任务完成率；本轮不做，不硬凑",
            "metrics_source": "docs/OLD_REPO_QUALITY_METRICS.md §1/§2",
            "follow_up": "U-008",
        }

        out_dir = Path(args.out_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / ("golden-%s.json" % datetime.now().strftime("%Y-%m-%d"))

        result = {
            "schema_version": "1.0",
            "corpus": {
                "path": str(corpus_path), "sha256": sha, "n_items": len(corpus),
                "scene_counts": dict(Counter(x.get("scene") for x in corpus)),
                "source_counts": dict(Counter(x.get("source") for x in corpus)),
            },
            "crisis": crisis,
            "emotion": emotion,
            "recall": recall,
            "dialogue_quality": dialogue,
            "ab_decisions": ab_decisions(emotion, recall),
            "metrics_source": "docs/OLD_REPO_QUALITY_METRICS.md（旧仓库 V2）；未自创指标",
            "output_file": str(out_file),
        }
        out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
        print_report(result)
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
