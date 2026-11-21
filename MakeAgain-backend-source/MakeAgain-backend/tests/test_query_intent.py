"""U-013 query-intent 门行为验收 V1-V5 + 越权/边界（不是"测试全绿"就算过）。

V1 开关开启：「今天晚饭吃什么」→ []（即使库里真有"晚饭"记忆也短路）
V2 不误伤：「失眠难过」/「想起豆豆」/「我养的猫还在吗」仍正常召回
V3 不下降：V2 golden 语料 18 query，Recall@K(gate on) >= Recall@K(gate off)
V4 逐字节回退：开关关 = 基线 cde75f8 legacy 原文（内嵌 oracle）
V5 全量 pytest（本文件只负责 V1-V4 + 越权/边界；V5 命令见交付报告）

口径说明：
- V2/V3 用确定性 hash embedding（USE_SEMANTIC_EMBEDDING=0），非训练语义模型；
  与 test_dual_recall.py 的口径一致，真实语义模型数字不在本文件声称范围内。
- V3 语料 100% synthetic-handwritten（见 eval/RECALL_CORPUS_README.md），不是真实用户数据。
"""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from memory import recall as recall_mod
from memory import store
from memory.models import MemoryEntry
from memory.query_intent import evaluate_query_intent, is_memory_irrelevant
from memory.vector_index import reset_index

BACKEND = Path(__file__).resolve().parents[1]
CORPUS = BACKEND / "eval" / "recall_corpus.json"
_FIXED_NOW = datetime(2026, 11, 21, 12, 0, 0)


# 基线 oracle：git show cde75f8:MakeAgain-backend/memory/recall.py 的 recall 原文
# （= USE_QUERY_INTENT 关闭且其余开关关闭时必须逐字节一致的行为）。
_BASELINE_SOURCE = '''from __future__ import annotations
import math
import re
from datetime import datetime, timezone
from config import get_settings
from memory.facts import facts_text
from memory.store import list_memories
_TOKEN_RE = re.compile(r"[一-鿿]|[a-zA-Z0-9]+")
def _tokens(text): return set(_TOKEN_RE.findall(text or ""))
def _relevance(q, e):
    if not q or not e: return 0.0
    inter = q & e
    return len(inter) / math.sqrt(len(q) * len(e)) if inter else 0.0
def _overlap_fraction(entry_tokens, recent_tokens):
    if not entry_tokens: return 0.0
    return len(entry_tokens & recent_tokens) / len(entry_tokens)
def recall(db, user_id, query, top_k=None, recent_text=None):
    s = get_settings()
    top_k = top_k or s.recall_top_k
    entries = list_memories(db, user_id)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    q_tokens = _tokens(query)
    recent_tokens = _tokens(recent_text) if recent_text else set()
    scored = []
    for e in entries:
        entry_tokens = _tokens(f"{facts_text(e.facts)} {e.summary or ''} {e.content}")
        if recent_tokens and _overlap_fraction(entry_tokens, recent_tokens) >= s.recall_recent_overlap_threshold:
            continue
        days = max(0.0, (now - e.ts).total_seconds() / 86400.0)
        recency = math.exp(-days / 30.0)
        importance = (e.importance or 5.0) / 10.0
        relevance = _relevance(q_tokens, entry_tokens)
        score = s.recall_alpha * recency + s.recall_beta * importance + s.recall_gamma * relevance
        scored.append({"entry": e, "score": score, "recency": recency, "relevance": relevance})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
'''


class _FixedDatetime:
    @staticmethod
    def now(tz=None):
        return _FIXED_NOW


def _baseline_recall(fixed_datetime=None):
    ns = {"__name__": "recall_baseline_cde75f8"}
    exec(compile(_BASELINE_SOURCE, "cde75f8:memory/recall.py", "exec"), ns)
    if fixed_datetime is not None:
        ns["datetime"] = fixed_datetime
    return ns["recall"]


def _dump(result: list[dict]) -> bytes:
    payload = [
        {"id": str(r["entry"].id), "content": r["entry"].content,
         "score": repr(r["score"]), "recency": repr(r["recency"]), "relevance": repr(r["relevance"])}
        for r in result
    ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")


@pytest.fixture
def idx_env(tmp_path, monkeypatch):
    path = str(tmp_path / "qi_idx.sqlite3")
    monkeypatch.setenv("DUAL_RECALL_INDEX_DB", path)
    reset_index()
    monkeypatch.delenv("USE_QUERY_INTENT", raising=False)
    monkeypatch.delenv("USE_DUAL_RECALL", raising=False)
    monkeypatch.delenv("USE_RECALL_FUSION", raising=False)
    monkeypatch.setenv("USE_SEMANTIC_EMBEDDING", "0")
    yield path
    reset_index()


@pytest.fixture
def db(isolation):
    from memory.db import SessionLocal

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _add(db, user_id, content, summary, *, facts=None, importance=5.0, days_ago=0):
    m = store.add_memory(db, user_id, content=content, summary=summary,
                         facts=facts or [], importance=importance)
    m.ts = _FIXED_NOW - timedelta(days=days_ago)
    db.commit()
    return m


def _index_all(db, user_id):
    entries = store.list_memories(db, user_id)
    return sum(1 for e in entries if recall_mod.index_memory(e, db=db)), len(entries)


def _load_v2_metrics_fn():
    """直接复用 eval/run_recall_v2.py 的 metrics（真实评测口径，不在测试里另造指标）。"""
    spec = importlib.util.spec_from_file_location("run_recall_v2_qi", CORPUS.parent / "run_recall_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.metrics


# ---------------------------------------------------------------------------
# 意图判定单元测试（边界，先于端到端）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("query,expected_irrelevant", [
    ("今天晚饭吃什么？", True),
    ("今天晚饭吃什么", True),
    ("晚饭吃什么", True),
    ("今晚吃什么", True),
    ("what to eat", True),
    ("明天午饭吃什么", True),
    ("天气怎么样", True),
    ("天气", False),         # 裸主题词不拦（可能是记忆里的名词）
    ("那天的天气怎么样", False),  # 过去时间词 + 杂务短语 → 放行，防误伤
    ("昨天晚饭包了饺子", False),
    ("", False),          # 空串保守放行，不误伤
    ("   ", False),
    ("失眠难过", False),
    ("想起豆豆", False),
    ("我养的猫还在吗", False),
    ("橘子喜欢待在哪儿？", False),
    ("今天晚饭吃什么，我想起妈妈做的红烧肉", False),  # 杂务词 + 记忆线索 → 放行
])
def test_intent_decision_boundary(query, expected_irrelevant):
    assert is_memory_irrelevant(query) is expected_irrelevant
    assert evaluate_query_intent(query).memory_related is (not expected_irrelevant)


def test_purity_no_db_or_user_input():
    """越权防护的结构性证据：判定函数只吃 query，不接受 DB/user_id。"""
    import inspect

    for fn in (evaluate_query_intent, is_memory_irrelevant):
        params = list(inspect.signature(fn).parameters)
        assert params == ["query"], params
    # 同一 query 任意次调用结果一致（无隐藏状态）
    assert [evaluate_query_intent("今天晚饭吃什么").reason for _ in range(5)] == [
        "mundane_topic_without_memory_cue"
    ] * 5


# ---------------------------------------------------------------------------
# V1 —— 无关 query 直接返回 []
# ---------------------------------------------------------------------------
def test_v1_irrelevant_query_returns_empty(db, idx_env, monkeypatch):
    uid = "QI-V1-U"
    # 故意埋一条含"晚饭"的记忆：证明门是直接短路，而不是靠 bm25/relevance 恰好为 0
    _add(db, uid, "昨天晚饭我们一起包了饺子", "晚饭", days_ago=1)
    _add(db, uid, "豆豆是一只小狗", "豆豆", days_ago=3)
    assert _index_all(db, uid)[0] == 2

    monkeypatch.setenv("USE_DUAL_RECALL", "1")
    monkeypatch.setenv("USE_QUERY_INTENT", "1")
    res = recall_mod.recall(db, uid, "今天晚饭吃什么？", top_k=3)
    assert res == [], f"负例仍召回: {[r['entry'].content for r in res]}"

    # 关掉门，同一 query 应有召回（证明非空库 + 命中路径本身是通的）
    monkeypatch.setenv("USE_QUERY_INTENT", "0")
    res_off = recall_mod.recall(db, uid, "今天晚饭吃什么？", top_k=3)
    assert res_off, "关掉门后应恢复召回，否则 V1 证据不成立"


# ---------------------------------------------------------------------------
# V2 —— 不误伤语义/回忆/指代 query
# ---------------------------------------------------------------------------
def test_v2_reminiscence_and_emotion_queries_not_harmed(db, idx_env, monkeypatch):
    uid = "QI-V2-U"
    target_sleep = _add(db, uid, "那晚彻夜睡不着，心里很痛苦", "回忆", days_ago=3)
    target_doudou = _add(db, uid, "豆豆是一只小狗，很可爱", "豆豆", days_ago=7)
    target_cat = _add(db, uid, "养了一只猫，叫橘子", "养猫", days_ago=120)
    _add(db, uid, "今天去超市买了苹果和香蕉", "买水果", days_ago=1)
    assert _index_all(db, uid)[0] == 4

    monkeypatch.setenv("USE_DUAL_RECALL", "1")
    monkeypatch.setenv("USE_QUERY_INTENT", "1")
    cases = [
        ("失眠难过", target_sleep.id),
        ("想起豆豆", target_doudou.id),
        ("我养的猫还在吗", target_cat.id),
    ]
    for q, want_id in cases:
        res = recall_mod.recall(db, uid, q, top_k=4)
        assert res, f"{q!r} 被门误杀，返回空"
        assert any(r["entry"].id == want_id for r in res), (
            f"{q!r} 未召回目标 {want_id}: {[r['entry'].content for r in res]}"
        )


# ---------------------------------------------------------------------------
# V3 —— V2 golden 18 query Recall@K 不下降
# ---------------------------------------------------------------------------
def test_v3_golden_recall_not_degraded(db, idx_env, monkeypatch):
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    memories, queries = corpus["memories"], corpus["queries"]
    assert len(queries) == 18

    for m in memories:
        db.add(MemoryEntry(
            id=m["id"], user_id="golden-eval-user", type=m.get("type", "chat"),
            content=m["content"], summary=m.get("summary"), facts=m.get("facts", []),
            importance=float(m.get("importance", 5.0)),
            ts=datetime.fromisoformat(m["ts"].replace("Z", "+00:00")).replace(tzinfo=None),
        ))
    db.commit()
    indexed = sum(1 for m in memories
                  if recall_mod.index_memory(db.get(MemoryEntry, m["id"]), db=db))
    assert indexed == len(memories)

    monkeypatch.setenv("USE_DUAL_RECALL", "1")
    metrics = _load_v2_metrics_fn()

    def run(gate: str):
        monkeypatch.setenv("USE_QUERY_INTENT", gate)
        return metrics(queries, lambda q, top_k=3: recall_mod.recall(
            db, "golden-eval-user", q, top_k=top_k), 3)

    base = run("0")
    gated = run("1")
    assert base["n_queries"] == 17 and gated["n_queries"] == 17  # q-16 负例单列
    assert gated["Recall@K"] >= base["Recall@K"], (
        f"Recall@K 下降: base={base['Recall@K']} gated={gated['Recall@K']}"
    )
    assert gated["MRR"] >= base["MRR"]
    # 负例：关着会注入，开着变空——正是 U-013 要修的缺陷
    neg_base = next(p for p in base["per_query"] if p["negative"])
    neg_gated = next(p for p in gated["per_query"] if p["negative"])
    assert neg_base["retrieved"], "基线负例本应有注入，否则该 golden 不具修复价值"
    assert neg_gated["retrieved"] == [], f"开启后负例仍注入: {neg_gated['retrieved']}"
    print(f"[V3] baseline R@K={base['Recall@K']} MRR={base['MRR']} "
          f"neg_inject={base['negative_query_injection_rate']}")
    print(f"[V3] gated    R@K={gated['Recall@K']} MRR={gated['MRR']} "
          f"neg_inject={gated['negative_query_injection_rate']}")


# ---------------------------------------------------------------------------
# V4 —— 开关关闭逐字节回退（oracle = git show cde75f8）
# ---------------------------------------------------------------------------
def test_v4_switch_off_byte_for_byte_identical_to_baseline(db, idx_env, monkeypatch):
    uid = "QI-V4-U"
    _add(db, uid, "汤圆是一只猫，很爱晒太阳", "养猫",
         facts=[{"kind": "object", "fact": "汤圆是一只猫", "confidence": 0.9}],
         importance=6.0, days_ago=1)
    _add(db, uid, "今天早上汤圆去世了，我很伤心", "猫去世",
         facts=[{"kind": "memory", "fact": "汤圆去世了", "confidence": 0.9}],
         importance=8.0, days_ago=0)
    _add(db, uid, "昨天晚饭包了饺子", "晚饭", days_ago=2)
    _index_all(db, uid)

    monkeypatch.delenv("USE_DUAL_RECALL", raising=False)
    monkeypatch.delenv("USE_QUERY_INTENT", raising=False)
    monkeypatch.setattr(recall_mod, "datetime", _FixedDatetime)
    baseline = _baseline_recall(_FixedDatetime)
    for q, recent in [
        ("汤圆", None), ("猫 去世", "汤圆 猫"), ("桂花糕", "天气 不错"),
        ("今天晚饭吃什么？", None), ("失眠难过", None),
    ]:
        base = _dump(baseline(db, uid, q, top_k=5, recent_text=recent))
        actual = _dump(recall_mod.recall(db, uid, q, top_k=5, recent_text=recent))
        assert actual == base, f"query={q!r}\nbase={base!r}\nactual={actual!r}"

    # 显式 0 / off 也必须与未设置一致
    for val in ("0", "off", "false", ""):
        monkeypatch.setenv("USE_QUERY_INTENT", val)
        assert _dump(recall_mod.recall(db, uid, "今天晚饭吃什么？", top_k=5)) == _dump(
            baseline(db, uid, "今天晚饭吃什么？", top_k=5))


# ---------------------------------------------------------------------------
# 越权 / 边界：门不改变 user scope，也不依赖 DB 状态
# ---------------------------------------------------------------------------
def test_scope_isolation_not_bypassed_by_gate(db, idx_env, monkeypatch):
    uid_a, uid_b = "QI-SCOPE-A", "QI-SCOPE-B"
    a_mem = _add(db, uid_a, "我养的猫叫橘子", "养猫", days_ago=2)
    assert _index_all(db, uid_a)[0] == 1

    monkeypatch.setenv("USE_DUAL_RECALL", "1")
    monkeypatch.setenv("USE_QUERY_INTENT", "1")
    # 有记忆的用户：相关 query 正常召回
    res_a = recall_mod.recall(db, uid_a, "我养的猫还在吗", top_k=3)
    assert any(r["entry"].id == a_mem.id for r in res_a)
    # 无记忆的用户：同 query 不得因门而拿到 A 的记忆（跨用户越权）
    res_b = recall_mod.recall(db, uid_b, "我养的猫还在吗", top_k=3)
    assert res_b == [], f"跨用户泄漏: {[r['entry'].content for r in res_b]}"
    # 无关 query 对两个用户都为空，门的判定与 user_id 无关
    assert recall_mod.recall(db, uid_a, "今天晚饭吃什么", top_k=3) == []
    assert recall_mod.recall(db, uid_b, "今天晚饭吃什么", top_k=3) == []


def test_switch_default_off_and_truthy_values(monkeypatch):
    for val in (None, "", "0", "off", "no", "random"):
        if val is None:
            monkeypatch.delenv("USE_QUERY_INTENT", raising=False)
        else:
            monkeypatch.setenv("USE_QUERY_INTENT", val)
        assert recall_mod._use_query_intent_enabled() is False
    for val in ("1", "true", "TRUE", "yes", "on", " On "):
        monkeypatch.setenv("USE_QUERY_INTENT", val)
        assert recall_mod._use_query_intent_enabled() is True
