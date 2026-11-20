"""G2 / U-012 真语义 embedding 行为验收 V1-V5（不是"测试全绿"就算过）。

V1 真语义召回：查询"难过失眠"召回无字面重叠的"彻夜睡不着"，kind=semantic_bge_small_zh
V2 语义区分：cos(养猫,猫去世) > cos(养猫,天气)（真实数字）
V3 实体图复用同一 embedder 单例（模型只加载一次）+ 0.95 阈值语义合并
V4 fastembed 导入失败 → 优雅回退 hash，不崩
V5 开关关闭 → 与基线 51a063c 逐字节一致（git show oracle）

运行：HF_HOME=D:\\12412\\huggingface python -m pytest -s tests/test_real_embedding.py -v
若本机无已下载模型则相关用例 skip 并标注 ENV_FAIL（不是功能失败）。
"""
from __future__ import annotations

import builtins
import json
import math
import os
import subprocess
import tempfile
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

from memory import entity_graph, vector_index
from memory.vector_index import DualIndex, get_index, reset_index

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_REF = "51a063c"  # 用户指定基线（235 passed）
HF_HOME = r"D:\12412\huggingface"
MODEL = "BAAI/bge-small-zh-v1.5"


def _cos(a, b) -> float:
    d = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return d / (na * nb) if na and nb else 0.0


@pytest.fixture
def semantic_env(monkeypatch):
    """打开真语义 + 离线缓存；模型不可用则 skip（ENV_FAIL）。"""
    monkeypatch.setenv("HF_HOME", HF_HOME)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    monkeypatch.delenv("USE_SEMANTIC_EMBEDDING", raising=False)
    monkeypatch.setenv("ENTITY_GRAPH_DB_PATH", ":memory:")
    vector_index._reset_semantic_state()
    entity_graph.set_embedder(None)
    entity_graph.reset()
    ok = vector_index.enable_semantic_embedding()
    if not ok:
        pytest.skip(f"ENV_FAIL: fastembed/{MODEL} unavailable (HF_HOME={HF_HOME})")
    yield
    vector_index._reset_semantic_state()
    entity_graph.set_embedder(None)
    entity_graph.reset()


# ---------------------------------------------------------------------------
# G2-V1 —— 真语义召回（无字面重叠）+ embedding_kind
# ---------------------------------------------------------------------------
def test_g2_v1_semantic_recall_no_literal_overlap(semantic_env, monkeypatch):
    tmp = tempfile.mkdtemp(prefix="g2-v1-")
    dbp = os.path.join(tmp, "v1.sqlite3")
    idx = get_index(dbp)
    try:
        target = "那晚彻夜睡不着，心里很痛苦，整夜都醒着"
        distractors = [
            "今天去超市买了苹果和香蕉",
            "天气很好，出门散了步",
            "桂花糕的做法记下来了",
        ]
        for i, text in enumerate([target] + distractors):
            idx.upsert(f"m{i}", "u-g2-v1", text,
                       ts=datetime.now(timezone.utc).replace(tzinfo=None))
        query = "难过失眠"  # 与 target 无共同汉字
        overlap = set(query) & set(target)
        print("\n[G2-V1] model:", MODEL, "| dim:", vector_index.SEMANTIC_DIM)
        print("[G2-V1] query:", query, "| target:", target)
        print("[G2-V1] literal char overlap:", overlap)
        assert not overlap, "查询与目标必须字面零重叠才能证明语义路"
        qvec = vector_index.embed_text(query)
        print("[G2-V1] embed_text dim:", len(qvec))
        assert len(qvec) == 512
        kw = idx.keyword_search(query, "u-g2-v1", k=4)
        hits = idx.vector_search(qvec, "u-g2-v1", k=4)
        print("[G2-V1] keyword_search (bm25):", json.dumps(kw, ensure_ascii=False))
        print("[G2-V1] vector_search (semantic):", json.dumps(hits, ensure_ascii=False))
        assert hits and hits[0]["id"] == "m0", "真语义路未召回目标"
        d = idx.diagnostics("m0")
        print("[G2-V1] embedding_kind:", vector_index.embedding_kind())
        print("[G2-V1] diagnostics:", json.dumps(d, ensure_ascii=False))
        assert vector_index.embedding_kind() == "semantic_bge_small_zh"
        assert d["embedding_kind"] == "semantic_bge_small_zh"
        # 维度隔离证明：语义写进 memory_vec_d512，hash 表 memory_vec 为空，查询不混用
        assert d["vec_total"] == 0
        assert d["dim_tables"].get("memory_vec_d512") == 4
        hash_hits = idx.vector_search([0.0] * 64, "u-g2-v1", k=4)
        print("[G2-V1] 64-dim query on hash table:", json.dumps(hash_hits, ensure_ascii=False))
        assert hash_hits == []
    finally:
        reset_index(dbp)


# ---------------------------------------------------------------------------
# G2-V2 —— 语义区分（真实 cos 数字）
# ---------------------------------------------------------------------------
def test_g2_v2_semantic_cos_numbers(semantic_env):
    va = vector_index.embed_text("养猫")
    vb = vector_index.embed_text("猫去世")
    vc = vector_index.embed_text("天气")
    c_same = _cos(va, vb)
    c_diff = _cos(va, vc)
    print("\n[G2-V2] dim:", len(va))
    print("[G2-V2] cos(养猫, 猫去世) = %.4f" % c_same)
    print("[G2-V2] cos(养猫, 天气)   = %.4f" % c_diff)
    print("[G2-V2] 差值 = %.4f | 区分成立:" % (c_same - c_diff), c_same > c_diff)
    assert c_same > c_diff


# ---------------------------------------------------------------------------
# G2-V3 —— 实体图复用同一 embedder（模型只加载一次）+ 0.95 语义合并
# ---------------------------------------------------------------------------
def test_g2_v3_entity_graph_shares_single_embedder(monkeypatch):
    import fastembed

    monkeypatch.setenv("HF_HOME", HF_HOME)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    monkeypatch.setenv("ENTITY_GRAPH_DB_PATH", ":memory:")
    vector_index._reset_semantic_state()
    entity_graph.set_embedder(None)
    entity_graph.reset()

    real_cls = fastembed.TextEmbedding
    counter = {"n": 0}

    class CountingTextEmbedding(real_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *a, **kw):
            counter["n"] += 1
            super().__init__(*a, **kw)

    monkeypatch.setattr(fastembed, "TextEmbedding", CountingTextEmbedding)
    try:
        ok1 = vector_index.enable_semantic_embedding()
        fn_vi = vector_index.get_semantic_embedder()
        ok2 = entity_graph.enable_semantic_embedding()
        fn_eg = entity_graph._embedder
        print("\n[G2-V3] vector_index.enable:", ok1,
              "| entity_graph.enable:", ok2,
              "| TextEmbedding 构造次数:", counter["n"])
        print("[G2-V3] 同一函数对象:", fn_eg is fn_vi,
              "| entity_graph.current_dim:", entity_graph.current_dim())
        assert ok1 and ok2
        assert counter["n"] == 1, "模型只应加载一次"
        assert fn_eg is fn_vi, "实体图必须复用同一 embedder 单例"

        # 0.95 阈值语义合并（cos≈0.9723 vs 负例 0.3446）
        text_a, text_b, text_neg = "我的小猫咪豆豆", "我的小猫豆豆", "兜兜"
        c_pos = _cos(fn_vi(text_a), fn_vi(text_b))
        c_neg = _cos(fn_vi(text_a), fn_vi(text_neg))
        print("[G2-V3] cos(%s, %s) = %.4f" % (text_a, text_b, c_pos))
        print("[G2-V3] cos(%s, %s) = %.4f" % (text_a, text_neg, c_neg))
        print("[G2-V3] threshold:", entity_graph._SIM_THRESHOLD)
        a = entity_graph.upsert_entity(text_a, "name", "g2-sem-1", user_id="u-g2-v3")
        b = entity_graph.upsert_entity(text_b, "name", "g2-sem-2", user_id="u-g2-v3")
        c = entity_graph.upsert_entity(text_neg, "name", "g2-sem-3", user_id="u-g2-v3")
        print("[G2-V3] upsert result:", json.dumps(
            {"a": a, "b": b, "c": c}, ensure_ascii=False))
        assert c_pos >= entity_graph._SIM_THRESHOLD > c_neg
        assert a["created"] is True
        assert b["created"] is False
        assert set(b["linked_memory_ids"]) == {"g2-sem-1", "g2-sem-2"}
        assert c["created"] is True  # 负例不合并
        assert entity_graph.current_dim() == 512
    finally:
        vector_index._reset_semantic_state()
        entity_graph.set_embedder(None)
        entity_graph.reset()


# ---------------------------------------------------------------------------
# G2-V3b —— 仅靠环境变量激活也联动实体图（lead 端到端开 USE_SEMANTIC_EMBEDDING=1 的路径）
# ---------------------------------------------------------------------------
def test_g2_v3b_env_switch_also_shares_embedder(monkeypatch):
    monkeypatch.setenv("HF_HOME", HF_HOME)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    monkeypatch.setenv("USE_SEMANTIC_EMBEDDING", "1")
    monkeypatch.setenv("ENTITY_GRAPH_DB_PATH", ":memory:")
    vector_index._reset_semantic_state()
    entity_graph.set_embedder(None)
    entity_graph.reset()
    try:
        vec = vector_index.embed_text("养猫")  # 触发懒加载
        fn = vector_index.get_semantic_embedder()
        print("\n[G2-V3b] USE_SEMANTIC_EMBEDDING=1 -> dim:", len(vec),
              "| entity_graph 共用同一对象:", entity_graph._embedder is fn,
              "| current_dim:", entity_graph.current_dim())
        assert len(vec) == 512
        assert entity_graph._embedder is fn
        assert entity_graph.current_dim() == 512
    finally:
        vector_index._reset_semantic_state()
        entity_graph.set_embedder(None)
        entity_graph.reset()


# ---------------------------------------------------------------------------
# G2-V4 —— fastembed 不可用 → 优雅回退 hash，不崩
# ---------------------------------------------------------------------------
def test_g2_v4_import_failure_falls_back_to_hash(monkeypatch):
    vector_index._reset_semantic_state()
    real_import = builtins.__import__

    def failing_import(name, *a, **kw):
        if name == "fastembed" or name.startswith("fastembed."):
            raise ImportError("SIMULATED: fastembed unavailable")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    try:
        ok = vector_index.enable_semantic_embedding()
        vec = vector_index.embed_text("难过失眠")
        print("\n[G2-V4] enable_semantic_embedding ->", ok)
        print("[G2-V4] embed_text len:", len(vec), "| kind:", vector_index.embedding_kind())
        print("[G2-V4] hash vec (前8维):", [round(x, 6) for x in vec[:8]])
        assert ok is False
        assert len(vec) == 64  # 基线 hash 维度
        assert vector_index.embedding_kind() == "deterministic_hash_nonsemantic"
    finally:
        vector_index._reset_semantic_state()


# ---------------------------------------------------------------------------
# G2-V5 —— 开关关闭逐字节一致（git show 51a063c oracle）
# ---------------------------------------------------------------------------
def test_g2_v5_byte_for_byte_rollback(monkeypatch):
    monkeypatch.setenv("USE_SEMANTIC_EMBEDDING", "0")
    vector_index._reset_semantic_state()
    entity_graph.set_embedder(None)

    probe = subprocess.run(
        ["git", "cat-file", "-e", BASE_REF], cwd=str(REPO_ROOT), capture_output=True)
    if probe.returncode != 0:
        pytest.skip(f"baseline commit {BASE_REF} not present in this repo history")
    src = subprocess.check_output(
        ["git", "show", f"{BASE_REF}:MakeAgain-backend/memory/vector_index.py"],
        cwd=str(REPO_ROOT), text=True, encoding="utf-8")
    oracle = types.ModuleType("vector_index_oracle_51a063c")
    exec(compile(src, "<baseline 51a063c>", "exec"), oracle.__dict__)

    texts = [
        "难过失眠", "彻夜睡不着", "养猫", "猫去世", "天气",
        "豆豆是一只小狗，很可爱", "桂花糕的做法记下来了", "", "Hello world 123",
    ]
    emb_equal = all(oracle.embed_text(t) == vector_index.embed_text(t) for t in texts)
    print("\n[G2-V5] oracle: git show %s:MakeAgain-backend/memory/vector_index.py" % BASE_REF)
    print("[G2-V5] embed_text(%d texts) byte-identical:" % len(texts), emb_equal)
    print("[G2-V5] sample 难过失眠:", [round(x, 6) for x in vector_index.embed_text("难过失眠")[:6]])
    assert emb_equal

    pa = os.path.join(tempfile.mkdtemp(prefix="g2-v5-oracle-"), "o.sqlite3")
    pb = os.path.join(tempfile.mkdtemp(prefix="g2-v5-current-"), "c.sqlite3")
    ia, ib = DualIndex(pa), DualIndex(pb)
    try:
        for i, text in enumerate(texts[1:], start=1):
            ts = datetime(2026, 1, i, 12, 0, tzinfo=timezone.utc).replace(tzinfo=None)
            ia.upsert(f"m{i}", "u", text, ts=ts, importance=float(i))
            ib.upsert(f"m{i}", "u", text, ts=ts, importance=float(i))
        qa, qb = oracle.embed_text("难过失眠"), vector_index.embed_text("难过失眠")
        va = ia.vector_search(qa, "u", k=5)
        vb = ib.vector_search(qb, "u", k=5)
        ka = ia.keyword_search("难过失眠", "u", k=5)
        kb = ib.keyword_search("难过失眠", "u", k=5)
        ma = ia.get_meta(["m2", "m3"])
        mb = ib.get_meta(["m2", "m3"])
        print("[G2-V5] vector_search identical:", va == vb, "|", json.dumps(va, ensure_ascii=False))
        print("[G2-V5] keyword_search identical:", ka == kb, "|", json.dumps(ka, ensure_ascii=False))
        print("[G2-V5] get_meta identical:", ma == mb)
        assert va == vb and ka == kb and ma == mb
        assert vector_index.embedding_kind() == "deterministic_hash_nonsemantic"
    finally:
        ia.close()
        ib.close()
