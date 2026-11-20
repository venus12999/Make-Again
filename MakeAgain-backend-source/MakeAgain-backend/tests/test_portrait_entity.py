"""WP-C 行为验收：实体双向链接 + evidence-grounded 画像（V1-V6）。

验收铁律：每条 V 都打印真实命令可复现的原始输出；全绿不是标准。
运行：pytest -s tests/test_portrait_entity.py -v
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from memory import entity_graph, portrait, store
from memory.db import Base, SessionLocal
from memory.models import Portrait, User

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_REF = "cde75f8"  # lead 指定基线（215 passed）


@pytest.fixture(autouse=True)
def _wp_c_isolation(monkeypatch):
    monkeypatch.setenv("ENTITY_GRAPH_DB_PATH", ":memory:")
    monkeypatch.setenv("USE_ENTITY_PORTRAIT", "0")
    entity_graph.reset()
    yield
    entity_graph.reset()
    monkeypatch.delenv("USE_ENTITY_PORTRAIT", raising=False)


@pytest.fixture
def db():
    session = SessionLocal()
    session.add(User(id="u-wp-c"))
    session.commit()
    yield session
    session.close()


def _link_doubao(mid: str):
    return entity_graph.link_facts(
        [{"kind": "user", "fact": f"记忆 {mid} 提到豆豆", "confidence": 0.9, "linked_memory_ids": []}],
        memory_id=mid, user_id="u-wp-c", entities=[("name", "豆豆")],
    )


class FakeClient:
    mock = False

    class settings:
        llm_fast_model = "fake-fast-model"

    def __init__(self, payload=None):
        self.payload = payload or {"likes": "猫", "owns_dog": "养狗"}
        self.prompt = None

    def chat_json(self, messages, **kwargs):
        self.prompt = messages[-1]["content"]
        return self.payload, {}


def _seed_merge_state(session, user_id="u-wp-c", fact="喜欢猫", n=20):
    """建画像水位线 + n 条后续记忆（满足 portrait_merge_every=20）。"""
    from memory.models import MemoryEntry
    store.upsert_portrait(session, user_id, "user", {"likes": "暂未提及"}, status="draft")
    watermark = session.execute(
        select(Portrait).where(Portrait.user_id == user_id, Portrait.kind == "user")
    ).scalar_one().updated_at
    for i in range(n):
        m = store.add_memory(
            session, user_id, content=f"第{i}次聊天：我说{fact}。",
            facts=[{"kind": "user", "fact": fact, "confidence": 0.9, "linked_memory_ids": []}],
        )
        m.ts = watermark + timedelta(hours=i + 1)
    session.commit()
    return watermark


def _snapshot(session, user_id="u-wp-c"):
    p = session.execute(
        select(Portrait).where(Portrait.user_id == user_id, Portrait.kind == "user")
    ).scalar_one()
    return {"content": p.content, "status": p.status, "version": p.version}


# ==================== V1：两轮都提"豆豆"，实体 linked_memory_ids 含 2 条 ====================
def test_v1_two_memories_linked_to_doubao(db):
    o1 = _link_doubao("mem-1")
    o2 = _link_doubao("mem-2")
    ent = entity_graph.get_entity("豆豆", user_id="u-wp-c")
    print("\n[V1] entity:", json.dumps(ent, ensure_ascii=False))
    print("[V1] fact2.linked_memory_ids:", o2[0]["linked_memory_ids"])
    assert set(ent["linked_memory_ids"]) == {"mem-1", "mem-2"}
    assert o1[0]["linked_memory_ids"] == []
    assert o2[0]["linked_memory_ids"] == ["mem-1"]


def test_v1_semantic_threshold_095(db):
    """V1 附证：精确文本不同但 vec0 余弦相似度 >= 0.95 时命中同一实体。"""
    entity_graph.set_embedder(lambda text: [1.0] + [0.0] * 31)  # 两文本向量完全相同
    try:
        a = entity_graph.upsert_entity("豆豆", "name", "sem-1", user_id="u-wp-c")
        b = entity_graph.upsert_entity("兜兜", "name", "sem-2", user_id="u-wp-c")
        print("\n[V1b] vec0 cosine=1.0 match result:", json.dumps(
            {"first": a, "second": b}, ensure_ascii=False))
        assert a["created"] is True
        assert b["created"] is False
        assert set(b["linked_memory_ids"]) == {"sem-1", "sem-2"}
        # 0.95 以下不合并（用正交向量证明阈值生效）
        entity_graph.set_embedder(lambda text: [1.0] + [0.0] * 31 if "豆" in text else [0.0, 1.0] + [0.0] * 30)
        c = entity_graph.upsert_entity("兜兜", "name", "sem-3", user_id="u-wp-c")
        print("[V1b] vec0 cosine=0.0 match result:", json.dumps(c, ensure_ascii=False))
        assert c["created"] is True
    finally:
        entity_graph.set_embedder(None)


# ==================== V2：删除一条记忆，实体反链同步移除（不悬空） ====================
def test_v2_delete_clears_reverse_links(db):
    _link_doubao("mem-1")
    _link_doubao("mem-2")
    cleaned = entity_graph.remove_memory("mem-1", user_id="u-wp-c")
    ent = entity_graph.get_entity("豆豆", user_id="u-wp-c")
    print("\n[V2] cleaned:", cleaned, "entity after delete:", json.dumps(ent, ensure_ascii=False))
    assert cleaned == 1
    assert ent["linked_memory_ids"] == ["mem-2"]
    assert "mem-1" not in ent["linked_memory_ids"]
    cleaned2 = entity_graph.remove_memory("mem-2", user_id="u-wp-c")
    print("[V2] cleaned2:", cleaned2, "entity after last delete:",
          entity_graph.get_entity("豆豆", user_id="u-wp-c"))
    assert entity_graph.get_entity("豆豆", user_id="u-wp-c") is None  # 反链空→实体删除


# ==================== V3：抗幻觉，prompt 没有的信息不编造 ====================
def test_v3_no_hallucination(db, monkeypatch):
    monkeypatch.setenv("USE_ENTITY_PORTRAIT", "1")
    assert portrait.entity_portrait_enabled() is True
    _seed_merge_state(db)
    client = FakeClient({"likes": "猫", "owns_dog": "养狗"})
    did = portrait.merge_if_due(db, "u-wp-c", client)
    result = store.get_portrait(db, "u-wp-c", "user")
    print("\n[V3] merged:", json.dumps(result, ensure_ascii=False))
    print("[V3] HARD RULES in prompt:", "HARD RULES" in client.prompt,
          "| copied/normalized rule:", "(a) a clean value copied" in client.prompt)
    assert did is True
    assert "HARD RULES" in client.prompt
    assert "(a) a clean value copied" in client.prompt
    assert result.get("likes") == "猫"           # fact「喜欢猫」支持
    assert "养狗" not in json.dumps(result, ensure_ascii=False)  # 编造值被拒
    assert "owns_dog" not in result
    # 基线 prompt 不含 HARD RULES（证明规则只在开关路径引入）
    assert "HARD RULES" not in portrait._merge_prompt({"likes": "暂未提及"}, ["喜欢猫"])


# ==================== V4：Observation quote 可溯源回原消息 ====================
def test_v4_observation_quote_traced(db):
    original = "昨天我说：豆豆很可爱，像家人一样，我常想起她。"
    m = store.add_memory(
        db, "u-wp-c", content=original,
        facts=[{"kind": "user", "fact": "豆豆很可爱", "confidence": 0.9, "linked_memory_ids": []}],
    )
    records = portrait._collect_fact_records(
        db, "u-wp-c", since=m.ts - timedelta(seconds=1))
    assert records
    obs = portrait.build_observation(
        title="豆豆是家人", content=records[0]["fact"],
        evidence=[{
            "memory_id": str(m.id), "quote": records[0]["quote"],
            "relevance": "逐字证据", "timestamp": records[0]["timestamp"],
        }],
    )
    print("\n[V4] observation:", json.dumps(obs, ensure_ascii=False))
    print("[V4] quote in original message:", obs["evidence"][0]["quote"] in original)
    assert obs["evidence"][0]["quote"] in original
    assert obs["evidence"][0]["memory_id"] == str(m.id)
    assert obs["evidence"][0]["timestamp"].endswith(("+00:00", "Z"))  # pydantic v2 json mode 用 Z
    assert obs["evidence_count"] == 1
    assert obs["trend"] in {"stable", "strengthening", "weakening", "new", "stale"}


# ==================== V5：link_facts 接口让 linked_memory_ids 不再恒空 ====================
def test_v5_link_facts_interface(db):
    out1 = portrait.link_facts(
        [{"fact": "豆豆很可爱", "linked_memory_ids": []}],
        memory_id="m1", user_id="u-wp-c", entities=[("name", "豆豆")])
    out2 = portrait.link_facts(
        [{"fact": "又想起豆豆", "linked_memory_ids": []}],
        memory_id="m2", user_id="u-wp-c", entities=[("name", "豆豆")])
    print("\n[V5] out1.linked_memory_ids:", out1[0]["linked_memory_ids"])
    print("[V5] out2.linked_memory_ids:", out2[0]["linked_memory_ids"])
    assert out1[0]["linked_memory_ids"] == []
    assert out2[0]["linked_memory_ids"] == ["m1"]  # 不再恒空
    # 可注入真实抽取器（lead 接 extract 路径的方式）
    out3 = portrait.link_facts(
        [{"fact": "豆豆陪着我", "linked_memory_ids": []}],
        memory_id="m3", user_id="u-wp-c",
        entity_extractor=lambda text, *a: [("name", "豆豆")] if "豆豆" in text else [],
    )
    print("[V5] out3.linked_memory_ids (injected extractor):", out3[0]["linked_memory_ids"])
    assert set(out3[0]["linked_memory_ids"]) == {"m1", "m2"}


# ==================== V6：关闭开关逐字节回退（git show 基线做 oracle） ====================
def test_v6_byte_for_byte_rollback(monkeypatch):
    monkeypatch.setenv("USE_ENTITY_PORTRAIT", "0")
    assert portrait.entity_portrait_enabled() is False
    # 该 oracle 依赖主仓基线 commit。同步到队友仓库后历史里没有该 SHA，
    # 此时优雅跳过（不是功能失败），避免跨仓库误判。
    probe = subprocess.run(
        ["git", "cat-file", "-e", BASE_REF],
        cwd=str(REPO_ROOT), capture_output=True,
    )
    if probe.returncode != 0:
        pytest.skip(f"baseline commit {BASE_REF} not present in this repo history (synced copy)")
    src = subprocess.check_output(
        ["git", "show", f"{BASE_REF}:MakeAgain-backend/memory/portrait.py"],
        cwd=str(REPO_ROOT), text=True, encoding="utf-8",
    )
    oracle = types.ModuleType("portrait_oracle_cde75f8")
    exec(compile(src, "<baseline cde75f8>", "exec"), oracle.__dict__)

    tmp = tempfile.mkdtemp(prefix="wp-c-v6-")
    engine2 = create_engine("sqlite:///" + tmp + "/v6.sqlite3")
    Base.metadata.create_all(engine2)
    S2 = sessionmaker(bind=engine2, autoflush=False, expire_on_commit=False)

    sa = S2()
    sa.add(User(id="u-v6"))
    sa.commit()
    _seed_merge_state(sa, user_id="u-v6")
    client_a = FakeClient()
    r1 = oracle.merge_if_due(sa, "u-v6", client_a)
    snap1 = _snapshot(sa, "u-v6")
    sa.close()

    sb = S2()
    sb.add(User(id="u-v6-b"))
    sb.commit()
    _seed_merge_state(sb, user_id="u-v6-b")
    client_b = FakeClient()
    r2 = portrait.merge_if_due(sb, "u-v6-b", client_b)
    snap2 = _snapshot(sb, "u-v6-b")
    sb.close()

    a = json.dumps([r1, snap1], sort_keys=True, ensure_ascii=False)
    b = json.dumps([r2, snap2], sort_keys=True, ensure_ascii=False)
    print("\n[V6] git show oracle:", f"git show {BASE_REF}:MakeAgain-backend/memory/portrait.py")
    print("[V6] oracle  (return,snapshot):", a)
    print("[V6] current (return,snapshot):", b)
    print("[V6] byte-identical:", a.encode("utf-8") == b.encode("utf-8"))
    assert a.encode("utf-8") == b.encode("utf-8")
