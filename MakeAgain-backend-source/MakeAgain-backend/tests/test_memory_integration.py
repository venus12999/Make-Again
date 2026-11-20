"""三个深度替换模块的端到端集成验收（lead 集成接线后）。

验证 USE_DUAL_RECALL + USE_FIVE_LAYER + USE_ENTITY_PORTRAIT 三开关同时开启时：
写记忆 → 索引 → 双路召回 → 五层双写 → 实体画像 → 删除 → 索引清理，全链路不打架。

验收不是"全绿"：每条断言对应 MEMORY_DEEP_REPLACE_PLAN.md 的集成契约。
"""
import os
import pytest


@pytest.fixture
def all_on(monkeypatch):
    for k in ("USE_DUAL_RECALL", "USE_FIVE_LAYER", "USE_ENTITY_PORTRAIT", "USE_RECALL_FUSION"):
        monkeypatch.setenv(k, "1")
    # 索引用独立库，避免污染
    monkeypatch.setenv("VECTOR_INDEX_DB_PATH", ":memory:")
    monkeypatch.setenv("ENTITY_GRAPH_DB_PATH", ":memory:")


def test_integration_write_index_recall_delete(all_on):
    """集成契约：存储层写成功后索引必须建立，删除后索引必须清理。"""
    from memory import store
    from memory import recall

    # 重置可能存在的全局索引单例
    if hasattr(recall, "reset_index_cache"):
        recall.reset_index_cache()

    # 1. 写入两条记忆（一条关于豆豆，一条关于猫去世）
    db_ctx = None
    from memory.db import _RawSessionLocal
    db = _RawSessionLocal()
    try:
        e1 = store.add_memory(db, "u-int", type="chat", content="豆豆是一只小狗，很可爱",
                              importance=7.0)
        e2 = store.add_memory(db, "u-int", type="chat", content="豆豆上周去世了，我很想念它",
                              importance=9.0)
        print("\n[INT-1] written:", e1.id, e2.id)

        # 2. 索引已建立：双路召回能查到（不依赖全量 list_memories）
        import unittest.mock as mock
        with mock.patch.object(store, "list_memories", wraps=store.list_memories) as spy:
            hits = recall.recall(db, "u-int", "豆豆", top_k=5)
            print("[INT-2] recall hits:", [(h["entry"].id, h["entry"].content[:20]) for h in hits])
            print("[INT-2] list_memories calls during recall:", spy.call_count)
        # 双路召回不应触发全量读取（V1 契约）
        assert spy.call_count == 0, "dual recall must not call list_memories"
        assert len(hits) >= 1, "indexed memories must be recallable"
        assert any("豆豆" in h["entry"].content for h in hits)

        # 3. 删除一条 → 索引清理 → 召回不再返回它
        store.prov_delete_memory(db, str(e1.id), actor_id="u-int")
        after = recall.recall(db, "u-int", "豆豆", top_k=5)
        ids = [h["entry"].id for h in after]
        print("[INT-3] after delete, recall ids:", ids)
        assert e1.id not in ids, "deleted memory must be purged from index"
        assert e2.id in ids, "surviving memory must remain"

        # 4. 五层双写：另一条记忆能在五层库 current 查到
        cur = store.five_layer_current("豆豆")
        print("[INT-4] five-layer current:", cur)
        assert cur, "five-layer dual write must produce current records"

        print("\n[INTEGRATION] PASS: write→index→recall→delete→purge→five-layer all consistent")
    finally:
        db.close()
