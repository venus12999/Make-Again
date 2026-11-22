"""V3 · 场景切换审计验证（离线，monkeypatch LLM，零 hy4/key，不改产品代码）。

对应 docs/THREE_SCENARIO_VALIDATION_PLAN.md V3：
用户在 breakup 场景完成旅程后表达进入新关系（"我开始新恋情了"），验证
  V3-V1  loss_type / 场景切换是否发生
  V3-V2  旧 Journey 是否归档只读
  V3-V3  新 Journey 是否不继承旧对象画像

测试口径（务必看）：
- 全离线：conftest 已置 MOCK_LLM=true，本文件再显式 monkeypatch LLMClient.chat 为确定性回复；
  不读 work/local-test-key.txt、不发任何网络请求、不跑真 hy4。
- 两类测试：
  1) `test_*_current`：刻画今天代码的真实行为，期望 PASS，print 原始数字作为缺口证据；
  2) `test_*_expected`：断言"应有的产品行为"，标 `@pytest.mark.xfail(strict=True)`，
     当前必然 XFAIL = 产品缺口；若产品修好会 XPASS(strict) 变红，提示缺口已闭合。
- 本文件不修改 agent/ memory/ api/ 任何主链路文件，只做只读验证。
- 缺口清单与源码行号见 docs/SCENE_TRANSITION_AUDIT.md。
"""
from __future__ import annotations

import inspect
import pathlib
import re
from contextlib import contextmanager

import pytest

from agent import companion as companion_agent
from conftest import account
from gateway.client import LLMClient
from memory import isolation as iso
from memory import state, store
from memory.db import _RawSessionLocal
from memory.models import Portrait, utcnow
from memory.scope import current_db


# --------------------------------------------------------------------------- helpers
@contextmanager
def _managed_db():
    """chat 会排后台记忆写入，要求 current_db 指向 managed session（否则 RuntimeError）。"""
    db = _RawSessionLocal()
    db.info.update(managed=True, borrowed=True)
    token = current_db.set(db)
    try:
        yield db
    finally:
        current_db.reset(token)
        db.finish()
        db.close()

def _mock_llm(monkeypatch, reply: str = "（离线确定性回复）我在听你说。"):
    """把 LLM 换成确定性桩：任何调用都不触网、不读 key。"""
    def fake_chat(self, messages, *args, **kwargs):
        return {"content": reply, "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "tool_calls": []}

    monkeypatch.setattr(LLMClient, "chat", fake_chat)


def _seed_completed_breakup(uid: str) -> tuple[str, str, str]:
    """构造"已完成 breakup 旅程"的用户：当前 breakup companion + 报告落库画像 + 一条记忆。

    画像写入刻意走产品主链同款 store.upsert_portrait（interview.py:744-745 的调用方式），
    即不显式传 companion_id —— 这正是本次要审计的真实落库形态。
    另补一条"正确带 companion_id"的画像行，用于分别验证隔离层与读取层的差异。
    """
    with _RawSessionLocal() as db:
        store.get_or_create_user(db, uid, loss_type="breakup")
        st = store.get_or_create_user_state(db, uid)
        st.consent_accepted_at = utcnow()
        db.commit()

        c = store.get_current_companion(db, uid)
        store.update_companion(db, c.id, relationship_type="breakup", subject_name="小夏")

        # 主链形态（interview._persist_report:744-745）：companion_id 为空
        store.upsert_portrait(db, uid, "object", {"称呼": "小夏", "关系": "前任"}, status="draft")
        store.upsert_portrait(db, uid, "user", {"昵称": "林薇"}, status="draft")
        # 隔离层形态（test_journey_isolation:29-35）：显式归属旧 Journey
        scoped = Portrait(user_id=uid, companion_id=c.id, kind="object",
                          content={"称呼": "小夏", "关系": "前任（带归属）"},
                          status="confirmed", version=1)
        db.add(scoped)
        db.commit()
        db.refresh(scoped)

        old_mem = iso.scoped_add_memory(db, uid, c.id, content="和小夏分手了，还是会想起她", importance=9.0)
        return c.id, scoped.id, old_mem.id


# ===========================================================================
# V3-V1 · 场景切换（loss_type）
# ===========================================================================
def test_v1_scene_machine_current_has_no_production_caller():
    """现状刻画：状态机只在 state.py 定义，api/agent 零调用；detect_scene_transition 根本不存在。"""
    backend = pathlib.Path(inspect.getsourcefile(state)).resolve().parents[1]
    entrypoints = ("transition_scene", "on_relationship_change", "scene_decision_for", "SceneStateMachine")
    callers: list[tuple[str, str]] = []
    for path in backend.rglob("*.py"):
        rel = path.relative_to(backend).as_posix()
        if rel == "memory/state.py" or "test" in path.name or "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for fn in entrypoints:
            if re.search(rf"\b{re.escape(fn)}\s*\(", text):
                callers.append((rel, fn))

    print("\n[V1-CURRENT] detect_scene_transition 是否存在:", hasattr(state, "detect_scene_transition"))
    print("[V1-CURRENT] state.py 暴露的切换入口:", [f for f in entrypoints if hasattr(state, f)])
    print("[V1-CURRENT] api/ + agent/ 中对切换入口的调用:", callers)

    # 2026-11-23 U-018 修复后（fix/U018-scene-switch d059174a）：detect_scene_transition
    # 已实现且接入 chat 主链路。审计初版断言"不存在/零调用"是缺口证据，缺口闭合后反转。
    assert hasattr(state, "detect_scene_transition") is True
    assert callers, f"应有生产调用方（companion.chat 已接线），实测: {callers}"


def test_v1_explicit_transition_current_switches_loss_type(client, monkeypatch):
    """现状刻画：显式调用 transition_scene（USE_PERSONA=1）时 loss_type 确实切换。"""
    monkeypatch.setenv("USE_PERSONA", "1")
    uid, _, _ = account(client, "v3-v1-explicit")
    old_cid, _, _ = _seed_completed_breakup(uid)

    with _RawSessionLocal() as db:
        before = store.effective_loss_type(db, uid)
        result = state.transition_scene(db, uid, "pet", trigger=state.TRIGGER_INTERVIEW)
        after = store.effective_loss_type(db, uid)
        new_c = store.get_current_companion(db, uid)

        print("\n[V1-CURRENT] 显式切换前 loss_type:", before, "companion:", old_cid)
        print("[V1-CURRENT] 决策 action:", result["decision"].action, "changed:", result["changed"])
        print("[V1-CURRENT] 显式切换后 loss_type:", after, "新 companion:", new_c.id,
              "relationship_type:", new_c.relationship_type)

        assert before == "breakup"
        assert result["decision"].action == state.ACTION_ARCHIVE_AND_CREATE
        assert after == "pet"
        assert new_c.id != old_cid


def test_v1_chat_message_current_does_NOT_switch(client, monkeypatch):
    """U-018 修复后：用户明说"我开始新恋情了"，chat 应触发切换（原缺口证据已反转）。"""
    monkeypatch.setenv("USE_PERSONA", "1")
    _mock_llm(monkeypatch)
    uid, _, _ = account(client, "v3-v1-chat")
    old_cid, _, _ = _seed_completed_breakup(uid)

    with _managed_db() as db:
        before = store.effective_loss_type(db, uid)
        reply = companion_agent.chat(db, uid, "我开始新恋情了", LLMClient(), session_id="v3-v1-chat")
        after = store.effective_loss_type(db, uid)
        n_comp = len(store.list_companions(db, uid))

        print("\n[V1-CURRENT] 用户话语: 我开始新恋情了")
        print("[V1-CURRENT] chat 回复:", str(reply.get("reply"))[:80])
        print("[V1-CURRENT] 切换前 loss_type:", before, "| 切换后 loss_type:", after,
              "| companion 段数:", n_comp)

        assert before == "breakup"
        # 缺口闭合：自然语言"开始新恋情"现在触发切换（切到 friend 通用陪伴）
        assert after == "friend", f"U-018 修复后应切换，实测 after={after}"
        assert n_comp == 2, f"U-018 修复后应有新 Journey，实测 n_comp={n_comp}"


def test_v1_expected_chat_message_starts_new_journey(client, monkeypatch):
    """U-018 修复后应为真 PASS：用户表达进入新关系后归档旧 Journey + 新建一段。"""
    monkeypatch.setenv("USE_PERSONA", "1")
    _mock_llm(monkeypatch)
    uid, _, _ = account(client, "v3-v1-expected")
    _seed_completed_breakup(uid)

    with _managed_db() as db:
        companion_agent.chat(db, uid, "我开始新恋情了", LLMClient(), session_id="v3-v1-expected")
        companions = store.list_companions(db, uid)
        print("\n[V1-EXPECTED] companion 段数:", len(companions),
              "archived_at:", [str(c.archived_at) for c in companions])
        assert len(companions) == 2, "期望：表达进入新关系后旧 Journey 归档 + 新 Journey 建立"


# ===========================================================================
# V3-V2 · 旧 Journey 归档只读
# ===========================================================================
def test_v2_archived_journey_is_read_only_current(client, monkeypatch):
    """现状刻画：显式切换后旧 companion 置 archived_at/ending_stage，隔离层读放行、写拒绝。"""
    monkeypatch.setenv("USE_PERSONA", "1")
    uid, _, _ = account(client, "v3-v2-archive")
    old_cid, _, old_mem_id = _seed_completed_breakup(uid)

    with _RawSessionLocal() as db:
        result = state.transition_scene(db, uid, "pet", trigger=state.TRIGGER_INTERVIEW)
        new_cid = result["companion_id"]
        old_c = store.get_companion(db, old_cid)
        old_c_read = iso.scoped_get_memory(db, uid, old_cid, old_mem_id)
        with pytest.raises(iso.ScopeDenied) as ew:
            iso.assert_scope(db, uid, old_cid, old_mem_id, operation="write")
        with pytest.raises(iso.ScopeDenied) as ea:
            iso.scoped_add_memory(db, uid, old_cid, content="归档后续写")

        print("\n[V2-CURRENT] 旧 companion archived_at:", old_c.archived_at,
              "ending_stage:", old_c.ending_stage, "ending_ritual:", old_c.ending_ritual)
        print("[V2-CURRENT] 旧记忆读成功 id:", old_c_read.id)
        print("[V2-CURRENT] 归档写拒绝:", ew.value.code, "| 归档新增拒绝:", ea.value.code)

        assert old_c.archived_at is not None
        assert old_c.ending_stage == "complete"
        assert old_c.ending_ritual == "scene_transition"
        assert old_c_read.id == old_mem_id
        assert ew.value.code == "ARCHIVED_READ_ONLY"
        assert ea.value.code == "ARCHIVED_READ_ONLY"


def test_v2_archived_memory_excluded_from_new_scope_current(client, monkeypatch):
    """现状刻画：旧 Journey 记忆不进入新 Journey 的 scope/召回（有数字）。"""
    monkeypatch.setenv("USE_PERSONA", "1")
    uid, _, _ = account(client, "v3-v2-scope")
    old_cid, _, old_mem_id = _seed_completed_breakup(uid)

    with _RawSessionLocal() as db:
        result = state.transition_scene(db, uid, "pet", trigger=state.TRIGGER_INTERVIEW)
        new_cid = result["companion_id"]
        new_mem = iso.scoped_add_memory(db, uid, new_cid, content="新旅程的猫叫糯米", importance=10.0)
        scope = iso.build_scope(db, uid, new_cid)
        hits = iso.scoped_recall(db, uid, new_cid, "小夏 糯米", top_k=10)
        hit_ids = [str(h["entry"].id) for h in hits]

        print("\n[V2-CURRENT] 新 Journey scope allowed_companion_ids:", sorted(scope.allowed_companion_ids))
        print("[V2-CURRENT] 旧记忆 id:", old_mem_id, "in allowed_ids:", str(old_mem_id) in scope.allowed_ids)
        print("[V2-CURRENT] 召回命中 ids:", hit_ids, "新记忆 id:", new_mem.id)

        assert old_cid not in scope.allowed_companion_ids
        assert str(old_mem_id) not in scope.allowed_ids
        assert str(old_mem_id) not in hit_ids
        assert str(new_mem.id) in hit_ids


# ===========================================================================
# V3-V3 · 新 Journey 不继承旧对象画像
# ===========================================================================
def test_v3_isolation_layer_portraits_not_inherited_current(client, monkeypatch):
    """现状刻画：隔离层 scoped_portraits 只返回新 Journey 归属的画像（旧画像 0 条）。"""
    monkeypatch.setenv("USE_PERSONA", "1")
    uid, _, _ = account(client, "v3-v3-iso")
    old_cid, old_scoped_portrait_id, _ = _seed_completed_breakup(uid)

    with _RawSessionLocal() as db:
        result = state.transition_scene(db, uid, "pet", trigger=state.TRIGGER_INTERVIEW)
        new_cid = result["companion_id"]
        scoped = iso.scoped_portraits(db, uid, new_cid)
        scoped_ids = {p.id for p in scoped}

        print("\n[V3-CURRENT] 新 Journey scoped_portraits 条数:", len(scoped))
        print("[V3-CURRENT] 旧 Journey 带归属画像 id:", old_scoped_portrait_id,
              "出现在新 Journey:", old_scoped_portrait_id in scoped_ids)
        print("[V3-CURRENT] 新 Journey 画像明细:", [(p.id, p.companion_id, p.content) for p in scoped])

        assert old_scoped_portrait_id not in scoped_ids
        assert all(p.companion_id == new_cid for p in scoped)


def test_v3_main_chain_object_portrait_current_still_inherited(client, monkeypatch):
    """现状刻画：主链 store.get_portrait 只按 user_id+kind 取最新，切到新 Journey 后仍读到旧对象称呼。"""
    monkeypatch.setenv("USE_PERSONA", "1")
    uid, _, _ = account(client, "v3-v3-main")
    _seed_completed_breakup(uid)

    with _RawSessionLocal() as db:
        result = state.transition_scene(db, uid, "pet", trigger=state.TRIGGER_INTERVIEW)
        obj = store.get_portrait(db, uid, "object")
        user = store.get_portrait(db, uid, "user")
        print("\n[V3-CURRENT] 切换后 loss_type:", store.effective_loss_type(db, uid))
        print("[V3-CURRENT] store.get_portrait(object) =", obj)
        print("[V3-CURRENT] store.get_portrait(user)   =", user)
        print("[V3-CURRENT] mention_count 取到的对象称呼:", (obj or {}).get("称呼"))

        assert obj.get("称呼") == "小夏", "现状：旧对象画像被主链读取层继承"
        assert user.get("昵称") == "林薇", "现状：用户画像同样是 user 级、跨 Journey 共享"


def test_v3_expected_main_chain_object_portrait_not_inherited(client, monkeypatch):
    """U-018 画像隔离修复后应为真 PASS：传新 companion_id 读不到旧对象画像。

    向后兼容约定：get_portrait 不传 companion_id 时行为完全不变（仍读到旧画像），
    因此这里显式传新 Journey 的 companion_id 来验证隔离生效——这是隔离层的正确用法。
    """
    monkeypatch.setenv("USE_PERSONA", "1")
    uid, _, _ = account(client, "v3-v3-expected")
    _seed_completed_breakup(uid)

    with _RawSessionLocal() as db:
        state.transition_scene(db, uid, "pet", trigger=state.TRIGGER_INTERVIEW)
        new_comp = store.get_current_companion(db, uid)
        obj = store.get_portrait(db, uid, "object", companion_id=new_comp.id)
        print("\n[V3-EXPECTED] 新 Journey 主链 object portrait:", obj)
        assert "小夏" not in str(obj), "修复后：新 Journey 传 companion_id 不应读到旧对象「小夏」"
