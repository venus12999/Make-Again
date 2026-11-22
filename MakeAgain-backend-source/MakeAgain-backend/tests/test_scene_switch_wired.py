"""U-018 · 场景切换主链路接线验证（A 包，离线零 hy4/key）。

覆盖：
  A-V1 自然语言"我开始新恋情了"触发切换（companion.chat 直接调用 + /api/chat 端到端）
  A-V2 普通倾诉 / 第三人称叙述不误触发
  A-V3 开关 USE_PERSONA 默认关 / 显式开
  A-V5 切换后旧 Journey 归档只读（复用现有 archive_companion，不另造逻辑）

口径：conftest 已置 MOCK_LLM=true，本文件再显式 monkeypatch LLMClient.chat 为确定性桩，
不读 work/local-test-key.txt、不发网络、不跑真 hy4。
画像按 companion_id 过滤属 B 包 GAP-V3，本文件不测（守包边界）。
"""
from __future__ import annotations

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
    def fake_chat(self, messages, *args, **kwargs):
        return {"content": reply, "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "tool_calls": []}
    monkeypatch.setattr(LLMClient, "chat", fake_chat)


def _seed_completed_breakup(uid: str) -> tuple[str, str]:
    """构造"已完成 breakup 旅程"的用户：当前 breakup companion + 一条归属旧 Journey 的记忆。"""
    with _RawSessionLocal() as db:
        store.get_or_create_user(db, uid, loss_type="breakup")
        st = store.get_or_create_user_state(db, uid)
        st.consent_accepted_at = utcnow()
        db.commit()
        c = store.get_current_companion(db, uid)
        store.update_companion(db, c.id, relationship_type="breakup", subject_name="小夏")
        scoped = Portrait(user_id=uid, companion_id=c.id, kind="object",
                          content={"称呼": "小夏", "关系": "前任"}, status="confirmed", version=1)
        db.add(scoped)
        db.commit()
        db.refresh(scoped)
        old_mem = iso.scoped_add_memory(db, uid, c.id, content="和小夏分手了，还是会想起她", importance=9.0)
        return c.id, old_mem.id


# --------------------------------------------------------------------------- A-V1 检测单元
def test_detect_scene_transition_positive_and_negative(monkeypatch):
    """纯函数检测：第一人称进入新关系命中；怀念/第三人称叙述不命中。"""
    monkeypatch.setenv("USE_PERSONA", "1")
    positives = ("我开始新恋情了", "我最近开始新的恋情了", "我们在一起了", "我谈恋爱了", "我交了个男朋友")
    negatives = (
        "我想起前任了，还是很难过",
        "我还是放不下他，心里空落落的",
        "前任开始新恋情了，我心里难受",
        "听说我前任开始新恋情了，我心里复杂",
        "我今天吃了顿饭，还行",
    )
    for msg in positives:
        d = state.detect_scene_transition(None, "u", msg, "breakup")
        print(f"[DETECT+] {msg} -> {d}")
        assert d["should_switch"] is True
        assert d["target_scene"] in state.VALID_SCENES
        assert d["new_loss_type"] == d["target_scene"]
    for msg in negatives:
        d = state.detect_scene_transition(None, "u", msg, "breakup")
        print(f"[DETECT-] {msg} -> {d}")
        assert d["should_switch"] is False
        assert d["target_scene"] is None


def test_detect_scene_transition_disabled_by_switch(monkeypatch):
    """USE_PERSONA 关（默认）时检测整体 no-op。"""
    monkeypatch.delenv("USE_PERSONA", raising=False)
    d = state.detect_scene_transition(None, "u", "我开始新恋情了", "breakup")
    assert d["should_switch"] is False
    assert "USE_PERSONA" in d["reason"]


# --------------------------------------------------------------------------- A-V1 chat 接线
def test_natural_language_switches_scene_via_chat(client, monkeypatch):
    """A-V1：用户说'我开始新恋情了'→ 旧 Journey 归档 + 新 Journey 建立（companion.chat 直调）。"""
    monkeypatch.setenv("USE_PERSONA", "1")
    _mock_llm(monkeypatch)
    uid, _, _ = account(client, "wired-v1-chat")
    old_cid, _ = _seed_completed_breakup(uid)

    with _managed_db() as db:
        before = store.effective_loss_type(db, uid)
        companion_agent.chat(db, uid, "我开始新恋情了", LLMClient(), session_id="wired-v1-chat")
        after = store.effective_loss_type(db, uid)
        companions = store.list_companions(db, uid)
        old = store.get_companion(db, old_cid)
        current = store.get_current_companion(db, uid)

        print("\n[Wired-V1] 切换前 loss_type:", before, "| 切换后:", after)
        print("[Wired-V1] companion 段数:", len(companions),
              "| archived_at:", [str(c.archived_at) for c in companions])
        assert before == "breakup"
        assert after == "friend"  # 新恋情→通用陪伴场景（不再引用旧对象）
        assert len(companions) == 2
        assert old.archived_at is not None
        assert current.id != old_cid and current.archived_at is None
        assert current.chat_session_id == "wired-v1-chat"  # 本轮会话 id 延续到新旅程


# --------------------------------------------------------------------------- A-V2 不误触发
def test_ordinary_grief_does_not_switch(client, monkeypatch):
    """A-V2：普通倾诉不切场景。"""
    monkeypatch.setenv("USE_PERSONA", "1")
    _mock_llm(monkeypatch)
    uid, _, _ = account(client, "wired-v2-grief")
    old_cid, _ = _seed_completed_breakup(uid)

    with _managed_db() as db:
        companion_agent.chat(db, uid, "我想起前任了，还是很难过", LLMClient(), session_id="wired-v2-grief")
        after = store.effective_loss_type(db, uid)
        companions = store.list_companions(db, uid)
        print("\n[Wired-V2] 普通倾诉后 loss_type:", after, "| companion 段数:", len(companions))
        assert after == "breakup"
        assert len(companions) == 1
        assert store.get_companion(db, old_cid).archived_at is None


def test_third_person_new_relationship_does_not_switch(client, monkeypatch):
    """A-V2：讲别人的新恋情不切场景。"""
    monkeypatch.setenv("USE_PERSONA", "1")
    _mock_llm(monkeypatch)
    uid, _, _ = account(client, "wired-v2-third")
    old_cid, _ = _seed_completed_breakup(uid)

    with _managed_db() as db:
        msg = "听说我前任开始新恋情了，我心里好难受"
        companion_agent.chat(db, uid, msg, LLMClient(), session_id="wired-v2-third")
        after = store.effective_loss_type(db, uid)
        print("\n[Wired-V2] 第三人称话语:", msg, "| loss_type:", after)
        assert after == "breakup"
        assert len(store.list_companions(db, uid)) == 1
        assert store.get_companion(db, old_cid).archived_at is None


# --------------------------------------------------------------------------- A-V3 开关
def test_switch_disabled_by_default(client, monkeypatch):
    """A-V3：USE_PERSONA 默认关时，即使说了触发语也不切换。"""
    monkeypatch.delenv("USE_PERSONA", raising=False)
    _mock_llm(monkeypatch)
    uid, _, _ = account(client, "wired-v3-off")
    old_cid, _ = _seed_completed_breakup(uid)

    with _managed_db() as db:
        companion_agent.chat(db, uid, "我开始新恋情了", LLMClient(), session_id="wired-v3-off")
        after = store.effective_loss_type(db, uid)
        print("\n[Wired-V3] 开关关时 loss_type:", after, "| companion 段数:", len(store.list_companions(db, uid)))
        assert after == "breakup"
        assert len(store.list_companions(db, uid)) == 1
        assert store.get_companion(db, old_cid).archived_at is None


# --------------------------------------------------------------------------- A-V5 归档只读
def test_archived_old_journey_still_read_only_after_chat_switch(client, monkeypatch):
    """A-V5：自然语言触发的切换同样复用 archive_companion，旧 Journey 归档只读。"""
    monkeypatch.setenv("USE_PERSONA", "1")
    _mock_llm(monkeypatch)
    uid, _, _ = account(client, "wired-v5-archive")
    old_cid, old_mem_id = _seed_completed_breakup(uid)

    with _managed_db() as db:
        companion_agent.chat(db, uid, "我开始新恋情了", LLMClient(), session_id="wired-v5-archive")
        new_cid = store.get_current_companion(db, uid).id
        old = store.get_companion(db, old_cid)
        read = iso.scoped_get_memory(db, uid, old_cid, old_mem_id)
        with pytest.raises(iso.ScopeDenied) as ew:
            iso.assert_scope(db, uid, old_cid, old_mem_id, operation="write")
        with pytest.raises(iso.ScopeDenied) as ea:
            iso.scoped_add_memory(db, uid, old_cid, content="归档后续写")
        scope = iso.build_scope(db, uid, new_cid)

        print("\n[Wired-V5] old archived_at:", old.archived_at,
              "| ending_stage:", old.ending_stage, "| ending_ritual:", old.ending_ritual)
        print("[Wired-V5] 归档读 id:", read.id, "| 写拒绝:", ew.value.code, "| 新增拒绝:", ea.value.code)
        print("[Wired-V5] 旧记忆 in 新 scope:", str(old_mem_id) in scope.allowed_ids)
        assert old.archived_at is not None
        assert old.ending_stage == "complete"
        assert old.ending_ritual == "scene_transition"
        assert read.id == old_mem_id
        assert ew.value.code == "ARCHIVED_READ_ONLY"
        assert ea.value.code == "ARCHIVED_READ_ONLY"
        assert str(old_mem_id) not in scope.allowed_ids


# --------------------------------------------------------------------------- /api/chat 端到端
def test_api_chat_switch_returns_new_companion(client, monkeypatch):
    """A-V1(API)：/api/chat 触发切换后回包是新陪伴 id，前端可继续请求（不 409）。"""
    monkeypatch.setenv("USE_PERSONA", "1")
    _mock_llm(monkeypatch)
    uid, headers, s = account(client, "wired-v1-api")
    old_cid, _ = _seed_completed_breakup(uid)

    resp = client.post("/api/chat", headers=headers, json={
        "user_id": uid, "message": "我开始新恋情了", "session_id": s["chat_session_id"],
    })
    print("\n[Wired-API] status:", resp.status_code, "| body:", str(resp.json())[:200])
    assert resp.status_code == 200, resp.json()
    assert resp.json()["companion_id"] != old_cid
    assert resp.json()["session_id"] == s["chat_session_id"]

    state_resp = client.get("/api/state", headers=headers)
    assert state_resp.status_code == 200
    payload = state_resp.json()
    assert len(payload["companions"]) == 2
    assert payload["current_companion_id"] != old_cid
    print("[Wired-API] /api/state relationship_type:", payload["relationship_type"],
          "| companions:", len(payload["companions"]))
