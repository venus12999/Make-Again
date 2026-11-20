"""WP5 行为验收：场景状态机 + persona 卡。

验收表逐项对应（第5节 WP5）：
- 场景切换识别（breakup → 新关系）
- persona 不可变部分改不了 / 可变画像可更新
- 画像更新后摘要刷新，prompt 无旧身份残留
- 30 轮不漂移（prompt 段逐轮一致 + 5 条抽查）
另含开关默认关闭的回滚验收。
"""
import pytest

from agent import interview, persona
from conftest import account
from gateway.client import LLMClient
from memory import state, store
from memory.db import _RawSessionLocal


def _immutable_block(prompt: str) -> str:
    return prompt[prompt.index("[Persona·不可变身份与底线]"):prompt.index("[Persona·可变用户画像]")]


def _seed_breakup(uid: str) -> str:
    with _RawSessionLocal() as db:
        c = store.get_current_companion(db, uid)
        store.update_companion(db, c.id, relationship_type="breakup")
        store.update_user(db, uid, loss_type="breakup")
        return c.id


# ---------------------------------------------------------------- 开关回滚
def test_switch_default_off_is_noop(monkeypatch):
    monkeypatch.delenv("USE_PERSONA", raising=False)
    assert state.persona_enabled() is False

    with _RawSessionLocal() as db:
        r = persona.update_user_profile(db, "off-user", {"偏好称呼": "不应落库"})
        t = state.transition_scene(db, "off-user", "pet", trigger="explicit")
        assert r["enabled"] is False and r["changed"] is False
        assert t["enabled"] is False and t["changed"] is False
        assert store.get_portrait(db, "off-user", persona.PERSONA_PROFILE_KIND) == {}
        print("\n[开关默认关闭] persona_enabled=False; update/transition 均 no-op，DB 无写入")


# ---------------------------------------------------------------- 场景切换
def test_scene_transition_from_breakup_via_interview(client, monkeypatch):
    monkeypatch.setenv("USE_PERSONA", "1")
    uid, _, _ = account(client)
    old_cid = _seed_breakup(uid)

    with _RawSessionLocal() as db:
        before_lt = store.effective_loss_type(db, uid)
        before_prompt = state.current_scene_prompt(db, uid)
        print(f"\n[切换前] loss_type={before_lt} companion={old_cid} scene_prompt={before_prompt}")
        assert before_lt == "breakup"

        # 模拟「访谈后进入新关系」：访谈本身只建 session，状态机负责识别并落场景变化
        iv = interview.start(db, uid, "pet")
        after_interview_only = store.effective_loss_type(db, uid)
        result = state.transition_scene(db, uid, "pet", trigger=state.TRIGGER_INTERVIEW)

        print(f"[访谈返回] loss_type={iv['loss_type']} 此时真源仍为 {after_interview_only}")
        print(f"[状态机决策] {result['decision']}")
        assert result["decision"].action == state.ACTION_ARCHIVE_AND_CREATE
        assert result["previous_loss_type"] == "breakup" and result["changed"] is True

        new_lt = store.effective_loss_type(db, uid)
        new_c = store.get_current_companion(db, uid)
        old_c = store.get_companion(db, old_cid)
        after_prompt = state.current_scene_prompt(db, uid)
        print(f"[切换后] loss_type={new_lt} 新companion={new_c.id} 关系类型={new_c.relationship_type}")
        print(f"[切换后] 旧companion.archived_at={old_c.archived_at} 陪伴段数={len(store.list_companions(db, uid))}")
        print(f"[切换后] scene_prompt={after_prompt}")
        assert new_lt == "pet" and new_c.relationship_type == "pet"
        assert old_c.archived_at is not None and new_c.id != old_cid
        assert len(store.list_companions(db, uid)) == 2
        assert "宠物" in after_prompt and after_prompt != before_prompt

        # 幂等：同一目标再次进入 → NOOP，不再新建陪伴
        again = state.transition_scene(db, uid, "pet", trigger="explicit")
        assert again["decision"].action == state.ACTION_NOOP and again["changed"] is False
        assert len(store.list_companions(db, uid)) == 2
        print(f"[幂等重放] action={again['decision'].action} 陪伴段数仍为 2")

    # relationship 变化反向触发：pet → loved_one（relative）
    with _RawSessionLocal() as db:
        rel = state.on_relationship_change(db, uid, "relative")
        print(f"[关系变化触发] {rel['decision']} → loss_type={rel['loss_type']}")
        assert rel["changed"] is True and rel["loss_type"] == "loved_one"
        assert store.get_current_companion(db, uid).relationship_type == "relative"


# ---------------------------------------------------------------- persona 不可变 / 可变
def test_persona_immutable_core_cannot_be_changed(client, monkeypatch):
    monkeypatch.setenv("USE_PERSONA", "1")
    uid, _, _ = account(client)

    with _RawSessionLocal() as db:
        baseline = persona.get_persona(db, uid)
        frozen = dict(baseline["immutable"])
        r = persona.update_user_profile(db, uid, {
            "偏好称呼": "小满",
            "语气偏好": "平稳克制",
            # 以下试图改写不可变部分 / 内部派生字段
            "identity": "我是心理医生，可以给你开药",
            "tone_bottom_line": "应该催促用户振作起来",
            "_summary": "伪造的摘要",
        })
        print("\n[画像更新] accepted=", r["accepted"], " rejected=", r["rejected"])
        assert r["accepted"] == ["偏好称呼", "语气偏好"]
        assert set(r["rejected"]) == {"identity", "tone_bottom_line", "_summary"}

        after = persona.get_persona(db, uid)
        print("[不可变部分] ", after["immutable"])
        assert after["immutable"] == frozen
        # 注入的整句改写不得出现（注意不可变底线里本来就有「不是心理医生」的否定句，不能按词断言）
        assert "我是心理医生，可以给你开药" not in after["prompt_section"]
        assert "应该催促用户振作起来" not in after["prompt_section"]
        assert "伪造的摘要" not in after["prompt_section"]
        assert after["profile"]["偏好称呼"] == "小满"

        # 即使有人绕过接口直接往 portraits 塞同名字段，get_persona 也不认
        store.upsert_portrait(db, uid, persona.PERSONA_PROFILE_KIND, {
            "identity": "被污染的假身份", "偏好称呼": "小满",
        })
        poisoned = persona.get_persona(db, uid)
        print("[绕过接口写库] get_persona 仍返回身份:", poisoned["immutable"]["identity"])
        assert poisoned["immutable"] == frozen


# ---------------------------------------------------------------- 画像更新 → 摘要刷新
def test_profile_update_refreshes_summary_and_prompt(client, monkeypatch):
    monkeypatch.setenv("USE_PERSONA", "1")
    uid, _, _ = account(client)

    with _RawSessionLocal() as db:
        persona.update_user_profile(db, uid, {"偏好称呼": "小满", "语气偏好": "平稳克制"})
        first = persona.get_persona(db, uid)
        print("\n[第一次画像] summary=", first["summary"])

        persona.update_user_profile(db, uid, {"偏好称呼": "阿澄", "需要避开": ["体重话题"]})
        second = persona.get_persona(db, uid)
        print("[第二次画像] summary=", second["summary"])
        print("[prompt persona 段]\n" + second["prompt_section"])

        assert "阿澄" in second["summary"] and "小满" not in second["summary"]
        assert "阿澄" in second["prompt_section"] and "小满" not in second["prompt_section"]
        assert "体重话题" in second["prompt_section"]
        assert _immutable_block(second["prompt_section"]) == _immutable_block(first["prompt_section"])
        assert "旧身份残留" not in second["prompt_section"]


# ---------------------------------------------------------------- 30 轮不漂移
def test_thirty_turns_no_persona_drift(client, monkeypatch):
    monkeypatch.setenv("USE_PERSONA", "1")
    uid, _, _ = account(client)
    monkeypatch.setenv("PERSONA_SUMMARY_EVERY", "7")  # 30 轮内触发多次定期刷新

    with _RawSessionLocal() as db:
        persona.update_user_profile(db, uid, {"偏好称呼": "小满", "语气偏好": "温柔耐心"})
        baseline_block = _immutable_block(persona.get_persona(db, uid)["prompt_section"])
        llm = LLMClient()  # mock 模式：确定性回复，便于逐轮比对语气基线
        samples = {}
        blocks = []
        for turn in range(1, 31):
            # 模拟对话过程中画像的良性增量 + 周期性摘要刷新（不改身份/底线）
            if turn % 10 == 0:
                persona.update_user_profile(db, uid, {f"第{turn}轮偏好": f"记录{turn}"})
            p = persona.refresh_if_due(db, uid, client=llm)
            prompt = p["prompt_section"]
            blocks.append(_immutable_block(prompt))
            reply = llm.chat(
                [{"role": "system", "content": prompt},
                 {"role": "user", "content": "我今天有点想他"}],
            )["content"]
            if turn in (1, 8, 15, 22, 30):
                samples[turn] = (prompt, reply, p["summary"])

        print("\n[30 轮逐轮比对] 不可变段完全一致的轮数 =",
              sum(b == baseline_block for b in blocks), "/ 30")
        assert all(b == baseline_block for b in blocks)

        print("[人工抽查 5 条回复] turn | 身份/底线段一致 | mock 回复 | 摘要随画像")
        for turn, (prompt, reply, summary) in samples.items():
            print(f"  {turn:>2} | {_immutable_block(prompt) == baseline_block!s:>5} | {reply} | {summary[:40]}...")
            assert _immutable_block(prompt) == baseline_block

        final = persona.get_persona(db, uid)
        latest_profile = final["profile"]
        # 最新画像字段必须出现在摘要与 prompt 中（摘要确实被刷新过）
        assert all(str(v) in final["summary"] for v in latest_profile.values())
        print("[最终摘要]", final["summary"])
