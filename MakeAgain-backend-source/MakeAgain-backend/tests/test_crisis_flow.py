"""WP4 · 危机触达闭环行为验收测试。

逐项对应 MASTER_PLAN_PARALLEL.md 第 5 节 WP4 行为验收表：
1. 输入"我想死" → 分级话术 + 热线 + 事件记录
2. 用户确认轮（不是直接结束）
3. level 2/3 不同话术 + level 3 人工复核入口
4. 冷却期（1 小时内重复高危不重复弹话术但仍记录）
5. 否定守卫 / 第三人称不误判
6. 开关默认关闭可回滚
"""
import pytest

from emotion import crisis, crisis_flow

T0 = 1_700_000_000.0  # 固定基准时间，冷却期用 now 参数确定性控制


@pytest.fixture
def flow_on(monkeypatch):
    monkeypatch.setenv("USE_CRISIS_FLOW", "1")
    crisis_flow.reset_crisis_state()
    yield
    crisis_flow.reset_crisis_state()


# ---- 1. 高危命中：分级 + 热线 + 落事件记录 ----------------------------------

def test_high_risk_grades_hotline_and_records(flow_on):
    r = crisis_flow.handle_crisis("我想死", "u1", session_id="s1", now=T0)
    assert r["is_crisis"] is True
    assert r["level"] == 3
    assert r["graded_script"] == "level3"
    assert "400-161-9995" in r["reply"]
    assert r["hotline"]
    evs = crisis_flow.get_events("u1")
    assert len(evs) == 1
    assert evs[0]["level"] == 3
    assert evs[0]["session_id"] == "s1"


# ---- 2. 用户确认轮 -----------------------------------------------------------

def test_confirm_round_is_not_direct_end(flow_on):
    r = crisis_flow.handle_crisis("我想自杀", "u2", now=T0)
    assert r["need_confirm"] is True
    assert "你现在安全吗" in r["confirm_prompt"]
    assert crisis_flow.get_pending_confirm("u2") is not None

    c = crisis_flow.confirm_crisis("u2", safe=True, now=T0 + 5)
    assert c["ok"] is True
    assert c["event"]["state"] == "confirmed_safe"
    assert crisis_flow.get_pending_confirm("u2") is None

    # 回答"不安全" → 升级人工复核
    crisis_flow.handle_crisis("我想死", "u2b", now=T0)
    c2 = crisis_flow.confirm_crisis("u2b", safe=False, now=T0 + 6)
    assert c2["event"]["state"] == "escalated"
    assert c2["event"]["review_required"] is True
    assert c2["event"]["review_entry"]["reason"] == "escalated_by_user_answer"


# ---- 3. level 2/3 不同话术 + level 3 人工复核入口 ----------------------------

def test_level2_and_level3_different_scripts_and_review(flow_on):
    r3 = crisis_flow.handle_crisis("我想死", "u3", now=T0)
    r2 = crisis_flow.handle_crisis("我撑不下去了,想解脱", "u4", now=T0)
    assert r2["level"] == 2 and r3["level"] == 3
    assert r2["graded_script"] == "level2" and r3["graded_script"] == "level3"
    assert r2["reply"] != r3["reply"]
    assert "400-161-9995" in r2["reply"] and "400-161-9995" in r3["reply"]

    assert r2["review_required"] is False
    assert r2["review_entry"] is None
    assert r3["review_required"] is True
    assert r3["review_entry"]["queue"] == "human_review"
    assert r3["review_entry"]["priority"] == "high"


# ---- 4. 冷却期 ---------------------------------------------------------------

def test_cooldown_suppresses_script_but_still_records(flow_on):
    r1 = crisis_flow.handle_crisis("我想死", "u5", now=T0)
    assert r1["cooldown_active"] is False
    assert r1["script_repeated"] is True

    r2 = crisis_flow.handle_crisis("我又想死了", "u5", now=T0 + 600)
    assert r2["cooldown_active"] is True
    assert r2["script_repeated"] is False
    assert r2["reply"] == crisis_flow.COOLDOWN_MESSAGE
    assert r2["reply"] != r1["reply"]  # 不是重复同样的完整话术（改为简短承接语）
    assert len(r2["reply"]) < len(r1["reply"])  # 简短化，避免骚扰
    assert len(crisis_flow.get_events("u5")) == 2  # 但仍记录两条

    # 超过 1 小时后重新弹完整话术
    r3 = crisis_flow.handle_crisis("我还是想死", "u5", now=T0 + 3600 + 1)
    assert r3["cooldown_active"] is False
    assert r3["script_repeated"] is True
    assert len(crisis_flow.get_events("u5")) == 3


# ---- 5. 否定守卫 / 第三人称不误判 --------------------------------------------

def test_negation_and_third_person_not_misjudged(flow_on):
    for text in ("我不想死", "妈妈自杀", "他跳楼了"):
        r = crisis_flow.handle_crisis(text, "u6", now=T0)
        assert r["is_crisis"] is False, text
        assert r["level"] == 0, text
        assert r["reply"] is None, text
        assert r["need_confirm"] is False, text
    assert crisis_flow.get_events("u6") == []


# ---- 6. 开关关闭可回滚 --------------------------------------------------------

def test_switch_off_is_legacy_rollback(monkeypatch):
    monkeypatch.setenv("USE_CRISIS_FLOW", "0")
    crisis_flow.reset_crisis_state()
    r = crisis_flow.handle_crisis("我想死", "u7", now=T0)
    assert r["is_crisis"] is True and r["level"] == 3
    assert r["reply"] == crisis.CRISIS_MESSAGE
    assert "need_confirm" not in r
    assert "review_required" not in r
    assert "graded_script" not in r
    assert crisis_flow.get_events() == []
    crisis_flow.reset_crisis_state()


# ---- 硬门槛零回归 -------------------------------------------------------------

def test_hard_gate_detect_unchanged(flow_on):
    assert crisis.detect("我想死") == {"level": 3, "hit": ["想死"], "is_crisis": True}
    assert crisis.detect("我撑不下去了,想解脱")["level"] == 2
    assert crisis.detect("我不想死")["level"] == 0
    assert crisis.detect("妈妈自杀")["level"] == 0
    assert crisis.detect("今天天气不错")["level"] == 0


def test_detect_with_flow_hook(flow_on):
    r = crisis.detect_with_flow("我想死", "u8", now=T0)
    assert r["level"] == 3 and r["need_confirm"] is True
