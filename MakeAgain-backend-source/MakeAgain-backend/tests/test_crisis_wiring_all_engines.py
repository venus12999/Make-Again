# -*- coding: utf-8 -*-
"""D-073 包 R2 · 危机闭环接线覆盖四个专属引擎（breakup/loved_one/pet/agent_v2）。

口径（自造 fixture，纯离线，不启动 8000、不碰真 key）：
  * 语料 "我想死" 是 synthetic 短句，命中 emotion.crisis.detect() 硬门槛 level=3；
  * 开关关 → 四引擎返回的 crisis 与 detect() 序列化后逐字节全等、无 flow、零落库；
  * 开关开 → 四引擎都挂 crisis["flow"] 且高危事件在 crisis_store 可查（独立会话真提交读回）；
  * 失败 swallow（落库/闭环探测）→ 危机话术与硬门槛字段不退化；
  * D-070 的 cc.maybe_compact 摘要注入链保持原样（源码级回归守卫）。
真后端 API / 真实模型危机评测 NOT_RUN；本测试只证明离线接线与回退语义。
"""
from __future__ import annotations

import inspect
import json

import pytest

# agent_v2 必须在 import 期安装 opentelemetry shim，先导入它（沿用 test_agent_v2 口径）
from agent import agent_v2 as v2
from agent import breakup, loved_one, pet
from agent.crisis_wiring import crisis_flow_enabled, detect_crisis
from emotion import crisis, crisis_flow
from emotion.crisis import CRISIS_MESSAGE
from gateway.client import LLMClient
from memory import crisis_store
from memory.db import _RawSessionLocal
from memory.scope import current_db

CRISIS_TEXT = "我想死"

# 四引擎统一调用口径：危机会在 LLM 前硬门槛返回，client 不会被消费。
ENGINES = [
    ("breakup", lambda db, u, sid: breakup.breakup_chat(db, u, CRISIS_TEXT, LLMClient(), sid)),
    ("loved_one", lambda db, u, sid: loved_one.loved_one_chat(db, u, CRISIS_TEXT, LLMClient(), sid)),
    ("pet", lambda db, u, sid: pet.pet_chat(db, u, CRISIS_TEXT, LLMClient(), sid)),
    ("agent_v2", lambda db, u, sid: v2.chat_v2(db, u, CRISIS_TEXT, LLMClient(), sid)),
]
ENGINE_IDS = [name for name, _ in ENGINES]


def _json(d) -> str:
    """稳定序列化，作为『逐字节一致』证据（键排序 + 不转义中文）。"""
    return json.dumps(d, ensure_ascii=False, sort_keys=True)


@pytest.fixture
def db():
    """与 test_agent_v2 同口径的隔离会话（managed+borrowed，teardown 真提交）。"""
    session = _RawSessionLocal()
    session.info.update(managed=True, borrowed=True)
    token = current_db.set(session)
    yield session
    current_db.reset(token)
    session.finish()
    session.close()


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    """每例从开关未设置 + 危机内存态干净开始；不污染其他测试。"""
    monkeypatch.delenv("USE_CRISIS_FLOW", raising=False)
    crisis_flow.reset_crisis_state()
    yield
    crisis_flow.reset_crisis_state()


def _events(user: str):
    """用独立会话读回，规避 managed session 的读快照（D-070 同类坑）。"""
    with _RawSessionLocal() as d:
        return crisis_store.list_crisis_events(d, user), crisis_store.count_crisis_events(d, user)


# ===========================================================================
# ① 开关关：四引擎 crisis 字段与 detect() 逐字节一致，零落库、无 flow
# ===========================================================================
@pytest.mark.parametrize("name,call", ENGINES, ids=ENGINE_IDS)
def test_off_engine_crisis_byte_identical(monkeypatch, db, name, call):
    monkeypatch.setenv("USE_CRISIS_FLOW", "0")
    user = f"off-{name}"
    out = call(db, user, f"s-off-{name}")
    assert out["reply"] == CRISIS_MESSAGE
    assert _json(out["crisis"]) == _json(crisis.detect(CRISIS_TEXT)), (
        name, _json(out["crisis"]), _json(crisis.detect(CRISIS_TEXT))
    )
    assert "flow" not in out["crisis"]
    rows, n = _events(user)
    assert n == 0 and rows == []


@pytest.mark.parametrize("switch", [None, "0", "false", "no", "OFF", ""])
def test_off_switch_values_helper_equals_detect(monkeypatch, db, switch):
    """共享入口本身：所有 falsy/未设置取值 + 危机/非危机语句都逐字节等价 detect。"""
    if switch is None:
        monkeypatch.delenv("USE_CRISIS_FLOW", raising=False)
    else:
        monkeypatch.setenv("USE_CRISIS_FLOW", switch)
    assert crisis_flow_enabled() is False
    for msg in (CRISIS_TEXT, "今天天气不错", "我不想死"):
        got = detect_crisis(db, "u-off-helper", msg, None)
        exp = crisis.detect(msg)
        assert _json(got) == _json(exp)
        assert "flow" not in got


def test_on_switch_but_non_crisis_still_equals_detect(monkeypatch, db):
    """开关开但硬门槛未命中时不得叠加闭环字段（与 H1 语义一致）。"""
    monkeypatch.setenv("USE_CRISIS_FLOW", "1")
    for msg in ("今天天气不错", "我想喝杯水"):
        assert _json(detect_crisis(db, "u-noncrisis", msg, None)) == _json(crisis.detect(msg))


# ===========================================================================
# ② 开关开：四引擎高危事件落 crisis_store 可查（复用 D-071 表 + save_crisis_event）
# ===========================================================================
@pytest.mark.parametrize("name,call", ENGINES, ids=ENGINE_IDS)
def test_on_engine_persists_crisis_event(monkeypatch, db, name, call):
    monkeypatch.setenv("USE_CRISIS_FLOW", "1")
    user = f"on-{name}"
    out = call(db, user, f"s-on-{name}")
    c = out["crisis"]
    assert c["is_crisis"] is True and c["level"] == 3
    assert c["hit"] == crisis.detect(CRISIS_TEXT)["hit"]  # 硬门槛字段没被闭环改写
    flow = c["flow"]
    assert flow["graded_script"] == "level3"
    assert flow["need_confirm"] is True
    assert flow["review_required"] is True
    assert flow["event"]["state"] == "pending_confirm"

    rows, n = _events(user)
    assert n == 1, rows
    row = rows[0]
    assert row["id"] == flow["event_id"]
    assert row["text"] == CRISIS_TEXT
    assert row["level"] == 3
    assert row["keywords"] == ["想死"]
    assert row["review_required"] is True
    # 用户可见话术不被 flow 替换（引擎仍用自身 CRISIS_MESSAGE 常量）
    assert out["reply"] == CRISIS_MESSAGE
    # 用户隔离：别人的事件数恒 0
    assert _events("someone-else-not-in-this-test")[1] == 0


# ===========================================================================
# 失败 swallow：落库 / 闭环探测失败都不得破坏危机话术与硬门槛字段
# ===========================================================================
@pytest.mark.parametrize("name,call", ENGINES, ids=ENGINE_IDS)
def test_save_failure_swallowed(monkeypatch, db, name, call):
    monkeypatch.setenv("USE_CRISIS_FLOW", "1")

    def boom(*a, **kw):
        raise RuntimeError("simulated DB failure")

    monkeypatch.setattr(crisis_store, "save_crisis_event", boom)
    out = call(db, f"fail-save-{name}", "s")
    assert out["reply"] == CRISIS_MESSAGE
    assert out["crisis"]["level"] == 3
    assert "flow" in out["crisis"]  # 内存闭环字段仍在，只丢持久化


@pytest.mark.parametrize("name,call", ENGINES, ids=ENGINE_IDS)
def test_flow_probe_failure_falls_back_to_detect(monkeypatch, db, name, call):
    monkeypatch.setenv("USE_CRISIS_FLOW", "1")

    def boom(*a, **kw):
        raise RuntimeError("simulated flow failure")

    monkeypatch.setattr(crisis, "detect_with_flow", boom)
    out = call(db, f"fail-probe-{name}", "s")
    assert out["reply"] == CRISIS_MESSAGE
    assert _json(out["crisis"]) == _json(crisis.detect(CRISIS_TEXT))
    assert "flow" not in out["crisis"]


# ===========================================================================
# ③ D-070 摘要注入链没被本次改动破坏（源码级守卫，逐行确认未动）
# ===========================================================================
def test_d070_compaction_injection_lines_intact():
    for mod, fn in ((breakup, "breakup_chat"), (loved_one, "loved_one_chat"), (pet, "pet_chat")):
        src = inspect.getsource(getattr(mod, fn))
        assert "turns, summary_block = cc.maybe_compact(db, user_id, session_id, turns, client)" in src, fn
        assert "turns[-1:]" in src, fn
        assert "[此前对话摘要]" in src and "summary_block" in src, fn
    # agent_v2 历史上从未接 cc.maybe_compact（D-070 只接 companion + 三场景）；
    # 本次只把它的 detect 换成共享入口，recent_text 的 turns[-1:] 口径保持不变。
    assert "turns[-1:]" in inspect.getsource(v2.chat_v2)
