"""WP1 · pydantic-ai 编排行为测试（新建文件，不改现有测试）。

覆盖第 5 节 WP1 行为验收表：typed tool 卡片、todo 参数校验重试、
空 tool_calls 时的照片关键词兜底、危机硬门槛、结构化动作备选方案、开关回滚。
"""
from __future__ import annotations

import json

import pytest

# 必须最先导入 agent_v2：它在 import 期安装 opentelemetry._events 兼容 shim。
from agent import agent_v2 as v2
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from agent import companion
from emotion.crisis import CRISIS_MESSAGE
from gateway.client import LLMClient


@pytest.fixture
def db():
    from memory.db import _RawSessionLocal
    from memory.scope import current_db

    session = _RawSessionLocal()
    session.info.update(managed=True, borrowed=True)
    token = current_db.set(session)
    yield session
    current_db.reset(token)
    session.finish()
    session.close()


def _text_model(text: str = "（mock）我在这里，愿意听你慢慢说。") -> FunctionModel:
    """模拟"模型 tool_calls 为空"：只吐正文，不发起任何工具调用。"""
    return FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(text)]))


def _decision_model(decision: dict) -> FunctionModel:
    """备选方案：直接吐出结构化动作决策 JSON。"""
    return FunctionModel(
        lambda messages, info: ModelResponse(
            parts=[TextPart(json.dumps(decision, ensure_ascii=False))]
        )
    )


def _five_then_three_model():
    """模拟模型第一次给出 5 个任务（非法），Pydantic 拒绝后重试给 3 个。"""
    state = {"calls": 0}

    def fn(messages, info):
        if messages and any(isinstance(p, ToolReturnPart) for p in getattr(messages[-1], "parts", [])):
            return ModelResponse(parts=[TextPart("好，我陪你从一小步开始。")])
        state["calls"] += 1
        tasks = ["任务一", "任务二", "任务三", "任务四", "任务五"] if state["calls"] == 1 \
            else ["下楼走十分钟", "喝一杯温水", "给朋友发条消息"]
        return ModelResponse(parts=[ToolCallPart("suggest_todo_list", {"tasks": tasks})])

    return FunctionModel(fn), state


def _boom_model(*args, **kwargs):
    raise AssertionError("危机命中时不得进入 agent / 构造模型")


# ---------------------------------------------------------------------------
# 开关：默认关闭，关闭时走原有链路
# ---------------------------------------------------------------------------
def test_switch_defaults_off(monkeypatch):
    monkeypatch.delenv("USE_PYDANTIC_AI", raising=False)
    assert companion._use_pydantic_ai() is False
    monkeypatch.setenv("USE_PYDANTIC_AI", "0")
    assert companion._use_pydantic_ai() is False
    monkeypatch.setenv("USE_PYDANTIC_AI", "false")
    assert companion._use_pydantic_ai() is False


def test_switch_on_routes_to_v2(monkeypatch):
    monkeypatch.setenv("USE_PYDANTIC_AI", "1")
    assert companion._use_pydantic_ai() is True
    monkeypatch.setattr(v2, "chat_v2", lambda *a, **kw: {"sentinel": "v2"})
    assert companion.chat(None, "u", "你好", None) == {"sentinel": "v2"}


def test_switch_off_keeps_legacy_item_payload(monkeypatch, db):
    """关闭时走 v1 关键词兜底：item_name 为空串（v2 typed tool 才会填「围巾」）。"""
    monkeypatch.delenv("USE_PYDANTIC_AI", raising=False)
    monkeypatch.setattr(v2, "chat_v2", lambda *a, **kw: pytest.fail("不应进入 v2"))
    result = companion.chat(db, "u_legacy", "他送我的那条围巾还在", LLMClient())
    assert result["tool"]["type"] == "item_keep"
    assert result["tool"]["item_name"] == ""


# ---------------------------------------------------------------------------
# 行为验收
# ---------------------------------------------------------------------------
def test_scarf_produces_typed_item_card(db):
    result = v2.chat_v2(db, "u_scarf", "他送我的那条围巾还在", LLMClient())
    tool = result["tool"]
    assert tool is not None
    assert tool["type"] in ("item_let_go", "item_keep")
    assert "围巾" in tool["item_name"]
    assert tool["upload"] is True
    assert result["crisis"]["is_crisis"] is False


def test_todo_five_tasks_rejected_and_retried_to_three(db):
    model, state = _five_then_three_model()
    result = v2.chat_v2(db, "u_todo", "我最近睡不好", LLMClient(), model=model)
    tool = result["tool"]
    assert tool is not None and tool["type"] == "todo_list"
    assert 1 <= len(tool["tasks"]) <= 3
    assert all(1 <= len(t["content"]) for t in tool["tasks"])
    # 第一次 5 任务被 Pydantic 拒绝，第二次 3 任务才真正执行工具。
    assert state["calls"] == 2


def test_empty_tool_calls_still_triggers_photo_keyword_fallback(db, monkeypatch):
    monkeypatch.setenv("WP1_USE_STRUCTURED_ACTION", "1")
    result = v2.chat_v2(
        db, "u_photo", "翻到一张老照片，突然很想他", LLMClient(),
        model=_text_model(), action_model=_decision_model({"action": "none", "reply": ""}),
    )
    assert result["tool"] is not None
    assert result["tool"]["type"] == "photo_upload"
    assert result["tool"]["upload"] is True


def test_crisis_never_enters_agent(monkeypatch, db):
    result = v2.chat_v2(db, "u_crisis", "我不想活了", LLMClient(), model=_boom_model)
    assert result["crisis"]["is_crisis"] is True
    assert result["crisis"]["level"] == 3
    assert result["reply"] == CRISIS_MESSAGE
    assert "400-161-9995" in CRISIS_MESSAGE
    assert result["emotion"] is None


def test_structured_action_decision_fallback(db, monkeypatch):
    """备选方案：typed tool 无调用时，结构化动作决策仍可产出 todo 卡片。"""
    monkeypatch.setenv("WP1_USE_STRUCTURED_ACTION", "1")
    result = v2.chat_v2(
        db, "u_action", "我最近睡不好，想做点什么", LLMClient(),
        model=_text_model(),
        action_model=_decision_model({
            "action": "todo", "reply": "我陪你从一小步开始。",
            "tasks": ["下楼走十分钟", "喝一杯温水"],
        }),
    )
    assert result["tool"] is not None and result["tool"]["type"] == "todo_list"
    assert len(result["tool"]["tasks"]) == 2


def test_structured_action_decision_rejects_four_tasks(db, monkeypatch):
    """备选方案的 output_type 同样受 Pydantic 约束：4 个任务被拒。"""
    monkeypatch.setenv("WP1_USE_STRUCTURED_ACTION", "1")
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        v2.ActionDecision(action="todo", reply="", tasks=["1", "2", "3", "4"])


def test_no_tools_followup_has_no_card(db):
    result = v2.chat_v2(
        db, "u_followup", "好，就这么办", LLMClient(), no_tools=True, model=_text_model(),
    )
    assert result["tool"] is None
    assert result["reply"]


def test_typed_tools_have_pydantic_constrained_schema():
    """4 个工具都是 @agent.tool 类型化函数，todo.tasks 的 maxLength=3 写入 JSON schema。"""
    tools = {t.name: t for t in v2._tool_agent()._function_toolset.tools.values()}
    assert set(tools) >= {
        "suggest_item_ritual", "suggest_photo_upload",
        "suggest_todo_list", "suggest_self_note",
    }
    schema = tools["suggest_todo_list"].function_schema.json_schema
    assert schema["properties"]["tasks"]["maxItems"] == 3
    assert schema["properties"]["tasks"]["minItems"] == 1
