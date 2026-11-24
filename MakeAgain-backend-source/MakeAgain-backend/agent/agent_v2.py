"""WP1 · pydantic-ai 编排（typed tools + deps 注入）。

设计要点：
- 4 个动作工具用 ``@agent.tool`` 类型化函数定义，参数由 Pydantic 校验
  （todo.tasks 限 1~3 条，非法参数由 pydantic-ai 自动重试）；执行逻辑复用既有
  item/note/photo/todo 的 ``execute_tool``，不重复实现业务规则。
- deps 注入 db / user_id / settings；危机硬门槛在 agent.run 之前，命中直接返回；
  关键词兜底与记忆回写沿用 companion 的既有实现。
- 备选方案：当 typed tool call 未产出卡片时，走"结构化动作决策"（output_type=
  ActionDecision）再做一次 Pydantic 校验，最后仍有信号则走零 LLM 关键词兜底。
- 本模块只在 ``USE_PYDANTIC_AI`` 开启时被 companion.chat 引入；关闭时主链路零改动。
"""
from __future__ import annotations

import json
import os
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Annotated, Any, Literal

# ---------------------------------------------------------------------------
# 兼容 shim：pydantic-ai 0.8.1 在 import 期引用 opentelemetry._events，
# 某些环境的 opentelemetry-api>=1.44 移除了这个私有模块。仅在缺失时注入最小实现，
# 不降级全局依赖，也不启用 OTel instrumentation（本项目未使用）。
# ---------------------------------------------------------------------------
def _ensure_otel_events_shim() -> None:
    try:
        import opentelemetry._events  # noqa: F401
        return
    except ImportError:
        pass
    module = types.ModuleType("opentelemetry._events")

    class Event:  # noqa: D101 - 仅为 import 期符号占位
        def __init__(self, name: str, body: dict | None = None, **kwargs: Any):
            self.name = name
            self.body = body or {}

    class EventLogger:  # noqa: D101
        def emit(self, *args: Any, **kwargs: Any) -> None:
            return None

    class EventLoggerProvider:  # noqa: D101
        def get_event_logger(self, *args: Any, **kwargs: Any) -> EventLogger:
            return EventLogger()

    def get_event_logger_provider() -> EventLoggerProvider:  # noqa: D103
        return EventLoggerProvider()

    module.Event = Event
    module.EventLogger = EventLogger
    module.EventLoggerProvider = EventLoggerProvider
    module.get_event_logger_provider = get_event_logger_provider
    sys.modules.setdefault("opentelemetry._events", module)


_ensure_otel_events_shim()

from pydantic import BaseModel, Field, model_validator  # noqa: E402
from pydantic_ai import Agent, RunContext  # noqa: E402
from pydantic_ai.messages import (  # noqa: E402
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel  # noqa: E402
from pydantic_ai.models.openai import OpenAIChatModel  # noqa: E402
from pydantic_ai.providers.openai import OpenAIProvider  # noqa: E402

from agent import companion as _base  # noqa: E402  复用关键词检测/提示构建，避免重复实现
from agent import item as _item  # noqa: E402
from agent import note as _note  # noqa: E402
from agent import photo as _photo  # noqa: E402
from agent import todo as _todo  # noqa: E402
from config import get_settings  # noqa: E402
from emotion.classifier import classify  # noqa: E402
from emotion.crisis import CRISIS_MESSAGE, detect  # noqa: E402
from emotion.tone import ANALYZE, SHARED_RULE, pick_tone  # noqa: E402
from memory import recall, session_context, store, temporal  # noqa: E402
from memory.async_write import schedule_memory_write  # noqa: E402


# ---------------------------------------------------------------------------
# deps / 结构化动作决策
# ---------------------------------------------------------------------------
@dataclass
class CompanionDeps:
    """pydantic-ai 的 deps：工具函数通过 ctx.deps 访问业务上下文。"""

    db: Any
    user_id: str
    settings: Any
    prompt: str = ""
    tool_results: list[dict] = field(default_factory=list)


class ActionDecision(BaseModel):
    """备选方案的结构化输出：模型不吐 tool_calls 时也能稳定产出动作。"""

    action: Literal["item", "photo", "todo", "note", "none"] = Field(
        description="本轮要弹出的卡片动作；没有明确动作时选 none"
    )
    reply: str = Field(default="", description="给用户的一段温柔、具体的回复")
    item_name: str = Field(default="", description="物品名称，如「他送的围巾」")
    intent: Literal["keep", "let_go"] = Field(default="keep", description="keep=留念，let_go=放下")
    item_description: str = Field(default="", description="物品故事片段，可为空")
    photo_title: str = Field(default="", description="照片标题/场景")
    scene_description: str = Field(default="", description="照片背后的场景，可为空")
    tasks: list[str] = Field(default_factory=list, max_length=3, description="1~3 个具体短句")
    target: Literal["self", "other"] = Field(default="self", description="self=写给自己，other=写给他")
    opening: str = Field(default="", description="日记开场白，可为空")

    @model_validator(mode="after")
    def _todo_must_have_tasks(self) -> "ActionDecision":
        # 与 typed tool 同样的约束：todo 必须给 1~3 条具体任务，否则判为非法、触发重试。
        if self.action == "todo" and not 1 <= len(self.tasks) <= 3:
            raise ValueError("todo 动作必须带 1~3 个任务")
        return self


_EXECUTORS = {
    "suggest_item_ritual": _item.execute_tool,
    "suggest_photo_upload": _photo.execute_tool,
    "suggest_todo_list": _todo.execute_tool,
    "suggest_self_note": _note.execute_tool,
}

_PAYLOADS = {
    "suggest_item_ritual": _item.tool_payload,
    "suggest_photo_upload": _photo.tool_payload,
    "suggest_todo_list": _todo.tool_payload,
    "suggest_self_note": _note.tool_payload,
}

_DECISION_TO_TOOL = {
    "item": "suggest_item_ritual",
    "photo": "suggest_photo_upload",
    "todo": "suggest_todo_list",
    "note": "suggest_self_note",
}


def _call_json(name: str, arguments: dict) -> dict:
    """把 typed 参数还原成既有 execute_tool 认识的 tool_call 结构（复用内部逻辑）。"""
    return {"function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}


def _run_tool(deps: CompanionDeps, name: str, arguments: dict) -> str:
    result = _EXECUTORS[name](deps.db, deps.user_id, _call_json(name, arguments))
    result["_tool"] = name
    deps.tool_results.append(result)
    return json.dumps({k: v for k, v in result.items() if k != "_tool"}, ensure_ascii=False)


def _pick_payload(results: list[dict], reply: str) -> dict | None:
    """与 companion._execute_tools 相同的"取第一个要展示动作"逻辑。"""
    for r in results:
        if not r.get("surface"):
            continue
        payload = _PAYLOADS[r.get("_tool")]([r], reply)
        if payload:
            return payload
    return None


# ---------------------------------------------------------------------------
# Agent 构造（lru_cache：schema 只构建一次，prompt 通过 deps/instructions 注入）
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _instructions_agent() -> Agent:
    def instructions(ctx: RunContext[CompanionDeps]) -> str:
        return ctx.deps.prompt

    return Agent(
        None,
        instructions=instructions,
        deps_type=CompanionDeps,
        output_type=str,
        instrument=False,
        retries=2,
        output_retries=2,
        end_strategy="early",
    )


@lru_cache(maxsize=1)
def _plain_agent() -> Agent:
    """no_tools=True 的收尾轮：不注册任何工具，行为对齐 v1 的 tools=None。"""
    def instructions(ctx: RunContext[CompanionDeps]) -> str:
        return ctx.deps.prompt

    return Agent(
        None,
        instructions=instructions,
        deps_type=CompanionDeps,
        output_type=str,
        instrument=False,
        end_strategy="early",
    )


@lru_cache(maxsize=1)
def _action_agent() -> Agent:
    """备选方案：结构化动作决策（不注册工具，直接用 output_type 约束）。"""
    def instructions(ctx: RunContext[CompanionDeps]) -> str:
        return ctx.deps.prompt

    return Agent(
        None,
        instructions=instructions,
        deps_type=CompanionDeps,
        output_type=ActionDecision,
        instrument=False,
        output_retries=2,
    )


def _register_tools(agent: Agent) -> Agent:
    """4 个 typed tool：参数即 schema，Pydantic 校验失败会自动回喂模型重试。"""

    @agent.tool(name="suggest_item_ritual")
    def suggest_item_ritual(
        ctx: RunContext[CompanionDeps],
        item_name: Annotated[str, Field(min_length=1, description="物品名称，如「他送的围巾」")],
        intent: Annotated[Literal["keep", "let_go"], Field(description="keep=留念；let_go=想放下")],
        item_description: Annotated[str, Field(description="用户提到的物品故事片段，可为空")] = "",
    ) -> str:
        return _run_tool(ctx.deps, "suggest_item_ritual", {
            "item_name": item_name, "intent": intent, "item_description": item_description,
        })

    @agent.tool(name="suggest_photo_upload")
    def suggest_photo_upload(
        ctx: RunContext[CompanionDeps],
        photo_title: Annotated[str, Field(min_length=1, description="照片标题/场景")],
        scene_description: Annotated[str, Field(description="照片背后的场景，可为空")] = "",
    ) -> str:
        return _run_tool(ctx.deps, "suggest_photo_upload", {
            "photo_title": photo_title, "scene_description": scene_description,
        })

    @agent.tool(name="suggest_todo_list")
    def suggest_todo_list(
        ctx: RunContext[CompanionDeps],
        tasks: Annotated[list[str], Field(
            min_length=1, max_length=3,
            description="1~3 个具体、马上能做的小任务，每个一句话",
        )],
    ) -> str:
        return _run_tool(ctx.deps, "suggest_todo_list", {"tasks": tasks})

    @agent.tool(name="suggest_self_note")
    def suggest_self_note(
        ctx: RunContext[CompanionDeps],
        target: Annotated[Literal["self", "other"], Field(
            description="self=写给自己；other=写给他的一封不寄出的信",
        )],
        opening: Annotated[str, Field(description="一句开场白，可为空")] = "",
    ) -> str:
        return _run_tool(ctx.deps, "suggest_self_note", {"target": target, "opening": opening})

    return agent


@lru_cache(maxsize=1)
def _tool_agent() -> Agent:
    return _register_tools(_instructions_agent())


# ---------------------------------------------------------------------------
# 模型：生产走 DeepSeek（provider 自行配置 base_url/key，不改 gateway/client.py）；
# mock/无 key 时走确定性离线模型，保证离线可跑、可 spike、可测试。
# ---------------------------------------------------------------------------
def _extract_user_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        for part in getattr(message, "parts", []):
            if isinstance(part, UserPromptPart):
                return part.content if isinstance(part.content, str) else ""
    return ""


def _is_tool_result(messages: list[Any]) -> bool:
    if not messages:
        return False
    return any(isinstance(p, ToolReturnPart) for p in getattr(messages[-1], "parts", []))


# 具体纪念物名词优先于「送我的/留下的」这类泛化信号，保证 item_name 是真正的物件名。
_ITEM_NOUNS = (
    "围巾", "毛衣", "手链", "项链", "戒指", "耳环", "镯子", "手表", "挂坠", "胸针",
    "玩偶", "娃娃", "毛绒", "杯子", "眼镜", "香水", "吉他", "相机", "相框", "日记本",
    "衣服", "外套", "衬衫", "帽子", "鞋子", "礼物", "信物", "纪念品", "遗物", "旧物",
)


def _pick_item_name(text: str) -> str:
    for noun in _ITEM_NOUNS:
        if noun in text:
            return noun
    return "那件物品"


def _pick_photo_title(text: str) -> str:
    for hint in _base._PHOTO_MENTION_HINTS:
        if hint in text:
            return hint
    return "那张照片"


def _offline_tool_fn(messages: list[Any], info: Any) -> ModelResponse:
    """确定性模拟：按关键词发一个 typed tool call（离线 spike/测试用，非真实 DeepSeek）。"""
    if _is_tool_result(messages):
        return ModelResponse(parts=[TextPart("（mock）我在这里，愿意听你慢慢说。")])
    text = _extract_user_text(messages)
    if _base._detect_item_mention(text):
        return ModelResponse(parts=[ToolCallPart("suggest_item_ritual", {
            "item_name": _pick_item_name(text), "intent": "keep", "item_description": "",
        })])
    if _base._detect_photo_mention(text):
        return ModelResponse(parts=[ToolCallPart("suggest_photo_upload", {
            "photo_title": _pick_photo_title(text), "scene_description": "",
        })])
    if _base._detect_self_note_mention(text):
        return ModelResponse(parts=[ToolCallPart("suggest_self_note", {"target": "self", "opening": ""})])
    if _base._detect_todo_mention(text):
        return ModelResponse(parts=[ToolCallPart("suggest_todo_list", {"tasks": ["下楼走十分钟", "喝一杯温水"]})])
    return ModelResponse(parts=[TextPart("（mock）我在这里，愿意听你慢慢说。")])


def _offline_action_fn(messages: list[Any], info: Any) -> ModelResponse:
    text = _extract_user_text(messages)
    if _base._detect_item_mention(text):
        decision = {"action": "item", "reply": "", "item_name": _pick_item_name(text), "intent": "keep"}
    elif _base._detect_photo_mention(text):
        decision = {"action": "photo", "reply": "", "photo_title": _pick_photo_title(text)}
    elif _base._detect_self_note_mention(text):
        decision = {"action": "note", "reply": "", "target": "self"}
    elif _base._detect_todo_mention(text):
        decision = {"action": "todo", "reply": "", "tasks": ["下楼走十分钟", "喝一杯温水"]}
    else:
        decision = {"action": "none", "reply": ""}
    return ModelResponse(parts=[TextPart(json.dumps(decision, ensure_ascii=False))])


def _offline() -> bool:
    settings = get_settings()
    return bool(settings.mock_llm) or not bool(settings.llm_api_key)


def _make_model(override: Any = None) -> Any:
    if override is not None:
        return override
    if _offline():
        return FunctionModel(_offline_tool_fn)
    settings = get_settings()
    # DeepSeek provider 在这个工作包内自行配置，不改 gateway/client.py / config.py。
    provider = OpenAIProvider(
        base_url=settings.llm_base_url.rstrip("/"),
        api_key=settings.llm_api_key,
    )
    return OpenAIChatModel(settings.llm_fast_model, provider=provider)


def _make_action_model(override: Any = None) -> Any:
    if override is not None:
        return override
    if _offline():
        return FunctionModel(_offline_action_fn)
    settings = get_settings()
    provider = OpenAIProvider(
        base_url=settings.llm_base_url.rstrip("/"),
        api_key=settings.llm_api_key,
    )
    return OpenAIChatModel(settings.llm_fast_model, provider=provider)


def _structured_action_enabled() -> bool:
    """备选方案子开关：默认开（typed tool 未出卡片时兜底一次结构化决策）。"""
    return os.getenv("WP1_USE_STRUCTURED_ACTION", "1").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def _history_from_turns(turns: list[dict]) -> list[Any]:
    history: list[Any] = []
    for turn in turns:
        role, content = turn.get("role"), turn.get("content") or ""
        if role == "user":
            history.append(ModelRequest.user_text_prompt(content))
        elif role == "assistant":
            history.append(ModelResponse(parts=[TextPart(content)]))
    return history


def _date_key() -> str:
    offset = get_settings().timezone_offset_hours
    return (datetime.now() + timedelta(hours=offset)).strftime("%Y-%m-%d")


def chat_v2(db, user_id: str, message: str, client, session_id: str | None = None,
            no_tools: bool = False, *, model: Any = None, action_model: Any = None) -> dict:
    """USE_PYDANTIC_AI=1 时的陪伴入口，返回与 companion._chat_generic 相同的 dict 契约。"""
    store.get_or_create_user(db, user_id)

    # 1. 危机硬门槛：agent.run 之前，命中直接返回（绝不进 LLM）。
    crisis = detect(message)
    if crisis["is_crisis"]:
        session_context.append_turn(user_id, "user", message, session_id)
        session_context.append_turn(user_id, "assistant", CRISIS_MESSAGE, session_id)
        store.save_chat_turn(db, user_id, session_id, message, CRISIS_MESSAGE)
        return {"reply": CRISIS_MESSAGE, "crisis": crisis, "emotion": None, "recalled": 0}

    # 2. 召回 + 画像 + 今日主题 + 语气 + 提示（与 v1 同一套上下文构建）
    portraits = {
        "user": store.get_portrait(db, user_id, "user"),
        "object": store.get_portrait(db, user_id, "object"),
    }
    turns = session_context.get_turns(user_id, session_id)
    # U-013: 「正在聊」判据用最近 1 轮，跟 companion 主链路统一
    recent_text = " ".join(t["content"] for t in turns[-1:])
    memories = recall.recall(db, user_id, message, recent_text=recent_text)
    daily_pick = store.get_daily_pick(db, user_id, _date_key())
    tone = pick_tone(db, user_id, message=message)
    context_block = _base._context_block(db, user_id, message, portraits, memories, daily_pick)
    prompt = _base.SYSTEM_PROMPT.format(
        tone=tone["prompt"], rule=SHARED_RULE, context=context_block,
    )
    if tone["tone"] == ANALYZE:
        victim = _base._victim_context(db, user_id, message)
        if victim:
            prompt += (
                f"\n\n[安全边界] 检测到用户在被伤害情境({', '.join(victim)}),"
                "本次分析不做任何归咎于用户的表述,转为共情 + 识别不健康关系 + 重建自我。"
            )

    deps = CompanionDeps(db=db, user_id=user_id, settings=client.settings, prompt=prompt)
    history = _history_from_turns(turns)
    run_kwargs = dict(deps=deps, message_history=history)

    # 3. typed tools 编排（no_tools 收尾轮走无工具 agent）
    if no_tools:
        result = _plain_agent().run_sync(message, model=_make_model(model), **run_kwargs)
        reply = result.output or ""
        tool = None
    else:
        result = _tool_agent().run_sync(message, model=_make_model(model), **run_kwargs)
        reply = result.output or ""
        tool = _pick_payload(deps.tool_results, reply)

        # 4. 备选方案：结构化动作决策 output_type（typed tool 未出卡片时）
        if tool is None and _structured_action_enabled():
            try:
                decision: ActionDecision = _action_agent().run_sync(
                    message, model=_make_action_model(action_model), **run_kwargs,
                )
                if decision and decision.action != "none":
                    name = _DECISION_TO_TOOL[decision.action]
                    args = {
                        "item": {"item_name": decision.item_name, "intent": decision.intent,
                                 "item_description": decision.item_description},
                        "photo": {"photo_title": decision.photo_title,
                                  "scene_description": decision.scene_description},
                        "todo": {"tasks": decision.tasks},
                        "note": {"target": decision.target, "opening": decision.opening},
                    }[decision.action]
                    local_result = _EXECUTORS[name](db, user_id, _call_json(name, args))
                    local_result["_tool"] = name
                    tool = _PAYLOADS[name]([local_result], reply)
                    if decision.reply and not reply:
                        reply = decision.reply
            except Exception:  # noqa: BLE001 - 备选方案失败不阻断主链路，继续走关键词兜底
                pass

        # 5. 零 LLM 关键词兜底：与 v1 完全相同的 4 条 detect + 卡片逻辑
        if tool is None:
            if _base._detect_photo_mention(message):
                tool = _photo.tool_payload(
                    [{"surface": True, "photo_title": "", "scene_description": ""}], reply)
            if tool is None and _base._detect_self_note_mention(message):
                tool = _note.tool_payload([{"surface": True, "target": "self", "opening": ""}], reply)
            if tool is None and _base._detect_todo_mention(message):
                tool = _todo.fallback_todo(db, user_id, client, message, reply)
            if tool is None and _base._detect_item_mention(message):
                tool = _item.tool_payload(
                    [{"surface": True, "intent": "keep", "item_name": "", "item_description": ""}], reply)

    # 6. 记忆回写：与 v1 完全相同（异步抽取 + 会话上下文 + 落库聊天历史）
    schedule_memory_write(user_id, message)
    session_context.append_turn(user_id, "user", message, session_id)
    session_context.append_turn(user_id, "assistant", reply, session_id)
    store.save_chat_turn(db, user_id, session_id, message, reply)

    usage = result.usage()
    return {
        "reply": reply,
        "crisis": crisis,
        "emotion": classify(message),
        "recalled": len(memories),
        "tone": tone["tone"],
        "tool": tool,
        "usage": {
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
        },
        "memory_context": context_block,
    }

