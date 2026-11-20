"""模型网关:OpenAI 兼容的 LLM 客户端,默认指向 deepseek-v4-pro。

传输层(WHTTP/SSE/重试/取消)交给官方 openai SDK,这里只做业务层封装:
- mock 模式:无 key 时返回确定性占位回复,保证链路离线可跑通;生产禁止 mock。
- chat / chat_json / chat_stream 接口保持不变,24 个调用方无需改动。
- USE_INSTRUCTOR=1 时 chat_json 走 instructor(动态模型校验 + 自动重试),
  失败或未开启则回退到原 chat()+parse_json;开关默认关闭,可随时回滚。
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Iterator

from config import get_settings
from gateway.schemas import parse_json


@lru_cache(maxsize=1)
def _structured_model():
    """构造 instructor 用的「动态 Pydantic 模型」。

    chat_json 的调用方(约 15 处)各有不同的字段契约,这里不预设字段,只用一个
    extra='allow' 的动态模型约束「顶层是非空 JSON 对象」;字段级校验仍留给调用方。
    instructor 负责 schema 注入、解析/校验失败后的重试。
    """
    from pydantic import BaseModel, ConfigDict, create_model, model_validator

    def _nonempty(values):
        if not isinstance(values, dict) or not values:
            raise ValueError("structured output must be a non-empty JSON object")
        return values

    class _Object(BaseModel):
        model_config = ConfigDict(extra="allow")

    return create_model(
        "StructuredOutput",
        __base__=_Object,
        __validators__={"_nonempty": model_validator(mode="before")(_nonempty)},
    )


class LLMError(Exception):
    """LLM 调用失败。"""


class LLMClient:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._sdk = None

    @property
    def mock(self) -> bool:
        if self.settings.environment == "production" and self.settings.mock_llm:
            raise RuntimeError("Production forbids mock gateway")
        if not self.settings.mock_llm and not self.settings.llm_api_key:
            raise RuntimeError("Gateway configuration missing")
        return self.settings.mock_llm

    def _client(self):
        """懒加载官方 SDK(传输/SSE/重试由 SDK 负责)。"""
        if self._sdk is None:
            from openai import OpenAI
            self._sdk = OpenAI(
                api_key=self.settings.llm_api_key,
                base_url=self.settings.llm_base_url.rstrip("/"),
                timeout=self.settings.llm_timeout,
                max_retries=2,
            )
        return self._sdk

    def chat(self, messages: list[dict], temperature: float = 0.7,
             max_tokens: int | None = None, model: str | None = None,
             tools: list[dict] | None = None) -> dict:
        """返回 {"content": str, "usage": {...}, "tool_calls": [...]}。

        model 为空时用冷路径模型(llm_model);热路径可传 settings.llm_fast_model。
        传入 tools 时启用 function calling,响应里的 tool_calls 原样透出;mock 下为空列表。
        """
        if self.mock:
            return {
                "content": self._mock_reply(messages),
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                "tool_calls": [],
            }
        kwargs: dict[str, Any] = {
            "model": model or self.settings.llm_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools
        try:
            resp = self._client().chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001 归一化为 LLMError
            raise LLMError(f"LLM 调用失败: {e}") from e
        message = resp.choices[0].message if resp.choices else None
        content = (getattr(message, "content", None) or "") if message else ""
        tool_calls = []
        if message and getattr(message, "tool_calls", None):
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
        usage = resp.usage
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
        } if usage else {"prompt_tokens": 0, "completion_tokens": 0}
        return {"content": content, "usage": usage_dict, "tool_calls": tool_calls}

    def _use_instructor(self) -> bool:
        """USE_INSTRUCTOR 开关:os.getenv("USE_INSTRUCTOR","0"),默认关闭可回滚。"""
        return os.getenv("USE_INSTRUCTOR", "0").strip().lower() in {"1", "true", "yes", "on"}

    def _chat_json_instructor(self, messages: list[dict], temperature: float,
                              max_tokens: int | None, model: str | None) -> tuple[dict, dict]:
        """instructor 路径:动态模型校验 + 失败自动重试,返回同样的契约。

        mock 模式不会走到这里;instructor 未安装或任何异常时,chat_json 回退到 parse_json。
        """
        import instructor

        # instructor 包裹官方 SDK:传输仍走 self._client() 的 base_url/key/timeout。
        wrapped = instructor.from_openai(self._client())
        kwargs: dict[str, Any] = {
            "model": model or self.settings.llm_model,
            "messages": messages,
            "temperature": temperature,
            "response_model": _structured_model(),
            "max_retries": 2,
            "strict": False,  # 允许动态模型携带任意额外字段
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        parsed_obj, completion = wrapped.chat.completions.create_with_completion(**kwargs)
        payload = parsed_obj.model_dump() if hasattr(parsed_obj, "model_dump") else dict(parsed_obj)

        usage = getattr(completion, "usage", None)
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
        } if usage else {"prompt_tokens": 0, "completion_tokens": 0}

        # TOOLS 模式下正文常为空,原始 JSON 在 tool_calls.arguments;都取不到就用解析结果兜底。
        raw = ""
        try:
            message = completion.choices[0].message
            raw = getattr(message, "content", None) or ""
            if not raw and getattr(message, "tool_calls", None):
                raw = message.tool_calls[0].function.arguments
        except (AttributeError, IndexError, TypeError):
            pass
        if not raw:
            raw = json.dumps(payload, ensure_ascii=False)
        return payload, {"usage": usage_dict, "raw": raw}

    def chat_json(self, messages: list[dict], temperature: float = 0.2,
                  max_tokens: int | None = None, model: str | None = None) -> tuple[Any | None, dict]:
        """要求模型返回 JSON 并解析;失败时 parsed 为 None,附带原始文本。

        USE_INSTRUCTOR=1 时先走 instructor(动态模型校验 + 重试);任何失败都回退到
        原有 chat() + parse_json 路径,保证约 15 处调用方的返回契约零改动。
        """
        if self._use_instructor() and not self.mock:
            try:
                return self._chat_json_instructor(messages, temperature, max_tokens, model)
            except Exception:  # noqa: BLE001 - 开关可回滚:instructor 失败一律回退
                pass
        result = self.chat(messages, temperature=temperature, max_tokens=max_tokens, model=model)
        parsed, raw = parse_json(result["content"])
        return parsed, {"usage": result["usage"], "raw": raw}

    def chat_stream(self, messages: list[dict], temperature: float = 0.7,
                    max_tokens: int | None = None, model: str | None = None) -> Iterator[str]:
        """流式返回 content 增量。reasoning 内容跳过,只吐正文。

        取消由调用方关闭迭代器(官方 SDK 的 stream 上下文支持中断)。
        """
        if self.mock:
            text = self._mock_reply(messages)
            for i in range(0, len(text), 8):
                yield text[i:i + 8]
            return
        kwargs: dict[str, Any] = {
            "model": model or self.settings.llm_model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        try:
            stream = self._client().chat.completions.create(**kwargs)
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                piece = getattr(delta, "content", None)
                if piece:
                    yield piece
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"LLM 流式调用失败: {e}") from e

    def _mock_reply(self, messages: list[dict]) -> str:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return "（mock）我在这里,愿意听你慢慢说。你可以告诉我现在最难受的是什么。"
