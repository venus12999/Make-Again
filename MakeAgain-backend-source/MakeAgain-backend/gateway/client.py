"""模型网关:OpenAI 兼容的 LLM 客户端,默认指向 deepseek-v4-pro。

传输层(WHTTP/SSE/重试/取消)交给官方 openai SDK,这里只做业务层封装:
- mock 模式:无 key 时返回确定性占位回复,保证链路离线可跑通;生产禁止 mock。
- chat / chat_json / chat_stream 接口保持不变,24 个调用方无需改动。
"""
from __future__ import annotations

import json
from typing import Any, Iterator

from config import get_settings
from gateway.schemas import parse_json


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

    def chat_json(self, messages: list[dict], temperature: float = 0.2,
                  max_tokens: int | None = None, model: str | None = None) -> tuple[Any | None, dict]:
        """要求模型返回 JSON 并解析;失败时 parsed 为 None,附带原始文本。"""
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
