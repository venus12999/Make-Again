"""gateway.client.chat_json 的 instructor 集成测试。

覆盖:开关默认值/回滚、返回契约不变、instructor 动态模型校验与重试、失败回退。
不修改任何现有测试文件;全部使用内存 MockTransport,不访问网络。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from gateway.client import LLMClient


def _settings(*, mock: bool = False):
    """最小化 settings,避免测试依赖真实 .env / 网络。"""
    return SimpleNamespace(
        environment="test",
        mock_llm=mock,
        llm_api_key="" if mock else "test-key",
        llm_base_url="http://localhost/v1",
        llm_model="test-model",
        llm_timeout=5.0,
    )


def _tool_response(payload: dict, call_no: int = 1) -> httpx.Response:
    """模拟 OpenAI 工具调用响应(instructor TOOLS 模式期望的形态)。"""
    return httpx.Response(200, json={
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": f"call_{call_no}",
                    "type": "function",
                    "function": {
                        "name": "StructuredOutput",
                        "arguments": json.dumps(payload, ensure_ascii=False),
                    },
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    })


def _client_with_transport(handler) -> LLMClient:
    client = LLMClient(settings=_settings())
    sdk = OpenAI(
        api_key="test-key",
        base_url="http://localhost/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client._sdk = sdk
    return client


MESSAGES = [{"role": "system", "content": "只输出 JSON"},
            {"role": "user", "content": "输出 {\"emotion\":\"难过\"}"}]


# --- 开关 ---

def test_use_instructor_defaults_off(monkeypatch):
    monkeypatch.delenv("USE_INSTRUCTOR", raising=False)
    assert LLMClient(settings=_settings())._use_instructor() is False


@pytest.mark.parametrize("value,expected", [
    ("0", False), ("false", False), ("", False),
    ("1", True), ("true", True), ("YES", True), ("on", True),
])
def test_use_instructor_switch_values(monkeypatch, value, expected):
    monkeypatch.setenv("USE_INSTRUCTOR", value)
    assert LLMClient(settings=_settings())._use_instructor() is expected


# --- 契约不变:关闭开关仍走 chat + parse_json ---

def test_chat_json_legacy_contract_when_switch_off(monkeypatch):
    monkeypatch.delenv("USE_INSTRUCTOR", raising=False)
    client = LLMClient(settings=_settings())
    usage = {"prompt_tokens": 3, "completion_tokens": 4}
    monkeypatch.setattr(client, "chat", lambda *a, **kw: {
        "content": "序言```json\n{\"a\": 1, \"b\": [2,3]}\n```",
        "usage": usage, "tool_calls": [],
    })
    parsed, meta = client.chat_json(MESSAGES)
    assert parsed == {"a": 1, "b": [2, 3]}
    assert meta["usage"] == usage
    assert "json" in meta["raw"]


def test_chat_json_mock_mode_skips_instructor(monkeypatch):
    """mock 模式无论如何都不走 instructor(没有真实 API)。"""
    monkeypatch.setenv("USE_INSTRUCTOR", "1")
    client = LLMClient(settings=_settings(mock=True))

    def must_not_run(*a, **kw):
        raise AssertionError("mock 模式不应调用 instructor")

    monkeypatch.setattr(client, "_chat_json_instructor", must_not_run)
    parsed, meta = client.chat_json(MESSAGES)
    assert parsed is None  # mock 正文不是 JSON
    assert set(meta) == {"usage", "raw"}


# --- instructor 路径 ---

def test_chat_json_instructor_contract(monkeypatch):
    monkeypatch.setenv("USE_INSTRUCTOR", "1")
    payload = {"emotion": "难过", "score": 30, "tags": ["失落"]}
    client = _client_with_transport(lambda req: _tool_response(payload))

    parsed, meta = client.chat_json(MESSAGES, temperature=0.1, max_tokens=100, model="fast")

    assert parsed == payload
    assert isinstance(parsed, dict)
    assert set(meta) == {"usage", "raw"}
    assert meta["usage"]["prompt_tokens"] == 11
    assert meta["usage"]["completion_tokens"] == 7
    assert json.loads(meta["raw"]) == payload  # raw 保留工具调用返回的原 JSON


def test_chat_json_instructor_retries_on_invalid_output(monkeypatch):
    """动态模型校验失败(空对象)时 instructor 自动重试并最终成功。"""
    monkeypatch.setenv("USE_INSTRUCTOR", "1")
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        payload = {} if calls["n"] == 1 else {"ok": True, "n": 2}
        return _tool_response(payload, call_no=calls["n"])

    client = _client_with_transport(handler)
    parsed, meta = client.chat_json(MESSAGES)

    assert parsed == {"ok": True, "n": 2}
    assert calls["n"] == 2  # 第一次被校验拒绝,重试一次
    assert set(meta) == {"usage", "raw"}


def test_chat_json_instructor_failure_falls_back(monkeypatch):
    """instructor 抛错时回退到原 chat()+parse_json,调用方无感。"""
    monkeypatch.setenv("USE_INSTRUCTOR", "1")
    client = LLMClient(settings=_settings())
    usage = {"prompt_tokens": 1, "completion_tokens": 1}

    def boom(*a, **kw):
        raise RuntimeError("instructor unavailable")

    monkeypatch.setattr(client, "_chat_json_instructor", boom)
    monkeypatch.setattr(client, "chat", lambda *a, **kw: {
        "content": '{"fallback": true}', "usage": usage, "tool_calls": [],
    })

    parsed, meta = client.chat_json(MESSAGES)
    assert parsed == {"fallback": True}
    assert meta["usage"] == usage
