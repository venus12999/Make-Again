"""包 H3 prompt caching 离线守卫（口径：fake SDK，不访问网络/不读真 key/不起后端）。

覆盖：稳定前缀哈希、USE_PROMPT_CACHE 默认关逐字节回退、开时派生稳定 session 头、
usage 只报网关真实回传的 cached_tokens（无回传=NO_DATA，绝不伪造命中）、
tools/function calling 路径不破坏、mock 路径零影响、chat_json 契约不变。

所有 cache_status 都是 fake 响应里人为构造的字段值，只证明代码读写口径正确，
**不代表真网关命中收益**（真模型 NOT_RUN，见返回说明）。
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from config import Settings
from gateway.client import CACHE_SESSION_HEADER, LLMClient, stable_prefix

SYS = {"role": "system", "content": "你是陪伴助手"}
RULES = {"role": "system", "content": "[场景规则] 危机规则钉在 system"}
PORTRAIT = {"role": "system", "content": "[用户画像] 小满 [思念对象画像] 林夏"}
USER = {"role": "user", "content": "我很难过"}
USER2 = {"role": "user", "content": "还记得林夏吗"}


def _messages():
    return [SYS, RULES, PORTRAIT, USER]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("USE_PROMPT_CACHE", raising=False)
    monkeypatch.delenv("LLM_EXTRA_HEADERS", raising=False)
    monkeypatch.delenv("USE_INSTRUCTOR", raising=False)


class _Completions:
    """fake SDK：记录每次 create 的 kwargs，返回构造好的响应。"""

    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _usage(cached="absent"):
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=9)
    if cached != "absent":
        usage.prompt_tokens_details = SimpleNamespace(cached_tokens=cached)
    return usage


def _tool_call():
    return [SimpleNamespace(
        id="call_1", type="function",
        function=SimpleNamespace(name="save_item", arguments='{"name":"橘子"}'),
    )]


def _client(*, cache_field=False, cached="absent", tool_calls=None, content="回复",
            usage=None):
    settings = SimpleNamespace(
        environment="test", mock_llm=False, llm_api_key="test-key",
        llm_base_url="http://localhost/v1", llm_model="test-model", llm_timeout=5.0,
        use_prompt_cache=cache_field,
    )
    client = LLMClient(settings=settings)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))],
        usage=usage if usage is not None else _usage(cached),
    )
    completions = _Completions(response)
    client._sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


# --- 开关默认关 ---

def test_config_default_off():
    assert Settings.model_fields["use_prompt_cache"].default is False


def test_switch_env_override_and_default(monkeypatch):
    c_off, _ = _client(cache_field=False)
    c_on, _ = _client(cache_field=True)
    assert c_off._use_prompt_cache() is False
    assert c_on._use_prompt_cache() is True
    monkeypatch.setenv("USE_PROMPT_CACHE", "0")
    assert c_on._use_prompt_cache() is False  # 显式 env 优先
    monkeypatch.setenv("USE_PROMPT_CACHE", "1")
    assert c_off._use_prompt_cache() is True


# --- 关闭时逐字节不变 ---

def test_off_kwargs_and_usage_byte_identical():
    msgs = _messages()
    client, comps = _client(cache_field=False, cached=48)
    result = client.chat(msgs)
    assert comps.calls[0] == {
        "model": "test-model", "messages": msgs, "temperature": 0.7,
    }
    assert comps.calls[0]["messages"] is msgs  # 原 list 对象透传，未拷贝未改写
    assert "extra_headers" not in comps.calls[0]
    assert result["usage"] == {"prompt_tokens": 100, "completion_tokens": 9}


def test_off_kwargs_with_tools_and_max_tokens_unchanged():
    msgs = _messages()
    tools = [{"type": "function", "function": {"name": "save_item"}}]
    client, comps = _client(cache_field=False)
    client.chat(msgs, temperature=0.1, max_tokens=32, tools=tools)
    assert comps.calls[0] == {
        "model": "test-model", "messages": msgs, "temperature": 0.1,
        "max_tokens": 32, "tools": tools,
    }


# --- 稳定前缀哈希 ---

def test_stable_prefix_only_leading_system_messages_and_hash():
    prefix, digest = stable_prefix(_messages())
    assert prefix == [SYS, RULES, PORTRAIT]
    canonical = json.dumps(prefix, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert digest == stable_prefix(_messages())[1]  # 确定性与顺序稳定

    # 只换 user 轮次：可缓存前缀没变，哈希必须不变
    assert stable_prefix([SYS, RULES, PORTRAIT, USER2])[1] == digest
    # 改 system 内容 / 顺序：前缀字节变了，哈希必须变
    changed = [SYS, RULES, {**PORTRAIT, "content": PORTRAIT["content"] + "!"}, USER]
    assert stable_prefix(changed)[1] != digest
    assert stable_prefix([RULES, SYS, PORTRAIT, USER])[1] != digest
    # 边界：首个非 system 即截断；空输入不炸
    assert stable_prefix([USER, SYS])[0] == []
    assert stable_prefix([])[0] == []
    assert stable_prefix(None)[0] == []


# --- 开时：真缓存机制（网关自定义 header + 真实 cached_tokens） ---

def test_on_adds_stable_session_header_from_prefix(monkeypatch):
    monkeypatch.setenv("USE_PROMPT_CACHE", "1")
    msgs = _messages()
    client, comps = _client()
    first = client.chat(msgs)
    client.chat([SYS, RULES, PORTRAIT, USER2])  # 只换轮换，前缀不变

    expected = {CACHE_SESSION_HEADER: "ses_" + stable_prefix(msgs)[1][:32]}
    assert comps.calls[0]["extra_headers"] == expected
    assert comps.calls[1]["extra_headers"] == expected
    assert first["usage"]["prefix_hash"] == stable_prefix(msgs)[1]


def test_on_respects_existing_session_header(monkeypatch):
    monkeypatch.setenv("USE_PROMPT_CACHE", "1")
    monkeypatch.setenv("LLM_EXTRA_HEADERS", json.dumps({
        CACHE_SESSION_HEADER: "ses_bench123", "User-Agent": "make-again-bench/1.0",
    }))
    client, comps = _client()
    client.chat(_messages())
    assert "extra_headers" not in comps.calls[0]  # 不覆盖 eval 已注入的 session


def test_on_bad_extra_headers_json_falls_back_to_derived_header(monkeypatch):
    monkeypatch.setenv("USE_PROMPT_CACHE", "1")
    monkeypatch.setenv("LLM_EXTRA_HEADERS", "not-json")
    client, comps = _client()
    client.chat(_messages())
    assert comps.calls[0]["extra_headers"][CACHE_SESSION_HEADER].startswith("ses_")


@pytest.mark.parametrize("cached,status,n", [(48, "HIT", 48), (0, "MISS", 0)])
def test_on_usage_reports_only_gateway_cached_tokens(monkeypatch, cached, status, n):
    monkeypatch.setenv("USE_PROMPT_CACHE", "1")
    client, _ = _client(cached=cached)
    usage = client.chat(_messages())["usage"]
    assert usage["cache_status"] == status
    assert usage["cache_read_tokens"] == n
    assert usage["prompt_tokens"] == 100 and usage["completion_tokens"] == 9


def test_on_usage_no_data_never_fakes_hit(monkeypatch):
    monkeypatch.setenv("USE_PROMPT_CACHE", "1")
    client, _ = _client()  # 响应没有 prompt_tokens_details
    usage = client.chat(_messages())["usage"]
    assert usage["cache_status"] == "NO_DATA"
    assert usage["cache_read_tokens"] == 0


def test_on_usage_accepts_dict_shaped_details(monkeypatch):
    monkeypatch.setenv("USE_PROMPT_CACHE", "1")
    usage_obj = SimpleNamespace(
        prompt_tokens=7, completion_tokens=3,
        prompt_tokens_details={"cached_tokens": 5},
    )
    client, _ = _client(usage=usage_obj)
    usage = client.chat(_messages())["usage"]
    assert usage["cache_status"] == "HIT" and usage["cache_read_tokens"] == 5


# --- 不破坏现有契约 ---

def test_on_tools_function_calling_path_intact(monkeypatch):
    monkeypatch.setenv("USE_PROMPT_CACHE", "1")
    tools = [{"type": "function", "function": {"name": "save_item"}}]
    client, comps = _client(tool_calls=_tool_call(), content=None)
    result = client.chat(_messages(), tools=tools)
    assert comps.calls[0]["tools"] is tools
    assert result["tool_calls"] == [{
        "id": "call_1", "type": "function",
        "function": {"name": "save_item", "arguments": '{"name":"橘子"}'},
    }]
    assert set(result["usage"]) == {
        "prompt_tokens", "completion_tokens", "cache_status", "cache_read_tokens", "prefix_hash",
    }


def test_on_chat_json_contract_unchanged(monkeypatch):
    monkeypatch.setenv("USE_PROMPT_CACHE", "1")
    client, _ = _client(cached=5, content='{"ok": true}')
    parsed, meta = client.chat_json(_messages())
    assert parsed == {"ok": True}
    assert set(meta) == {"usage", "raw"}
    assert meta["usage"]["cache_status"] == "HIT"


def test_mock_mode_zero_impact_even_with_switch_on(monkeypatch):
    monkeypatch.setenv("USE_PROMPT_CACHE", "1")
    settings = SimpleNamespace(
        environment="test", mock_llm=True, llm_api_key="",
        llm_base_url="http://localhost/v1", llm_model="test-model", llm_timeout=5.0,
        use_prompt_cache=True,
    )
    client = LLMClient(settings=settings)
    result = client.chat(_messages())
    assert result["usage"] == {"prompt_tokens": 0, "completion_tokens": 0}
    assert result["tool_calls"] == []
