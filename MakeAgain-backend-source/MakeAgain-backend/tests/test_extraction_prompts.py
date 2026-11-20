"""R2 抽取 prompt 替换验收：开关双向一致、复制资产可用、中文语料不丢事实。

全部使用内存 FakeClient，不访问网络；conftest 的 autouse fixture 继续生效。
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from emotion.classifier import classify as _classify
from memory import extract as ext
from memory import mem_prompts

# --- 基线原样副本：来自 main 45f3351 的 memory/extract.py:23-84（逐字复制，用于逐字节对照） ---
_PLACE_TAGS = ["学校", "教室", "家里", "她家", "教堂", "寺庙", "医院", "公司", "公园", "海边", "咖啡店", "拜佛"]
_TIME_TAGS = ["凌晨", "深夜", "夜里", "晚上", "傍晚", "下午", "中午", "上午", "早上", "周末", "周六", "周日"]


def _first_tag(text, tags):
    for t in tags:
        if t in text:
            return t
    return None


def _importance(emotion):
    s = abs(emotion.get("valence", 0)) * 5 + emotion.get("arousal", 0.5) * 5
    return max(3.0, min(10.0, s))


def _baseline_extract_turn(client, user_text, *, object_hint=None):
    """main 45f3351 原始 extract_turn（原文件第 23-84 行，逐字）。"""
    emotion = _classify(user_text, client)
    place_tag = _first_tag(user_text, _PLACE_TAGS)
    time_tag = _first_tag(user_text, _TIME_TAGS)

    if client.mock:
        return {
            "facts": [],
            "emotion": emotion,
            "summary": user_text[:120],
            "importance": _importance(emotion),
            "place_tag": place_tag,
            "time_tag": time_tag,
        }

    hint = f"[场景提示] {object_hint}\n\n" if object_hint else ""
    messages = [
        {"role": "system", "content": "你是记忆抽取助手,只输出 JSON,不要多余文字。"},
        {"role": "user", "content": (
            "从下面的倾诉中抽取结构化信息,输出 JSON,字段:\n"
            "{\"facts\":[{\"kind\":\"user|object|memory|emotion_trigger|goal|time\",\"fact\":\"...\",\"confidence\":0.8}],\n"
            " \"summary\":\"一句话概括\",\n"
            " \"importance\":0到10(对疗愈进程的重要程度),\n"
            f" \"place_tag\":\"地点,只能从这些词里选:{'/'.join(_PLACE_TAGS)},没有则为 null\",\n"
            f" \"time_tag\":\"时间,只能从这些词里选:{'/'.join(_TIME_TAGS)},没有则为 null\"}}\n\n"
            "facts 的 kind 含义:user=用户自身信息,object=思念对象信息,memory=共同记忆细节,"
            "emotion_trigger=情绪触发点,goal=疗愈目标,time=事件发生的真实时间(如「去年生日」「三年前」「纪念日」,"
            "指事件发生的时间,不是聊天时刻;没有则省略)。\n\n"
            + hint
            + f"文本:\n{user_text}"
        )},
    ]
    parsed, _ = client.chat_json(messages, temperature=0.2, model=client.settings.llm_fast_model)
    if not parsed:
        return {
            "facts": [], "emotion": emotion, "summary": user_text[:120],
            "importance": _importance(emotion),
            "place_tag": place_tag, "time_tag": time_tag,
        }

    facts = parsed.get("facts")
    if not isinstance(facts, list):
        facts = []
    importance = float(parsed.get("importance", _importance(emotion)))

    llm_place = parsed.get("place_tag")
    llm_time = parsed.get("time_tag")
    return {
        "facts": facts,
        "emotion": emotion,
        "summary": parsed.get("summary") or user_text[:120],
        "importance": max(0.0, min(10.0, importance)),
        "place_tag": llm_place if llm_place in _PLACE_TAGS else place_tag,
        "time_tag": llm_time if llm_time in _TIME_TAGS else time_tag,
    }


# --- 内存 FakeClient：第 1 个响应留给 classify（返回 None 走词典兜底），后续为抽取响应 ---
class FakeClient:
    def __init__(self, responses, *, mock=False):
        self.responses = list(responses)
        self.calls = []
        self._mock = mock
        self.settings = SimpleNamespace(llm_fast_model="fast-model")

    @property
    def mock(self):
        return self._mock

    def chat_json(self, messages, temperature=0.2, max_tokens=None, model=None):
        self.calls.append({
            "messages": messages,
            "temperature": temperature,
            "model": model,
        })
        if not self.responses:
            raise AssertionError("unexpected chat_json call")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item, {"usage": {"prompt_tokens": 1, "completion_tokens": 1}, "raw": ""}


def _b(obj):
    """逐字节对照用的确定性序列化（UTF-8 字节）。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


LEGACY_PAYLOAD = {
    "facts": [{"kind": "memory", "fact": "去年生日他送我的手表，我一直戴着", "confidence": 0.8}],
    "summary": "回忆去年生日的礼物",
    "importance": 7.5,
    "place_tag": "家里",
    "time_tag": None,
}


# --- ① 开关默认关闭 + 关闭时逐字节一致 ---

def test_switch_defaults_off(monkeypatch):
    monkeypatch.delenv("USE_MEM_PROMPTS", raising=False)
    assert ext.use_mem_prompts() is False


@pytest.mark.parametrize("value,expected", [
    ("0", False), ("false", False), ("", False), ("off", False),
    ("1", True), ("true", True), ("YES", True), ("on", True),
])
def test_switch_values(monkeypatch, value, expected):
    monkeypatch.setenv("USE_MEM_PROMPTS", value)
    assert ext.use_mem_prompts() is expected


def test_off_path_byte_identical_to_baseline(monkeypatch):
    monkeypatch.delenv("USE_MEM_PROMPTS", raising=False)
    corpus = [
        "去年生日他送我的手表,我一直戴着",
        "上周六在公园散步时想起奶奶做的红烧肉",
        "我养的猫叫橘子,去年冬天它走了,我到现在还留着它的项圈",
    ]
    for text in corpus:
        for hint in (None, "逝者是亲人,别写前任/前女友"):
            got_client = FakeClient([None, LEGACY_PAYLOAD])
            base_client = FakeClient([None, LEGACY_PAYLOAD])
            got = ext.extract_turn(got_client, text, object_hint=hint)
            expected = _baseline_extract_turn(base_client, text, object_hint=hint)
            assert _b(got) == _b(expected), (text, hint)
            assert len(got_client.calls) == len(base_client.calls) == 2
            for got_call, base_call in zip(got_client.calls, base_client.calls):
                assert _b(got_call["messages"]) == _b(base_call["messages"])
                assert got_call["temperature"] == base_call["temperature"]
                assert got_call["model"] == base_call["model"]

    # mock 模式同样逐字节一致
    mock_new, mock_base = FakeClient([], mock=True), FakeClient([], mock=True)
    assert _b(ext.extract_turn(mock_new, corpus[0])) == _b(_baseline_extract_turn(mock_base, corpus[0]))


# --- ② 开关开启：复制 prompt 上链路，输出带 attributed_to / linked_memory_ids ---

def test_on_path_uses_copied_prompts_and_keeps_shape(monkeypatch):
    monkeypatch.setenv("USE_MEM_PROMPTS", "1")
    payload = {
        "memory": [
            {"id": "0", "text": "用户在去年生日收到他送的手表，一直戴着", "kind": "memory",
             "confidence": 0.9, "attributed_to": "user", "linked_memory_ids": ["uuid-1"]},
            {"id": "1", "text": "用户的橘子猫去年冬天去世，用户至今保留它的项圈", "kind": "object",
             "confidence": 0.95, "attributed_to": "user", "linked_memory_ids": []},
        ],
        "summary": "回忆礼物与离世的猫",
        "importance": 9.0,
        "place_tag": "家里",
        "time_tag": "周日",
    }
    client = FakeClient([None, payload])
    got = ext.extract_turn(client, "去年生日他送我的手表,我一直戴着;橘子猫去年冬天走了", object_hint="逝者是亲人")
    assert set(got) == {"facts", "emotion", "summary", "importance", "place_tag", "time_tag"}
    assert len(got["facts"]) == 2
    assert all("attributed_to" in f and "linked_memory_ids" in f for f in got["facts"])
    assert got["facts"][0]["attributed_to"] == "user"
    assert got["facts"][0]["linked_memory_ids"] == ["uuid-1"]
    assert got["facts"][1]["kind"] == "object"
    assert got["importance"] == 9.0
    assert got["place_tag"] == "家里" and got["time_tag"] == "周日"

    system_prompt = client.calls[1]["messages"][0]["content"]
    user_prompt = client.calls[1]["messages"][1]["content"]
    assert "# ROLE" in system_prompt and "Exhaustive Extraction Checklist" in system_prompt
    assert "linked_memory_ids" in system_prompt and "attributed_to" in system_prompt
    assert mem_prompts.DO_NOT_ESCAPE_UNICODE.strip() in system_prompt
    assert "Language Requirement" in user_prompt
    assert "## Observation Date" in user_prompt
    assert "NEVER extract any of the following" in user_prompt  # graphiti 负面清单注入
    assert "逝者是亲人" in user_prompt  # object_hint 仍生效
    assert client.calls[1]["model"] == "fast-model" and client.calls[1]["temperature"] == 0.2


def test_new_path_failure_falls_back_to_legacy(monkeypatch):
    monkeypatch.setenv("USE_MEM_PROMPTS", "1")
    client = FakeClient([None, RuntimeError("LLM down"), LEGACY_PAYLOAD])
    got = ext.extract_turn(client, "去年生日他送我的手表,我一直戴着")
    assert _b(got) == _b(_baseline_extract_turn(FakeClient([None, LEGACY_PAYLOAD]),
                                               "去年生日他送我的手表,我一直戴着"))
    assert "attributed_to" not in got["facts"][0]  # 回退后是基线事实形态
    assert len(client.calls) == 3


def test_empty_extraction_on_long_text_falls_back(monkeypatch):
    monkeypatch.setenv("USE_MEM_PROMPTS", "1")
    text = "去年生日他送我的手表我一直戴着,上周六又在公园想起奶奶做的红烧肉,那味道这辈子忘不掉"
    client = FakeClient([None, {"memory": []}, LEGACY_PAYLOAD])
    got = ext.extract_turn(client, text)
    assert got["facts"] == LEGACY_PAYLOAD["facts"]
    assert len(client.calls) == 3


# --- ③ 冲突 prompt 判 ADD/UPDATE/DELETE/NONE ---

def test_conflict_prompt_decides_add_update_delete_none():
    old = [
        {"id": "0", "text": "用户养了一只叫橘子的猫"},
        {"id": "1", "text": "用户喜欢奶酪披萨"},
        {"id": "2", "text": "用户是软件工程师"},
        {"id": "3", "text": "用户在 Shopify 工作"},
    ]
    facts = ["橘子猫去年去世了", "爱吃的其实是鸡肉披萨", "用户是软件工程师", "用户在字节工作"]
    payload = {"memory": [
        {"id": "0", "text": "用户的橘子猫去年去世了", "event": "UPDATE", "old_memory": "用户养了一只叫橘子的猫"},
        {"id": "1", "text": "用户喜欢鸡肉披萨", "event": "DELETE"},
        {"id": "2", "text": "用户是软件工程师", "event": "NONE"},
        {"id": "3", "text": "用户在字节工作", "event": "UPDATE", "old_memory": "用户在 Shopify 工作"},
        {"id": "4", "text": "用户开始学吉他", "event": "ADD"},
    ]}
    client = FakeClient([payload])
    result = ext.resolve_memory_conflicts(client, old, facts)
    assert [m["event"] for m in result] == ["UPDATE", "DELETE", "NONE", "UPDATE", "ADD"]
    system_prompt = client.calls[0]["messages"][0]["content"]
    for token in ("ADD", "UPDATE", "DELETE", "NONE"):
        assert token in system_prompt
    assert "Contradicts" not in system_prompt or "contradicts" in system_prompt
    assert "橘子猫去年去世了" in client.calls[0]["messages"][1]["content"]


# --- ④ 中文语料抽取不丢事实 ---

def test_chinese_corpus_no_fact_loss(monkeypatch):
    monkeypatch.setenv("USE_MEM_PROMPTS", "1")
    text = ("以前家里养了只叫橘子的猫,它陪了我十年。去年冬天它走了,我到现在还觉得家里空落落的。"
            "上周六在公园散步时想起奶奶做的红烧肉,那是这辈子最想念的味道。"
            "今年清明我想去她墓前坐坐,也把围巾带去。")
    kept = [
        "用户家里养过一只叫橘子的猫，陪伴了十年",
        "用户的橘子猫去年冬天去世，用户至今觉得家里空落落",
        "用户上周六在公园散步时想起奶奶做的红烧肉，那是最想念的味道",
        "用户今年清明想去奶奶墓前坐坐，并带上她织的围巾",
    ]
    payload = {
        "memory": [
            {"id": str(i), "text": t, "kind": "memory", "attributed_to": "user",
             "linked_memory_ids": ["uuid-cat"] if i == 1 else []}
            for i, t in enumerate(kept)
        ],
        "summary": "思念橘子猫和奶奶",
        "importance": 9.5,
        "place_tag": "公园",
        "time_tag": "周六",
    }
    client = FakeClient([None, payload])
    got = ext.extract_turn(client, text)
    extracted = [f["fact"] for f in got["facts"]]
    assert len(extracted) == 4
    for fact in kept:
        assert fact in extracted, fact
    assert got["summary"] == "思念橘子猫和奶奶"
    assert "养过" in extracted[0] and "去世" in extracted[1]  # 冲突事实两条都保留
    assert text in client.calls[1]["messages"][1]["content"]  # 原文完整送进 prompt


# --- ⑤ 复制资产本身（graphiti 负面清单 / schema / to_prompt_json） ---

def test_copied_graphiti_assets():
    assert mem_prompts.to_prompt_json({"猫": "橘子"}, ensure_ascii=False) == '{"猫": "橘子"}'
    assert mem_prompts.DO_NOT_ESCAPE_UNICODE.strip() == "Do not escape unicode characters."
    messages = mem_prompts.extract_message({
        "entity_types": "人物/地点",
        "previous_episodes": [],
        "episode_content": "奶奶: 橘子去年走了",
        "custom_extraction_instructions": "中文负面清单",
    })
    assert messages[0].role == "system" and "NEVER extract abstract concepts" in messages[0].content
    assert "Bare relational or kinship terms" in messages[1].content
    assert "橘子去年走了" in messages[1].content
    chat = mem_prompts.ChatPrompt.from_messages(messages)
    assert "Do not escape unicode characters." in chat.as_messages()[0].content
    assert "Do not escape unicode characters." not in chat.as_messages(append_unicode_note=False)[0].content


def test_copied_edge_schema_with_provenance():
    raw = {"edges": [{
        "source_entity_name": "用户", "target_entity_name": "橘子",
        "relation_type": "OWNED_AND_MOURNED", "fact": "用户的猫橘子去年冬天去世",
        "valid_at": "2024-12-01T00:00:00Z", "invalid_at": None, "episode_indices": [0],
    }]}
    result = ext.validate_extracted_edges(raw)
    assert result[0]["valid_at"] == "2024-12-01T00:00:00Z"
    assert result[0]["invalid_at"] is None
    assert result[0]["episode_indices"] == [0]
    assert set(mem_prompts.Edge.model_fields) >= {
        "source_entity_name", "target_entity_name", "relation_type", "fact",
        "valid_at", "invalid_at", "episode_indices",
    }
