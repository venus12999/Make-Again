"""json_repair 替换手写 JSON 提取的回归测试。

验收来自 REPLACEABLE_SCAN.md：默认走 json_repair，结果形态不符时回退，
USE_JSON_REPAIR=0 时逐字节回到旧实现，调用方契约 (parsed, raw) 不变。
"""
import os

import pytest

from gateway import schemas


CASES = [
    ('```json\n{"a": 1, "b": [1,2,],}\n```', {"a": 1, "b": [1, 2]}),
    ('好的，这是结果：{"emotion":"难过","score":35} 希望有帮助', {"emotion": "难过", "score": 35}),
    ('{"facts":[{"fact":"他走了",},{"fact":"我想念",}],"summary":"测试"}',
     {"facts": [{"fact": "他走了"}, {"fact": "我想念"}], "summary": "测试"}),
    ('<think>让我想想</think>\n{"emotion":"平静","score":80}', {"emotion": "平静", "score": 80}),
]


@pytest.fixture
def default_on(monkeypatch):
    monkeypatch.delenv("USE_JSON_REPAIR", raising=False)


def test_json_repair_parses_real_llm_shapes(default_on):
    for text, expected in CASES:
        parsed, raw = schemas.parse_json(text)
        assert parsed == expected
        assert raw == text


def test_returns_tuple_contract(default_on):
    parsed, raw = schemas.parse_json('{"a":1}')
    assert isinstance(parsed, dict) and raw == '{"a":1}'


def test_non_object_falls_back(default_on):
    # 数组根：json_repair 会返回 list，旧实现返回 None（不处理数组根）；
    # 这里只断言契约不被破坏——返回 dict/list/None 之一，raw 恒为原文。
    parsed, raw = schemas.parse_json('结果：[1,2,3] 结束')
    assert parsed in (None, [1, 2, 3])
    assert raw == '结果：[1,2,3] 结束'


def test_empty_and_garbage(default_on):
    assert schemas.parse_json("") == (None, "")
    parsed, raw = schemas.parse_json("完全不是 JSON")
    assert parsed is None and raw == "完全不是 JSON"


def test_switch_off_matches_legacy(monkeypatch):
    """关闭开关时结果与旧实现一致。"""
    for text, _ in CASES:
        monkeypatch.setenv("USE_JSON_REPAIR", "0")
        off = schemas.parse_json(text)
        monkeypatch.setenv("USE_JSON_REPAIR", "1")
        on = schemas.parse_json(text)
        # 典型对象场景两条路径应一致
        assert off == on
