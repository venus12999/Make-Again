"""scan_pii 的 Presidio 接入测试(新文件,不改动既有测试)。"""
import pytest

from agent import moderation


class _FakeResult:
    def __init__(self, entity_type: str, start: int, end: int):
        self.entity_type = entity_type
        self.start = start
        self.end = end


class _FakeAnalyzer:
    def analyze(self, text, language):
        assert language == "zh"
        return [_FakeResult("CONTACT", 0, 2), _FakeResult("LOCATION", 4, 6)]


class _FakeClient:
    mock = False

    class settings:
        llm_fast_model = "fake-fast-model"

    def chat_json(self, messages, temperature=0.1, model=None):
        return ({"pii": [{"type": "name", "text": "李四"}]}, {})


def test_toggle_defaults_off(monkeypatch):
    monkeypatch.delenv("USE_PRESIDIO", raising=False)
    assert moderation._presidio_enabled() is False


def test_regex_fast_path_keeps_existing_flag_format(monkeypatch):
    monkeypatch.delenv("USE_PRESIDIO", raising=False)
    result = moderation.scan_pii("我电话13800138000")
    assert result["clean"] is False
    assert {"type": "contact", "text": "13800138000"} in result["flags"]


def test_clean_text_passes_by_default(monkeypatch):
    monkeypatch.delenv("USE_PRESIDIO", raising=False)
    assert moderation.scan_pii("奶奶走后我心里空空的") == {"clean": True, "flags": []}


def test_regex_hit_skips_presidio(monkeypatch):
    def _boom():
        raise AssertionError("正则命中时不应初始化 Presidio")

    monkeypatch.setenv("USE_PRESIDIO", "1")
    monkeypatch.setattr(moderation, "_get_presidio_analyzer", _boom)
    result = moderation.scan_pii("联系 a@b.com")
    assert result["clean"] is False


def test_presidio_results_map_to_existing_flag_names(monkeypatch):
    monkeypatch.setenv("USE_PRESIDIO", "1")
    monkeypatch.setattr(moderation, "_get_presidio_analyzer", lambda: _FakeAnalyzer())
    result = moderation.scan_pii("张三住在北京")
    assert result["clean"] is False
    assert {f["type"]: f["text"] for f in result["flags"]} == {
        "contact": "张三", "location": "北京",
    }


def test_llm_semantic_fallback_still_runs(monkeypatch):
    monkeypatch.delenv("USE_PRESIDIO", raising=False)
    result = moderation.scan_pii("昨天和李四聊了很久", _FakeClient())
    assert result["clean"] is False
    assert {"type": "name", "text": "李四"} in result["flags"]


def test_presidio_analyze_failure_degrades_gracefully(monkeypatch):
    def _broken():
        raise RuntimeError("model unavailable")

    monkeypatch.setenv("USE_PRESIDIO", "1")
    monkeypatch.setattr(moderation, "_get_presidio_analyzer", _broken)
    assert moderation.scan_pii("今天天气很好") == {"clean": True, "flags": []}


def test_real_presidio_analyzer_on_clean_text(monkeypatch):
    pytest.importorskip("presidio_analyzer")
    monkeypatch.setenv("USE_PRESIDIO", "1")
    moderation._get_presidio_analyzer.cache_clear()
    try:
        assert moderation.scan_pii("今天天气很好，想出去走走") == {
            "clean": True, "flags": [],
        }
    finally:
        moderation._get_presidio_analyzer.cache_clear()
