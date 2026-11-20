"""WP2 · faster-whisper 本地 ASR 单元测试。

这里只验证开关 / 回退 / 契约 / 懒加载的**逻辑**,用假模型替身,不下载模型、不访问网络。
真实中文语音转写与 15s 耗时的证据由 work/ 下的独立验收脚本产出(见交付报告)。
"""
from __future__ import annotations

import io
import wave
from pathlib import Path

import pytest

from config import Settings
from gateway import speech
from gateway.speech import SpeechError


class _FakeSegment:
    def __init__(self, text: str):
        self.text = text


class _FakeInfo:
    language = "zh"
    duration = 15.0


class _FakeModel:
    """假装是 WhisperModel:返回一段固定中文转写。"""

    def __init__(self, text: str = "我今天很难过"):
        self.text = text
        self.calls: list[tuple] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return iter([_FakeSegment(self.text)]), _FakeInfo()


@pytest.fixture(autouse=True)
def _asr_env(monkeypatch, tmp_path):
    """每个用例隔离开关与模型缓存,避免互相污染。"""
    for key in ("USE_LOCAL_ASR", "LOCAL_ASR_MODEL", "LOCAL_ASR_LANGUAGE"):
        monkeypatch.delenv(key, raising=False)
    speech._LOCAL_ASR_MODELS.clear()
    yield tmp_path
    speech._LOCAL_ASR_MODELS.clear()


def _client(monkeypatch, *, mock_speech: bool = False) -> speech.SpeechClient:
    return speech.SpeechClient(
        settings=Settings(
            environment="test",
            mock_speech=mock_speech,
            maas_api_key="" if mock_speech else "unit-test-key",
            storage_local_dir="data/uploads",
        )
    )


def _wav(path: Path) -> Path:
    """写一个 1 秒 16k 单声道静音 WAV(内容不重要,模型是替身)。"""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)
    return path


def test_switch_defaults_off_and_remote_path_unchanged(monkeypatch, tmp_path):
    """开关默认关闭:行为与改造前一致,且不触碰本地引擎。"""
    c = _client(monkeypatch)
    calls = []

    def fake_post(path, payload):
        calls.append((path, payload))
        return {"output": {"text": "远程转写结果"}}

    monkeypatch.setattr(c, "_post", fake_post)
    monkeypatch.setattr(
        speech, "_get_local_asr_model",
        lambda: pytest.fail("开关关闭时不应加载本地模型"),
    )
    result = c.transcribe("https://example.com/a.wav")
    assert result == {"text": "远程转写结果", "mock": False,
                      "raw": {"output": {"text": "远程转写结果"}}}
    assert calls == [("/v1/wand/asrproxy/sync_transcribe",
                      {"model": c.settings.asr_model,
                       "input_url": "https://example.com/a.wav"})]


def test_mock_contract_unchanged_when_switch_off(monkeypatch):
    """mock 且开关关闭:返回体逐字段与改造前相同。"""
    c = _client(monkeypatch, mock_speech=True)
    assert c.transcribe("https://example.com/a.wav") == {
        "text": "", "mock": True, "input_url": "https://example.com/a.wav"
    }


def test_local_engine_transcribes_chinese_and_holds_contract(monkeypatch, tmp_path):
    """USE_LOCAL_ASR=1:本地引擎输出中文,契约仍是 {text, mock, raw}。"""
    monkeypatch.setenv("USE_LOCAL_ASR", "1")
    audio = _wav(tmp_path / "zh.wav")
    model = _FakeModel("我今天很难过，但是会慢慢好起来")
    monkeypatch.setattr(speech, "_get_local_asr_model", lambda: (model, "tiny"))

    c = _client(monkeypatch)
    monkeypatch.setattr(c, "_post", lambda *a, **k: pytest.fail("本地成功时不应调远程"))
    result = c.transcribe(str(audio))

    assert set(result) == {"text", "mock", "raw"}
    assert result["text"] == "我今天很难过，但是会慢慢好起来"
    assert result["mock"] is False
    assert result["raw"]["engine"] == "local"
    assert result["raw"]["model"] == "tiny"
    assert result["raw"]["transcribe_ms"] >= 0
    assert model.calls[0][0] == audio  # 本地路径直接喂给模型


@pytest.mark.parametrize("mode", ["no_model", "empty_result"])
def test_local_failure_falls_back_to_remote(monkeypatch, tmp_path, mode):
    """无模型 / 本地空结果:自动回退远程 MaaS,不返回空串。"""
    monkeypatch.setenv("USE_LOCAL_ASR", "1")
    audio = _wav(tmp_path / "zh.wav")
    if mode == "no_model":
        def boom():
            raise SpeechError("本地模型未下载")
        monkeypatch.setattr(speech, "_get_local_asr_model", boom)
    else:
        monkeypatch.setattr(speech, "_get_local_asr_model",
                            lambda: (_FakeModel(""), "tiny"))

    c = _client(monkeypatch)
    remote_calls = []

    def fake_post(path, payload):
        remote_calls.append((path, payload))
        return {"output": {"text": "远程兜底转写"}}

    monkeypatch.setattr(c, "_post", fake_post)
    result = c.transcribe(str(audio))
    assert result["text"] == "远程兜底转写"
    assert result["mock"] is False
    assert set(result) == {"text", "mock", "raw"}
    assert len(remote_calls) == 1


def test_never_returns_empty_when_both_engines_empty(monkeypatch, tmp_path):
    """本地失败且远程也空:明确报错,绝不返回空文本假装成功。"""
    monkeypatch.setenv("USE_LOCAL_ASR", "1")
    audio = _wav(tmp_path / "zh.wav")

    def boom():
        raise SpeechError("本地模型未下载")
    monkeypatch.setattr(speech, "_get_local_asr_model", boom)

    c = _client(monkeypatch)
    monkeypatch.setattr(c, "_post", lambda *a, **k: {"text": ""})
    with pytest.raises(SpeechError, match="均未返回文本"):
        c.transcribe(str(audio))


def test_local_failure_in_mock_mode_raises(monkeypatch, tmp_path):
    """本地失败 + mock 模式无法回退:明确报错,不返回空串。"""
    monkeypatch.setenv("USE_LOCAL_ASR", "1")
    audio = _wav(tmp_path / "zh.wav")

    def boom():
        raise SpeechError("本地模型未下载")
    monkeypatch.setattr(speech, "_get_local_asr_model", boom)

    c = _client(monkeypatch, mock_speech=True)
    with pytest.raises(SpeechError, match="无法回退远程"):
        c.transcribe(str(audio))


def test_model_is_lazy_singleton_and_whitelisted(monkeypatch):
    """模型懒加载 + 单例;切换 LOCAL_ASR_MODEL 重建;非白名单值回退 tiny。"""
    import faster_whisper

    created = []

    class FakeWhisper:
        def __init__(self, name, **kwargs):
            created.append((name, kwargs))

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisper)

    a, name_a = speech._get_local_asr_model()
    b, _ = speech._get_local_asr_model()
    assert a is b and name_a == "tiny" and len(created) == 1

    monkeypatch.setenv("LOCAL_ASR_MODEL", "base")
    c, name_c = speech._get_local_asr_model()
    assert c is not a and name_c == "base" and len(created) == 2
    assert created[-1][1] == {"device": "cpu", "compute_type": "int8"}

    monkeypatch.setenv("LOCAL_ASR_MODEL", "not-a-real-model")
    d, name_d = speech._get_local_asr_model()
    assert d is a and name_d == "tiny" and len(created) == 2


def test_http_audio_is_downloaded_into_bytesio(monkeypatch):
    """远程预签名 URL:下载字节流后交给本地引擎(不落临时文件)。"""
    monkeypatch.setenv("USE_LOCAL_ASR", "1")
    model = _FakeModel("远程音频的转写")
    monkeypatch.setattr(speech, "_get_local_asr_model", lambda: (model, "tiny"))

    class FakeResponse:
        status_code = 200
        content = b"RIFFfake-audio-bytes"

    class FakeHttpClient:
        def get(self, url):
            assert url == "https://cos.example.com/signed.mp3?x=1"
            return FakeResponse()

    monkeypatch.setattr(speech, "_get_shared_client", lambda: FakeHttpClient())
    c = _client(monkeypatch)
    result = c.transcribe("https://cos.example.com/signed.mp3?x=1")
    source = model.calls[0][0]
    assert isinstance(source, io.BytesIO) and source.getvalue() == FakeResponse.content
    assert result["text"] == "远程音频的转写"
