"""语音网关:TTS(文本转语音)+ ASR(语音识别),经腾讯 MaaS 代理转发 MiniMax / 混元。

同步接口。无 key 或 mock_speech 时进入 mock(返回明确占位,不真正生成音频 / 调外部服务)。
"""
from __future__ import annotations

import io
import os
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import httpx

from config import get_settings


class SpeechError(Exception):
    """语音服务调用失败。"""


# 共享 httpx 连接池(线程安全):复用与 MaaS 语音网关的连接,避免每请求重开 TLS。
_shared_client: httpx.Client | None = None
_shared_client_lock = threading.Lock()


def _get_shared_client() -> httpx.Client:
    global _shared_client
    if _shared_client is None:
        with _shared_client_lock:
            if _shared_client is None:
                _shared_client = httpx.Client(
                    timeout=get_settings().speech_timeout,
                    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                )
    return _shared_client


def _extract(obj: Any, *keys: str) -> Any:
    """在未知响应结构里递归找第一个匹配 key 的值(防上游字段变动)。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys:
                return v
        for v in obj.values():
            found = _extract(v, *keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _extract(item, *keys)
            if found is not None:
                return found
    return None


# ---- 本地 ASR(faster-whisper)----
# 开关默认关闭:首次使用需下载模型,默认关便于回滚。开启后本地失败一律回退远程 MaaS,
# 两端都拿不到文本时抛 SpeechError,绝不返回空串假装成功。
_LOCAL_ASR_SAFE_MODELS = frozenset({"tiny", "base", "small", "medium", "large-v3"})
_LOCAL_ASR_DEFAULT_MODEL = "tiny"
_LOCAL_ASR_MODELS: dict[str, Any] = {}
_LOCAL_ASR_LOCK = threading.Lock()


def _local_asr_enabled() -> bool:
    """USE_LOCAL_ASR=1/true/yes/on 时开启本地 ASR;默认关闭。"""
    return os.getenv("USE_LOCAL_ASR", "0").strip().lower() in ("1", "true", "yes", "on")


def _local_asr_model_name() -> str:
    """读取模型名并做白名单校验;非白名单值一律退回 tiny(小体积优先)。"""
    name = os.getenv("LOCAL_ASR_MODEL", _LOCAL_ASR_DEFAULT_MODEL).strip().lower()
    return name if name in _LOCAL_ASR_SAFE_MODELS else _LOCAL_ASR_DEFAULT_MODEL


def _get_local_asr_model():
    """懒加载单例:同名模型只加载一次;LOCAL_ASR_MODEL 变化时重新加载。"""
    name = _local_asr_model_name()
    model = _LOCAL_ASR_MODELS.get(name)
    if model is None:
        with _LOCAL_ASR_LOCK:
            model = _LOCAL_ASR_MODELS.get(name)
            if model is None:
                from faster_whisper import WhisperModel
                # CPU + int8:无 GPU 依赖、内存占用最小,适合本地优先的延迟目标。
                model = WhisperModel(name, device="cpu", compute_type="int8")
                _LOCAL_ASR_MODELS[name] = model
    return model, name


def _local_asr_audio_source(input_url: str, settings):
    """把 input_url 解析成 faster-whisper 可读的本地路径或内存字节流。"""
    if not isinstance(input_url, str) or not input_url.strip():
        raise SpeechError("ASR input_url 为空")
    raw = input_url.strip()
    if raw.startswith("local://"):
        key = unquote(raw[len("local://"):]).lstrip("/")
        root = Path(settings.storage_local_dir).resolve()
        path = (root / key).resolve()
        if root not in path.parents or not path.is_file():
            raise SpeechError(f"本地音频不存在: {key}")
        return path
    # 先按本地文件判断:Windows 盘符路径(D:\x.wav)会被 urlparse 误判成 scheme。
    if Path(raw).is_file():
        return Path(raw)
    parsed = urlparse(raw)
    if parsed.scheme in ("http", "https"):
        resp = _get_shared_client().get(raw)
        if resp.status_code != 200:
            raise SpeechError(f"下载音频失败: HTTP {resp.status_code}")
        if not resp.content:
            raise SpeechError("下载的音频为空")
        bio = io.BytesIO(resp.content)
        bio.name = Path(unquote(parsed.path)).name or "audio"
        return bio
    if parsed.scheme in ("", "file"):
        path = Path(unquote(parsed.path if parsed.scheme == "file" else raw))
        if not path.is_file():
            raise SpeechError(f"本地音频不存在: {path}")
        return path
    raise SpeechError(f"不支持的音频地址: {raw[:120]}")


class SpeechClient:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._client = _get_shared_client()

    @property
    def mock(self) -> bool:
        if self.settings.environment == "production" and self.settings.mock_speech: raise RuntimeError("Production forbids mock gateway")
        if not self.settings.mock_speech and not self.settings.maas_api_key: raise RuntimeError("Gateway configuration missing")
        return self.settings.mock_speech

    def _post(self, path: str, payload: dict[str, Any]) -> dict:
        url = self.settings.maas_base_url.rstrip("/") + path
        headers = {
            "Authorization": f"Bearer {self.settings.maas_api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = self._client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as e:
            raise SpeechError(f"语音服务网络错误: {e}") from e
        if resp.status_code != 200:
            detail = resp.text[:300]
            try:
                err = (resp.json().get("error") or {})
                detail = err.get("message_zh") or err.get("message") or detail
            except ValueError:
                pass
            raise SpeechError(f"语音服务返回 {resp.status_code}: {detail}")
        try:
            return resp.json()
        except ValueError:
            return {"raw": resp.text}

    def tts(self, text: str, voice_id: str | None = None, speed: float = 1.0,
            vol: float = 1.0, pitch: float = 0.0) -> dict:
        """文本 → 语音,返回 {audio_url, mock, raw};audio_url 为可直接播放的音频地址。"""
        if self.mock:
            return {"audio_url": None, "mock": True, "text": text}

        payload = {
            "model": self.settings.tts_model,
            "text": text,
            "voice_setting": {
                "voice_id": voice_id or self.settings.tts_voice_id,
                "speed": speed,
                "vol": vol,
                "pitch": pitch,
            },
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": "mp3",
                "channel": 1,
            },
            "subtitle_enable": False,
            "output_format": "url",
        }
        data = self._post("/v1/wand/minimax-tts/sync_tts", payload)
        url = _extract(data, "audio", "audio_url")
        if url is None:
            url = _extract(data, "url")
        return {"audio_url": url, "mock": False, "raw": data}

    def transcribe(self, input_url: str) -> dict:
        """音频/视频 URL → 转写文本,返回 {text, mock, raw};text 优先取 output.text 全文。

        USE_LOCAL_ASR=1 时先用 faster-whisper 本地转写;本地失败/无模型/空结果
        一律回退远程 MaaS。开启本地引擎后若两端都拿不到文本,抛 SpeechError,
        绝不返回空串假装成功。开关关闭时行为与改造前完全一致。
        """
        local_enabled = _local_asr_enabled()
        local_error: Exception | None = None
        if local_enabled:
            try:
                return self._transcribe_local(input_url)
            except Exception as e:  # 本地引擎任何异常都不得中断业务,回退远程
                local_error = e

        if self.mock:
            if local_enabled:
                raise SpeechError(f"本地 ASR 失败且处于 mock 模式,无法回退远程: {local_error}")
            return {"text": "", "mock": True, "input_url": input_url}

        payload = {"model": self.settings.asr_model, "input_url": input_url}
        data = self._post("/v1/wand/asrproxy/sync_transcribe", payload)
        text = None
        if isinstance(data, dict) and isinstance(data.get("output"), dict):
            text = data["output"].get("text")
        if not text:
            text = _extract(data, "text", "transcript")
        if local_enabled and not text:
            raise SpeechError(f"本地 ASR 与远程 MaaS 均未返回文本(本地错误: {local_error})")
        return {"text": text or "", "mock": False, "raw": data}

    def _transcribe_local(self, input_url: str) -> dict:
        """本地 faster-whisper 转写。失败抛 SpeechError,由 transcribe 回退远程。"""
        started = time.perf_counter()
        try:
            audio = _local_asr_audio_source(input_url, self.settings)
            model, model_name = _get_local_asr_model()
            model_loaded = time.perf_counter()
            language = os.getenv("LOCAL_ASR_LANGUAGE", "").strip() or None
            segments, info = model.transcribe(audio, language=language, beam_size=5)
            text = "".join(seg.text for seg in segments).strip()
        except SpeechError:
            raise
        except Exception as e:
            raise SpeechError(f"本地 ASR 失败: {e}") from e
        if not text:
            raise SpeechError("本地 ASR 转写结果为空")
        finished = time.perf_counter()
        return {
            "text": text,
            "mock": False,
            "raw": {
                "engine": "local",
                "model": model_name,
                "language": getattr(info, "language", None) or language,
                "duration": getattr(info, "duration", None),
                "load_ms": round((model_loaded - started) * 1000, 1),
                "transcribe_ms": round((finished - model_loaded) * 1000, 1),
                "total_ms": round((finished - started) * 1000, 1),
            },
        }
