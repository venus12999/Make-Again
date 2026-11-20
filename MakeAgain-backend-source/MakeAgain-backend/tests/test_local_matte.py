"""本地 rembg 抠图降级单测:开关、许可安全模型白名单、单例复用、云失败回退。

用假的 rembg 模块注入 sys.modules,避免测试真的下载 ONNX 模型。
"""
import io
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from gateway import storage
from gateway.storage import ObjectStore, StorageError


@pytest.fixture(autouse=True)
def clean_matte_env(monkeypatch):
    """每个用例从「开关关闭、模型默认」的干净状态开始。"""
    monkeypatch.delenv("USE_LOCAL_MATTE", raising=False)
    monkeypatch.delenv("LOCAL_MATTE_MODEL", raising=False)


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 100, 50)).save(buf, format="PNG")
    return buf.getvalue()


class _FakeSession:
    def __init__(self, model):
        self.model = model


@pytest.fixture
def fake_rembg(monkeypatch):
    """注入假 rembg:记录 new_session / remove 调用,remove 返回真实 RGBA PNG 字节。"""
    calls = {"new_session": [], "remove": []}
    fake = types.ModuleType("rembg")

    def new_session(model_name=None, **kw):
        calls["new_session"].append(model_name)
        return _FakeSession(model_name)

    def remove(data, session=None, **kw):
        calls["remove"].append((data, session.model if session else None))
        buf = io.BytesIO()
        Image.new("RGBA", (8, 8), (200, 100, 50, 255)).save(buf, format="PNG")
        return buf.getvalue()

    fake.new_session = new_session
    fake.remove = remove
    monkeypatch.setitem(sys.modules, "rembg", fake)
    monkeypatch.setattr(storage, "_local_matte_session", None)
    monkeypatch.setattr(storage, "_local_matte_session_model", None)
    return calls


def _local_store_with_image() -> tuple[ObjectStore, str]:
    """造一个 mock 存储(无 COS),并在本地目录写入一张原图。"""
    obj = ObjectStore()
    key = "item/test-matte-key.png"
    dest = Path(obj.settings.storage_local_dir) / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_png_bytes())
    return obj, key


def test_default_off_raises_and_does_not_load_rembg(monkeypatch, fake_rembg):
    """开关默认关闭:无 COS 仍抛 StorageError(可回滚),且完全不碰 rembg/模型。"""
    obj, key = _local_store_with_image()
    with pytest.raises(StorageError, match="USE_LOCAL_MATTE"):
        obj.ai_matte(key)
    assert fake_rembg["new_session"] == []
    assert fake_rembg["remove"] == []


def test_switch_on_uses_local_rembg_and_caches_session(monkeypatch, fake_rembg):
    """开关打开:返回 RGBA PNG 字节(契约不变),session 懒加载且仅加载一次。"""
    monkeypatch.setenv("USE_LOCAL_MATTE", "1")
    obj, key = _local_store_with_image()
    out1 = obj.ai_matte(key)
    out2 = obj.ai_matte(key)
    assert out1[:8] == b"\x89PNG\r\n\x1a\n"
    assert Image.open(io.BytesIO(out1)).mode == "RGBA"
    assert fake_rembg["new_session"] == ["u2netp"]
    assert len(fake_rembg["remove"]) == 2
    assert all(model == "u2netp" for _, model in fake_rembg["remove"])


@pytest.mark.parametrize("env_value", ["1", "true", "TRUE", "yes", "on"])
def test_switch_truthy_values(monkeypatch, fake_rembg, env_value):
    monkeypatch.setenv("USE_LOCAL_MATTE", env_value)
    obj, key = _local_store_with_image()
    assert obj.ai_matte(key)[:8] == b"\x89PNG\r\n\x1a\n"


def test_paid_default_model_is_blocked_by_whitelist(monkeypatch, fake_rembg):
    """显式配置商用付费的 bria-rmbg 也必须被白名单拦下,退回许可安全的 u2netp。"""
    monkeypatch.setenv("USE_LOCAL_MATTE", "1")
    monkeypatch.setenv("LOCAL_MATTE_MODEL", "bria-rmbg")
    obj, key = _local_store_with_image()
    obj.ai_matte(key)
    assert fake_rembg["new_session"] == ["u2netp"]


def test_isnet_general_use_is_whitelisted(monkeypatch, fake_rembg):
    monkeypatch.setenv("USE_LOCAL_MATTE", "1")
    monkeypatch.setenv("LOCAL_MATTE_MODEL", "isnet-general-use")
    obj, key = _local_store_with_image()
    obj.ai_matte(key)
    assert fake_rembg["new_session"] == ["isnet-general-use"]


def test_unknown_model_falls_back_to_safe_default(monkeypatch, fake_rembg):
    monkeypatch.setenv("USE_LOCAL_MATTE", "1")
    monkeypatch.setenv("LOCAL_MATTE_MODEL", "some-unvetted-model")
    obj, key = _local_store_with_image()
    obj.ai_matte(key)
    assert fake_rembg["new_session"] == ["u2netp"]


def _mock_cos_store(monkeypatch):
    """造一个「有 COS 凭据」的存储,并关闭 mock_storage 以进入云抠图分支。"""
    obj, key = _local_store_with_image()
    monkeypatch.setattr(obj, "_cos", object())
    monkeypatch.setattr(obj, "_bucket", "fake-bucket")
    monkeypatch.setattr(obj.settings, "mock_storage", False)
    return obj, key


def _cloud_down(monkeypatch):
    def boom(self, full_key):
        raise StorageError("cloud down")
    monkeypatch.setattr(ObjectStore, "_ai_matte_ci", boom)


def test_cloud_failure_falls_back_to_local_when_enabled(monkeypatch, fake_rembg):
    """有 COS 但云调用失败 + 开关打开 → 走本地 rembg。"""
    monkeypatch.setenv("USE_LOCAL_MATTE", "1")
    obj, key = _mock_cos_store(monkeypatch)
    _cloud_down(monkeypatch)
    assert obj.ai_matte(key)[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(fake_rembg["remove"]) == 1


def test_cloud_failure_reraises_when_switch_off(monkeypatch, fake_rembg):
    monkeypatch.delenv("USE_LOCAL_MATTE", raising=False)
    obj, key = _mock_cos_store(monkeypatch)
    _cloud_down(monkeypatch)
    with pytest.raises(StorageError, match="cloud down"):
        obj.ai_matte(key)
    assert fake_rembg["remove"] == []


def test_mock_storage_skips_cloud_even_with_cos_client(monkeypatch, fake_rembg):
    """mock_storage 为真时,即便有 COS client 也不走云,直接本地 rembg。"""
    monkeypatch.setenv("USE_LOCAL_MATTE", "1")
    obj, key = _local_store_with_image()
    monkeypatch.setattr(obj, "_cos", object())
    monkeypatch.setattr(obj, "_bucket", "fake-bucket")
    monkeypatch.setattr(obj.settings, "mock_storage", True)

    def should_not_be_called(self, full_key):
        raise AssertionError("mock storage must not call cloud matting")

    monkeypatch.setattr(ObjectStore, "_ai_matte_ci", should_not_be_called)
    assert obj.ai_matte(key)[:8] == b"\x89PNG\r\n\x1a\n"


def test_local_matte_rejects_path_traversal(monkeypatch, fake_rembg):
    monkeypatch.setenv("USE_LOCAL_MATTE", "1")
    obj, _key = _local_store_with_image()
    with pytest.raises(StorageError, match="非法存储 key"):
        obj.ai_matte("../outside.png")
