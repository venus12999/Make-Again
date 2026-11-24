"""R1 离线守卫：cc.compact 摘要调用 max_tokens 与 3 次重试/失败保护语义。

口径：纯离线 fake client（无真 key、无 8000）；
- 验证修复后 max_tokens 用 COMPACTION_SUMMARY_MAX_TOKENS（≥2000，不再硬编码 1500）；
- 验证"前2次空、第3次非空"与"全空"两路径的逐字节回退语义不变；
- SUMMARY_EMPTY / SUMMARY_MALFORMED / SUMMARY_TOO_SHORT 三种失败保护不变；
- D-070 的 maybe_compact 持续注入链不触碰（源码守卫见 test_context_block.py / test_wiring.py，此处再断言一次引用）。
"""
from __future__ import annotations

from pathlib import Path
import types

from memory import context_compaction as cc


BACKEND = Path(__file__).resolve().parents[1]


class _FakeSettings:
    llm_fast_model = "hy4-flash-free"


class _ScriptedClient:
    """按脚本返回 content/抛异常；记录每次 chat 的完整 kwargs。"""

    def __init__(self, script: list[dict | Exception]):
        self.settings = _FakeSettings()
        self.script = script
        self.calls: list[dict] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"n": len(self.calls), "kwargs": dict(kwargs)})
        item = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        return {"content": item.get("content", ""), "usage": {"prompt_tokens": 10,
                "completion_tokens": 2, "total_tokens": 12}}


def _messages() -> list[dict]:
    # 末条是 user（切点稳定）；调用时 keep_recent_tokens=1 强制只留末条，其余进摘要。
    return [
        {"role": "user", "content": "林夏今天下午跟我说的分手，在一起三年零四个月。"},
        {"role": "assistant", "content": "听起来这段关系对你很重要，慢慢说。"},
        {"role": "user", "content": "她每次豆浆都半勺糖，票根我还留着。"},
        {"role": "user", "content": "你还记得她叫什么吗？"},
    ]


def _compact(client, msgs=None):
    return cc.compact(msgs if msgs is not None else _messages(), client,
                      "test-session", keep_recent_tokens=1)


GOOD_SUMMARY = """## 这段关系/话题
- 用户在经历分手，对象是林夏，在一起三年零四个月。

## 已聊过的重要事实
- 她每次豆浆半勺糖，票根还留着。"""


def test_summary_max_tokens_constant_is_raised():
    """修复后 max_tokens 常量必须 ≥2000（key 文件 RECOMMENDED_MAX_TOKENS）且不再是 1500。"""
    mt = cc.COMPACTION_SUMMARY_MAX_TOKENS
    assert isinstance(mt, int) and mt >= 2000, f"摘要 max_tokens 常量太低: {mt}"
    src = (BACKEND / "memory" / "context_compaction.py").read_text(encoding="utf-8")
    assert "max_tokens=1500" not in src, "硬编码 1500 未清除"


def test_happy_path_single_success_uses_constant():
    client_script = [{"content": GOOD_SUMMARY}]
    client = _ScriptedClient(client_script)
    r = _compact(client)
    assert r.error is None and r.entry is not None
    assert r.entry.summary == GOOD_SUMMARY
    assert len(client.calls) == 1
    kw = client.calls[0]["kwargs"]
    assert kw["max_tokens"] == cc.COMPACTION_SUMMARY_MAX_TOKENS
    assert kw["temperature"] == 0.0
    assert kw["model"] == client.settings.llm_fast_model


def test_first_two_empty_third_nonempty():
    """真跑踩的路径：前2次空、第3次非空 → 拿到摘要、共3次调用、max_tokens 一致。"""
    client = _ScriptedClient([{"content": ""}, {"content": "   "}, {"content": GOOD_SUMMARY}])
    r = _compact(client)
    assert r.error is None and r.entry is not None
    assert r.entry.summary == GOOD_SUMMARY
    assert len(client.calls) == 3, f"应重试到第3次，实际 {len(client.calls)}"
    assert all(c["kwargs"]["max_tokens"] == cc.COMPACTION_SUMMARY_MAX_TOKENS for c in client.calls)


def test_all_empty_falls_back_byte_for_byte():
    """全空 → SUMMARY_EMPTY，不落库，kept_messages 与入参逐字节相等。"""
    msgs = _messages()
    client = _ScriptedClient([{"content": ""}])
    r = _compact(client, msgs)
    assert r.entry is None
    assert r.error == "SUMMARY_EMPTY"
    assert r.kept_messages == msgs  # 逐字节回退
    assert len(client.calls) == 3, "3 次重试全保留"


def test_exception_then_success_still_retries():
    """异常也算一次失败，后续调用仍继续（重试语义不变）。"""
    client = _ScriptedClient([RuntimeError("LLM 调用失败: timeout"),
                              {"content": ""}, {"content": GOOD_SUMMARY}])
    r = _compact(client)
    assert r.error is None and r.entry is not None
    assert len(client.calls) == 3


def test_all_exception_error_label():
    client = _ScriptedClient([RuntimeError("LLM 调用失败: timeout")])
    r = _compact(client)
    assert r.entry is None
    assert r.error.startswith("SUMMARY_CALL_FAILED")
    assert len(client.calls) == 3


def test_malformed_and_too_short_protections_intact():
    """失败保护三种标签语义不变（不改摘要格式/切点/存储）。"""
    bad_fence = _ScriptedClient([{"content": "```json\n{\"a\": 1}\n```"}])
    r1 = _compact(bad_fence)
    assert r1.entry is None and r1.error == "SUMMARY_MALFORMED"
    assert r1.kept_messages == _messages()

    too_short = _ScriptedClient([{"content": "太短了"}])
    r2 = _compact(too_short)
    assert r2.entry is None and r2.error == "SUMMARY_TOO_SHORT"
    assert r2.kept_messages == _messages()


def test_d070_chain_source_still_intact():
    """D-070：cc.maybe_compact 与 summary_block 持续注入链本次未触碰。"""
    for rel in ("agent/companion.py", "agent/breakup.py", "agent/loved_one.py", "agent/pet.py"):
        src = (BACKEND / rel).read_text(encoding="utf-8")
        assert "cc.maybe_compact(" in src, f"{rel} 的 maybe_compact 接线丢失"
    src = (BACKEND / "agent" / "companion.py").read_text(encoding="utf-8")
    assert "if summary_block:" in src
    assert '[此前对话摘要]' in src
    # 产品代码本身：每轮持续注入的语义在 cc.maybe_compact 内仍在
    csrc = (BACKEND / "memory" / "context_compaction.py").read_text(encoding="utf-8")
    assert "summary = prev.summary" in csrc
    assert "已落库的摘要继续带着" in csrc
