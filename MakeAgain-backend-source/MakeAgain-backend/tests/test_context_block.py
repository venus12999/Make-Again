"""包 E 离线回归守卫：稳定分层 / mute_topic 开关 / D-070 摘要链 / recent_text / AI 披露。

口径：synthetic-handwritten + 离线模拟（fake 对象 + monkeypatch），
不访问网络、不读真实 key、不启动后端，绝不冒充真人验证。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from agent import companion
from memory import temporal

BACKEND = Path(__file__).resolve().parents[1]


class _FakeEntry:
    def __init__(self, summary: str, eid: int, ts=None, facts=None):
        self.summary = summary
        self.content = ""
        self.id = eid
        self.ts = ts
        self.facts = facts or []


class _FakeDailyPick:
    opening = "今天聊聊那段一起看过的海"


@pytest.fixture(autouse=True)
def _stub_temporal(monkeypatch):
    """时间线两段注入用离线替身；只影响本模块测试，monkeypatch 自动还原。"""
    monkeypatch.setattr(
        temporal, "recall_daily_summaries",
        lambda db, uid, msg: "[近期状态轨迹] 最近睡得好一点",
    )
    monkeypatch.setattr(
        temporal, "recall_by_time",
        lambda db, uid, msg, skip_ids=None: "[相关往事] 3月12日 一起看过海",
    )


def _inputs():
    portraits = {
        "user": {"称呼": "小满", "失去类型": "分手"},
        "object": {"称呼": "林夏", "关系": "前任"},
    }
    memories = [{"entry": _FakeEntry("她喜欢海", eid=7, ts=datetime(2026, 3, 12))}]
    return portraits, memories, _FakeDailyPick()


def test_context_block_tiers_stable_semi_volatile():
    """P0：双画像 → 相关记忆 → 今日主题+时间线；原有信息一条不丢。"""
    portraits, memories, pick = _inputs()
    out = companion._context_block(object(), "u-1", "想到那天", portraits, memories, pick)
    assert out.index("[用户画像]") < out.index("[思念对象画像]")
    assert out.index("[思念对象画像]") < out.index("[相关记忆")
    assert out.index("[相关记忆") < out.index("[今日启发]")
    assert out.index("[今日启发]") < out.index("[近期状态轨迹]")
    assert out.index("[近期状态轨迹]") < out.index("[相关往事]")
    for token in ("小满", "林夏", "她喜欢海", "今天聊聊那段一起看过的海", "3月12日 一起看过海"):
        assert token in out, f"信息丢失：{token}"


def test_mute_topic_default_off_zero_behavior_change(monkeypatch):
    """P1：开关默认关；当前无现成字段，_mute_topics 恒空，输出逐字不变。"""
    portraits, memories, pick = _inputs()
    monkeypatch.delenv("USE_MUTE_TOPIC", raising=False)
    assert companion._mute_topic_enabled() is False
    base = companion._context_block(object(), "u-1", "hi", portraits, memories, pick)
    monkeypatch.setenv("USE_MUTE_TOPIC", "1")
    assert companion._mute_topic_enabled() is True
    assert companion._mute_topics(object(), "u-1") == []
    assert companion._context_block(object(), "u-1", "hi", portraits, memories, pick) == base


def test_mute_topic_filter_hook_when_wired(monkeypatch):
    """接线点验证：未来 user state 提供 mute 列表后，过滤钩子按子串生效。"""
    monkeypatch.setenv("USE_MUTE_TOPIC", "1")
    monkeypatch.setattr(companion, "_mute_topics", lambda db, uid: ["海"])
    portraits, memories, pick = _inputs()
    out = companion._context_block(object(), "u-1", "hi", portraits, memories, pick)
    assert "她喜欢海" not in out
    assert "[思念对象画像]" in out  # 稳定区不受影响


def test_d070_summary_chain_intact():
    """D-070 守卫：压缩摘要仍每轮作为独立 system 消息注入，未被本次改动破坏。"""
    src = (BACKEND / "agent" / "companion.py").read_text(encoding="utf-8")
    assert "from memory import context_compaction as cc" in src
    assert "turns, summary_block = cc.maybe_compact(" in src
    assert "if summary_block:" in src
    assert 'msgs.append({"role": "system", "content": "[此前对话摘要]' in src
    assert "turns[-1:]" in src
    bsrc = (BACKEND / "agent" / "breakup.py").read_text(encoding="utf-8")
    assert "cc.maybe_compact(" in bsrc
    assert '" ".join(t["content"] for t in turns[-1:])' in bsrc


def test_chat_response_has_disclosure_offline_mock(monkeypatch):
    """GAP-D1：/api/chat 响应新增 disclosure，复用 ending/call 文案；旧字段仍在。"""
    from api import routes

    class _Companion:
        id = "c-1"
        chat_session_id = "s-1"
        archived_at = None

    class _Row:
        def __init__(self, rid: int):
            self.id = rid
            self.ts = datetime(2026, 11, 24, 8, 0, 0)

    class _Scalars:
        def all(self):
            return [_Row(2), _Row(1)]

    class _DB:
        info = {"request_id": "req-1"}

        def scalars(self, stmt):
            return _Scalars()

    monkeypatch.setattr(routes.store, "get_current_companion", lambda db, uid: _Companion())
    monkeypatch.setattr(
        routes.companion, "chat",
        lambda db, uid, msg, client, session_id=None, no_tools=False: {
            "reply": "mock 回复", "crisis": {}, "emotion": None, "recalled": 0,
        },
    )
    resp = routes.chat(
        routes.ChatReq(user_id="u-1", message="你好", session_id="s-1"), db=_DB()
    )
    assert resp["disclosure"] == routes.call_mod._DISCLOSURE
    assert resp["status"] == "completed"
    assert resp["session_id"] == "s-1"
