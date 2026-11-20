"""R4 · 时序与趋势替换验收（dateparser + hindsight trend + mem0 expiration）。

- 开关 USE_MATURE_TEMPORAL 默认关闭；关闭时 parse_time_ref / compute_state 与基线逐字节一致。
- 开启时用真实 dateparser 解析中文语料；解析失败/未覆盖自动回退基线词典。
- compute_trend 用构造证据序列验收 STALE/NEW/STRENGTHENING/WEAKENING/STABLE。
- 过期判定边界：今天不过期、昨天过期。
"""
from datetime import datetime, timedelta, timezone

import pytest

from config import get_settings
from memory import state, store, temporal, trends
from memory.db import _RawSessionLocal


# 改动前由 work/r4_baseline_golden.py 实跑采集（见 work/r4_baseline_golden.txt）
BASELINE_PARSE_GOLDEN = {
    "去年生日": {"lo": 270, "hi": 460, "text": ["去年", "生日"]},
    "三年前": {"lo": 766.5, "hi": 1642.5, "text": []},
    "上个月": {"lo": 15, "hi": 50, "text": ["上个月"]},
    "前天": None,
    "前几天": {"lo": 1, "hi": 10, "text": ["前几天"]},
    "两个月前": {"lo": 42.0, "hi": 90.0, "text": []},
    "2020年": None,
    "今天天气不错": None,
    "": None,
    "十年前的事": {"lo": 2555.0, "hi": 5475.0, "text": []},
}


@pytest.fixture
def mature_off(monkeypatch):
    monkeypatch.delenv("USE_MATURE_TEMPORAL", raising=False)


@pytest.fixture
def mature_on(monkeypatch):
    monkeypatch.setenv("USE_MATURE_TEMPORAL", "1")


# ---------------------------------------------------------------- 开关 & 回退

def test_switch_default_off_and_truthy_values(monkeypatch):
    for value in (None, "", "0", "false", "no", "off"):
        if value is None:
            monkeypatch.delenv("USE_MATURE_TEMPORAL", raising=False)
        else:
            monkeypatch.setenv("USE_MATURE_TEMPORAL", value)
        assert temporal.mature_temporal_enabled() is False
    for value in ("1", "true", "YES", " on "):
        monkeypatch.setenv("USE_MATURE_TEMPORAL", value)
        assert temporal.mature_temporal_enabled() is True


def test_switch_off_is_byte_identical_to_baseline(mature_off):
    """逐字节对照：改动前实跑采集的基线输出 vs 开关关闭后的输出（repr 完全相等）。"""
    assert temporal.parse_time_ref("") is None
    for query, expected in BASELINE_PARSE_GOLDEN.items():
        got = temporal.parse_time_ref(query)
        assert repr(got) == repr(expected), f"{query!r}: {got!r} != {expected!r}"
    print("\n[开关关·逐字节对照] " + "; ".join(
        f"{q!r}->{temporal.parse_time_ref(q)!r}" for q in BASELINE_PARSE_GOLDEN
    ))


def test_dateparser_failure_falls_back_to_legacy(mature_on, monkeypatch):
    def boom(*_a, **_kw):
        raise RuntimeError("dateparser unavailable")
    monkeypatch.setattr(temporal, "_dateparser_parse", boom)
    assert temporal.parse_time_ref("三年前") == BASELINE_PARSE_GOLDEN["三年前"]
    assert temporal.parse_time_ref("去年生日") == BASELINE_PARSE_GOLDEN["去年生日"]


# ---------------------------------------------------------------- ② dateparser 真实语料

def test_dateparser_parses_chinese_corpora(mature_on):
    pytest.importorskip("dateparser")
    expected_ranges = {
        "去年生日": (355, 375),   # 去年 ≈ 365 天
        "三年前": (1080, 1110),   # 3 年 ≈ 1095 天
        "上个月": (27, 32),       # 自然月上一个月 ≈ 30 天
        "前天": (2, 2),           # dateparser zh.py "2 day ago": ["前天"]
    }
    for query, (lo_ok, hi_ok) in expected_ranges.items():
        ref = temporal.parse_time_ref(query)
        print(f"\n[dateparser 实解析] {query!r} -> {ref!r}")
        assert ref is not None
        assert lo_ok <= ref["lo"] <= hi_ok, (query, ref)
        assert ref["lo"] == ref["hi"]  # 精确日期 → 退化成单点窗口，基线返回形态不变
        assert isinstance(ref["text"], list)
    assert "生日" in temporal.parse_time_ref("去年生日")["text"]
    assert temporal.parse_time_ref("前天")["text"] == ["前天"]
    # 未覆盖的标准词回退基线词典（「上上个月」长词优先，dateparser 不认→回退）
    assert temporal.parse_time_ref("上上个月") == temporal._legacy_parse_time_ref("上上个月")
    assert temporal.parse_time_ref("前几天") == {"lo": 1, "hi": 10, "text": ["前几天"]}


# ---------------------------------------------------------------- ③ hindsight compute_trend

def _evidence(now, days_ago, count=1):
    return [
        trends.ObservationEvidence(
            memory_id=f"m{i}", quote="那时候我连白天都熬不住",
            timestamp=now - timedelta(days=days_ago),
        )
        for i in range(count)
    ]


def test_compute_trend_constructed_sequences():
    now = datetime(2026, 11, 21, 12, 0, 0)
    cases = [
        ("STALE(全是 100 天前)", _evidence(now, 100, 3), trends.Trend.STALE),
        ("NEW(全是近 5 天)", _evidence(now, 5, 3) + _evidence(now, 6, 2), trends.Trend.NEW),
        ("STRENGTHENING(近期密/远期疏)", _evidence(now, 5, 10) + _evidence(now, 60, 1), trends.Trend.STRENGTHENING),
        ("WEAKENING(近期疏/远期密)", _evidence(now, 5, 1) + _evidence(now, 60, 10), trends.Trend.WEAKENING),
        ("STABLE(密度相当)", _evidence(now, 5, 2) + _evidence(now, 60, 4), trends.Trend.STABLE),
        ("空证据", [], trends.Trend.STALE),
    ]
    for name, evidence, expected in cases:
        got = trends.compute_trend(evidence, now=now)
        print(f"\n[compute_trend] {name}: evidence={len(evidence)} -> {got.value}")
        assert got is expected

    assert trends.Trend.STRENGTHENING.value == "strengthening"
    assert trends.Trend.STALE.value == "stale"
    assert trends.TREND_SCORE_DELTA[trends.Trend.STRENGTHENING] == 2.0


# ---------------------------------------------------------------- ④ mem0 过期判定边界

def test_payload_expiration_boundary(mature_on):
    # mem0 原文用 datetime.now(timezone.utc).date() 作「今天」，这里按同一口径构造边界。
    today = datetime.now(tz=timezone.utc).date()
    yesterday = today - timedelta(days=1)
    assert temporal._payload_is_expired({"expiration_date": today.isoformat()}) is False
    assert temporal._payload_is_expired({"expiration_date": yesterday.isoformat()}) is True
    print(f"\n[过期边界·开启] today={today} -> False; yesterday={yesterday} -> True")
    assert temporal.payload_is_expired({"expiration_date": yesterday.isoformat()}) is True
    assert temporal._payload_is_expired({"expiration_date": "not-a-date"}) is False
    assert temporal._payload_is_expired({}) is False
    assert temporal._payload_is_expired(None) is False


def test_payload_expiration_switch_off(mature_off):
    yesterday = (datetime.now(tz=timezone.utc).date() - timedelta(days=1)).isoformat()
    assert temporal.payload_is_expired({"expiration_date": yesterday}) is False
    # 原函数（纯移植）本身仍可测，但入口在开关关闭时不调用它
    assert temporal._payload_is_expired({"expiration_date": yesterday}) is True


# ---------------------------------------------------------------- ① compute_state 逐字节对照 + 趋势增强

def _seed_state_fixture():
    offset = get_settings().timezone_offset_hours
    local_noon = (datetime.now() + timedelta(hours=offset)).replace(hour=12, minute=0, second=0, microsecond=0)
    with _RawSessionLocal() as db:
        uid = store.get_or_create_user(db, "r4-trend-user").id
        for days_ago, score, emo in [(40, 30, "难过"), (20, 40, "难过"),
                                     (10, 60, "平静"), (3, 75, "释怀"), (1, 80, "平静")]:
            m = store.add_memory(db, uid, content=f"d{days_ago}", summary=f"d{days_ago}",
                                 emotion={"emotion": emo, "score": score})
            m.ts = local_noon - timedelta(hours=offset, days=days_ago)
            db.commit()
        return uid


BASELINE_STATE_GOLDEN = {
    "baseline": 57.0, "trend": 36.7, "volatility": 19.4,
    "acute_ratio": 0.4, "calm_ratio": 0.6, "risk": 0.33,
    "reconcile": 66.6, "stage": 2, "stage_label": "趋稳期",
    "n_days": 5, "n_memories": 5,
}


def test_compute_state_switch_off_matches_baseline_and_on_enhances(mature_off, monkeypatch):
    uid = _seed_state_fixture()
    with _RawSessionLocal() as db:
        off = state.compute_state(db, uid)
        print(f"\n[compute_state·开关关] {off!r}")
        assert off == BASELINE_STATE_GOLDEN  # 改动前实跑采集的基线值

        monkeypatch.setenv("USE_MATURE_TEMPORAL", "1")
        on = state.compute_state(db, uid)
        print(f"[compute_state·开关开] {on!r}")
        assert set(on) == set(off)                 # 返回形态不变
        assert on["trend"] == 38.7                 # 证据趋势 STRENGTHENING → +2.0
        assert on["baseline"] == off["baseline"]
        assert on["volatility"] == off["volatility"]

        monkeypatch.delenv("USE_MATURE_TEMPORAL", raising=False)
        off_again = state.compute_state(db, uid)
        assert repr(off_again) == repr(off)         # 关回去逐字节一致
