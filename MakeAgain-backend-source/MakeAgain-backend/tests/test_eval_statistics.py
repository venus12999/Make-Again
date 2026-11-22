"""S1 统计门禁行为证据（构造性测试，口径：手写模拟分数，非真实用户数据）。

覆盖 S1-V1~V7：
- 纯统计函数（bootstrap / sign test / MDE）单测
- V1 repetitions 保留逐 case 多次分数
- V2 同 case 配对 delta + 符号检验 p 值
- V3 抖动 vs 稳定提升/退化 的构造性区分
- V4 危机 recall 红线不许退化
- V5 stdout 含 n/MDE，n<30 标功效不足
- V7 --no-repetitions 与旧行为 stdout 逐字节一致（golden 来自 git show HEAD 的真实旧代码）
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "eval"))
import iterate as it  # noqa: E402


# ---------------------------------------------------------------- 纯函数单测
def test_bootstrap_ci_deterministic_and_ordered():
    tight = it.bootstrap_ci([4.0, 4.1, 3.9, 4.0, 4.0], seed=7)
    wide = it.bootstrap_ci([1.0, 5.0, 2.0, 5.0, 1.0], seed=7)
    assert it.bootstrap_ci([4.0, 4.1, 3.9, 4.0, 4.0], seed=7) == tight  # 可复现
    assert tight[0] <= 4.0 <= tight[1]
    assert wide[0] <= 2.8 <= wide[1]
    assert (wide[1] - wide[0]) > (tight[1] - tight[0])  # 方差大 -> CI 更宽
    assert it.bootstrap_ci([], seed=7) == (None, None)
    assert it.bootstrap_ci([3.0], seed=7) == (3.0, 3.0)


def test_sign_test_exact_values():
    all_pos = it.sign_test([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert all_pos["pos"] == 10 and all_pos["neg"] == 0
    assert abs(all_pos["p_value"] - 2 / 1024) < 1e-6
    assert all_pos["significant"] is True

    mixed = it.sign_test([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
    assert mixed["p_value"] == 1.0 and mixed["significant"] is False

    zeros = it.sign_test([0, 0, 0])
    assert zeros["p_value"] == 1.0 and zeros["zero"] == 3

    with_zero = it.sign_test([1, -1, 0, 0])
    assert with_zero["pos"] == 1 and with_zero["neg"] == 1 and with_zero["zero"] == 2


def test_mde_formula_and_sample_size_effect():
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - 0.05 / 2) + NormalDist().inv_cdf(0.80)
    m = it.mde(100, 1.0)
    assert m is not None and m == round(z / 10, 4)
    assert it.mde(25, 1.0) > m  # n 越小 MDE 越大（越难检出）
    assert it.mde(1, 1.0) is None


# ---------------------------------------------------------------- V3 构造性
def _per_case(vals):
    return {f"c{i + 1:02d}": [float(v), float(v), float(v)] for i, v in enumerate(vals)}


def test_v3_mean_up_but_within_noise_is_hold():
    """均值 +0.2 但配对 delta 正负交错、CI 重叠 -> 持平（抖动）。"""
    base_vals = [3.0, 5.0] * 10  # 均值 4.0，高方差
    deltas = [1.0] * 12 + [-1.0] * 8  # 12 正 8 负 -> 符号检验不显著
    after_vals = [b + d for b, d in zip(base_vals, deltas)]  # 均值 4.2
    r = it.compare_metric("judge_avg", _per_case(base_vals), _per_case(after_vals))
    assert r["delta_mean"] == pytest.approx(0.2, abs=1e-6)
    assert r["sign"]["significant"] is False
    assert r["ci_dir"] == 0
    assert r["verdict"] == "持平"


def test_v3_stable_improvement_is_improved():
    base_vals = [3.0] * 20
    after_vals = [4.5] * 20  # 全部 +1.5，CI 不重叠且符号检验全正
    r = it.compare_metric("judge_avg", _per_case(base_vals), _per_case(after_vals))
    assert r["ci_dir"] == 1
    assert r["sign"]["pos"] == 20 and r["sign"]["p_value"] < 0.01
    assert r["verdict"] == "改善"


def test_v3_stable_degradation_is_degraded():
    base_vals = [5.0] * 20
    after_vals = [3.0] * 20
    r = it.compare_metric("judge_avg", _per_case(base_vals), _per_case(after_vals))
    assert r["ci_dir"] == -1
    assert r["verdict"] == "退化（建议不合入）"
    assert it.overall_verdict([r["verdict"]]) == "❌ 有核心指标退化，建议不合入/回滚"


# ---------------------------------------------------------------- V4 危机红线
def test_v4_crisis_recall_degradation_blocked():
    base = {f"c{i:02d}": [1.0, 1.0, 1.0] for i in range(12)}
    after = {f"c{i:02d}": [1.0, 1.0, 1.0] for i in range(12)}
    for i in range(4):  # 4/12 危机 case 漏检 -> recall 下降
        after[f"c{i:02d}"] = [0.0, 0.0, 0.0]
    r = it.compare_metric("crisis_detect_rate", base, after, crisis_line=True)
    assert r["delta_mean"] < 0
    assert "退化" in r["verdict"] and "不合入" in r["verdict"]
    assert it.overall_verdict([r["verdict"]]) == "❌ 有核心指标退化，建议不合入/回滚"


def test_v4_crisis_no_change_stays_hold():
    base = {f"c{i:02d}": [1.0] * 3 for i in range(12)}
    r = it.compare_metric("crisis_detect_rate", base, base, crisis_line=True)
    assert r["verdict"] == "持平"


# ---------------------------------------------------------------- V5 n/MDE/功效
def test_v5_n_mde_and_low_power_warning():
    base = _per_case([3.0] * 17)
    after = _per_case([3.0] * 17)
    r = it.compare_metric("judge_avg", base, after)
    txt = it.format_metric_report(r)
    assert "n=17" in txt and "MDE" in txt and "功效不足" in txt

    base_hi = _per_case([3.0] * 40)
    r_hi = it.compare_metric("judge_avg", base_hi, _per_case([3.0] * 40))
    txt_hi = it.format_metric_report(r_hi)
    assert "n=40" in txt_hi and "功效不足" not in txt_hi


# ---------------------------------------------------------------- V1 重复次数
def test_v1_repetitions_keep_every_score(monkeypatch):
    """mock run_batch：同一 case 三次给不同分，验证逐 case 三次分数都保留。"""
    calls = {"n": 0}

    def fake_run_batch(name, use_raw, client):
        calls["n"] += 1
        k = calls["n"]
        out = it.BACKEND.parent / "outputs" / "benchmarks" / "demo.json"
        rows = [{"id": "c1", "judge": {"score": 3 + k}},
                {"id": "c2", "judge": {"score": 5 - k}}]
        out.write_text(json.dumps({"summary": {"n": 2, "judge_avg": 0,
                                               "emotion_hit_rate": 0, "crisis_detect_rate": 0},
                                   "results": rows}, ensure_ascii=False), encoding="utf-8")

    class FakeRB:
        BATCHDIR = BACKEND / "eval" / "batches"
        OUTDIR = BACKEND.parent / "outputs" / "benchmarks"

        @staticmethod
        def load_key():
            return {}

        @staticmethod
        def setup(conf):
            pass

        @staticmethod
        def run_batch(name, use_raw, client):
            fake_run_batch(name, use_raw, client)

    sys.modules["run_benchmarks"] = FakeRB
    monkeypatch.setattr(it, "run_batches", it.run_batches)  # 真用新实现
    (FakeRB.BATCHDIR / "demo.json").write_text('{"items":[]}', encoding="utf-8")
    try:
        out = it.run_batches(["demo"], repetitions=3)
    finally:
        (FakeRB.BATCHDIR / "demo.json").unlink(missing_ok=True)
        (FakeRB.OUTDIR / "demo.json").unlink(missing_ok=True)
        sys.modules.pop("run_benchmarks", None)
    pc = out["demo"]["per_case"]["judge_avg"]
    assert pc["c1"] == [4.0, 5.0, 6.0]
    assert pc["c2"] == [4.0, 3.0, 2.0]
    assert out["demo"]["repetitions"] == 3


# ---------------------------------------------------------------- V2 配对输出
def test_v2_paired_delta_and_sign_output(monkeypatch, capsys):
    base = {
        "ts": "T", "git": {"commit": "c", "branch": "b"},
        "batches": {"demo": {
            "n": 3, "judge_avg": 3.0, "emotion_hit_rate": 0.0, "crisis_detect_rate": 1.0,
            "per_case": {
                "judge_avg": {"c1": [3.0] * 3, "c2": [4.0] * 3, "c3": [2.0] * 3},
                "crisis_detect_rate": {"c1": [1.0] * 3, "c2": [1.0] * 3, "c3": [1.0] * 3},
            },
        }},
        "note": "",
    }
    bp = BACKEND.parent / "outputs" / "benchmarks" / "_v2_test_baseline.json"
    bp.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")
    hp = BACKEND.parent / "outputs" / "benchmarks" / "_v2_test_history.jsonl"
    hp.write_text("", encoding="utf-8")
    monkeypatch.setattr(it, "BASELINE", bp)
    monkeypatch.setattr(it, "HISTORY", hp)
    monkeypatch.setattr(it, "git_info", lambda: {"commit": "c", "branch": "b"})
    monkeypatch.setattr(it, "prompt_versions", lambda: {})
    now = {"demo": {
        "n": 3, "judge_avg": 3.33, "emotion_hit_rate": 0.0, "crisis_detect_rate": 1.0,
        "per_case": {
            "judge_avg": {"c1": [3.0] * 3, "c2": [5.0] * 3, "c3": [2.0] * 3},
            "crisis_detect_rate": {"c1": [1.0] * 3, "c2": [1.0] * 3, "c3": [1.0] * 3},
        },
    }}
    monkeypatch.setattr(it, "run_batches", lambda names, repetitions: now)
    try:
        it.cmd_compare_statistical(SimpleNamespace(batch=None, note="", repetitions=3))
    finally:
        bp.unlink(missing_ok=True)
        hp.unlink(missing_ok=True)
    out = capsys.readouterr().out
    assert "c1: " in out and "mean=" in out  # V1 逐 case 多次分数
    assert "delta列表" in out  # V2 配对 delta
    assert "配对符号检验" in out and "p=" in out  # V2 符号检验 p 值
    assert "n=3" in out and "MDE" in out and "功效不足" in out  # V5


# ---------------------------------------------------------------- V7 逐字节回退
GOLDEN = (
    "[compare] running 1 batches ...\n"
    "\n"
    "======================================================================\n"
    "对照基线: 2026-01-01T00:00:00+00:00 commit=abc1234\n"
    "======================================================================\n"
    "\n"
    "[demo]\n"
    "  judge_avg               3.0 ->    3.5  (+0.500)  改善\n"
    "  emotion_hit_rate        0.0 ->    0.0  (+0.000)  持平\n"
    "  crisis_detect_rate      1.0 ->    1.0  (+0.000)  持平\n"
    "\n"
    "======================================================================\n"
    "总判定: ✅ 改善，可合入\n"
    "======================================================================\n"
)


def test_v7_no_repetitions_byte_identical(monkeypatch, capsys):
    base = {
        "ts": "2026-01-01T00:00:00+00:00",
        "git": {"commit": "abc1234", "branch": "main"},
        "batches": {"demo": {"n": 2, "judge_avg": 3.0,
                             "emotion_hit_rate": 0.0, "crisis_detect_rate": 1.0}},
        "note": "",
    }
    bp = BACKEND.parent / "outputs" / "benchmarks" / "_v7_test_baseline.json"
    bp.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")
    hp = BACKEND.parent / "outputs" / "benchmarks" / "_v7_test_history.jsonl"
    hp.write_text("", encoding="utf-8")
    monkeypatch.setattr(it, "BASELINE", bp)
    monkeypatch.setattr(it, "HISTORY", hp)
    monkeypatch.setattr(it, "git_info", lambda: {"commit": "xyz9999", "branch": "test"})
    monkeypatch.setattr(it, "prompt_versions", lambda: {})
    now = {"demo": {"n": 2, "judge_avg": 3.5,
                    "emotion_hit_rate": 0.0, "crisis_detect_rate": 1.0}}
    monkeypatch.setattr(it, "run_batches_legacy", lambda names: {k: dict(v) for k, v in now.items()})
    try:
        it.cmd_compare(SimpleNamespace(batch=None, note="golden", repetitions=3,
                                       no_repetitions=True))
    finally:
        bp.unlink(missing_ok=True)
        hp.unlink(missing_ok=True)
    assert capsys.readouterr().out == GOLDEN  # 与 git show HEAD 的真实旧代码输出逐字节一致
