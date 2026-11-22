"""S1 统计门禁行为证据（构造性测试，口径：手写模拟分数，非真实用户数据）。

覆盖 S1-V1~V7：
- 纯统计函数（bootstrap / sign test / MDE）单测
- V1 repetitions 保留逐 case 多次分数
- V2 同 case 配对 delta + 符号检验 p 值
- V3 抖动 vs 稳定提升/退化 的构造性区分
- V4 危机 recall 红线不许退化
- V5 stdout 含 n/MDE，n<30 标功效不足
- V7 --no-repetitions 与旧行为 stdout 逐字节一致（golden 来自 git show HEAD 的真实旧代码）

2026-11-23 方法升级追加（构造数据，非真实用户/模型输出）：
- C-V2 稳健判定真实数字（n/配对符号检验 p/噪声带/判定）
- C-V3 噪声门槛：19正1负、均值+0.6 低于噪声带 0.784 -> 建议不合入
- C-V4 --no-robust 回退升级前统计判定；--no-repetitions 仍逐字节回退
- C-V5 全量 pytest 无回归（见 docs/EVAL_METHOD_UPGRADE.md 验收命令）
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


# ================================================================
# 2026-09 方法升级（C-V1~V5）：URL/日期/权重见 docs/EVAL_METHOD_UPGRADE.md
# 全部为构造性手写分数，非真实用户数据 / 非真实模型输出。
# ================================================================
def test_c_noise_gate_pure_numbers():
    """噪声带 = 1.96*sd(delta)/sqrt(n)，构造样本给出可追数字。"""
    deltas = [1.0] * 19 + [-7.0]
    g = it.noise_gate(deltas, 0.6, min_improve=0.1)
    assert g["noise_se"] == 0.4
    assert g["noise_band"] == 0.784
    assert g["gain_above_noise"] is False
    assert g["eval_noise_too_high"] is True  # 0.784 >= 0.1 -> 建议加样本再开跑
    g2 = it.noise_gate([1.5] * 8, 1.5, min_improve=0.1)
    assert g2["noise_band"] == 0.0 and g2["gain_above_noise"] is True
    assert it.noise_gate([1.0], 1.0)["noise_band"] is None  # n<2 不可用


def test_c_robust_gate_real_numbers(capsys):
    """C-V2：稳健判定在构造样本上产出真实数字（n/配对/符号检验/噪声带/判定）。"""
    base = _per_case([7.0] * 20)
    after = _per_case([8.0] * 19 + [0.0])
    r = it.compare_metric("judge_avg", base, after)
    print(f"[C-V2] n={r['n']} delta={r['delta_mean']} "
          f"sign_pos={r['sign']['pos']} sign_neg={r['sign']['neg']} "
          f"p={r['sign']['p_value']} noise_band={r['noise']['noise_band']} "
          f"verdict={r['verdict']}")
    out = capsys.readouterr().out
    assert "[C-V2] n=20 delta=0.6 sign_pos=19 sign_neg=1 p=4e-05" in out
    assert "noise_band=0.784" in out and "增益在噪声内" in out
    assert r["b_mean"] == 7.0 and r["a_mean"] == 7.6
    assert r["bootstrap_caveat"] is not None  # n=20 < 100


def test_c_noise_gate_blocks_gain_within_noise():
    """C-V3：19正1负、均值+0.6 但低于噪声带 0.784 -> 建议不合入；关开关恢复旧判定。"""
    base = _per_case([7.0] * 20)
    after = _per_case([8.0] * 19 + [0.0])
    robust = it.compare_metric("judge_avg", base, after)
    legacy = it.compare_metric("judge_avg", base, after, robust_gate=False)
    assert robust["verdict"] == "持平（增益在噪声内，建议不合入）"
    assert "噪声内" in robust["reason"]
    assert legacy["verdict"] == "改善"  # 升级前：CI+符号检验同向即判改善
    assert legacy["noise"] is None and legacy["bootstrap_caveat"] is None
    assert it.overall_verdict([robust["verdict"]]) == (
        "⚠ 增益在噪声内，建议不合入（先加重复次数/样本量再开跑）")


def test_c_small_sample_caveat_only_below_100():
    """evalstats N<100 才标 bootstrap 不可靠；大样本保留 bootstrap 作对照不加警告。"""
    small = it.compare_metric("judge_avg", _per_case([3.0] * 17), _per_case([3.0] * 17))
    large = it.compare_metric("judge_avg", _per_case([3.0] * 120), _per_case([3.0] * 120))
    assert "bootstrap CI 不可靠" in it.format_metric_report(small)
    assert "bootstrap CI 不可靠" not in it.format_metric_report(large)
    assert small["bootstrap_caveat"] and large["bootstrap_caveat"] is None
    assert small["noise"]["noise_band"] == 0.0


def test_c_overfit_warning_threshold_pure():
    assert it.overfit_warning(0) is None
    assert it.overfit_warning(1) is None
    assert "过拟合风险" in it.overfit_warning(2)
    assert "3 次" in it.overfit_warning(3)


def _c_compare_ctx(monkeypatch, ts, history_lines):
    """构造 compare 的基线/历史/当前批次（全部构造数据，不读真实 key）。"""
    base = {
        "ts": ts, "git": {"commit": "c", "branch": "b"},
        "batches": {"demo": {"n": 20, "judge_avg": 7.0,
                             "per_case": {"judge_avg": {
                                 f"c{i + 1:02d}": [7.0] * 3 for i in range(20)}}}},
        "note": "",
    }
    bp = BACKEND.parent / "outputs" / "benchmarks" / f"_c_{ts.replace('-', '_')}_baseline.json"
    hp = BACKEND.parent / "outputs" / "benchmarks" / f"_c_{ts.replace('-', '_')}_history.jsonl"
    bp.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")
    hp.write_text("\n".join(history_lines) + ("\n" if history_lines else ""), encoding="utf-8")
    monkeypatch.setattr(it, "BASELINE", bp)
    monkeypatch.setattr(it, "HISTORY", hp)
    monkeypatch.setattr(it, "git_info", lambda: {"commit": "c", "branch": "b"})
    monkeypatch.setattr(it, "prompt_versions", lambda: {})
    now = {"demo": {"n": 20, "judge_avg": 7.6, "per_case": {"judge_avg": {
        **{f"c{i + 1:02d}": [8.0] * 3 for i in range(19)},
        "c20": [0.0] * 3}}}}
    monkeypatch.setattr(it, "run_batches", lambda names, repetitions: now)
    return bp, hp


def test_c_overfit_marker_in_compare_output(monkeypatch, capsys):
    """同基线同批 compare 2 次后，第 3 次应打印过拟合风险并写入快照。"""
    ts = "T-OVERFIT"
    lines = [json.dumps({"event": "compare", "ts": f"T{i}",
                         "git": {"commit": "c", "branch": "b"},
                         "vs_baseline": ts, "batches": {"demo": {}}, "verdict": "持平"},
                        ensure_ascii=False) for i in (1, 2)]
    bp, hp = _c_compare_ctx(monkeypatch, ts, lines)
    try:
        it.cmd_compare_statistical(SimpleNamespace(batch=None, note="", repetitions=3,
                                                   no_robust=False))
        written = hp.read_text(encoding="utf-8").splitlines()
    finally:
        bp.unlink(missing_ok=True)
        hp.unlink(missing_ok=True)
    out = capsys.readouterr().out
    assert "[overfit] 该批已被用于 2 次调参对比，存在过拟合风险" in out
    assert "噪声门槛: 噪声带=0.7840" in out
    assert "同批多次调参，存在过拟合风险" in out
    snap = json.loads(written[-1])
    assert snap["statistical"]["prior_compares_same_batch"] == 2
    assert "过拟合风险" in snap["statistical"]["overfit_warning"]
    assert snap["statistical"]["methods"] == ["sign_test_primary", "bootstrap_ci_reference",
                                              "noise_gate", "overfit_marker"]


def test_c_no_robust_switch_restores_old_statistical_output(monkeypatch, capsys):
    """C-V4：--no-robust 走升级前判定，报告无新标注且增益不再被噪声门拦。"""
    ts = "T-NO-ROBUST"
    bp, hp = _c_compare_ctx(monkeypatch, ts, [])
    try:
        it.cmd_compare_statistical(SimpleNamespace(batch=None, note="", repetitions=3,
                                                   no_robust=True))
        written = hp.read_text(encoding="utf-8").splitlines()
    finally:
        bp.unlink(missing_ok=True)
        hp.unlink(missing_ok=True)
    out = capsys.readouterr().out
    assert "robust_gate=False" in out
    assert "噪声门槛" not in out and "bootstrap CI 不可靠" not in out and "[overfit]" not in out
    assert "(+0.600)  改善" in out
    snap = json.loads(written[-1])
    assert snap["statistical"]["robust_gate"] is False
    assert snap["statistical"]["prior_compares_same_batch"] == 0
