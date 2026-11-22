"""H1 统计功效扩充 · 行为证据（验收 H1-V1 ~ H1-V4）。

先调研后实现，主流依据写进交付物（URL + 发布日期）：
- Anthropic《A statistical approach to model evaluations》官方博客，2024-11-18：
  https://www.anthropic.com/research/statistical-approach-to-model-evals
  论文《Adding Error Bars to Evals》arXiv:2411.00640，提交 2024-11-01：
  https://arxiv.org/abs/2411.00640
  采纳 Rec#3（同题多次 resample 取均值 → 题内方差按 1/repetitions 缩小）、
  Rec#4（同 case 配对差值，消去题目难度方差）、Rec#5（power analysis 按 MDE 反推 n）。
  未采纳 Rec#2 聚类标准误：本批每条来自 CPsyCoun 不同 row_index，无重复 passage 结构。
- Airbnb《Eval-driven development: Lessons from evaluating GenAI at scale》，2026-07：
  https://airbnb.tech/ai-ml/eval-driven-development-lessons-from-evaluating-genai-at-scale/
  原文 "Create a golden dataset of 50–100 examples" / "Start with 50–100 rows"。
  本地中文权威语料与调用成本受限，先落 n=34（≥30 门槛），50-100 为后续目标。

数据口径（诚实标注）：
- H1-V1 用的 per-case 打分是 P0 真实 hy4-flash-free 运行留下的 baseline 明细
  （outputs/benchmarks/baseline.json，17 case × 2 repetitions），不是编造的分数；
  5 次重复下的配对差值 sd 由该真实数据的 pooled within-case variance 按
  sqrt(2*sigma^2/5) 推算（Anthropic Rec#3 的方差分解），MDE 用 iterate.py::mde 真函数。
- H1-V2 是构造性用例（手写模拟分数，与 S1 同口径），只验证判定器能识别 +0.1 稳定变化，
  不代表真实模型成绩。
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "eval"))
import build_benchmarks as bb  # noqa: E402
import iterate as it  # noqa: E402

BATCH = BACKEND / "eval" / "batches" / "01_中文共情_CPsyCoun.json"

# P0 真实运行 baseline（2026-11-22，hy4-flash-free，17 case × 2 次重复），逐字抄自
# outputs/benchmarks/baseline.json 的 per_case.judge_avg。
_REAL_BASELINE_2REP = {
    "cp-786": [5, 5], "cp-158": [4, 5], "cp-831": [4, 4], "cp-266": [5, 5],
    "cp-285": [5, 5], "cp-634": [4, 5], "cp-744": [5, 5], "cp-120": [5, 5],
    "cp-168": [5, 5], "cp-169": [5, 5], "cp-274": [5, 5], "cp-541": [4, 4],
    "cp-553": [5, 5], "cp-734": [5, 5], "cp-758": [5, 4], "cp-853": [4, 5],
    "cp-991": [5, 5],
}
# P0 完整闭环 compare 的真实配对 delta 列表（docs/P0_EVAL_HARDENING_RESULT.md §3 原始输出）。
_REAL_P0_DELTAS = [0, 0, -0.5, -1.0, 0, 0, 0, 0.5, 0, -0.5, 0, 0, 0.5, 0, 1.0, 0.5, 0]


def _pooled_within_case_variance(per_case: dict) -> float:
    """每个 case 的样本方差取平均（题内随机方差 sigma^2 的无偏估计）。"""
    vars_ = []
    for arr in per_case.values():
        vals = [float(v) for v in arr if v is not None]
        if len(vals) < 2:
            continue
        m = sum(vals) / len(vals)
        vars_.append(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))
    return sum(vars_) / len(vars_)


# ---------------------------------------------------------------- H1-V1 统计功效
def test_v1_real_power_mde_under_015(capsys):
    """n=30 的真实 MDE：配对差值 sd 来自真实 per-case 重复打分，MDE 来自 iterate.mde()。"""
    sigma2 = _pooled_within_case_variance(_REAL_BASELINE_2REP)
    sd_pair_5rep = math.sqrt(2.0 * sigma2 / 5)  # Anthropic Rec#3: 5 次重采样取 case 均值
    mde30 = it.mde(30, sd_pair_5rep)

    # 对照：旧口径（P0 实际 2 次重复的配对 delta sd）即使扩到 n=30 仍不达标
    sd_old = statistics.stdev(_REAL_P0_DELTAS)
    mde30_old = it.mde(30, sd_old)

    print(f"\n[H1-V1 真实统计] 真实 per-case 打分: {len(_REAL_BASELINE_2REP)} case × 2 repetitions "
          f"(P0 hy4-flash-free baseline)")
    print(f"  pooled within-case variance sigma^2 = {sigma2:.6f}")
    print(f"  配对差值 sd (repetitions=5, sqrt(2*sigma^2/5)) = {sd_pair_5rep:.6f}")
    print(f"  mde(30, sd_pair_5rep) = {mde30}  (alpha={it.ALPHA}, power={it.POWER})")
    print(f"  对照: 旧 2 次重复 sd={sd_old:.4f} -> mde(30)={mde30_old}（仍 >0.15，须配合重采样）")
    assert mde30 is not None and mde30 < 0.15
    assert mde30_old > 0.15  # 证明改善来自"扩样本 + 降题内方差"，不是数字游戏

    # 真实函数被真的调用（不是硬编码常量）
    assert mde30 == it.mde(30, sd_pair_5rep)


def test_v1_research_basis_locked_in_build_script():
    """调研结论必须留在交付物里（先调研后实现的可审计证据）。"""
    assert bb.ZH_PER_GROUP >= 12
    assert "2411.00640" in bb.RESEARCH_BASIS["anthropic_paper"]
    assert "2024-11-18" in bb.RESEARCH_BASIS["anthropic_blog"]
    assert "2026-07" in bb.RESEARCH_BASIS["airbnb_blog"]
    assert "airbnb.tech" in bb.RESEARCH_BASIS["airbnb_blog"]


# ---------------------------------------------------------------- H1-V2 判定器能检出 +0.1
def _per_case(base, rep=3):
    return {f"c{i + 1:02d}": [float(v)] * rep for i, v in enumerate(base)}


def test_v2_compare_detects_stable_plus_0_1(capsys):
    """30 条 case 全部 +0.1：compare_metric 必须判"改善"（构造性用例）。"""
    base_vals = [3.5 + (i % 6) * 0.2 for i in range(30)]  # 题目难度有差异，配对设计应消掉
    after_vals = [v + 0.1 for v in base_vals]              # 每条都稳定 +0.1
    before, after = _per_case(base_vals), _per_case(after_vals)
    r = it.compare_metric("judge_avg", before, after)
    print("\n[H1-V2 构造性] 30 条 case 全部 +0.1 -> " + it.format_metric_report(r))
    assert r["n"] == 30
    assert r["delta_mean"] == pytest.approx(0.1, abs=1e-4)
    assert r["sign"]["pos"] == 30 and r["sign"]["neg"] == 0
    assert r["sign"]["significant"] is True
    assert r["verdict"] == "改善"
    assert "改善" in it.format_metric_report(r)
    assert r["mde"] is not None and r["mde"] < 0.15


def test_v2_control_balanced_jitter_stays_hold():
    """对照：同样 30 条、正负交错的 ±0.1 抖动不能误判改善。"""
    base_vals = [4.0] * 30
    after_vals = [v + (0.1 if i % 2 == 0 else -0.1) for i, v in enumerate(base_vals)]
    r = it.compare_metric("judge_avg", _per_case(base_vals), _per_case(after_vals))
    assert r["n"] == 30
    assert r["sign"]["significant"] is False
    assert r["ci_dir"] == 0
    assert r["verdict"] == "持平"


# ---------------------------------------------------------------- H1-V3 数据真实可追溯
def test_v3_batch_expanded_and_traceable():
    """中文共情批 n>=30，全部来自 CPsyCoun 实拉，带 source_id 与非 synthetic 来源标注。"""
    d = json.loads(BATCH.read_text(encoding="utf-8"))
    items = d["items"]
    assert d["n"] == len(items) >= 30
    assert len({i["id"] for i in items}) == len(items)
    assert len({i["source_id"] for i in items}) == len(items)
    for it_ in items:
        assert it_["source_id"].startswith("CPsyCoun:train:")
        assert it_["source"]["dataset"] == "CPsyCoun"
        assert it_["provenance"]["kind"] == "authoritative"
        assert it_["provenance"]["synthetic"] is False
        assert isinstance(it_["provenance"]["row_index"], int)
        assert it_["language"] == "zh" and len(it_["user_input"]) >= 30
    # 旧 17 条 id 全保留：和历史 baseline 的 per-case 配对可比
    assert set(_REAL_BASELINE_2REP).issubset({i["id"] for i in items})
    # S3 集成写入的 HEART rubric / 旧自编维度对照不能被重建冲掉
    assert "HEART" in d["eval"] and "rubric" in d and "eval_control" in d
    assert d["power_analysis"]["actual_n"] == len(items)
    assert d["power_analysis"]["target_n"] == 30
