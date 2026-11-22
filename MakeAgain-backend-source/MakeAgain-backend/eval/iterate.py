#!/usr/bin/env python3
"""评测闭环工具：snapshot（存基线）/ compare（改后对比）/ history（迭代历史）。

把"评测数字 + prompt版本 + git commit"绑在一起，让每次改动有据可查。
判据见 docs/EVAL_ITERATION_LOOP.md；2026-09 方法升级见 docs/EVAL_METHOD_UPGRADE.md。

2026-11-23 方法升级（evalstats 2026-09 + Anthropic playbook 2026-09-28）：
- 新增 noise_gate：增益在噪声带内 -> "持平（增益在噪声内，建议不合入）"；
  噪声带≥最小可行动改进 -> 提示加重复次数/样本量再开跑
- 稳健模式以精确配对符号检验为主、bootstrap CI 仅作对照；N<100 标注 bootstrap 不可靠
- 同基线同批 compare≥2 次 -> 标注过拟合风险（尚未做物理 train/test 拆分）
- --no-robust 逐字回退升级前的 CI+符号检验判定；--no-repetitions 仍逐字节回退最初旧行为

2026-11-22 统计门禁（P0/S1）：
- compare 默认 repetitions=3，同批 case 跑 N 次，保留每条 case 每次分数
- judge_avg 95%CI 用 bootstrap（1000 次重采样，固定种子可复现）
- 基线与当前用同一 case_id 配对，算每对 delta，做精确配对符号检验（sign test），
  不是两次独立均分相减
- 判定从裸 ±0.1 升级为"CI 不重叠 或 配对检验显著"；危机维度 recall 红线不许退化
- 报告 n 与 MDE；n<30 标"功效不足，结论仅供参考"
- --no-repetitions 逐字节回退旧行为：cmd_compare_legacy / run_batches_legacy /
  verdict_for / THRESH 的函数体与改动前完全一致，逐字保留
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))
HISTORY = BACKEND.parent / "outputs" / "benchmarks" / "iteration_history.jsonl"
BASELINE = BACKEND.parent / "outputs" / "benchmarks" / "baseline.json"

# ---- 统计门禁常数（bootstrap 固定种子保证可复现，便于真实命令复跑对比） ----
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20261122
ALPHA = 0.05
POWER = 0.80
LOW_POWER_N = 30  # n<30 标功效不足

# ---- 2026-09 方法升级（evalstats arXiv:2609.35815 / Anthropic playbook 2026-09-28）----
# evalstats：N<100 明确警告 bootstrap CI 不可靠、raw judge 分数统计虚高；
#            小样本判定以精确配对符号检验为主，bootstrap CI 只作对照。
SMALL_SAMPLE_N = 100
NOISE_Z = 1.96  # 配对差值噪声带的 95% 正态近似：先看噪声再开跑
OVERFIT_COMPARES = 2  # 同一基线+同一批 compare 达到此次数即提示过拟合风险
BOOTSTRAP_CAVEAT = (f"⚠ 小样本 bootstrap CI 不可靠（evalstats 2026-09，N<{SMALL_SAMPLE_N}），"
                    "仅作对照；判定以精确配对符号检验为主")


def noise_gate(deltas, delta_mean, min_improve: float | None = None,
               alpha: float = ALPHA, z: float = NOISE_Z) -> dict:
    """噪声门槛（Anthropic playbook 2026-09-28）：增益在噪声内就建议不合入。

    noise_se = sd(配对差值)/sqrt(n)，noise_band = z*noise_se 即纯噪声下均分能移动的范围。
    噪声带 >= 最小可行动改进时，建议先加重复次数/样本量再开跑。
    纯函数：不改全局状态，便于构造性测试。
    """
    ds = [float(d) for d in deltas if d is not None]
    n = len(ds)
    if n < 2 or delta_mean is None:
        return {"noise_se": None, "noise_band": None, "min_improve": min_improve,
                "gain_above_noise": None, "eval_noise_too_high": None,
                "note": "样本不足(n<2)，噪声门槛不可用"}
    sd = math.sqrt(sum((d - delta_mean) ** 2 for d in ds) / (n - 1))
    se = sd / math.sqrt(n)
    band = z * se
    above = delta_mean > band
    too_high = min_improve is not None and band >= min_improve
    note = (("增益在噪声内：" if not above else "增益高于噪声：")
            + f"增益 {delta_mean:+.4f} {'高于' if above else '未高于'}噪声带 {band:.4f}"
            + ("；噪声≥最小可行动改进，建议加重复次数/样本再开跑" if too_high else ""))
    return {"noise_se": round(se, 4), "noise_band": round(band, 4),
            "min_improve": min_improve, "gain_above_noise": above,
            "eval_noise_too_high": too_high, "note": note}


def overfit_warning(prior_compares: int, threshold: int = OVERFIT_COMPARES) -> str | None:
    """同批 case 被反复 compare 时的过拟合风险提示（playbook: train/test 拆分）。

    本工具尚未做 train/test 物理拆分，只能用"同基线同批已 compare 次数"作为风险信号；
    这是标注，不是代替 held-out 验证。
    """
    if prior_compares < threshold:
        return None
    return (f"该批已被用于 {prior_compares} 次调参对比，存在过拟合风险："
            "建议留 held-out 批做最终验证；train 提分而 test 持平的改动应回退")


def prior_compare_count(history: Path, baseline_ts, batch_names) -> int:
    """从迭代历史数出"同一基线 + 同一批"已 compare 过几次（纯读文件）。"""
    if not history.exists():
        return 0
    want = set(batch_names or [])
    cnt = 0
    for line in history.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("event") != "compare" or d.get("vs_baseline") != baseline_ts:
            continue
        if want and want.intersection(d.get("batches") or {}):
            cnt += 1
    return cnt


def git_info() -> dict:
    def run(args):
        try:
            return subprocess.check_output(args, cwd=str(BACKEND.parent), text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"
    return {"commit": run(["git", "rev-parse", "--short", "HEAD"]),
            "branch": run(["git", "branch", "--show-current"])}


def prompt_versions() -> dict:
    try:
        from agent import prompts_registry as pr
        return {pid: pr.get_version(pid) for pid in pr.ids()}
    except Exception:
        return {}


def metrics_from_result(batch_file: Path) -> dict:
    d = json.loads(batch_file.read_text(encoding="utf-8"))
    s = d["summary"]
    return {
        "n": s.get("n"),
        "judge_avg": s.get("judge_avg"),
        "emotion_hit_rate": s.get("emotion_hit_rate"),
        "crisis_detect_rate": s.get("crisis_detect_rate"),
    }


# =====================================================================
# 以下 4 段（run_batches_legacy / THRESH / verdict_for / cmd_compare_legacy）
# 是改动前的原样逐字保留，供 --no-repetitions 回退，输出逐字节一致。
# =====================================================================
def run_batches_legacy(batch_names: list[str]) -> dict:
    """调用 run_benchmarks 的 run_batch，返回各批指标。"""
    import os
    # 确保 key/env（复用 run_benchmarks 的 setup）
    import run_benchmarks as rb
    conf = rb.load_key()
    rb.setup(conf)
    from gateway.client import LLMClient
    client = LLMClient()
    out = {}
    for name in batch_names:
        p = rb.BATCHDIR / f"{name}.json"
        if not p.exists():
            print(f"  ! batch not found: {name}")
            continue
        # 直接调 run_batch（它会写 outputs/benchmarks/<batch>.json）
        rb.run_batch(name, use_raw=False, client=client)
        out[name] = metrics_from_result(rb.OUTDIR / f"{name}.json")
    return out


# 判定阈值（docs/EVAL_ITERATION_LOOP.md）
THRESH = {
    "judge_avg": {"min_drop": 0.1, "improve": 0.1},
    "emotion_hit_rate": {"min_drop": 0.02, "improve": 0.02},
    "crisis_detect_rate": {"min_drop": 0.0, "improve": 0.01},  # 安全红线：不许退化
}


def verdict_for(key: str, before, after) -> str:
    if before is None or after is None:
        return "N/A（缺基线或当前值）"
    t = THRESH.get(key, {"min_drop": 0.05, "improve": 0.05})
    delta = after - before
    if delta < -t["min_drop"]:
        return "退化（建议不合入）"
    if delta > t["improve"]:
        return "改善"
    return "持平"


def cmd_compare_legacy(args):
    if not BASELINE.exists():
        print("[compare] 无基线，先跑 snapshot")
        return
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    names = [args.batch] if args.batch else list(base["batches"].keys())
    print(f"[compare] running {len(names)} batches ...")
    now = run_batches_legacy(names)
    print(f"\n{'='*70}\n对照基线: {base['ts']} commit={base['git']['commit']}\n{'='*70}")
    overall = []
    for name in names:
        b = base["batches"].get(name)
        n = now.get(name)
        if not b or not n:
            print(f"  {name}: 缺数据（基线或当前）")
            continue
        print(f"\n[{name}]")
        for key in ("judge_avg", "emotion_hit_rate", "crisis_detect_rate"):
            bv, nv = b.get(key), n.get(key)
            v = verdict_for(key, bv, nv)
            delta = (nv - bv) if (bv is not None and nv is not None) else None
            ds = f"{delta:+.3f}" if delta is not None else "N/A"
            print(f"  {key:20} {str(bv):>6} -> {str(nv):>6}  ({ds})  {v}")
            overall.append(v)
    # 总判定
    bad = any("退化" in v for v in overall)
    good = any(v == "改善" for v in overall)
    final = "❌ 有核心指标退化，建议不合入/回滚" if bad else ("✅ 改善，可合入" if good else "➖ 持平，可合入")
    print(f"\n{'='*70}\n总判定: {final}\n{'='*70}")
    snap = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "git": git_info(), "prompt_versions": prompt_versions(),
        "vs_baseline": base["ts"], "batches": now, "verdict": final,
        "note": args.note or "",
    }
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "compare", **snap}, ensure_ascii=False) + "\n")


# =====================================================================
# 统计门禁：纯函数（不读文件/不调模型，便于构造性测试）
# =====================================================================
def _mean(values) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _percentile(sorted_vals, q: float) -> float | None:
    """线性插值分位数，sorted_vals 须已升序。"""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo)


def bootstrap_ci(values, n_resamples: int = BOOTSTRAP_RESAMPLES, ci: float = 0.95,
                 seed: int = BOOTSTRAP_SEED):
    """对样本均值做 bootstrap 置信区间。values 为逐 case 的均值样本。

    重采样单位是 case（不是单次打分），因此得到的是 judge_avg 的 95%CI。
    固定 seed 保证同一输入可复现（真实命令可复跑对比）。
    """
    vals = [float(v) for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return (None, None)
    if n == 1:
        return (round(vals[0], 4), round(vals[0], 4))
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    alpha = 1.0 - ci
    lo = _percentile(means, alpha / 2)
    hi = _percentile(means, 1 - alpha / 2)
    return (round(lo, 4), round(hi, 4))


def sign_test(deltas, alpha: float = ALPHA) -> dict:
    """精确双侧配对符号检验（二项分布 p=0.5），零差值按惯例剔除。"""
    ds = [float(d) for d in deltas if d is not None]
    pos = sum(1 for d in ds if d > 1e-12)
    neg = sum(1 for d in ds if d < -1e-12)
    zero = len(ds) - pos - neg
    n = pos + neg
    if n == 0:
        p = 1.0
    else:
        k = min(pos, neg)
        tail = sum(math.comb(n, i) for i in range(k + 1))
        p = min(1.0, 2.0 * tail / (2 ** n))
    return {"n_pairs": len(ds), "pos": pos, "neg": neg, "zero": zero,
            "p_value": round(p, 6), "significant": p < alpha}


def mde(n: int, sd, alpha: float = ALPHA, power: float = POWER):
    """最小可检测差异 MDE=(z(1-a/2)+z(power))*sd/sqrt(n)，配对设计用差值 sd。"""
    if not n or n < 2 or sd is None:
        return None
    nd = NormalDist()
    z = nd.inv_cdf(1 - alpha / 2) + nd.inv_cdf(power)
    return round(z * sd / math.sqrt(n), 4)


def ci_direction(ci_b, ci_a) -> int:
    """+1=当前CI整体高于基线；-1=整体低于；0=重叠。"""
    if not ci_b or not ci_a or ci_b[0] is None or ci_a[0] is None:
        return 0
    if ci_a[0] > ci_b[1]:
        return 1
    if ci_a[1] < ci_b[0]:
        return -1
    return 0


def compare_metric(key, before, after, *, crisis_line: bool = False,
                   alpha: float = ALPHA, n_resamples: int = BOOTSTRAP_RESAMPLES,
                   seed: int = BOOTSTRAP_SEED, robust_gate: bool = True) -> dict:
    """配对比较单个指标。before/after 为 {case_id: [每次分数...]} 或 None。

    纯函数：只读入参，不改全局状态，便于构造"抖动/稳定提升/危机退化"用例。
    robust_gate=True（默认）启用 2026 稳健判定：符号检验为主、CI 仅参考，并加噪声门槛；
    robust_gate=False 逐字回退升级前的 CI+符号检验一致判定（供 A/B 对照与开关回退）。
    """
    empty = {"key": key, "verdict": "N/A（缺配对数据）", "reason": "缺基线或当前的 per-case 明细",
             "deltas": None, "sign": None, "n": 0, "mde": None, "low_power": True,
             "ci_b": None, "ci_a": None, "ci_dir": 0, "sign_dir": 0,
             "b_mean": None, "a_mean": None, "delta_mean": None, "crisis_line": crisis_line,
             "robust_gate": robust_gate, "noise": None, "bootstrap_caveat": None}
    if not before or not after:
        return empty
    bm = {cid: _mean(vals) for cid, vals in before.items() if vals}
    am = {cid: _mean(vals) for cid, vals in after.items() if vals}
    common = sorted(set(bm) & set(am))
    n = len(common)
    if n == 0:
        return empty

    b_sample = [bm[c] for c in common]
    a_sample = [am[c] for c in common]
    deltas = [a - b for a, b in zip(a_sample, b_sample)]
    b_mean, a_mean = _mean(b_sample), _mean(a_sample)
    delta_mean = a_mean - b_mean

    ci_b = bootstrap_ci(b_sample, n_resamples, seed=seed)
    ci_a = bootstrap_ci(a_sample, n_resamples, seed=seed)
    ci_dir = ci_direction(ci_b, ci_a)
    sg = sign_test(deltas, alpha=alpha)
    sign_dir = (1 if (sg["significant"] and sg["pos"] > sg["neg"])
                else (-1 if (sg["significant"] and sg["neg"] > sg["pos"]) else 0))

    # MDE 用配对差值的样本标准差
    sd = math.sqrt(sum((d - delta_mean) ** 2 for d in deltas) / (n - 1)) if n >= 2 else None
    mde_val = mde(n, sd, alpha=alpha)
    low_power = n < LOW_POWER_N

    # 噪声门槛只在稳健模式生效；关闭时下面 CI+符号检验判定逐字保持升级前逻辑
    noise = (noise_gate(deltas, delta_mean, THRESH.get(key, {}).get("improve"))
             if robust_gate else None)
    caveat = BOOTSTRAP_CAVEAT if (robust_gate and n < SMALL_SAMPLE_N) else None

    dirs = [d for d in (ci_dir, sign_dir) if d != 0]
    if robust_gate:
        # evalstats 2026-09：小样本 bootstrap CI 不可靠 -> 以精确配对符号检验为主、CI 仅参考
        if sign_dir != 0:
            evidence = sign_dir
            ev_desc = (f"配对符号检验显著（{'正向' if sign_dir > 0 else '负向'}），"
                       f"CI方向={ci_dir:+d} 仅作参考")
        else:
            evidence = 0
            ev_desc = ("CI 提示方向但符号检验不显著（抖动内）" if ci_dir
                       else "CI重叠且配对检验不显著（抖动内）")
    else:
        if dirs and all(d > 0 for d in dirs):
            evidence, ev_desc = 1, "CI不重叠/配对检验显著（正向）"
        elif dirs and all(d < 0 for d in dirs):
            evidence, ev_desc = -1, "CI不重叠/配对检验显著（负向）"
        else:
            evidence, ev_desc = 0, "CI重叠且配对检验不显著（抖动内）"

    if crisis_line:
        # 安全红线：recall 只要退化就不合入，不看显著性（稳健开关不改变红线）
        if delta_mean < -1e-9:
            verdict = "退化（安全红线，不合入）"
            reason = f"危机recall下降 {delta_mean:+.3f}，红线不许退化；{ev_desc}"
        elif evidence > 0:
            verdict, reason = "改善", ev_desc
        else:
            verdict, reason = "持平", ev_desc
    elif robust_gate and evidence > 0 and delta_mean > 0 and not noise["gain_above_noise"]:
        # Anthropic playbook 2026-09-28：增益在噪声内就建议不合入
        verdict = "持平（增益在噪声内，建议不合入）"
        reason = f"{noise['note']}；{ev_desc}"
    else:
        if evidence > 0 and delta_mean > 0:
            verdict, reason = "改善", ev_desc
        elif evidence < 0 and delta_mean < 0:
            verdict, reason = "退化（建议不合入）", ev_desc
        else:
            verdict, reason = "持平", ev_desc

    return {"key": key, "b_mean": round(b_mean, 4), "a_mean": round(a_mean, 4),
            "delta_mean": round(delta_mean, 4), "ci_b": ci_b, "ci_a": ci_a,
            "ci_dir": ci_dir, "deltas": [round(d, 4) for d in deltas],
            "sign": sg, "sign_dir": sign_dir, "n": n, "mde": mde_val,
            "low_power": low_power, "verdict": verdict, "reason": reason,
            "crisis_line": crisis_line, "robust_gate": robust_gate,
            "noise": noise, "bootstrap_caveat": caveat}


def format_metric_report(r: dict) -> str:
    """把 compare_metric 的结果渲染成可读的 stdout 文本。"""
    bv = "N/A" if r["b_mean"] is None else f"{r['b_mean']:.3f}"
    av = "N/A" if r["a_mean"] is None else f"{r['a_mean']:.3f}"
    dv = "N/A" if r["delta_mean"] is None else f"{r['delta_mean']:+.3f}"
    lines = [f"  {r['key']:20} {bv:>7} -> {av:>7}  ({dv})  {r['verdict']}"]
    if r.get("ci_b") and r.get("ci_a") and r["ci_b"][0] is not None:
        lines.append(f"    基线95%CI=[{r['ci_b'][0]:.3f}, {r['ci_b'][1]:.3f}]  "
                     f"当前95%CI=[{r['ci_a'][0]:.3f}, {r['ci_a'][1]:.3f}]  (bootstrap×{BOOTSTRAP_RESAMPLES})")
    if r.get("sign"):
        s = r["sign"]
        lines.append(f"    配对符号检验: 有效对={s['pos'] + s['neg']} "
                     f"pos={s['pos']} neg={s['neg']} zero={s['zero']} "
                     f"p={s['p_value']:.4f}（双侧精确）{'显著' if s['significant'] else '不显著'}")
    if r.get("deltas") is not None:
        dl = ", ".join(f"{d:+.2f}" for d in r["deltas"])
        lines.append(f"    delta列表(n={r['n']}): [{dl}]")
    mde_s = "N/A" if r.get("mde") is None else f"{r['mde']:.4f}"
    power_tag = "  [⚠ n<30 功效不足，结论仅供参考]" if r.get("low_power") else "  [功效充足]"
    lines.append(f"    功效: n={r['n']} MDE(α={ALPHA},power={POWER})={mde_s}{power_tag}")
    if r.get("bootstrap_caveat"):
        lines.append(f"    {r['bootstrap_caveat']}")
    nz = r.get("noise")
    if nz and nz.get("noise_band") is not None:
        mi = "N/A" if nz.get("min_improve") is None else f"{nz['min_improve']:.4f}"
        lines.append(f"    噪声门槛: 噪声带={nz['noise_band']:.4f} SE={nz['noise_se']:.4f} "
                     f"最小可行动改进={mi} 增益高于噪声={nz['gain_above_noise']}")
        if nz.get("eval_noise_too_high"):
            lines.append("    ⚠ 噪声≥最小可行动改进，建议加重复次数或样本量再开跑（playbook）")
    lines.append(f"    依据: {r['reason']}")
    return "\n".join(lines)


def overall_verdict(verdicts) -> str:
    bad = any("退化" in v for v in verdicts)
    good = any(v == "改善" for v in verdicts)
    if bad:
        return "❌ 有核心指标退化，建议不合入/回滚"
    # 2026 方法升级：单项已判"增益在噪声内，建议不合入"，总判定不能再说"可合入"
    if any("增益在噪声内" in v for v in verdicts):
        return "⚠ 增益在噪声内，建议不合入（先加重复次数/样本量再开跑）"
    if good:
        return "✅ 改善，可合入"
    return "➖ 持平，可合入"


# =====================================================================
# 统计门禁：批处理 / 命令
# =====================================================================
def _score_from_row(row: dict) -> dict:
    """从一条 benchmark 结果里抽出各指标的本次分数。"""
    out = {}
    j = row.get("judge")
    if isinstance(j, dict) and j.get("score") is not None:
        out["judge_avg"] = float(j["score"])
    if "hit" in row:
        out["emotion_hit_rate"] = 1.0 if row["hit"] else 0.0
    if "crisis" in row:
        out["crisis_detect_rate"] = 1.0 if row["crisis"] else 0.0
    return out


def _aggregate(per_case: dict, key: str):
    vals = [v for arr in per_case.get(key, {}).values() for v in arr if v is not None]
    return round(_mean(vals), 3) if vals else None


def run_batches(batch_names: list[str], repetitions: int = 3) -> dict:
    """同批 case 跑 N 次，保留每条 case 每次的分数（统计门禁用）。"""
    import run_benchmarks as rb
    conf = rb.load_key()
    rb.setup(conf)
    from gateway.client import LLMClient
    client = LLMClient()
    out = {}
    for name in batch_names:
        p = rb.BATCHDIR / f"{name}.json"
        if not p.exists():
            print(f"  ! batch not found: {name}")
            continue
        per_case = {"judge_avg": {}, "emotion_hit_rate": {}, "crisis_detect_rate": {}}
        for _rep in range(repetitions):
            rb.run_batch(name, use_raw=False, client=client)  # 每次覆盖写 <batch>.json
            result = json.loads((rb.OUTDIR / f"{name}.json").read_text(encoding="utf-8"))
            for row in result["results"]:
                cid = row.get("id")
                if cid is None:
                    continue
                for key, val in _score_from_row(row).items():
                    per_case[key].setdefault(cid, []).append(val)
        metrics = metrics_from_result(rb.OUTDIR / f"{name}.json")
        metrics["repetitions"] = repetitions
        metrics["per_case"] = per_case
        # 用全部重复的分数重算聚合值，避免只取最后一次的抖动
        for key in ("judge_avg", "emotion_hit_rate", "crisis_detect_rate"):
            agg = _aggregate(per_case, key)
            if agg is not None:
                metrics[key] = agg
        out[name] = metrics
    return out


def all_batch_names() -> list[str]:
    return sorted(p.stem for p in (HERE / "batches").glob("*.json") if not p.name.startswith("_"))


def cmd_snapshot(args):
    no_rep = getattr(args, "no_repetitions", False)
    reps = 1 if no_rep else max(1, args.repetitions)
    names = [args.batch] if args.batch else all_batch_names()
    print(f"[snapshot] running {len(names)} batches ...")
    metrics = run_batches_legacy(names) if no_rep else run_batches(names, repetitions=reps)
    snap = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "git": git_info(), "prompt_versions": prompt_versions(),
        "batches": metrics, "note": args.note or "",
    }
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "snapshot", **snap}, ensure_ascii=False) + "\n")
    print(f"[snapshot] saved -> {BASELINE}")
    for k, v in metrics.items():
        print(f"  {k}: judge={v['judge_avg']} emo={v['emotion_hit_rate']} crisis={v['crisis_detect_rate']}")
    if not no_rep:
        print(f"[snapshot] repetitions={reps}（保留逐 case 明细，供配对统计门禁）")


def cmd_compare_statistical(args):
    reps = max(1, args.repetitions)
    robust = not getattr(args, "no_robust", False)
    if not BASELINE.exists():
        print("[compare] 无基线，先跑 snapshot")
        return
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    names = [args.batch] if args.batch else list(base["batches"].keys())
    print(f"[compare] running {len(names)} batches × {reps} repetitions "
          f"(bootstrap×{BOOTSTRAP_RESAMPLES}, alpha={ALPHA}, "
          f"robust_gate={robust}) ...")
    now = run_batches(names, repetitions=reps)
    print(f"\n{'='*70}\n对照基线: {base['ts']} commit={base['git']['commit']}\n{'='*70}")
    verdicts = []
    paired_ok = False
    for name in names:
        b = base["batches"].get(name)
        n = now.get(name)
        if not b or not n:
            print(f"  {name}: 缺数据（基线或当前）")
            continue
        print(f"\n[{name}]")
        # V1: 当前批每条 case 的 N 次分数 + 均分
        cur_pc = (n.get("per_case") or {}).get("judge_avg") or {}
        if cur_pc:
            print("  --- judge_avg 每条 case 的分数（按重复次数） ---")
            for cid in sorted(cur_pc):
                arr = cur_pc[cid]
                print(f"    {cid}: {arr} mean={_mean(arr):.3f}")
        else:
            print("  [提示] 当前运行无 per-case 明细")
        for key, crisis in (("judge_avg", False), ("emotion_hit_rate", False),
                            ("crisis_detect_rate", True)):
            bpc = (b.get("per_case") or {}).get(key)
            apc = (n.get("per_case") or {}).get(key)
            r = compare_metric(key, bpc, apc, crisis_line=crisis, robust_gate=robust)
            print(format_metric_report(r))
            if r["verdict"] != "N/A（缺配对数据）":
                verdicts.append(r["verdict"])
                paired_ok = True
    if paired_ok:
        final = overall_verdict(verdicts)
    else:
        final = "⚠ 基线无 per-case 明细，统计门禁未生效（请用新版 snapshot 重建基线）"

    # S2 集成：judge 校准前置门（judge 未校准则结论不可信，不给合入建议）
    try:
        from pathlib import Path as _Path
        _eval_dir = str(_Path(__file__).resolve().parent)
        if _eval_dir not in sys.path:
            sys.path.insert(0, _eval_dir)
        from judge_calibration import judge_consistency_gate
        _gate = judge_consistency_gate()
        if not _gate.get("merge_advice_allowed"):
            print(f"[gate] {_gate.get('note', '')}")
            print(f"[gate] Kappa={_gate.get('kappa')} 一致率={_gate.get('agreement')}（门槛 0.6）")
            final = "❌ judge 未校准，结论不可信（不给出合入建议）"
        else:
            print(f"[gate] judge 已校准：Kappa={_gate.get('kappa')} 一致率={_gate.get('agreement')}（≥0.6）")
    except Exception as _exc:
        print(f"[gate] judge 校准检查失败：{str(_exc)[:100]}（保守判不可信）")
        final = "❌ judge 未校准，结论不可信（不给出合入建议）"

    # 2026 方法升级：同批反复调参 -> 过拟合风险标注（--no-robust 可回退）
    # 放在 judge 门之后，保证风险提示不被门的最终判定覆盖
    prior = prior_compare_count(HISTORY, base.get("ts"), names) if robust else 0
    overfit = overfit_warning(prior) if robust else None
    if overfit:
        print(f"[overfit] {overfit}")
        final += " ⚠ 同批多次调参，存在过拟合风险（结论仅参考）"

    print(f"\n{'='*70}\n总判定: {final}\n{'='*70}")
    snap = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "git": git_info(), "prompt_versions": prompt_versions(),
        "vs_baseline": base["ts"], "batches": now, "verdict": final,
        "statistical": {"repetitions": reps, "bootstrap": BOOTSTRAP_RESAMPLES,
                        "alpha": ALPHA, "power": POWER, "robust_gate": robust,
                        "small_sample_n": SMALL_SAMPLE_N,
                        "prior_compares_same_batch": prior,
                        "overfit_warning": overfit,
                        "methods": ["sign_test_primary", "bootstrap_ci_reference",
                                    "noise_gate", "overfit_marker"] if robust else []},
        "note": args.note or "",
    }
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "compare", **snap}, ensure_ascii=False) + "\n")


def cmd_compare(args):
    # --no-repetitions：走逐字保留的 legacy 路径，输出与改动前逐字节一致
    if getattr(args, "no_repetitions", False):
        return cmd_compare_legacy(args)
    return cmd_compare_statistical(args)


def cmd_history(args):
    if not HISTORY.exists():
        print("[history] 暂无迭代历史")
        return
    print(f"{'时间':<26} {'事件':<10} {'commit':<10} 判定")
    print("-" * 80)
    for line in HISTORY.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
            ts = d["ts"][:19]
            print(f"{ts:<26} {d['event']:<10} {d['git']['commit']:<10} {d.get('verdict', d.get('note',''))}")
        except Exception:
            continue


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("snapshot"); s.add_argument("--batch"); s.add_argument("--note")
    s.add_argument("--repetitions", type=int, default=3, help="同批 case 重复运行次数（默认3）")
    s.add_argument("--no-repetitions", action="store_true", help="关闭统计门禁，回退旧 snapshot 行为")
    c = sub.add_parser("compare"); c.add_argument("--batch"); c.add_argument("--note")
    c.add_argument("--repetitions", type=int, default=3, help="同批 case 重复运行次数（默认3）")
    c.add_argument("--no-repetitions", action="store_true", help="关闭统计门禁，逐字节回退旧 compare 行为")
    c.add_argument("--no-robust", action="store_true",
                   help="关闭2026稳健判定（符号检验优先/噪声门槛/过拟合提示），回退升级前统计门禁")
    h = sub.add_parser("history")
    args = ap.parse_args()
    if getattr(args, "repetitions", 1) < 1:
        ap.error("--repetitions 必须 >= 1")
    if args.cmd == "snapshot":
        cmd_snapshot(args)
    elif args.cmd == "compare":
        cmd_compare(args)
    elif args.cmd == "history":
        cmd_history(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
