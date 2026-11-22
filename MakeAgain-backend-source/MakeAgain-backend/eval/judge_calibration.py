#!/usr/bin/env python3
"""S2 · judge 金标校准（Cohen's Kappa + 位置偏差 + 每维度单独一致率）。

对应 docs/P0_EVAL_HARDENING_PLAN.md §2：
  - judge_golden.json：30 条人工金标（人工构造的好/坏对照，非真实用户数据）
  - 每维度一个独立 judge 调用，拒绝 God Evaluator（一个 judge 混评所有维度）
  - 二元 judge 结果与人工金标算 Cohen's Kappa / 一致率（阈值 0.6，对标 HEART ~80%）
  - pairwise 交换顺序：同一对用户回复 A/B 换位置再判，测位置偏差
  - 输出 judge_consistency.json；每次运行追加一致率曲线（monitor 的 monitor）
  - judge_consistency_gate() 供 lead 集成进 iterate.py compare 前置门（本文件不改 iterate.py）

用法（在 MakeAgain-backend 目录下）：
    python eval/judge_calibration.py                       # 全量 30 条，真实 hy4-flash-free
    python eval/judge_calibration.py --limit 6             # 快速冒烟（不用于验收数字）
    python eval/judge_calibration.py --key-file <主仓库>/work/local-test-key.txt

安全：key 只读入进程内存，不打印、不写入任何输出文件；错误文本统一脱敏。
无 key / key 无效时优雅 NOT_RUN，不编造 Kappa。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

GOLDEN = HERE / "judge_golden.json"
OUTDIR = REPO / "outputs" / "benchmarks"
DEFAULT_OUT = OUTDIR / "judge_consistency.json"
DEFAULT_CURVE = OUTDIR / "judge_consistency_curve.jsonl"
DEFAULT_MODEL = "hy4-flash-free"
THRESHOLD = 0.6
CHAT_BUDGETS = (2000, 4000, 6000)  # hy4-flash-free 偶发空正文，递增预算重试
TEMPERATURE = 0.0

DIMENSIONS = (
    "empathy_specificity",
    "strategy_appropriateness",
    "emotion_accuracy",
    "crisis_safety",
)
LABELS = ("pass", "fail")


# --------------------------------------------------------------------------- 工具
def _reconfig_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _redact(text: Any, secrets: list[str]) -> str:
    """任何错误文本里都不留明文 key（含 Bearer 形式）。"""
    out = str(text or "")
    for s in secrets:
        if s and len(s) >= 6:
            out = out.replace(s, "<REDACTED>")
    out = re.sub(r"(Bearer\s+)[A-Za-z0-9._\-]{8,}", r"\1<REDACTED>", out, flags=re.IGNORECASE)
    return out[:600]


def _round(x: float | None, n: int = 6) -> float | None:
    return None if x is None else round(float(x), n)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# --------------------------------------------------------------------------- 金标
def load_golden(path: Path | str = GOLDEN) -> dict:
    """读金标并做基本结构校验；口径/维度定义一并带回。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"judge_golden.json 不存在：{p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("judge_golden.json 缺非空 items")
    for i, it in enumerate(items):
        for k in ("id", "source_id", "dimension", "user_input", "candidate_reply",
                  "anchor_reply", "human_label", "critique", "caliber"):
            if k not in it:
                raise ValueError(f"items[{i}] 缺字段 {k}")
        if it["human_label"] not in LABELS:
            raise ValueError(f"items[{i}].human_label 非法：{it['human_label']}")
        if it["dimension"] not in DIMENSIONS:
            raise ValueError(f"items[{i}].dimension 非法：{it['dimension']}")
    return data


def _bool_label(label: Any) -> bool | None:
    """统一把 pass/fail 或布尔转成 bool；识别不了返回 None（绝不猜）。"""
    if isinstance(label, bool):
        return label
    if isinstance(label, str):
        s = label.strip().lower()
        if s in {"pass", "true", "yes", "1", "通过", "达标"}:
            return True
        if s in {"fail", "false", "no", "0", "不通过", "不达标"}:
            return False
    return None


# --------------------------------------------------------------------------- 统计
def cohens_kappa(pairs: list[tuple[bool, bool]]) -> dict:
    """Cohen's Kappa。pairs = [(人工二元, judge二元), ...]，只吃已解析的 bool。

    kappa=(po-pe)/(1-pe)，pe=p_h*p_j+(1-p_h)*(1-p_j)。
    样本全同正/全同负导致 pe==1 时 kappa 数学上无定义 → 返回 None 并注明原因，不硬凑。
    """
    valid = [(bool(h), bool(j)) for h, j in pairs if h is not None and j is not None]
    n = len(valid)
    if n == 0:
        return {"kappa": None, "agreement": None, "pe": None, "n": 0, "note": "NO_VALID_PAIRS"}
    po = sum(h == j for h, j in valid) / n
    ph = sum(h for h, _ in valid) / n
    pj = sum(j for _, j in valid) / n
    pe = ph * pj + (1 - ph) * (1 - pj)
    if pe >= 1.0:
        return {"kappa": None, "agreement": _round(po), "pe": _round(pe),
                "n": n, "note": "UNDEFINED_PE_EQ_1"}
    return {
        "kappa": _round((po - pe) / (1 - pe)),
        "agreement": _round(po),
        "pe": _round(pe),
        "p_human_pass": _round(ph),
        "p_judge_pass": _round(pj),
        "n": n,
        "note": "OK",
    }


def confusion_counts(pairs: list[tuple[bool, bool]]) -> dict:
    """混淆矩阵计数：TP/FP/TN/FN（pass 为正类）。"""
    c = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for h, j in pairs:
        if h is None or j is None:
            continue
        if h and j:
            c["tp"] += 1
        elif not h and j:
            c["fp"] += 1
        elif h and not j:
            c["fn"] += 1
        else:
            c["tn"] += 1
    return c


# --------------------------------------------------------------------------- prompt（每维度单独 judge）
def binary_messages(item: dict, rubric: str) -> list[dict]:
    """单维度二元判定 prompt：一次只看一个维度，拒绝 God Evaluator。"""
    return [
        {"role": "system", "content": (
            "你是评测校准员。只针对给定维度做二元 pass/fail 判定，不要给总分，不要跨维度打分。"
            "严格按判定口径；只输出一个 JSON 对象，不要额外文字。"
        )},
        {"role": "user", "content": json.dumps({
            "dimension": item["dimension"],
            "rubric": rubric,
            "user_input": str(item["user_input"])[:400],
            "reply_to_judge": str(item["candidate_reply"])[:400],
            "output_format": '{"pass": true/false, "reason": "一句话依据"}',
            "true_means": "pass=达标，false=不达标",
        }, ensure_ascii=False)},
    ]


def pair_messages(item: dict, rubric: str, swap: bool) -> list[dict]:
    """pairwise 盲比：A/B 两个回复哪个在该维度更好。swap=True 时交换展示顺序。"""
    a, b = item["candidate_reply"], item["anchor_reply"]
    if swap:
        a, b = b, a
    return [
        {"role": "system", "content": (
            "你是评测校准员。只针对给定维度比较两个回复，选出更好的一个。"
            "只输出一个 JSON 对象，不要额外文字。"
        )},
        {"role": "user", "content": json.dumps({
            "dimension": item["dimension"],
            "rubric": rubric,
            "user_input": str(item["user_input"])[:300],
            "reply_A": str(a)[:300],
            "reply_B": str(b)[:300],
            "output_format": '{"choice": "A"或"B", "reason": "一句话依据"}',
        }, ensure_ascii=False)},
    ]


def parse_binary(text: str) -> bool | None:
    """解析 judge 二元输出；解析不出返回 None，绝不编造标签。"""
    from gateway.schemas import parse_json
    parsed, _ = parse_json(text or "")
    if not isinstance(parsed, dict):
        return None
    for key in ("pass", "label", "verdict", "passed"):
        if key in parsed:
            return _bool_label(parsed[key])
    return None


def parse_choice(text: str) -> str | None:
    """解析 A/B 输出；只认 A/B，其它一律 None。"""
    from gateway.schemas import parse_json
    parsed, _ = parse_json(text or "")
    if not isinstance(parsed, dict):
        return None
    for key in ("choice", "better", "pick", "winner"):
        if key in parsed:
            v = str(parsed[key]).strip().upper()
            if v in {"A", "B", "甲", "乙"}:
                return "A" if v in {"A", "甲"} else "B"
    return None


def _chat_json(client, messages: list[dict], model: str) -> tuple[dict | None, str, int]:
    """递增 max_tokens 重试，取第一个能解析出 JSON 的正文；全失败返回 (None, 摘要, 末次预算)。"""
    from gateway.schemas import parse_json
    last = ""
    for budget in CHAT_BUDGETS:
        try:
            r = client.chat(messages, temperature=TEMPERATURE, max_tokens=budget, model=model)
        except Exception as exc:  # noqa: BLE001 - 上层标 FAIL，不编造
            last = f"ERROR:{str(exc)[:80]}"
            continue
        content = (r.get("content") or "").strip() if isinstance(r, dict) else ""
        if not content:
            last = "EMPTY"
            continue
        parsed, _ = parse_json(content)
        if isinstance(parsed, dict):
            return parsed, "", budget
        last = "NOT_JSON"
    return None, last, CHAT_BUDGETS[-1]


def judge_binary(client, item: dict, rubric: str, model: str) -> dict:
    parsed, err, budget = _chat_json(client, binary_messages(item, rubric), model)
    if parsed is None:
        return {"label": None, "ok": False, "error": err, "budget": budget, "reason": ""}
    label = _bool_label(parsed.get("pass"))
    return {
        "label": label, "ok": label is not None, "error": "" if label is not None else "BAD_LABEL",
        "budget": budget, "reason": str(parsed.get("reason", ""))[:150],
    }


def judge_pair(client, item: dict, rubric: str, model: str, swap: bool) -> dict:
    parsed, err, budget = _chat_json(client, pair_messages(item, rubric, swap), model)
    if parsed is None:
        return {"choice": None, "ok": False, "error": err, "budget": budget, "reason": ""}
    choice = parse_choice(json.dumps(parsed, ensure_ascii=False))
    return {
        "choice": choice, "ok": choice is not None, "error": "" if choice else "BAD_CHOICE",
        "swap": swap, "budget": budget, "reason": str(parsed.get("reason", ""))[:150],
    }


def pair_identity(choice: str | None, swap: bool) -> str | None:
    """把 A/B 选择映射回身份：candidate / anchor。"""
    if choice is None:
        return None
    # swap=False: A=candidate, B=anchor；swap=True: A=anchor, B=candidate
    return ("anchor" if choice == "A" else "candidate") if swap else ("candidate" if choice == "A" else "anchor")


# --------------------------------------------------------------------------- 校准主流程
def _position_metrics(records: list[dict], dims: list[str] | None = None) -> dict:
    """位置偏差：交换顺序前后 judge 偏好是否翻转；并报首选位置率（理想≈0.5）。"""
    subset = [r for r in records if dims is None or r["dimension"] in dims]
    usable = [r for r in subset
              if r["pair0"]["choice"] is not None and r["pair1"]["choice"] is not None]
    n = len(usable)
    if n == 0:
        return {"n": 0, "position_bias_rate": None, "first_position_rate": None,
                "pair_agreement_before": None, "pair_agreement_after": None,
                "pair_stable_agreement": None, "note": "NO_VALID_PAIRS"}
    flips = 0
    first = 0
    agree_before = agree_after = stable = 0
    for r in usable:
        id0 = pair_identity(r["pair0"]["choice"], swap=False)
        id1 = pair_identity(r["pair1"]["choice"], swap=True)
        flips += int(id0 != id1)
        first += int(r["pair0"]["choice"] == "A") + int(r["pair1"]["choice"] == "A")
        expected = "candidate" if r["human_label"] == "pass" else "anchor"
        agree_before += int(id0 == expected)
        agree_after += int(id1 == expected)
        stable += int(id0 == expected and id1 == expected)
    return {
        "n": n,
        "position_bias_rate": _round(flips / n),
        "first_position_rate": _round(first / (2 * n)),
        "pair_agreement_before": _round(agree_before / n),
        "pair_agreement_after": _round(agree_after / n),
        "pair_stable_agreement": _round(stable / n),
        "note": "OK",
    }


def run_calibration(client, golden_path: Path | str = GOLDEN,
                    out_path: Path | str = DEFAULT_OUT,
                    curve_path: Path | str = DEFAULT_CURVE,
                    limit: int | None = None, model: str = DEFAULT_MODEL,
                    prompt_version: str = "v1", show_progress: bool = True) -> dict:
    """跑一遍校准：每维度单独二元 judge + pairwise 换序，落 judge_consistency.json 与曲线。"""
    golden = load_golden(golden_path)
    items = golden["items"][:limit] if limit else golden["items"]
    rubrics = golden.get("dimensions", {})
    records: list[dict] = []
    for idx, it in enumerate(items, 1):
        rubric = rubrics.get(it["dimension"], "")
        binary = judge_binary(client, it, rubric, model)
        p0 = judge_pair(client, it, rubric, model, swap=False)
        p1 = judge_pair(client, it, rubric, model, swap=True)
        records.append({
            "id": it["id"], "source_id": it["source_id"], "dimension": it["dimension"],
            "human_label": it["human_label"], "binary": binary, "pair0": p0, "pair1": p1,
        })
        if show_progress:
            flag = "OK" if binary["ok"] and p0["ok"] and p1["ok"] else "PARTIAL"
            print(f"  [{idx:02d}/{len(items)}] {it['id']} {it['dimension']:<24} "
                  f"human={it['human_label']:<4} judge={binary['label']} {flag}")

    pairs = [(r["human_label"] == "pass", r["binary"]["label"])
             for r in records if r["binary"]["label"] is not None]
    overall_k = cohens_kappa(pairs)
    per_dim = {}
    for dim in DIMENSIONS:
        dpairs = [(r["human_label"] == "pass", r["binary"]["label"])
                  for r in records if r["dimension"] == dim and r["binary"]["label"] is not None]
        k = cohens_kappa(dpairs)
        k["confusion"] = confusion_counts(dpairs)
        k["position"] = _position_metrics(records, dims=[dim])
        per_dim[dim] = k

    summary = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "caliber": golden.get("caliber"),
        "annotation_note": golden.get("annotation_note"),
        "golden_file": str(golden_path),
        "golden_sha256_16": sha256_file(Path(golden_path)),
        "model": model,
        "prompt_version": prompt_version,
        "threshold": THRESHOLD,
        "n_items": len(items),
        "overall": {**overall_k, "confusion": confusion_counts(pairs),
                    "position": _position_metrics(records)},
        "per_dimension": per_dim,
        "records": records,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    curve = {
        "ts": summary["generated_at"], "n_items": len(items), "model": model,
        "prompt_version": prompt_version, "golden_sha256_16": summary["golden_sha256_16"],
        "kappa": overall_k["kappa"], "agreement": overall_k["agreement"],
        "position_bias_rate": summary["overall"]["position"]["position_bias_rate"],
        "per_dimension_agreement": {d: per_dim[d]["agreement"] for d in DIMENSIONS},
    }
    curve_path = Path(curve_path)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    with curve_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(curve, ensure_ascii=False) + "\n")
    summary["curve_file"] = str(curve_path)

    if show_progress:
        k = overall_k["kappa"]
        pos = summary["overall"]["position"]
        print(f"\n[calibration] n={len(items)} kappa={k} agreement={overall_k['agreement']} "
              f"position_bias={pos['position_bias_rate']} first_pos_rate={pos['first_position_rate']}")
        for d in DIMENSIONS:
            v = per_dim[d]
            print(f"  {d:<24} n={v['n']:>2} kappa={str(v['kappa']):>8} agreement={str(v['agreement']):>8} "
                  f"pos_bias={v['position']['position_bias_rate']}")
        print(f"[calibration] -> {out}")
        print(f"[calibration] curve append -> {curve_path}")
    return summary


# --------------------------------------------------------------------------- compare 前置门（供 lead 集成）
def judge_consistency_gate(consistency: dict | None = None,
                           path: Path | str | None = None,
                           threshold: float = THRESHOLD) -> dict:
    """compare 前置门：一致率 < 阈值 → 标注「judge未校准，结论不可信」，不给合入建议。

    入参可为 judge_consistency.json 的 dict，或其路径；都不给时读默认路径。
    不修改 iterate.py；lead 在 cmd_compare 里调用并据 merge_advice_allowed 决定是否给合入建议。
    """
    if consistency is None:
        p = Path(path) if path else DEFAULT_OUT
        if not p.is_file():
            return _gate(None, None, threshold, "JUDGE_DATA_MISSING",
                         "judge未校准，结论不可信：找不到 judge_consistency.json，compare 不给出合入建议")
        try:
            consistency = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return _gate(None, None, threshold, "JUDGE_DATA_BAD",
                         f"judge未校准，结论不可信：一致性文件解析失败（{_redact(exc, [])}），compare 不给出合入建议")
    if not isinstance(consistency, dict):
        return _gate(None, None, threshold, "JUDGE_DATA_BAD",
                     "judge未校准，结论不可信：一致性数据结构非法，compare 不给出合入建议")
    overall = consistency.get("overall") or consistency
    kappa = overall.get("kappa") if isinstance(overall, dict) else None
    agreement = overall.get("agreement") if isinstance(overall, dict) else None
    k = float(kappa) if isinstance(kappa, (int, float)) else None
    a = float(agreement) if isinstance(agreement, (int, float)) else None
    if k is None or a is None:
        return _gate(k, a, threshold, "JUDGE_DATA_INCOMPLETE",
                     "judge未校准，结论不可信：Kappa/一致率缺失或不可解析，compare 不给出合入建议")
    calibrated = k >= threshold and a >= threshold
    if calibrated:
        note = (f"judge已校准（Kappa={k:.3f}，一致率={a:.3f}，阈值{threshold}），"
                "compare 结论可给出合入建议")
        return _gate(k, a, threshold, "JUDGE_CALIBRATED", note)
    note = (f"judge未校准，结论不可信：Kappa={k:.3f}、一致率={a:.3f} 低于阈值{threshold}，"
            "compare 不给出合入建议")
    return _gate(k, a, threshold, "JUDGE_NOT_CALIBRATED", note)


def _gate(kappa, agreement, threshold, verdict, note) -> dict:
    calibrated = verdict == "JUDGE_CALIBRATED"
    return {
        "verdict": verdict,
        "calibrated": calibrated,
        "kappa": kappa,
        "agreement": agreement,
        "threshold": threshold,
        "note": note,
        "merge_advice_allowed": calibrated,
    }


# --------------------------------------------------------------------------- CLI
def read_key(key_file: Path) -> tuple[dict, str, str]:
    """只读 KEY/BASE_URL 到内存；不打印明文。"""
    if not key_file.is_file():
        return {}, "NOT_RUN", (
            f"key 文件不存在：{key_file}。按铁律标 NOT_RUN 不硬凑；"
            "可用 --key-file 指定主仓库 work/local-test-key.txt"
        )
    kv: dict[str, str] = {}
    try:
        for line in key_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip().strip('"').strip("'")
    except Exception as exc:  # noqa: BLE001
        return {}, "NOT_RUN", f"读取 key 文件失败：{_redact(exc, [])}"
    api_key = os.getenv("LLM_API_KEY") or kv.get("KEY", "")
    base_url = os.getenv("LLM_BASE_URL") or kv.get("BASE_URL", "")
    if not api_key:
        return {}, "NOT_RUN", "key 文件无 KEY= 且 env 无 LLM_API_KEY；标 NOT_RUN 不硬凑"
    if not base_url:
        return {}, "NOT_RUN", "key 文件无 BASE_URL= 且 env 无 LLM_BASE_URL；标 NOT_RUN 不硬凑"
    return {"api_key": api_key, "base_url": base_url, "source": str(key_file)}, "RUN", "key 仅入内存"


def build_client(key: dict, model: str):
    """构造真实 LLMClient；明文 key 只进进程 env 与 Settings，绝不打印/入库。"""
    os.environ["LLM_BASE_URL"] = key["base_url"]
    os.environ["LLM_API_KEY"] = key["api_key"]
    os.environ["MOCK_LLM"] = "false"
    os.environ["USE_INSTRUCTOR"] = "0"
    os.environ["ENVIRONMENT"] = "test"
    os.environ["LLM_EXTRA_HEADERS"] = json.dumps({
        "x-opencode-session": "judge-cal-" + uuid.uuid4().hex[:8],
        "User-Agent": "make-again-judge-calibration/1.0",
    }, ensure_ascii=False)
    from config import Settings
    from gateway.client import LLMClient
    settings = Settings(
        environment="test", mock_llm=False,
        llm_api_key=key["api_key"], llm_base_url=key["base_url"],
        llm_model=model, llm_fast_model=model,
        llm_timeout=float(os.getenv("LLM_TIMEOUT") or "120"),
    )
    return LLMClient(settings=settings)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S2 judge 金标校准（真实 hy4，temperature=0）")
    p.add_argument("--golden", default=str(GOLDEN))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--curve", default=str(DEFAULT_CURVE))
    p.add_argument("--key-file", default=str(REPO / "work" / "local-test-key.txt"))
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用，验收须全量）")
    p.add_argument("--model", default=os.getenv("LLM_FAST_MODEL") or DEFAULT_MODEL)
    p.add_argument("--prompt-version", default="v1")
    p.add_argument("--gate-only", action="store_true", help="只打印前置门判定，不跑模型")
    return p.parse_args()


def main() -> None:
    _reconfig_stdout()
    args = parse_args()
    if args.gate_only:
        gate = judge_consistency_gate(path=args.out)
        print("[gate] " + json.dumps(gate, ensure_ascii=False))
        return
    key, status, reason = read_key(Path(args.key_file))
    if status != "RUN":
        print(f"[calibration] NOT_RUN：{_redact(reason, [])}")
        return
    client = build_client(key, args.model)
    print(f"[calibration] golden={args.golden} model={args.model} "
          f"limit={args.limit or 'all'} (key 来源仅记路径，不记明文)")
    run_calibration(client, golden_path=args.golden, out_path=args.out, curve_path=args.curve,
                    limit=args.limit or None, model=args.model,
                    prompt_version=args.prompt_version)


if __name__ == "__main__":
    main()
