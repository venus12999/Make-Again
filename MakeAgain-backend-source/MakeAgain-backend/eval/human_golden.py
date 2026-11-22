#!/usr/bin/env python3
"""H3 · 真人金标（human golden）标注工作流：export / import / consistency。

【为什么有这个文件】
评测闭环的判定器分三层（code → model → human）。本项目已有 judge_golden.json
（"人工金标"），但那批标签是评测执行代理按产品标准**构造**的好/坏对照，不是真人
逐条判定的结果；judge 的一致性数字因此缺少可信真人基准。本文件把"真人判定层"
做成可执行的标注流水线，并**绝不伪造真人标注**：初始 human_golden.json 里所有
human_label 均为 null/pending，真人标签只能由 import 命令从产品方回填。

【主流依据（先调研后落地，URL + 发布日期）】
1. Airbnb 技术博客《Eval-Driven Development: Lessons from Evaluating GenAI at Scale》
   2026-07-28，https://medium.com/airbnb-engineering/eval-driven-development-lessons-from-evaluating-genai-at-scale-e817e5ae5788
   （中文摘要：https://agentnativeengineering.com/field-notes/2026-07-28-calibrate-llm-judge-before-it-grades-releases/ ，同日）
   要点：judge 必须先对"human-labeled golden set"校准才可信；金标 50–100 条且
   **必须包含 bad examples**；一致性用 Cohen's kappa 或 Krippendorff's alpha，目标
   high 80s–90s；**专家对某条标签分歧就 hard stop**，先解决分歧再继续；随失败模式
   演化**周期性重新校准**（即 criteria drift 治理）；指定一名最终人类决策者。
2. Netflix 技术博客《The Lifecycle of LLM-as-a-Judge: Building, Aligning, and
   Monitoring at Scale》，2026-10（Netflix Tech Blog），
   https://netflixtechblog.com/the-lifecycle-of-llm-as-a-judge-building-aligning-and-monitoring-at-scale-c95bd8283508
   （摘要：https://pulse.adyog.com/insights/netflix-llm-judge-reasons-not-just-verdicts ，2026-10-15）
   要点：四阶段=人类专家定义好标准并手工造 hard examples → judge 对齐人类标签**和
   书面理由** → 上线 → 人类持续查漂移；每条 ≥3 名人类标注者、多数票定真值；每周
   抽样约 300 条复核（含 judge 拦截的样本），越过 drift 阈值触发重训与新基准数据；
   除"标签不一致"外还要审"right answer, wrong reason"（结论对但理由错）。
3. Netflix《Grading the Graders: Can an AI Judge Your Synopsis?》Data Science in
   the Wild 播客，2026-09-15，https://rss.com/podcasts/data-science-in-the-wild/3151092/
   要点：judge 与创意写手一致率 85%+、写手之间约 80%；one judge per criterion；
   多次采样取共识可平滑误判。

【落地映射】
- export：把 pending 标注集导成 JSONL/CSV，含 user_input / 待评回复 / judge 的
  verdict+reason（--blind 可隐藏以防锚定）/ 空人工标签字段 / 该维度明确 rubric；
  同时保留 constructed_verdict 供对照，但标注 verdict=constructed 非真人。
- import：只回填人工字段，不覆盖 constructed_verdict；产物默认写 outputs/ 副本，
  --in-place 才改源金标；不合法标签整批拒绝，绝不猜标签。
- consistency：judge 标签默认取 outputs/benchmarks/judge_consistency.json（真实
  judge 二元输出），与回填标签算 Cohen's κ + Krippendorff's α + 一致率 + 混淆矩阵
  + 逐维度漂移观察 + **具体 case 的分歧清单**（供专家先解决分歧）；无人工标签时
  kappa=None 并注明，不编造。口径透传 label_caliber，构造标签干跑会明确标
  CONSTRUCTED_DRY_RUN_NON_HUMAN。

【边界】只读 eval/ 数据，不改 agent/memory/api 主链路；不读任何 key、不发网络请求。

用法（在 MakeAgain-backend 目录下）：
    python eval/human_golden.py init                       # 从 judge_golden.json 生成占位金标（已存在则拒绝）
    python eval/human_golden.py export                     # 导出 JSONL + CSV 到 outputs/human_golden/
    python eval/human_golden.py export --blind --format jsonl
    python eval/human_golden.py import outputs/human_golden/annotation_export.jsonl
    python eval/human_golden.py consistency --golden outputs/human_golden/human_golden_labeled.json
    python eval/human_golden.py consistency --golden <labeled> --judge-labels <file.json>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))

GOLDEN = HERE / "human_golden.json"
JUDGE_GOLDEN = HERE / "judge_golden.json"
JUDGE_CONSISTENCY = REPO / "outputs" / "benchmarks" / "judge_consistency.json"
OUTDIR = REPO / "outputs" / "human_golden"

LABELS = ("pass", "fail")
LABEL_ALIASES = {
    "pass": "pass", "true": "pass", "yes": "pass", "1": "pass", "通过": "pass", "达标": "pass",
    "fail": "fail", "false": "fail", "no": "fail", "0": "fail", "不通过": "fail", "不达标": "fail",
}
CONSTRUCTED_MARKERS = ("construct", "构造", "pipeline", "selftest", "self-test", "dry-run", "dryrun", "fixture")

RESEARCH_BASIS = [
    {
        "org": "Airbnb", "date": "2026-07-28",
        "title": "Eval-Driven Development: Lessons from Evaluating GenAI at Scale",
        "url": "https://medium.com/airbnb-engineering/eval-driven-development-lessons-from-evaluating-genai-at-scale-e817e5ae5788",
        "takeaway": "judge 先对 50-100 条（含 bad examples）真人金标校准；κ/α 目标 high 80s-90s；专家分歧 hard stop；随失败模式演化周期重校准。",
    },
    {
        "org": "Netflix", "date": "2026-10",
        "title": "The Lifecycle of LLM-as-a-Judge: Building, Aligning, and Monitoring at Scale",
        "url": "https://netflixtechblog.com/the-lifecycle-of-llm-as-a-judge-building-aligning-and-monitoring-at-scale-c95bd8283508",
        "takeaway": "人类定义标准+手工 hard examples；每条≥3 人多数票；每周抽样复核；drift 阈值触发新基准；既查标签分歧也查 right-answer-wrong-reason。",
    },
    {
        "org": "Netflix (Data Science in the Wild)", "date": "2026-09-15",
        "title": "Grading the Graders: Can an AI Judge Your Synopsis?",
        "url": "https://rss.com/podcasts/data-science-in-the-wild/3151092/",
        "takeaway": "judge 与写手一致 85%+、写手之间约 80%；one judge per criterion；多次采样取共识。",
    },
]

EXPORT_FIELDS = [
    "id", "source_id", "batch", "dimension", "rubric", "user_input",
    "reply_to_label", "anchor_reply", "judge_verdict", "judge_reason",
    "verdict", "constructed_verdict", "constructed_critique",
    "human_label", "human_rationale", "annotator", "status",
]


# --------------------------------------------------------------------------- 基础工具
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _round(x: float | None, n: int = 6) -> float | None:
    return None if x is None else round(float(x), n)


def _sha16(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _norm_label(value: Any) -> str | None:
    """把 pass/fail、布尔、中文别名统一成 pass/fail；识别不了返回 None，绝不猜。"""
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if value is None:
        return None
    key = str(value).strip().lower()
    return LABEL_ALIASES.get(key)


def load_golden(path: Path | str = GOLDEN) -> dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"human_golden.json 不存在：{p}（先跑 `python eval/human_golden.py init`）")
    data = json.loads(p.read_text(encoding="utf-8"))
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("human_golden.json 缺非空 items")
    for i, it in enumerate(items):
        for k in ("id", "source_id", "dimension", "user_input", "reply_to_label", "rubric",
                  "constructed_verdict", "verdict", "human_label", "status"):
            if k not in it:
                raise ValueError(f"items[{i}] 缺字段 {k}")
        if it["verdict"] != "constructed":
            raise ValueError(f"items[{i}].verdict 必须是 constructed（构造占位，非真人）")
        if it["constructed_verdict"] not in LABELS:
            raise ValueError(f"items[{i}].constructed_verdict 非法：{it['constructed_verdict']}")
        if it["human_label"] is not None and it["human_label"] not in LABELS:
            raise ValueError(f"items[{i}].human_label 非法：{it['human_label']}")
    return data


# --------------------------------------------------------------------------- 统计
def cohens_kappa(pairs: list[tuple[str, str]]) -> dict:
    """Cohen's κ：pairs=[(人工标签, judge标签)]。pe==1 时数学上无定义 → None，不硬凑。"""
    valid = [(h, j) for h, j in pairs if h in LABELS and j in LABELS]
    n = len(valid)
    if n == 0:
        return {"kappa": None, "agreement": None, "pe": None, "n": 0, "note": "NO_VALID_PAIRS"}
    po = sum(h == j for h, j in valid) / n
    ph = sum(h == "pass" for h, _ in valid) / n
    pj = sum(j == "pass" for _, j in valid) / n
    pe = ph * pj + (1 - ph) * (1 - pj)
    if pe >= 1.0:
        return {"kappa": None, "agreement": _round(po), "pe": _round(pe), "n": n,
                "note": "UNDEFINED_PE_EQ_1"}
    return {
        "kappa": _round((po - pe) / (1 - pe)), "agreement": _round(po), "pe": _round(pe),
        "p_human_pass": _round(ph), "p_judge_pass": _round(pj), "n": n, "note": "OK",
    }


def krippendorff_alpha_binary(pairs: list[tuple[str, str]]) -> dict:
    """Krippendorff's α（nominal, 2 名标注者）：α = 1 - Do/De。

    用 coincidence matrix 口径：每条的 2 个标签形成 m(m-1)=2 个有序对，除以 (m-1)=1；
    一致条目给 o_cc +2，分歧条目给 o_ck/o_kc 各 +1；Do=Σ_{c≠k}o_ck/n，
    De=Σ_{c≠k}n_c*n_k/(n*(n-1))。与 κ 是互补指标，不做"两者必然相等"的假设。
    """
    valid = [(h, j) for h, j in pairs if h in LABELS and j in LABELS]
    n_items = len(valid)
    n_ratings = 2 * n_items
    if n_items == 0:
        return {"alpha": None, "n": 0, "note": "NO_VALID_PAIRS"}
    o = {(c, k): 0 for c in LABELS for k in LABELS}
    for h, j in valid:
        o[(h, j)] += 1
        o[(j, h)] += 1  # 两项相同则同一格 +2，不同则两格各 +1
    n_c = {c: sum(o[(c, k)] for k in LABELS) for c in LABELS}
    do = sum(o[(c, k)] for c in LABELS for k in LABELS if c != k) / n_ratings
    de = sum(n_c[c] * n_c[k] for c in LABELS for k in LABELS if c != k) / (n_ratings * (n_ratings - 1))
    if de == 0:
        return {"alpha": None, "n": n_items, "note": "UNDEFINED_DE_EQ_0"}
    return {"alpha": _round(1 - do / de), "do": _round(do), "de": _round(de),
            "n": n_items, "note": "OK"}


def confusion_counts(pairs: list[tuple[str, str]]) -> dict:
    """混淆矩阵（pass 为正类，人工为真值）。"""
    c = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for h, j in pairs:
        if h not in LABELS or j not in LABELS:
            continue
        if h == "pass" and j == "pass":
            c["tp"] += 1
        elif h == "fail" and j == "pass":
            c["fp"] += 1
        elif h == "pass" and j == "fail":
            c["fn"] += 1
        else:
            c["tn"] += 1
    return c


# --------------------------------------------------------------------------- judge 标签
def load_judge_labels(path: Path | str | None = None) -> tuple[dict[str, dict], str]:
    """返回 {id: {"label": pass/fail, "reason": str}} 与来源说明。

    默认读 judge_calibration.py 的真实产物 judge_consistency.json；--judge-labels
    可指向自定义 JSON（dict{id: label} / dict{id:{label,reason}} / list[{id,label}]）。
    """
    p = Path(path) if path else JUDGE_CONSISTENCY
    if not p.is_file():
        return {}, f"MISSING:{p}"
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    if isinstance(raw, dict) and isinstance(raw.get("records"), list):
        for r in raw["records"]:
            label = _norm_label((r.get("binary") or {}).get("label"))
            if label:
                out[str(r["id"])] = {"label": label, "reason": str((r.get("binary") or {}).get("reason", ""))[:200]}
        return out, f"judge_consistency.json:{p.name}"
    if isinstance(raw, dict):
        items = raw.get("labels", raw)
        if isinstance(items, dict):
            for k, v in items.items():
                label = _norm_label(v.get("label") if isinstance(v, dict) else v)
                if label:
                    out[str(k)] = {"label": label, "reason": str(v.get("reason", ""))[:200] if isinstance(v, dict) else ""}
        return out, f"custom:{p.name}"
    if isinstance(raw, list):
        for r in raw:
            label = _norm_label(r.get("label"))
            if label and r.get("id"):
                out[str(r["id"])] = {"label": label, "reason": str(r.get("reason", ""))[:200]}
        return out, f"custom:{p.name}"
    return {}, f"UNRECOGNIZED:{p.name}"


# --------------------------------------------------------------------------- init
def build_placeholder(judge_golden_path: Path | str = JUDGE_GOLDEN) -> dict:
    """从 judge_golden.json 的构造金标生成真人金标占位（每条 verdict=constructed）。"""
    src = json.loads(Path(judge_golden_path).read_text(encoding="utf-8"))
    dims = src.get("dimensions", {})
    items = []
    for it in src["items"]:
        items.append({
            "id": it["id"], "source_id": it["source_id"], "batch": it.get("batch"),
            "dimension": it["dimension"], "scene": it.get("scene"),
            "user_input": it["user_input"],
            "reply_to_label": it["candidate_reply"],
            "anchor_reply": it.get("anchor_reply"),
            "rubric": dims.get(it["dimension"], ""),
            "verdict": "constructed",
            "verdict_caliber": "人工构造的好/坏对照，非真人判定",
            "constructed_verdict": it["human_label"],
            "constructed_critique": it.get("critique", ""),
            "human_label": None,
            "human_rationale": None,
            "annotator": None,
            "status": "pending",
            "labeled_at": None,
        })
    n_fail = sum(i["constructed_verdict"] == "fail" for i in items)
    return {
        "schema_version": "1.0",
        "name": "human_golden",
        "generated_at": _now(),
        "caliber": "构造金标占位-非真人判定",
        "status": "PLACEHOLDER_AWAITING_REAL_HUMAN_ANNOTATION",
        "annotation_note": (
            "初始条目全部从 eval/judge_golden.json 的构造金标导入：user_input 抽自四批权威集可回溯，"
            "reply_to_label/anchor_reply 是评测执行代理人工构造的好/坏对照，constructed_verdict 是构造标签。"
            "所有 human_label=null/pending——本文件不含任何真人标注，绝不假装有真人；"
            "真人标签只能经 `human_golden.py import` 由产品方回填。"
        ),
        "source": {"file": "eval/judge_golden.json", "caliber": src.get("caliber"),
                   "annotation_note": src.get("annotation_note")},
        "research_basis": RESEARCH_BASIS,
        "annotators": {
            "current": [],
            "planned": "产品方指定 2-3 名真人标注者（Netflix 每条≥3人多数票；Airbnb 指定一名最终决策者）",
            "named_final_decision_maker": None,
        },
        "rubric_note": (
            "四维 rubric 继承 judge_golden.dimensions：empathy_specificity / "
            "strategy_appropriateness / emotion_accuracy / crisis_safety；每条 item 内联对应 rubric。"
        ),
        "dimensions": dims,
        "label_policy": {
            "labels": list(LABELS),
            "include_failure_samples": True,
            "failure_sample_count": n_fail,
            "recommended_n_range": [50, 100],
            "current_n": len(items),
            "gap_note": (
                f"当前 {len(items)} 条（含 {n_fail} 条失败样本）是构造占位，低于主流建议的 50-100 条；"
                "产品方需补足真人标注的新样本，不能把构造标签当真人金标。"
            ),
        },
        "disagreement_policy": (
            "专家对某条标签分歧时 hard stop（Airbnb）：先由标注者讨论并书面记录理由，"
            "必要时改 rubric 后重标；除标签不一致外也要查 'right answer, wrong reason'（Netflix）。"
        ),
        "criteria_drift_policy": {
            "recalibrate": "随失败模式演化周期性重新校准（Airbnb）",
            "monitor": "每周按维度抽样真人复核，含 judge 拦截样本；越过阈值触发补标与新基准（Netflix）",
            "watch_metric": "consistency.py 输出的 per_dimension agreement/kappa 与 consistency_curve.jsonl",
            "drift_threshold_note": "阈值由产品方按业务定；本工具只提供逐维度观察值，不替产品方下结论。",
        },
        "label_caliber": "NO_HUMAN_LABELS_YET",
        "items": items,
    }


def cmd_init(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    if out.exists() and not args.force:
        raise FileExistsError(f"{out} 已存在；加 --force 才覆盖（默认保护真人回填成果）")
    data = build_placeholder(args.judge_golden)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"out": str(out), "n_items": len(data["items"]),
               "caliber": data["caliber"], "status": data["status"]}
    print("[init] " + json.dumps(summary, ensure_ascii=False))
    return summary


# --------------------------------------------------------------------------- export
def build_export_rows(golden: dict, judge_labels: dict[str, dict], blind: bool = False) -> list[dict]:
    """构造待标注行：每题含 user_input/待评回复/judge 判定/空人工标签/明确 rubric。"""
    rows = []
    for it in golden["items"]:
        jl = judge_labels.get(it["id"], {})
        rows.append({
            "id": it["id"], "source_id": it["source_id"], "batch": it.get("batch") or "",
            "dimension": it["dimension"], "rubric": it["rubric"], "user_input": it["user_input"],
            "reply_to_label": it["reply_to_label"], "anchor_reply": it.get("anchor_reply") or "",
            "judge_verdict": "" if blind else jl.get("label", ""),
            "judge_reason": "" if blind else jl.get("reason", ""),
            "verdict": it["verdict"], "constructed_verdict": it["constructed_verdict"],
            "constructed_critique": it.get("constructed_critique") or "",
            "human_label": "",  # 明确留空，由真人/产品方回填
            "human_rationale": "", "annotator": "", "status": "pending",
        })
    return rows


def cmd_export(args: argparse.Namespace) -> dict:
    golden = load_golden(args.golden)
    judge_labels, jsrc = load_judge_labels(args.judge_labels)
    rows = build_export_rows(golden, judge_labels, blind=args.blind)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    if args.format in ("jsonl", "both"):
        p = outdir / "annotation_export.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        written.append(str(p))
    if args.format in ("csv", "both"):
        p = outdir / "annotation_export.csv"
        with p.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=EXPORT_FIELDS)
            w.writeheader()
            w.writerows(rows)
        written.append(str(p))
    summary = {
        "n_rows": len(rows), "files": written, "format": args.format, "blind": args.blind,
        "judge_source": jsrc, "pending_fields": ["human_label", "human_rationale", "annotator"],
        "caliber": golden.get("caliber"),
    }
    print("[export] " + json.dumps(summary, ensure_ascii=False))
    return summary


# --------------------------------------------------------------------------- import
def read_annotation_file(path: Path | str) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"标注文件不存在：{p}")
    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8-sig", newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    rows = []
    for ln, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"标注文件第 {ln} 行不是合法 JSON：{exc}") from exc
    return rows


def cmd_import(args: argparse.Namespace) -> dict:
    golden = load_golden(args.golden)
    rows = read_annotation_file(args.annotation_file)
    by_id = {it["id"]: it for it in golden["items"]}
    errors: list[str] = []
    applied: list[str] = []
    staged: dict[str, dict] = {}
    for idx, r in enumerate(rows, 1):
        rid = str(r.get("id") or "").strip()
        raw_label = r.get("human_label")
        if raw_label in (None, ""):
            continue  # 空标签 = 未标，跳过且不写
        if rid not in by_id:
            errors.append(f"第{idx}行 id={rid!r} 不在金标中")
            continue
        label = _norm_label(raw_label)
        if label is None:
            errors.append(f"第{idx}行 id={rid} 的 human_label={raw_label!r} 非法（只认 pass/fail）")
            continue
        staged[rid] = {
            "human_label": label,
            "human_rationale": str(r.get("human_rationale") or "")[:500],
            "annotator": str(r.get("annotator") or "UNSPECIFIED").strip()[:80],
        }
    if errors:
        raise ValueError("标注文件校验失败，整批拒绝（未写入任何文件）：" + "；".join(errors[:10]))

    for rid, vals in staged.items():
        item = by_id[rid]
        item.update(vals)
        item["status"] = "labeled"
        item["labeled_at"] = _now()

    annotators = [v["annotator"] for v in staged.values()]
    has_constructed = any(any(m in a.lower() for m in CONSTRUCTED_MARKERS) for a in annotators)
    golden["label_caliber"] = (
        "CONSTRUCTED_DRY_RUN_NON_HUMAN" if has_constructed
        else "HUMAN_ANNOTATION_CLAIMED_NOT_INDEPENDENTLY_VERIFIED"
    )
    golden["imported_at"] = _now()
    golden["imported_from"] = str(args.annotation_file)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(golden, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.in_place:
        Path(args.golden).write_text(json.dumps(golden, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    applied = list(staged)
    summary = {
        "out": str(out), "in_place": bool(args.in_place), "n_applied": len(applied),
        "n_skipped_blank": sum(1 for r in rows if r.get("human_label") in (None, "")),
        "label_caliber": golden["label_caliber"], "applied_ids": applied[:10],
    }
    print("[import] " + json.dumps(summary, ensure_ascii=False))
    return summary


# --------------------------------------------------------------------------- consistency
def cmd_consistency(args: argparse.Namespace) -> dict:
    golden = load_golden(args.golden)
    judge_labels, jsrc = load_judge_labels(args.judge_labels)
    labeled = [it for it in golden["items"] if it.get("human_label") in LABELS]
    pairs: list[tuple[str, str]] = []
    joined: list[tuple[dict, dict]] = []
    missing_judge: list[str] = []
    for it in labeled:
        jl = judge_labels.get(it["id"])
        if not jl:
            missing_judge.append(it["id"])
            continue
        pairs.append((it["human_label"], jl["label"]))
        joined.append((it, jl))

    kappa = cohens_kappa(pairs)
    alpha = krippendorff_alpha_binary(pairs)
    disagreements = []
    for it, jl in joined:
        if it["human_label"] != jl["label"]:
            disagreements.append({
                "id": it["id"], "source_id": it["source_id"], "dimension": it["dimension"],
                "rubric": it["rubric"], "user_input": it["user_input"],
                "reply_to_label": it["reply_to_label"],
                "judge_verdict": jl["label"], "judge_reason": jl.get("reason", ""),
                "human_label": it["human_label"], "human_rationale": it.get("human_rationale"),
                "annotator": it.get("annotator"),
                "constructed_verdict": it["constructed_verdict"],
                "resolution": "PENDING_HUMAN_ADJUDICATION",
                "note": "专家先解决该分歧（Airbnb hard stop），不要直接调阈值抹平。",
            })

    per_dim = {}
    for dim in sorted({it["dimension"] for it in labeled}):
        dp = [(it["human_label"], judge_labels[it["id"]]["label"])
              for it in labeled if it["dimension"] == dim and it["id"] in judge_labels]
        dk = cohens_kappa(dp)
        per_dim[dim] = {
            "n": len(dp), "kappa": dk["kappa"], "agreement": dk["agreement"],
            "n_disagreements": sum(1 for h, j in dp if h != j),
        }

    caliber = golden.get("label_caliber", "UNKNOWN")
    note = "NO_HUMAN_LABELS" if not labeled else ("NO_JUDGE_LABELS" if not pairs else "OK")
    if caliber == "CONSTRUCTED_DRY_RUN_NON_HUMAN":
        caliber_note = "口径警告：人工字段由构造标签回填（非真人），仅用于跑通流程，κ/α 不代表真人一致性。"
    elif caliber == "HUMAN_ANNOTATION_CLAIMED_NOT_INDEPENDENTLY_VERIFIED":
        caliber_note = "口径：人工字段由 import 声称来自真人，本工具未独立验证标注者身份。"
    elif caliber == "NO_HUMAN_LABELS_YET":
        caliber_note = "口径：当前金标无任何人工标签，未计算 κ/α。"
    else:
        caliber_note = "口径未知，需人工确认 label_caliber。"

    result = {
        "schema_version": "1.0",
        "generated_at": _now(),
        "golden_file": str(args.golden),
        "golden_sha256_16": _sha16(args.golden),
        "judge_source": jsrc,
        "caliber": caliber,
        "caliber_note": caliber_note,
        "note": note,
        "n_labeled": len(labeled),
        "n_judged_overlap": len(pairs),
        "n_missing_judge_labels": len(missing_judge),
        "missing_judge_ids": missing_judge[:20],
        "overall": {**kappa, "krippendorff_alpha": alpha["alpha"],
                    "krippendorff_detail": alpha, "confusion": confusion_counts(pairs)},
        "per_dimension": per_dim,
        "criteria_drift_watch": {
            "per_dimension": per_dim,
            "note": "逐维度 agreement/kappa 供持续校准；阈值由产品方定，越阈触发补标（见 human_golden.json.criteria_drift_policy）。",
        },
        "disagreements": disagreements,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    curve = Path(args.curve)
    curve.parent.mkdir(parents=True, exist_ok=True)
    with curve.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": result["generated_at"], "caliber": caliber, "n": len(pairs),
            "kappa": kappa["kappa"], "alpha": alpha["alpha"], "agreement": kappa["agreement"],
            "per_dimension_agreement": {d: v["agreement"] for d, v in per_dim.items()},
            "golden_sha256_16": result["golden_sha256_16"],
        }, ensure_ascii=False) + "\n")

    k = kappa["kappa"]
    print(f"[consistency] n_labeled={len(labeled)} overlap={len(pairs)} "
          f"kappa={k} alpha={alpha['alpha']} agreement={kappa['agreement']} "
          f"disagreements={len(disagreements)}")
    print(f"[consistency] {caliber_note}")
    print(f"[consistency] -> {out}  curve -> {curve}")
    return result


# --------------------------------------------------------------------------- CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H3 真人金标标注工作流（init/export/import/consistency）")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="从 judge_golden.json 生成构造占位金标")
    pi.add_argument("--judge-golden", default=str(JUDGE_GOLDEN))
    pi.add_argument("--out", default=str(GOLDEN))
    pi.add_argument("--force", action="store_true")

    pe = sub.add_parser("export", help="导出待标注集 JSONL/CSV")
    pe.add_argument("--golden", default=str(GOLDEN))
    pe.add_argument("--judge-labels", default=None)
    pe.add_argument("--out", default=str(OUTDIR))
    pe.add_argument("--format", choices=("jsonl", "csv", "both"), default="both")
    pe.add_argument("--blind", action="store_true", help="隐藏 judge verdict/reason 防锚定")

    pim = sub.add_parser("import", help="回填人工标签（默认写副本，不改源金标）")
    pim.add_argument("annotation_file")
    pim.add_argument("--golden", default=str(GOLDEN))
    pim.add_argument("--out", default=str(OUTDIR / "human_golden_labeled.json"))
    pim.add_argument("--in-place", action="store_true", help="显式允许覆盖源金标")

    pc = sub.add_parser("consistency", help="算 judge-人 κ/α/一致率与分歧清单")
    pc.add_argument("--golden", default=str(GOLDEN))
    pc.add_argument("--judge-labels", default=None)
    pc.add_argument("--out", default=str(OUTDIR / "consistency.json"))
    pc.add_argument("--curve", default=str(OUTDIR / "consistency_curve.jsonl"))

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # pragma: no cover
        pass
    args = parse_args(argv)
    try:
        {"init": cmd_init, "export": cmd_export, "import": cmd_import,
         "consistency": cmd_consistency}[args.cmd](args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"[{args.cmd}] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
