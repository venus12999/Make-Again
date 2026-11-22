#!/usr/bin/env python3
"""从权威评测集抽取分类分批样本（保证权威+广），落 eval/batches/。

每批一个 JSON，可独立反复跑。每批覆盖一个能力维度，每类 8-12 条。
数据源：CPsyCoun / ESConv+FailedESConv / GoEmotions / CueSupport-MH，全部 HF/Zenodo 实拉。

H1 统计功效扩充（2026-11-23）：先调研主流做法，再落地。调研结论（URL + 发布日期）：

1) Anthropic《A statistical approach to model evaluations》官方博客，发布 2024-11-18：
   https://www.anthropic.com/research/statistical-approach-to-model-evals
   配套论文《Adding Error Bars to Evals: A Statistical Approach to Language Model
   Evaluations》(Evan Miller)，arXiv:2411.00640，提交 2024-11-01：
   https://arxiv.org/abs/2411.00640
   五条建议与本批落点的对应：
   - #1 CLT → 报 SEM / 95%CI（iterate.py 的 bootstrap_ci 已覆盖）；
   - #2 聚类标准误 → 同源条目不能当独立样本（CPsyCoun 同一段对话只取一条，
     pick() 用 used 集合去重；本批按 row_index 唯一，不做段落级聚类）；
   - #3 降低题内方差 → 同一题 resample 多次取 question-level 均值
     （Inspect epochs 思路）：配对比较用 repetitions=5 的逐 case 均值，
     使配对差值 sd 按 1/sqrt(repetitions) 缩小；
   - #4 配对差异 → 同一 case 的 before/after 配对，消除题目难度方差
     （iterate.py::compare_metric 的 sign test + 配对 delta）；
   - #5 power analysis → 实验前按目标 MDE 定 n（本批 n=34≥30，
     MDE=(z(1-a/2)+z(power))*sd_pair/sqrt(n)，见 batches/01 的 power_analysis 元信息）。

2) Airbnb 工程博客《Eval-driven development: Lessons from evaluating GenAI at scale》，
   发布 2026-07：
   https://airbnb.tech/ai-ml/eval-driven-development-lessons-from-evaluating-genai-at-scale/
   原文 "Create a golden dataset of 50–100 examples" / "Start with 50–100 rows"，
   且必须含坏样本、judge 一致率要到 high 80s-90s。本地中文权威语料与调用成本受限，
   本步先做到 n=34（≥30 功效门槛），50-100 列为后续目标，不伪装成已达到。

口径：中文共情批全部来自 CPsyCoun 实拉（真实语料，非 synthetic），每条带 source_id
与 provenance 标注；旧 17 条 id 与内容逐字保留，保证和历史 baseline 的 per-case 配对可比。
用法：python eval/build_benchmarks.py --batch 01_中文共情_CPsyCoun 只重建中文批，
不覆盖已翻译的其余批次；不带参数维持原全量重建行为。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "batches"
OUTDIR.mkdir(exist_ok=True)
BASE = "https://datasets-server.huggingface.co"

ZH_BATCH = "01_中文共情_CPsyCoun"
ZH_PER_GROUP = 12  # 旧值 10 实际只落地 17 条；12 → 4 组共 34 条（≥30）
RESEARCH_BASIS = {
    "anthropic_blog": "https://www.anthropic.com/research/statistical-approach-to-model-evals (2024-11-18)",
    "anthropic_paper": "https://arxiv.org/abs/2411.00640 (arXiv 2411.00640, 2024-11-01)",
    "airbnb_blog": "https://airbnb.tech/ai-ml/eval-driven-development-lessons-from-evaluating-genai-at-scale/ (2026-07)",
}


def rows_api(ds, cfg, split, offset, length=100, retries=3):
    url = f"{BASE}/rows?dataset={urllib.parse.quote(ds, safe='')}&config={cfg}&split={split}&offset={offset}&length={length}"
    # HF datasets-server 偶发 502/504；实拉 3134 条要翻 32 页，单页失败不能让整批缩水
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=40) as r:
                return json.load(r).get("rows", [])
        except Exception as exc:
            if attempt == retries - 1:
                print(f"  ! fetch {ds}@{offset}: {str(exc)[:60]}")
                return []
            time.sleep(3 + attempt * 5)  # 429 限流时退避更久
    return []


def fetch_all(ds, cfg, split, total, batch=100):
    out = []
    for off in range(0, total, batch):
        chunk = rows_api(ds, cfg, split, off, batch)
        if not chunk:
            break
        out.extend(r["row"] for r in chunk)
    return out[:total]


_FILLER = ("谢谢", "嗯，", "好的，", "我明白了", "我会", "不用谢", "hello", "hi ", "hey", "good afternoon", "thank you", "i am ok", "i'm ok", "thanks")


def substantive(text):
    s = text.strip()
    return len(s) >= 30 and not s.lower().startswith(_FILLER)


def pick(rows, fields, groups, per_group):
    picked, used = [], set()
    for cat, kws in groups.items():
        n = 0
        for i, r in enumerate(rows):
            if i in used:
                continue
            txt = " ".join(str(r.get(f, "") or "") for f in fields)
            if any(k in txt for k in kws) and substantive(txt):
                picked.append((cat, i, r))
                used.add(i)
                n += 1
                if n >= per_group:
                    break
    return picked


def save_batch(name, meta, items):
    """落一批；同名旧文件的元信息（rubric / eval_control / eval 等）逐字保留，只更新 n 与新增元信息。

    这样重建数据不会丢掉 S3 集成时写入的 HEART 5 维 rubric 与旧自编维度对照字段。
    """
    p = OUTDIR / f"{name}.json"
    if p.exists():
        old = json.loads(p.read_text(encoding="utf-8"))
        merged = {k: v for k, v in old.items() if k not in ("batch", "items")}
        merged.update(meta)
        meta = merged
    obj = {"batch": name, **meta, "items": items}
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {p.name}: {len(items)} items")
    return obj


def build_zh_batch(per_group: int = ZH_PER_GROUP):
    """CPsyCoun 中文共情批：实拉 → pick(per_group) → 带 source_id / provenance 落盘。

    per_group 从 10 提到 12：4 组从实际落地 17 条扩到 34 条（旧 17 条 id 全保留，
    新增 17 条来自同一批 HF 实拉数据的不同 row_index，非 synthetic）。
    """
    cp_groups = {
        "分手失恋": ["分手", "失恋", "前任", "男朋友", "女朋友"],
        "亲人离世": ["去世", "离世", "父亲", "母亲", "外公", "奶奶", "闺蜜去世"],
        "情绪困扰": ["焦虑", "难过", "压力", "失眠", "抑郁"],
        "人际关系": ["朋友", "同事", "丈夫", "婆婆", "家人", "关系紧张"],
    }
    print("[1/4] CPsyCoun 中文多场景 ...")
    cache = os.environ.get("CPSYCOUN_ROWS_CACHE")
    if cache and Path(cache).exists():
        # 复用此前从同一 HF datasets-server rows API 实拉的 3134 行快照，避免限流重拉；
        # 快照内容仍是真实语料，不是 synthetic。
        cp = json.loads(Path(cache).read_text(encoding="utf-8"))
        print(f"  (cached real pull: {cache}, {len(cp)} rows)")
    else:
        cp = fetch_all("CAS-SIAT-XinHai/CPsyCoun", "default", "train", 3134)
    items = []
    for cat, i, r in pick(cp, ["input", "instruction"], cp_groups, per_group):
        sid = f"cp-{i}"
        items.append({
            "id": sid,
            "source_id": f"CPsyCoun:train:{i}",
            "scene": cat,
            "language": "zh",
            "user_input": (r.get("input") or r.get("instruction") or "")[:600],
            "reference": (r.get("output") or "")[:400],
            "source": {"dataset": "CPsyCoun", "paper": "arXiv 2405.16433 Findings ACL 2024", "id": sid},
            "provenance": {"kind": "authoritative", "synthetic": False,
                           "dataset": "CPsyCoun", "split": "train", "row_index": i,
                           "fetch": "HF datasets-server rows API (2026-11-23 实拉)"},
        })
    meta = {
        "source": "CPsyCoun",
        "paper": "Findings ACL 2024",
        "n": len(items),
        "power_analysis": {
            "research_basis": RESEARCH_BASIS,
            "target_n": 30, "actual_n": len(items),
            "alpha": 0.05, "power": 0.80, "repetitions": 5,
            "formula": "MDE=(z(1-alpha/2)+z(power))*sd_paired_diff/sqrt(n)  [iterate.py::mde]",
            "note": " Anthropic Rec#3 同题多次重采样取均值降低题内方差；Rec#4 同 case 配对差值；"
                    "Rec#5 power analysis 定 n。compare 用 --repetitions 5 才能兑现 MDE<0.15；"
                    "默认 3 次时功效仍不足。Airbnb 50-100 起步为后续目标。",
        },
    }
    if not (OUTDIR / f"{ZH_BATCH}.json").exists():
        meta["eval"] = "llm_judge 共情+安全"  # 全新落盘时的默认值；已有文件由 save_batch 保留 HEART 口径
    return save_batch(ZH_BATCH, meta, items)


def main(only: str | None = None):
    if only is not None and only != ZH_BATCH:
        raise SystemExit(f"unsupported single batch: {only}（本脚本仅支持单独重建 {ZH_BATCH}）")
    all_meta = {"schema_version": "1.0", "note": "权威集分类分批，每批独立可反复跑；英文样本需翻译后用于中文产品", "batches": []}

    # ===== 批次1: 中文共情（CPsyCoun）=====
    if only is None or only == ZH_BATCH:
        build_zh_batch()
        all_meta["batches"].append(ZH_BATCH)
    if only == ZH_BATCH:
        # 单批重建：_index.json 保留其余批次，不误删
        idx = OUTDIR / "_index.json"
        if idx.exists():
            for b in json.loads(idx.read_text(encoding="utf-8")).get("batches", []):
                if b not in all_meta["batches"]:
                    all_meta["batches"].append(b)
        idx.write_text(json.dumps(all_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[done] single batch rebuilt: {ZH_BATCH}")
        return

    # ===== 批次2: 支持策略（ESConv + FailedESConv）=====
    print("[2/4] ESConv 支持策略 ...")
    es = fetch_all("thu-coai/esconv", "default", "train", 500)
    es_items = []
    for cat, kws in {"分手": ["breakup", "ex-girlfriend", "ex-boyfriend", "split up"],
                     "丧亲/失去": ["death", "passed away", "lost my", "died"],
                     "抑郁情绪": ["depressed", "depression", "sad", "lonely"]}.items():
        n = 0
        for i, r in enumerate(es):
            if n >= 8:
                break
            raw = r.get("text", "")
            try:
                obj = json.loads(raw)
                situation = obj.get("situation", "")
                dialog = obj.get("dialog", [])
                first_usr = next((d.get("text", "") for d in dialog if d.get("speaker") == "usr"), "")
                strategy = [d.get("strategy") for d in dialog if d.get("speaker") == "sys" and d.get("strategy")][:3]
            except (json.JSONDecodeError, TypeError):
                continue
            text = situation + " " + first_usr
            if any(k.lower() in text.lower() for k in kws) and substantive(first_usr):
                es_items.append({
                    "id": f"es-{i}", "scene": cat, "language": "en", "needs_translation": True,
                    "user_input": first_usr[:600], "context": situation[:400],
                    "reference_strategies": strategy, "source": {"dataset": "ESConv", "paper": "arXiv 2106.01144 ACL 2021", "id": f"es-{i}"},
                })
                n += 1
    save_batch("02_支持策略_ESConv", {"source": "ESConv", "paper": "ACL 2021", "eval": "支持策略+共情（翻译后测）", "n": len(es_items)}, es_items)
    all_meta["batches"].append("02_支持策略_ESConv")

    # ===== 批次3: 情绪识别（GoEmotions）=====
    print("[3/4] GoEmotions 细粒度情绪 ...")
    ge = fetch_all("google-research-datasets/go_emotions", "simplified", "validation", 500)
    # simplified 标签 id->name：已从 HF datasets-server features 实测核实（2026-11-22）
    # 之前凭记忆硬编码 {7:grief,...} 全错（7实为curiosity），导致 gold 标签错误、hit=0.25
    label_names = {16: "grief", 25: "sadness", 2: "anger", 3: "annoyance", 18: "love", 19: "nervousness"}
    ge_items = []
    counts = Counter()
    for r in ge:
        labels = r.get("labels") or []
        for lid, lname in label_names.items():
            if lid in labels and counts[lname] < 6 and (r.get("text") or "").strip():
                ge_items.append({
                    "id": f"ge-{r.get('id','?')}", "scene": f"情绪-{lname}", "language": "en", "needs_translation": True,
                    "user_input": (r.get("text") or "")[:400], "expected_label": lname,
                    "source": {"dataset": "GoEmotions", "paper": "arXiv 2005.00547 ACL 2020", "id": f"ge-{r.get('id','?')}"},
                })
                counts[lname] += 1
                break
    save_batch("03_情绪识别_GoEmotions", {"source": "GoEmotions", "paper": "ACL 2020", "eval": "情绪分类（翻译后测）", "n": len(ge_items)}, ge_items)
    all_meta["batches"].append("03_情绪识别_GoEmotions")

    # ===== 批次4: 危机安全（CueSupport-MH）=====
    print("[4/4] CueSupport-MH 危机安全 ...")
    try:
        zurl = "https://zenodo.org/api/records/19699087/files/trace-mh.zip/content"
        with urllib.request.urlopen(zurl, timeout=120) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        zname = [n for n in z.namelist() if "camera_ready" in n and n.endswith(".json")][0]
        cue = json.loads(z.read(zname))
        cue_items = []
        for risk, resp, cat in [("high_risk", "supportive", "高危-支持性"), ("high_risk", "harmful", "高危-有害回复"),
                                 ("moderate_risk", "supportive", "中危-支持性"), ("moderate_risk", "dismissive", "中危-忽视")]:
            n = 0
            for r in cue:
                if n >= 6:
                    break
                if r.get("risk_label") == risk and r.get("response_label") == resp and (r.get("post") or "").strip():
                    cue_items.append({
                        "id": f"cue-{r.get('id','?')}", "scene": cat, "language": "en", "needs_translation": True,
                        "user_input": (r.get("post") or "")[:600], "reference": (r.get("response") or "")[:400],
                        "risk_label": risk, "response_label": resp,
                        "source": {"dataset": "CueSupport-MH", "paper": "Zenodo 10.5281/zenodo.19699087", "id": f"cue-{r.get('id','?')}"},
                    })
                    n += 1
        save_batch("04_危机安全_CueSupport", {"source": "CueSupport-MH", "paper": "Zenodo", "eval": "危机升级+回复安全性（翻译后测）", "n": len(cue_items)}, cue_items)
        all_meta["batches"].append("04_危机安全_CueSupport")
    except Exception as exc:
        print(f"  ! CueSupport failed: {str(exc)[:80]}")

    (OUTDIR / "_index.json").write_text(json.dumps(all_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {len(all_meta['batches'])} batches in {OUTDIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=None,
                    help=f"只重建指定批次（当前支持 {ZH_BATCH}），不覆盖已翻译的其它批次")
    main(ap.parse_args().batch)
