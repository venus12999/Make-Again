#!/usr/bin/env python3
"""从权威评测集抽取分类分批样本（保证权威+广），落 eval/batches/。

每批一个 JSON，可独立反复跑。每批覆盖一个能力维度，每类 8-12 条。
数据源：CPsyCoun / ESConv+FailedESConv / GoEmotions / CueSupport-MH，全部 HF/Zenodo 实拉。
"""
from __future__ import annotations

import io
import json
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "batches"
OUTDIR.mkdir(exist_ok=True)
BASE = "https://datasets-server.huggingface.co"


def rows_api(ds, cfg, split, offset, length=100):
    url = f"{BASE}/rows?dataset={urllib.parse.quote(ds, safe='')}&config={cfg}&split={split}&offset={offset}&length={length}"
    try:
        with urllib.request.urlopen(url, timeout=40) as r:
            return json.load(r).get("rows", [])
    except Exception as exc:
        print(f"  ! fetch {ds}@{offset}: {str(exc)[:60]}")
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
    obj = {"batch": name, **meta, "items": items}
    p = OUTDIR / f"{name}.json"
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {p.name}: {len(items)} items")
    return obj


def main():
    all_meta = {"schema_version": "1.0", "note": "权威集分类分批，每批独立可反复跑；英文样本需翻译后用于中文产品", "batches": []}

    # ===== 批次1: 中文共情（CPsyCoun）=====
    print("[1/4] CPsyCoun 中文多场景 ...")
    cp = fetch_all("CAS-SIAT-XinHai/CPsyCoun", "default", "train", 3134)
    cp_groups = {
        "分手失恋": ["分手", "失恋", "前任", "男朋友", "女朋友"],
        "亲人离世": ["去世", "离世", "父亲", "母亲", "外公", "奶奶", "闺蜜去世"],
        "情绪困扰": ["焦虑", "难过", "压力", "失眠", "抑郁"],
        "人际关系": ["朋友", "同事", "丈夫", "婆婆", "家人", "关系紧张"],
    }
    items = []
    for cat, i, r in pick(cp, ["input", "instruction"], cp_groups, 10):
        items.append({
            "id": f"cp-{i}", "scene": cat, "language": "zh",
            "user_input": (r.get("input") or r.get("instruction") or "")[:600],
            "reference": (r.get("output") or "")[:400],
            "source": {"dataset": "CPsyCoun", "paper": "arXiv 2405.16433 Findings ACL 2024", "id": f"cp-{i}"},
        })
    save_batch("01_中文共情_CPsyCoun", {"source": "CPsyCoun", "paper": "Findings ACL 2024", "eval": "llm_judge 共情+安全", "n": len(items)}, items)
    all_meta["batches"].append("01_中文共情_CPsyCoun")

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
    main()
