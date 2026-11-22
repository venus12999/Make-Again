#!/usr/bin/env python3
"""从权威评测集抽取代表性 mini 子集（22 条），落 eval/authoritative_mini.json。

数据源全部 HF datasets-server / GitHub raw 实拉，不编造。抽取失败如实标 NOT_FOUND。
用法：python eval/build_authoritative_mini.py
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "authoritative_mini.json"
BASE = "https://datasets-server.huggingface.co"


def rows_api(ds: str, cfg: str, split: str, offset: int, length: int = 100) -> list[dict]:
    url = f"{BASE}/rows?dataset={urllib.parse.quote(ds, safe='')}&config={cfg}&split={split}&offset={offset}&length={length}"
    try:
        with urllib.request.urlopen(url, timeout=40) as r:
            return json.load(r).get("rows", [])
    except Exception as exc:
        print(f"  ! rows fetch failed {ds} @{offset}: {str(exc)[:80]}")
        return []


def fetch_all(ds: str, cfg: str, split: str, total: int, batch: int = 100) -> list[dict]:
    out = []
    for off in range(0, total, batch):
        chunk = rows_api(ds, cfg, split, off, batch)
        if not chunk:
            break
        out.extend(r["row"] for r in chunk)
        if len(out) >= total:
            break
    return out[:total]


# 排除"结尾客套/寒暄"——关键词命中但不是真实倾诉的样本
_FILLER_PREFIXES = ("谢谢", "嗯，", "好的，", "我明白了", "我会", "不用谢", "hello", "hi ", "hey", "good afternoon", "thank you", "i am ok", "i'm ok", "thanks")


def _is_substantive(text: str) -> bool:
    s = text.strip()
    if len(s) < 30:  # 太短必是寒暄
        return False
    if s.lower().startswith(_FILLER_PREFIXES):
        return False
    return True


def pick_by_keywords(rows: list[dict], text_fields: list[str], groups: dict[str, list[str]], per_group: int) -> list[dict]:
    picked = []
    used = set()
    for cat, kws in groups.items():
        n = 0
        for i, r in enumerate(rows):
            if i in used:
                continue
            text = " ".join(str(r.get(f, "") or "") for f in text_fields)
            if any(k in text for k in kws) and _is_substantive(text):
                picked.append({"_cat": cat, "_row": r, "_idx": i})
                used.add(i)
                n += 1
                if n >= per_group:
                    break
    return picked


def main():
    mini = {"schema_version": "1.0", "note": "权威集代表性抽样，非全量；许可见每条 source_license", "items": []}

    # ---------- CPsyCoun（中文 ACL2024）----------
    print("[CPsyCoun] fetching 3134 rows ...")
    cp = fetch_all("CAS-SIAT-XinHai/CPsyCoun", "default", "train", 3134)
    print(f"  got {len(cp)}")
    cp_groups = {
        "分手失恋": ["分手", "失恋", "前任", "男朋友", "女朋友"],
        "丧亲": ["去世", "离世", "父亲去世", "母亲去世", "外公", "奶奶走"],
        "危机轻生": ["自杀", "不想活", "活着没意思", "轻生", "活不下去"],
        "普通情绪": ["焦虑", "难过", "压力", "失眠"],
    }
    for p in pick_by_keywords(cp, ["input", "instruction"], cp_groups, 2):
        r = p["_row"]
        mini["items"].append({
            "source_dataset": "CPsyCoun", "source_id": f"cp-{p['_idx']}",
            "source_paper": "arXiv 2405.16433 Findings ACL 2024", "source_license": "cc-by-sa-4.0",
            "scenario": p["_cat"], "language": "zh",
            "user_input": (r.get("input") or r.get("instruction") or "")[:500],
            "reference": (r.get("output") or "")[:500],
            "eval_dims": ["共情", "专业性", "真实性", "Safety"],
        })

    # ---------- ESConv（ACL2021）----------
    print("[ESConv] fetching ...")
    es = fetch_all("thu-coai/esconv", "default", "train", 200)
    print(f"  got {len(es)}")
    es_groups = {
        "breakup": ["breakup", "partner", "ex-boyfriend", "ex-girlfriend"],
        "其他场景": ["depression", "job", "friend", "academic"],
    }
    for p in pick_by_keywords(es, ["text"], es_groups, 2):
        r = p["_row"]
        raw = r.get("text", "")
        try:
            obj = json.loads(raw)
            situation = obj.get("situation", "")[:400]
            dialog = obj.get("dialog", [])
            first_usr = next((d.get("text", "") for d in dialog if d.get("speaker") == "usr"), "")
        except (json.JSONDecodeError, TypeError):
            situation, first_usr = raw[:400], ""
        mini["items"].append({
            "source_dataset": "ESConv", "source_id": f"es-{p['_idx']}",
            "source_paper": "arXiv 2106.01144 ACL 2021", "source_license": "academic-research-only",
            "scenario": p["_cat"], "language": "en",
            "user_input": (first_usr or situation)[:500], "context": situation,
            "reference": "", "eval_dims": ["支持策略", "共情", "相关性", "helpfulness"],
        })

    # ---------- GoEmotions（ACL2020）----------
    print("[GoEmotions] fetching ...")
    ge = fetch_all("google-research-datasets/go_emotions", "simplified", "validation", 200)
    print(f"  got {len(ge)}")
    ge_added = {"grief": 0, "sadness": 0}
    for r in ge:
        labels = r.get("labels") or []
        # labels 是 ClassLabel index；grief/sadness 在 simplified 配置里的 id
        for cat, lid in [("grief", 7), ("sadness", 18)]:
            if ge_added[cat] < 2 and lid in labels:
                mini["items"].append({
                    "source_dataset": "GoEmotions", "source_id": f"ge-{r.get('id','?')}",
                    "source_paper": "arXiv 2005.00547 ACL 2020", "source_license": "apache-2.0",
                    "scenario": f"情绪识别-{cat}", "language": "en",
                    "user_input": (r.get("text") or "")[:500], "reference": cat,
                    "eval_dims": ["情绪识别"], "expected_label": cat,
                })
                ge_added[cat] += 1
    print(f"  GoEmotions picked: {ge_added}")

    # ---------- CueSupport-MH（Zenodo，危机安全）----------
    print("[CueSupport-MH] downloading Zenodo zip ...")
    try:
        import io
        import zipfile
        zurl = "https://zenodo.org/api/records/19699087/files/trace-mh.zip/content"
        with urllib.request.urlopen(zurl, timeout=120) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        zname = [n for n in z.namelist() if "camera_ready" in n and n.endswith(".json")][0]
        cue_rows = json.loads(z.read(zname))
        print(f"  got {len(cue_rows)}")
        # 看 risk_label / response_label 的实际取值
        from collections import Counter
        print("  risk labels:", dict(Counter(str(r.get("risk_label")) for r in cue_rows).most_common(6)))
        print("  response labels:", dict(Counter(str(r.get("response_label")) for r in cue_rows).most_common(6)))
        for cat, resp in [("危机-supportive", "supportive"), ("危机-harmful", "harmful")]:
            n = 0
            for r in cue_rows:
                rl = str(r.get("risk_label", "")).lower()
                rp = str(r.get("response_label", "")).lower()
                # 高危 + 对应回复标签
                if ("high" in rl or "高" in rl or "severe" in rl) and resp in rp:
                    mini["items"].append({
                        "source_dataset": "CueSupport-MH", "source_id": f"cue-{r.get('id','?')}",
                        "source_paper": "Zenodo 10.5281/zenodo.19699087", "source_license": "cc-by-4.0",
                        "scenario": cat, "language": "en",
                        "user_input": (r.get("post") or "")[:500],
                        "reference": (r.get("response") or "")[:500],
                        "risk_label": r.get("risk_label"), "response_label": r.get("response_label"),
                        "eval_dims": ["危机升级", "回复安全性"],
                    })
                    n += 1
                    if n >= 2:
                        break
            print(f"  {cat}: {n}")
    except Exception as exc:
        print(f"  ! CueSupport-MH failed: {str(exc)[:100]} (标 NOT_FOUND，不补自编题)")

    OUT.write_text(json.dumps(mini, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {len(mini['items'])} items -> {OUT}")
    from collections import Counter
    print("  by source:", dict(Counter(i["source_dataset"] for i in mini["items"])))
    print("  by scenario:", dict(Counter(i["scenario"] for i in mini["items"])))


if __name__ == "__main__":
    main()
