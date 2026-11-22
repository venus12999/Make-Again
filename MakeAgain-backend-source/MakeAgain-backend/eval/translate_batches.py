#!/usr/bin/env python3
"""把英文批次的用户输入翻译成中文（hy4-flash-free），保留原文与 source_id 可追溯。

翻译后每条加 user_input_zh；runner 默认用 zh 测（中文产品），可切回原文测语言边界。
key 从 work/local-test-key.txt 读，不打印不入库。
用法：python eval/translate_batches.py [--batch 02_支持策略_ESConv]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))
BATCHDIR = HERE / "batches"


def load_key():
    p = BACKEND.parent / "work" / "local-test-key.txt"
    conf = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.startswith("#"):
            k, _, v = line.partition("=")
            conf[k.strip()] = v.strip()
    return conf


def setup(conf):
    import uuid
    os.environ.update(
        LLM_BASE_URL=conf["BASE_URL"], LLM_API_KEY=conf["KEY"],
        LLM_FAST_MODEL="hy4-flash-free", LLM_MODEL="hy4",
        MOCK_LLM="false", ENVIRONMENT="development",
        LLM_EXTRA_HEADERS=json.dumps({
            "x-opencode-session": "translate-" + uuid.uuid4().hex[:8],
            "User-Agent": "make-again-translate/1.0",
        }),
        DATABASE_URL="sqlite:///:memory:", MEDIA_SIGNING_KEY="x" * 40,
        STORAGE_LOCAL_DIR=str(BACKEND / "work" / "mini-media"),
        PUBLIC_BASE_URL="http://testserver",
    )


def translate(client, text: str) -> str:
    prompt = f"""把下面这段英文求助者的倾诉翻译成中文。要求：
- 保留口语感、情绪强度和第一人称
- 人名/专有名词保留英文或音译
- 只输出中文译文，不要解释

英文：{text[:600]}

中文："""
    r = client.chat([{"role": "user", "content": prompt}], temperature=0.0,
                    model=client.settings.llm_fast_model, max_tokens=1500)
    out = (r.get("content") or "").strip()
    return out or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default=None)
    args = ap.parse_args()

    conf = load_key()
    setup(conf)
    from gateway.client import LLMClient
    client = LLMClient()

    files = ([BATCHDIR / f"{args.batch}.json"] if args.batch else
             sorted(BATCHDIR.glob("*.json")))
    for p in files:
        if p.name.startswith("_"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        to_translate = [x for x in d["items"] if x.get("language") == "en" and not x.get("user_input_zh")]
        if not to_translate:
            print(f"[skip] {p.name}: no items to translate")
            continue
        print(f"\n[{p.name}] translating {len(to_translate)} items ...")
        for i, item in enumerate(to_translate, 1):
            zh = translate(client, item["user_input"])
            item["user_input_zh"] = zh
            print(f"  {i}/{len(to_translate)}: {zh[:60]}")
            time.sleep(0.2)
        # 翻译 reference 如有
        for item in d["items"]:
            if item.get("language") == "en" and item.get("reference") and not item.get("reference_zh"):
                item["reference_zh"] = translate(client, item["reference"])
        p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] {p.name}")


if __name__ == "__main__":
    main()
