#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dataset 版本化 + 失败 case 带标签回归漏斗（纯离线工具，零 LLM / 零 key）。

补 EVAL_CATEGORIES.md §维度三 两个缺口：
  1) dataset 版本化：snapshot 记录当前评测集的哈希/条数/批次清单/时间，
     追加写 outputs/dataset_versions.jsonl（一行一版）；
  2) 失败 case → 带标签回归漏斗：compare 列出两版本间新增/删除/修改的 case；
     to-regression 从 error_analysis 的带 taxonomy 标签失败 case 导出回归用例，
     每条带 source_id / provenance(authoritative|synthetic-handwritten) 口径，
     并能反查到 dataset 版本（可追溯）。

口径与边界（诚实标注）：
  - 纯离线：只用 Python 标准库，不 import 任何 gateway / LLM 客户端，不读 work/local-test-key.txt；
  - 确定性：同样的输入 + 同样的 ts 产生逐字节一致的快照；重复运行是追加新版本，不覆盖；
  - 不改任何生产代码，不碰 D-070 的 context_compaction 链路；
  - to-regression 默认只回流 verdict=real 的真产品失败（与 eval/error_analysis.py 同口径，
    mismatch=评测集错配不算产品失败，D-041 的 25 条全是 mismatch → 默认导出 0 条是诚实结果）；
    需要把带标签的失败都导出时用 --verdict all，并如实标记 verdict。
  - 口径声明：本工具是 synthetic-handwritten/离线模拟的机制工具，不是真人验证或真模型评测。

用法：
  python eval/dataset_version.py snapshot [--note "..."] [--ts ISO8601]
  python eval/dataset_version.py compare  [--from 1 --to 2] [--out PATH]
  python eval/dataset_version.py to-regression [--verdict real|all] [--version 2] [--out PATH]
  python eval/dataset_version.py --help

所有默认路径都可通过 --batches-dir / --versions / --failures / --out 覆盖，便于离线测试用临时目录。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
PROJECT = BACKEND.parent

BATCH_DIR = HERE / "batches"
VERSIONS_FILE = PROJECT / "outputs" / "dataset_versions.jsonl"
FAILURE_CASES = PROJECT / "outputs" / "error_analysis" / "failure_cases.json"
REGRESSION_OUT = PROJECT / "outputs" / "error_analysis" / "regression_cases.json"

SCHEMA_VERSION = "1.0"
KOUJING = ("离线机制工具（synthetic-handwritten/离线模拟口径）：零 LLM、零 key、零真人验证；"
           "hash 与判定均为确定性本地计算")


# ---------------------------------------------------------------------------
# 纯函数（无副作用，便于构造性离线测试）
# ---------------------------------------------------------------------------
def canonical_hash(obj) -> str:
    """稳定哈希：canonical JSON（key 排序、无 ASCII 转义）+ sha256。"""
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _case_key(batch: str, case_id) -> str:
    return f"{batch}::{case_id}"


def source_id_of(item: dict, batch: str = "") -> str:
    """取 case 的 source_id：优先 item.source_id，再 source.id，再 batch::id。"""
    sid = item.get("source_id")
    if sid:
        return str(sid)
    src = item.get("source")
    if isinstance(src, dict) and src.get("id"):
        return str(src["id"])
    if item.get("id") is not None:
        return _case_key(batch, item["id"])
    return ""


def provenance_of(item: dict) -> str:
    """口径：authoritative / synthetic-handwritten / unknown（不自造权威标签）。"""
    prov = item.get("provenance")
    if isinstance(prov, dict):
        kind = prov.get("kind")
        if isinstance(kind, str) and kind:
            return kind
        if prov.get("synthetic") is True:
            return "synthetic-handwritten"
    src = item.get("source")
    if isinstance(src, dict):
        ds = str(src.get("dataset") or "")
        if ds == "synthetic-handwritten" or "synthetic" in ds.lower():
            return "synthetic-handwritten"
        if ds:
            return "authoritative"
    return "unknown"


def _text_of(item: dict) -> str:
    for key in ("user_input_zh", "user_input", "text", "prompt"):
        val = item.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def flatten_cases(batches_dir: Path) -> dict:
    """把评测批次目录摊平成 {case_key: case_record}；忽略 _ 开头的索引文件。"""
    cases: dict[str, dict] = {}
    if not batches_dir.exists():
        return cases
    for path in sorted(batches_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        batch = str(doc.get("batch") or path.stem)
        for item in doc.get("items") or []:
            if not isinstance(item, dict) or item.get("id") is None:
                continue
            key = _case_key(batch, item["id"])
            cases[key] = {
                "case_key": key,
                "batch": batch,
                "id": str(item["id"]),
                "source_id": source_id_of(item, batch),
                "provenance": provenance_of(item),
                "scene": item.get("scene") or item.get("scenario") or "",
                "case_hash": canonical_hash(item),       # 整条 case（含元信息）
                "text_hash": canonical_hash(_text_of(item)),  # 仅正文，区分内容改动/元信息改动
            }
    return cases


def batch_manifest(batches_dir: Path) -> list:
    """批次清单：每批的条数 + 内容哈希（不依赖文件排版/键顺序）。"""
    out = []
    if not batches_dir.exists():
        return out
    for path in sorted(batches_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        items = doc.get("items") or []
        out.append({
            "batch": str(doc.get("batch") or path.stem),
            "file": path.name,
            "n": len(items),
            "declared_n": doc.get("n"),
            "sha256": canonical_hash(items),
        })
    return out


def make_snapshot(batches_dir: Path, *, note: str = "", ts: str | None = None,
                  version: int | None = None) -> dict:
    """构造一条版本记录（纯函数；不写文件）。ts/version 可注入以保证测试可复现。"""
    cases = flatten_cases(batches_dir)
    digest = canonical_hash({k: v["case_hash"] for k, v in sorted(cases.items())})
    return {
        "schema_version": SCHEMA_VERSION,
        "口径": KOUJING,
        "version": version,
        "ts": ts or datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(batches_dir),
        "n_batches": len({c["batch"] for c in cases.values()}) if cases else 0,
        "n_cases": len(cases),
        "sha256": digest,
        "batch_manifest": batch_manifest(batches_dir),
        "cases": {k: v["case_hash"] for k, v in sorted(cases.items())},
        "case_meta": {k: {m: v[m] for m in ("batch", "id", "source_id", "provenance",
                                             "scene", "text_hash")}
                      for k, v in sorted(cases.items())},
        "note": note or "",
    }


def read_versions(path: Path) -> list:
    """读 jsonl，跳过空行与坏行（坏行不静默吞掉判定：只跳过非 JSON 行）。"""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict):
            out.append(d)
    return out


def append_version(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":")) + "\n")


def next_version(versions: list) -> int:
    nums = [int(v.get("version", 0)) for v in versions
            if str(v.get("version", "")).lstrip("-").isdigit()]
    return max(nums, default=0) + 1


def resolve_version(versions: list, ref: str | int | None) -> dict | None:
    """版本引用：整数=版本号；last/prev/first；-1/-2=倒数第 N；否则按 ts 前缀匹配。"""
    if not versions:
        return None
    if ref is None or ref == "last":
        return versions[-1]
    if ref == "prev":
        return versions[-2] if len(versions) >= 2 else None
    if ref == "first":
        return versions[0]
    if isinstance(ref, int) or (isinstance(ref, str) and ref.lstrip("-").isdigit()):
        n = int(ref)
        if n < 0:
            return versions[n] if len(versions) >= abs(n) else None
        for v in versions:
            if int(v.get("version", -1)) == n:
                return v
        return None
    s = str(ref)
    for v in versions:
        if str(v.get("ts", "")).startswith(s):
            return v
    return None


def diff_versions(v_from: dict, v_to: dict) -> dict:
    """对比两版本：列出新增/删除/修改的 case（纯函数）。"""
    a = v_from.get("cases") or {}
    b = v_to.get("cases") or {}
    added = sorted(set(b) - set(a))
    deleted = sorted(set(a) - set(b))
    common = sorted(set(a) & set(b))
    modified = [k for k in common if a[k] != b[k]]
    ma = v_from.get("case_meta") or {}
    mb = v_to.get("case_meta") or {}
    detail = []
    for k in modified:
        ta = (ma.get(k) or {}).get("text_hash")
        tb = (mb.get(k) or {}).get("text_hash")
        detail.append({"case_key": k,
                       "from_hash": a[k], "to_hash": b[k],
                       "text_changed": bool(ta != tb),
                       "source_id": (mb.get(k) or ma.get(k) or {}).get("source_id", ""),
                       "provenance": (mb.get(k) or ma.get(k) or {}).get("provenance", "")})
    return {
        "口径": KOUJING,
        "from_version": v_from.get("version"), "from_ts": v_from.get("ts"),
        "to_version": v_to.get("version"), "to_ts": v_to.get("ts"),
        "from_sha256": v_from.get("sha256"), "to_sha256": v_to.get("sha256"),
        "n_added": len(added), "n_deleted": len(deleted), "n_modified": len(modified),
        "added": added, "deleted": deleted, "modified": modified,
        "modified_detail": detail,
        "same_content": (v_from.get("sha256") == v_to.get("sha256")),
    }


def export_regression(failures: dict, versions: list, *,
                      verdict_filter: str = "real", version_ref="last") -> dict:
    """失败 case → 带 taxonomy 标签的回归用例（纯函数）。

    verdict_filter="real" 只回流真产品失败；"all" 导出全部带标签失败（仍保留 verdict 字段）。
    能匹配到 dataset 版本快照时补 source_id / provenance / dataset_version/dataset_link。
    """
    chosen = resolve_version(versions, version_ref)
    meta = (chosen or {}).get("case_meta") or {}
    cases_meta = (chosen or {}).get("cases") or {}
    cases = failures.get("cases") or []
    out_cases, n_skipped = [], 0
    for c in cases:
        if not isinstance(c, dict) or (c.get("id") is None):
            continue
        verdict = str(c.get("verdict") or "unannotated")
        if verdict_filter != "all" and verdict != verdict_filter:
            n_skipped += 1
            continue
        batch = str(c.get("batch") or "")
        cid = str(c["id"])
        local_key = _case_key(batch, cid)
        linked = local_key in meta
        m = meta.get(local_key) or {}
        # taxonomy：人工裁定 category 优先，未标注则用 auto_category（并如实标注来源）
        cat = c.get("category")
        cat_from = "manual"
        if not cat:
            cat = c.get("auto_category")
            cat_from = "auto"
        out_cases.append({
            "source_id": m.get("source_id") or c.get("source_id") or local_key,
            "failure_id": cid,
            "batch": batch,
            "case_key": local_key,
            "dataset_version": chosen.get("version") if chosen else None,
            "dataset_link": "linked" if linked else "unresolved",
            "provenance": m.get("provenance") or "unknown",
            "scenario": c.get("scene") or m.get("scene") or "",
            "taxonomy_category": cat or "other",
            "taxonomy_source": cat_from,
            "auto_category": c.get("auto_category") or "",
            "verdict": verdict,
            "note": c.get("note") or "",
            "content_hash": cases_meta.get(local_key) or "",
            "text": c.get("text") or "",
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "口径": KOUJING + "；导出口径=error_analysis taxonomy 标签，verdict=" + verdict_filter,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_failure_file": "outputs/error_analysis/failure_cases.json",
        "dataset_version": chosen.get("version") if chosen else None,
        "dataset_ts": chosen.get("ts") if chosen else None,
        "verdict_filter": verdict_filter,
        "n_cases": len(out_cases),
        "n_skipped_by_filter": n_skipped,
        "n_unresolved_link": sum(1 for c in out_cases if c["dataset_link"] == "unresolved"),
        "taxonomy": failures.get("taxonomy") or {},
        "cases": out_cases,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _add_paths(ap: argparse.ArgumentParser, *, with_out: bool = False) -> None:
    ap.add_argument("--batches-dir", type=Path, default=BATCH_DIR,
                    help=f"评测批次目录（默认 {BATCH_DIR}）")
    ap.add_argument("--versions", type=Path, default=VERSIONS_FILE,
                    help=f"版本 jsonl（默认 {VERSIONS_FILE}）")


def cmd_snapshot(args) -> int:
    versions = read_versions(args.versions)
    rec = make_snapshot(args.batches_dir, note=args.note or "", ts=args.ts)
    rec["version"] = next_version(versions)
    append_version(args.versions, rec)
    print(f"[snapshot] v{rec['version']} ts={rec['ts']} sha256={rec['sha256'][:16]}… "
          f"batches={len(rec['batch_manifest'])} cases={rec['n_cases']}")
    print(f"[snapshot] 追加一行 -> {args.versions}")
    return 0


def cmd_compare(args) -> int:
    versions = read_versions(args.versions)
    if len(versions) < 2 and args.from_ref is None and args.to_ref is None:
        print(f"[compare] 至少需要两个版本才能对比（当前 {len(versions)} 个），先跑两次 snapshot")
        return 2
    if args.from_ref is None and args.to_ref is None:
        va, vb = versions[-2], versions[-1]
    else:
        va = resolve_version(versions, args.from_ref if args.from_ref is not None else "first")
        vb = resolve_version(versions, args.to_ref if args.to_ref is not None else "last")
    if va is None or vb is None:
        print("[compare] 版本引用无法解析；现有版本："
              + ", ".join(f"v{v.get('version')}@{v.get('ts')}" for v in versions))
        return 2
    diff = diff_versions(va, vb)
    print(f"[compare] v{va.get('version')} -> v{vb.get('version')}  "
          f"sha256 {str(va.get('sha256'))[:12]}… -> {str(vb.get('sha256'))[:12]}…  "
          f"{'内容一致' if diff['same_content'] else '内容不同'}")
    print(f"  新增 {diff['n_added']} / 删除 {diff['n_deleted']} / 修改 {diff['n_modified']}")
    for label, key in (("新增", "added"), ("删除", "deleted"), ("修改", "modified")):
        for ck in diff[key][:20]:
            print(f"    [{label}] {ck}")
        if diff[key][20:]:
            print(f"    ... 另 {len(diff[key]) - 20} 条")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[compare] diff -> {args.out}")
    return 0


def cmd_to_regression(args) -> int:
    if not args.failures.exists():
        print(f"[to-regression] 找不到失败 case 文件：{args.failures}")
        return 2
    failures = json.loads(args.failures.read_text(encoding="utf-8"))
    versions = read_versions(args.versions)
    reg = export_regression(failures, versions,
                            verdict_filter=args.verdict, version_ref=args.version)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[to-regression] verdict={args.verdict}: {reg['n_cases']} 条回归用例 -> {args.out}")
    print(f"  过滤跳过 {reg['n_skipped_by_filter']} 条；dataset v{reg['dataset_version']} 链接"
          f"成功 {reg['n_cases'] - reg['n_unresolved_link']}/{reg['n_cases']}；"
          f"口径 {reg['口径']}")
    if reg["n_cases"] == 0:
        print("  [提示] 0 条属实：error_analysis 若全为 verdict=mismatch（评测集错配），"
              "按口径不回流；要导出全部带标签失败请加 --verdict all。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="dataset_version.py",
        description="dataset 版本化 + 失败 case 带标签回归漏斗（纯离线，零 LLM/零 key）")
    sub = ap.add_subparsers(dest="cmd")

    s = sub.add_parser("snapshot", help="记录当前 dataset 哈希/条数/批次清单/时间，追加一行版本")
    _add_paths(s)
    s.add_argument("--note", default="")
    s.add_argument("--ts", default=None, help="注入 ISO8601 时间戳（仅测试复现用；默认当前 UTC）")
    s.set_defaults(func=cmd_snapshot)

    c = sub.add_parser("compare", help="对比两个版本，列出新增/删除/修改的 case")
    _add_paths(c)
    c.add_argument("--from", dest="from_ref", default=None,
                   help="来源版本：版本号(如 1)、last/prev/first、ts 前缀（默认倒数第二版）")
    c.add_argument("--to", dest="to_ref", default=None,
                   help="目标版本（默认最后一版）")
    c.add_argument("--out", type=Path, default=None, help="可选：把 diff 写成 JSON")
    c.set_defaults(func=cmd_compare)

    r = sub.add_parser("to-regression", help="带 taxonomy 标签失败 case → 可追溯 dataset 的回归用例")
    _add_paths(r)
    r.add_argument("--failures", type=Path, default=FAILURE_CASES,
                   help=f"失败 case 文件（默认 {FAILURE_CASES}）")
    r.add_argument("--verdict", choices=["real", "all"], default="real",
                   help="默认只回流 verdict=real；all 导出全部带标签失败（仍保留 verdict）")
    r.add_argument("--version", dest="version", default="last",
                   help="关联的 dataset 版本（默认 last）")
    r.add_argument("--out", type=Path, default=REGRESSION_OUT,
                   help=f"回归用例输出（默认 {REGRESSION_OUT}）")
    r.set_defaults(func=cmd_to_regression)
    return ap


def main(argv: list | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "cmd", None):
        ap.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
