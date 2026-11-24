# -*- coding: utf-8 -*-
"""dataset 版本化 + 失败 case 回归漏斗的离线测试（口径：synthetic-handwritten 离线模拟）。

全部使用 tmp_path 临时目录/临时文件，绝不写 outputs/；零 LLM、零 key、零真人数据。
覆盖：snapshot 追加/确定性、compare 增删改、to-regression 标签+口径+可追溯、--help。
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND / "eval"))

MODULE = BACKEND / "eval" / "dataset_version.py"


# --------------------------------------------------------------------------- helpers
def _item(cid, text, *, synthetic=False, scene="分手"):
    if synthetic:
        prov = {"kind": "synthetic-handwritten", "synthetic": True,
                "dataset": "synthetic-handwritten"}
        src = {"dataset": "synthetic-handwritten", "id": cid}
        sid = f"syn::{cid}"
    else:
        prov = {"kind": "authoritative", "synthetic": False, "dataset": "CPsyCoun"}
        src = {"dataset": "CPsyCoun", "id": cid}
        sid = f"CPsyCoun:train:{cid}"
    return {"id": cid, "source_id": sid, "scene": scene, "user_input_zh": text,
            "source": src, "provenance": prov}


def _write_batch(batches, name, items):
    doc = {"batch": name, "source": "CPsyCoun", "n": len(items), "items": items}
    (batches / f"{name}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                                          encoding="utf-8")


def _seed_batches(tmp_path):
    batches = tmp_path / "batches"
    batches.mkdir()
    _write_batch(batches, "01_共情", [
        _item("cp-1", "我们分手了，我很难过"),
        _item("cp-2", "猫走了，我很想念它", synthetic=True, scene="宠物离世"),
    ])
    _write_batch(batches, "02_情绪", [
        _item("ge-1", "我今天真的很高兴", scene="情绪-joy"),
    ])
    return batches


def _run_cli(*args):
    return subprocess.run([sys.executable, str(MODULE), *[str(a) for a in args]],
                          cwd=str(BACKEND), capture_output=True, text=True,
                          encoding="utf-8")


def _snap(batches, versions, ts, note=""):
    p = _run_cli("snapshot", "--batches-dir", batches, "--versions", versions,
                 "--ts", ts, "--note", note)
    assert p.returncode == 0, p.stdout + p.stderr
    return p


# --------------------------------------------------------------------------- tests
def test_help_lists_three_subcommands():
    p = _run_cli("--help")
    assert p.returncode == 0, p.stderr
    for cmd in ("snapshot", "compare", "to-regression"):
        assert cmd in p.stdout, f"--help 缺子命令 {cmd}"
    assert "零 LLM" in p.stdout


def test_canonical_hash_is_key_order_insensitive_and_content_sensitive():
    import dataset_version as dv
    a = {"b": 1, "a": {"y": 2, "x": 3}}
    b = {"a": {"x": 3, "y": 2}, "b": 1}
    assert dv.canonical_hash(a) == dv.canonical_hash(b)
    c = dict(a); c["b"] = 2
    assert dv.canonical_hash(a) != dv.canonical_hash(c)


def test_provenance_and_source_id_extraction():
    import dataset_version as dv
    auth = _item("cp-1", "x")
    syn = _item("pet-1", "y", synthetic=True)
    unknown = {"id": "u-1"}
    assert dv.provenance_of(auth) == "authoritative"
    assert dv.provenance_of(syn) == "synthetic-handwritten"
    assert dv.provenance_of(unknown) == "unknown"
    assert dv.source_id_of(auth, "01_共情") == "CPsyCoun:train:cp-1"
    assert dv.source_id_of(unknown, "01_共情") == "01_共情::u-1"


def test_snapshot_appends_one_line_per_version_and_is_deterministic(tmp_path):
    import dataset_version as dv
    batches, versions = _seed_batches(tmp_path), tmp_path / "versions.jsonl"
    _snap(batches, versions, "2026-11-24T00:00:00+00:00", note="first")
    _snap(batches, versions, "2026-11-24T00:00:01+00:00", note="second")

    lines = [ln for ln in versions.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2, "重复运行必须追加而不是覆盖（一行一版）"
    v1, v2 = (json.loads(ln) for ln in lines)
    assert (v1["version"], v2["version"]) == (1, 2)
    assert v1["ts"] == "2026-11-24T00:00:00+00:00"
    assert v1["sha256"] == v2["sha256"], "同一数据集两次快照哈希必须一致（可重复）"
    assert v1["n_cases"] == 3
    assert len(v1["batch_manifest"]) == 2
    for field in ("sha256", "n_cases", "n_batches", "batch_manifest", "ts"):
        assert field in v1, f"版本记录缺字段 {field}"
    # 纯函数同输入同 ts 逐字节一致（version 由调用方注入）
    r1 = dv.make_snapshot(batches, ts="2026-11-24T00:00:00+00:00", version=1)
    r2 = dv.make_snapshot(batches, ts="2026-11-24T00:00:00+00:00", version=1)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)
    # 合成条目的口径必须落到版本记录里（可追溯）
    assert v1["case_meta"]["01_共情::cp-2"]["provenance"] == "synthetic-handwritten"
    assert v1["case_meta"]["01_共情::cp-1"]["source_id"] == "CPsyCoun:train:cp-1"


def test_compare_lists_added_deleted_modified(tmp_path):
    batches, versions = _seed_batches(tmp_path), tmp_path / "versions.jsonl"
    diff_file = tmp_path / "diff.json"
    _snap(batches, versions, "2026-11-24T00:00:00+00:00")

    # 制造三类变更：删 cp-2、加 ge-new、改 cp-1 正文
    doc = json.loads((batches / "01_共情.json").read_text(encoding="utf-8"))
    doc["items"] = [_item("cp-1", "我们分手了，我很难过（改）", scene="分手")]
    (batches / "01_共情.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    doc2 = json.loads((batches / "02_情绪.json").read_text(encoding="utf-8"))
    doc2["items"].append(_item("ge-new", "又一天过去了", scene="情绪-sadness"))
    (batches / "02_情绪.json").write_text(json.dumps(doc2, ensure_ascii=False), encoding="utf-8")
    _snap(batches, versions, "2026-11-24T00:00:01+00:00")

    p = _run_cli("compare", "--versions", versions, "--from", "1", "--to", "2",
                 "--out", diff_file)
    assert p.returncode == 0, p.stdout + p.stderr
    diff = json.loads(diff_file.read_text(encoding="utf-8"))
    assert diff["deleted"] == ["01_共情::cp-2"]
    assert diff["added"] == ["02_情绪::ge-new"]
    assert diff["modified"] == ["01_共情::cp-1"]
    assert diff["n_added"] == diff["n_deleted"] == diff["n_modified"] == 1
    detail = diff["modified_detail"][0]
    assert detail["text_changed"] is True
    assert detail["source_id"] == "CPsyCoun:train:cp-1"
    assert diff["same_content"] is False


def test_diff_versions_pure_no_file_io():
    import dataset_version as dv
    a = {"version": 1, "sha256": "a", "cases": {"b::1": "h1", "b::2": "h2"},
         "case_meta": {"b::1": {"text_hash": "t1"}}}
    b = {"version": 2, "sha256": "b", "cases": {"b::1": "H1", "b::3": "h3"},
         "case_meta": {"b::1": {"text_hash": "t1"}}}
    d = dv.diff_versions(a, b)
    assert d["added"] == ["b::3"] and d["deleted"] == ["b::2"]
    assert d["modified"] == ["b::1"]
    assert d["modified_detail"][0]["text_changed"] is False, "仅元信息变化不算正文改动"


def test_to_regression_filters_excludes_mismatch_and_traces_dataset(tmp_path):
    batches, versions = _seed_batches(tmp_path), tmp_path / "versions.jsonl"
    _snap(batches, versions, "2026-11-24T00:00:00+00:00")

    failures = tmp_path / "failure_cases.json"
    failures.write_text(json.dumps({
        "taxonomy": {"judge_low": "judge低分", "emotion_boundary": "情绪边界"},
        "cases": [
            {"id": "cp-1", "batch": "01_共情", "verdict": "real", "category": "judge_low",
             "auto_category": "judge_low", "note": "真失败", "text": "我们分手了，我很难过"},
            {"id": "cp-2", "batch": "01_共情", "verdict": "mismatch", "category": "emotion_boundary",
             "auto_category": "emotion_boundary", "note": "错配不回流", "text": "猫走了"},
            {"id": "not-in-dataset", "batch": "99_未知", "verdict": "real", "category": "judge_low",
             "auto_category": "judge_low", "note": "未入库", "text": "孤立 case"},
        ],
    }, ensure_ascii=False), encoding="utf-8")

    out = tmp_path / "regression.json"
    p = _run_cli("to-regression", "--failures", failures, "--versions", versions,
                 "--version", "1", "--out", out)
    assert p.returncode == 0, p.stdout + p.stderr
    reg = json.loads(out.read_text(encoding="utf-8"))
    assert reg["n_cases"] == 2, "只回流 verdict=real（mismatch 是评测集错配）"
    assert reg["n_skipped_by_filter"] == 1
    assert reg["dataset_version"] == 1
    ids = {c["failure_id"] for c in reg["cases"]}
    assert ids == {"cp-1", "not-in-dataset"}
    linked = {c["failure_id"]: c for c in reg["cases"]}
    assert linked["cp-1"]["dataset_link"] == "linked"
    assert linked["cp-1"]["source_id"] == "CPsyCoun:train:cp-1"
    assert linked["cp-1"]["provenance"] == "authoritative"
    assert linked["cp-1"]["taxonomy_category"] == "judge_low"
    assert linked["cp-1"]["taxonomy_source"] == "manual"
    assert linked["not-in-dataset"]["dataset_link"] == "unresolved"
    assert reg["n_unresolved_link"] == 1

    # --verdict all 导出全部带标签失败，但保留 verdict 字段供人工区分
    out_all = tmp_path / "regression_all.json"
    p2 = _run_cli("to-regression", "--failures", failures, "--versions", versions,
                  "--verdict", "all", "--out", out_all)
    assert p2.returncode == 0, p2.stdout + p2.stderr
    reg_all = json.loads(out_all.read_text(encoding="utf-8"))
    assert reg_all["n_cases"] == 3
    assert {c["verdict"] for c in reg_all["cases"]} == {"real", "mismatch"}


def test_to_regression_synthetic_provenance_is_carried(tmp_path):
    """回归用例必须带 synthetic-handwritten 口径（自造语料不冒充真人/权威）。"""
    import dataset_version as dv
    batches, versions = _seed_batches(tmp_path), tmp_path / "versions.jsonl"
    _snap(batches, versions, "2026-11-24T00:00:00+00:00")
    vs = dv.read_versions(versions)
    failures = {"cases": [
        {"id": "cp-2", "batch": "01_共情", "verdict": "real",
         "category": "judge_low", "auto_category": "judge_low", "text": "猫走了"},
    ]}
    reg = dv.export_regression(failures, vs)
    case = reg["cases"][0]
    assert case["provenance"] == "synthetic-handwritten"
    assert case["source_id"] == "syn::cp-2"
    assert case["dataset_link"] == "linked"


def test_compare_needs_two_versions_exits_nonzero(tmp_path):
    versions = tmp_path / "versions.jsonl"
    p = _run_cli("compare", "--versions", versions)
    assert p.returncode == 2
    assert "至少需要两个版本" in p.stdout
