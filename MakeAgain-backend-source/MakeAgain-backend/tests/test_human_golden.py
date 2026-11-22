"""H3 真人金标标注工作流的行为测试（离线、确定性，不碰真实模型/key）。

口径：这里所有"人工标签"都是构造标签干跑（annotator 含 PIPELINE_SELFTEST_CONSTRUCTED），
只验证 export/import/consistency 流水线正确性与"不伪造真人"的诚实边界。
真实 judge 标签来自已落盘的输出产物 outputs/benchmarks/judge_consistency.json（真实 hy4 跑出来的），
本测试不重新调用模型、不需要 key。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "eval"))

import pytest  # noqa: E402

import human_golden as hg  # noqa: E402


def _export(tmp_path: Path, blind: bool = False) -> dict:
    return hg.cmd_export(SimpleNamespace(
        golden=str(hg.GOLDEN), judge_labels=None, out=str(tmp_path / "export"),
        format="both", blind=blind,
    ))


def _dry_run_annotation(tmp_path: Path, flip_id: str | None = None) -> Path:
    """用 constructed_verdict 当 human_label 生成构造标签干跑标注文件（非真人）。"""
    rows = [json.loads(x) for x in (tmp_path / "export" / "annotation_export.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    path = tmp_path / "annotation_constructed.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            label = r["constructed_verdict"]
            if r["id"] == flip_id:  # 故意制造一处分歧，验证分歧清单
                label = "fail" if label == "pass" else "pass"
                r["human_rationale"] = "自检故意翻转：验证分歧清单能列出具体 case"
            r["human_label"] = label
            r["annotator"] = "PIPELINE_SELFTEST_CONSTRUCTED(构造标签,非真人)"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _import(tmp_path: Path, ann: Path, golden: Path | None = None) -> dict:
    return hg.cmd_import(SimpleNamespace(
        golden=str(golden or hg.GOLDEN), annotation_file=str(ann),
        out=str(tmp_path / "labeled.json"), in_place=False,
    ))


def _consistency(tmp_path: Path, golden: Path) -> dict:
    return hg.cmd_consistency(SimpleNamespace(
        golden=str(golden), judge_labels=None,
        out=str(tmp_path / "consistency.json"), curve=str(tmp_path / "curve.jsonl"),
    ))


# ------------------------------------------------------------------ H3 金标结构
def test_golden_is_constructed_placeholder_not_real_human():
    g = hg.load_golden()
    items = g["items"]
    assert g["schema_version"] == "1.0"
    assert g["status"] == "PLACEHOLDER_AWAITING_REAL_HUMAN_ANNOTATION"
    assert "非真人" in g["caliber"]
    assert g["label_caliber"] == "NO_HUMAN_LABELS_YET"
    assert g["annotators"]["current"] == []
    assert g["annotators"]["named_final_decision_maker"] is None
    assert len(items) == 30
    assert all(i["verdict"] == "constructed" for i in items)
    assert all("非真人" in i["verdict_caliber"] for i in items)
    assert all(i["human_label"] is None and i["status"] == "pending" for i in items)
    assert all(i["rubric"] for i in items)
    # 失败样本必须保留（Airbnb: golden MUST include bad examples）
    n_fail = sum(i["constructed_verdict"] == "fail" for i in items)
    assert n_fail == g["label_policy"]["failure_sample_count"] == 14
    assert g["label_policy"]["recommended_n_range"] == [50, 100]
    assert "30" in g["label_policy"]["gap_note"]
    # 主流出处 URL + 日期写进金标
    assert any(r["org"] == "Airbnb" and r["date"] == "2026-07-28" for r in g["research_basis"])
    assert any(r["org"].startswith("Netflix") and "netflixtechblog.com" in r["url"]
               for r in g["research_basis"])


# ------------------------------------------------------------------ V1 export
def test_export_writes_jsonl_and_csv_with_rubric_empty_labels_and_judge(tmp_path):
    summary = _export(tmp_path)
    assert summary["n_rows"] == 30
    assert summary["format"] == "both"
    jl = tmp_path / "export" / "annotation_export.jsonl"
    cs = tmp_path / "export" / "annotation_export.csv"
    assert jl.is_file() and cs.is_file()
    rows = [json.loads(x) for x in jl.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 30
    for r in rows:
        assert r["rubric"] and r["user_input"] and r["reply_to_label"]
        assert r["human_label"] == "" and r["human_rationale"] == "" and r["annotator"] == ""
        assert r["status"] == "pending"
        assert r["verdict"] == "constructed"
        assert r["judge_verdict"] in ("pass", "fail")  # 默认带真实 judge 结论
    assert {"pass", "fail"} <= {r["constructed_verdict"] for r in rows}
    csv_text = cs.read_text(encoding="utf-8")
    assert "human_label" in csv_text.splitlines()[0] and "rubric" in csv_text.splitlines()[0]


def test_export_blind_hides_judge_verdict(tmp_path):
    _export(tmp_path, blind=True)
    rows = [json.loads(x) for x in (tmp_path / "export" / "annotation_export.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    assert all(r["judge_verdict"] == "" and r["judge_reason"] == "" for r in rows)


# ------------------------------------------------------------------ V2 import + consistency
def test_import_then_consistency_computes_real_kappa_with_constructed_labels(tmp_path):
    _export(tmp_path)
    ann = _dry_run_annotation(tmp_path)
    imp = _import(tmp_path, ann)
    assert imp["n_applied"] == 30
    assert imp["label_caliber"] == "CONSTRUCTED_DRY_RUN_NON_HUMAN"
    # 默认写副本，源金标未被改动
    assert hg.load_golden()["label_caliber"] == "NO_HUMAN_LABELS_YET"
    labeled = json.loads((tmp_path / "labeled.json").read_text(encoding="utf-8"))
    assert all(i["human_label"] in ("pass", "fail") for i in labeled["items"])

    res = _consistency(tmp_path, tmp_path / "labeled.json")
    assert res["note"] == "OK"
    assert res["n_labeled"] == 30 and res["n_judged_overlap"] == 30
    assert res["caliber"] == "CONSTRUCTED_DRY_RUN_NON_HUMAN"
    assert "非真人" in res["caliber_note"]
    k = res["overall"]["kappa"]
    assert isinstance(k, (int, float)) and 0.0 < k < 1.0  # 真实算出的数字，非硬编码
    assert isinstance(res["overall"]["krippendorff_alpha"], (int, float))
    assert 0.0 < res["overall"]["agreement"] < 1.0
    c = res["overall"]["confusion"]
    assert c["tp"] + c["fp"] + c["tn"] + c["fn"] == 30
    assert len(res["per_dimension"]) == 4
    assert (tmp_path / "consistency.json").is_file()
    assert (tmp_path / "curve.jsonl").read_text(encoding="utf-8").strip()
    # κ 与逐条独立重算一致（防止口径错配）
    judges, _ = hg.load_judge_labels()
    pairs = [(i["human_label"], judges[i["id"]]["label"]) for i in labeled["items"]]
    assert res["overall"]["kappa"] == hg.cohens_kappa(pairs)["kappa"]


# ------------------------------------------------------------------ V3 分歧清单
def test_disagreement_list_contains_concrete_case(tmp_path):
    _export(tmp_path)
    flip = "gc-001"  # 构造 pass，故意翻成 fail，与真实 judge(pass) 形成分歧
    ann = _dry_run_annotation(tmp_path, flip_id=flip)
    _import(tmp_path, ann)
    res = _consistency(tmp_path, tmp_path / "labeled.json")
    ids = [d["id"] for d in res["disagreements"]]
    assert flip in ids
    case = next(d for d in res["disagreements"] if d["id"] == flip)
    assert case["source_id"] and case["rubric"] and case["user_input"] and case["reply_to_label"]
    assert case["judge_verdict"] == "pass" and case["human_label"] == "fail"
    assert case["human_rationale"] == "自检故意翻转：验证分歧清单能列出具体 case"
    assert case["resolution"] == "PENDING_HUMAN_ADJUDICATION"
    assert "constructed" in case["constructed_verdict"] or case["constructed_verdict"] in ("pass", "fail")
    assert res["per_dimension"][case["dimension"]]["n_disagreements"] >= 1


# ------------------------------------------------------------------ 诚实边界
def test_no_human_labels_never_invents_kappa(tmp_path):
    res = _consistency(tmp_path, hg.GOLDEN)
    assert res["note"] == "NO_HUMAN_LABELS"
    assert res["n_labeled"] == 0 and res["n_judged_overlap"] == 0
    assert res["overall"]["kappa"] is None
    assert res["disagreements"] == []
    assert "任何人工标签" in res["caliber_note"]


def test_import_rejects_bad_rows_without_writing(tmp_path):
    _export(tmp_path)
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"id": "gc-001", "human_label": "pass"}, ensure_ascii=False) + "\n"
                 + json.dumps({"id": "NOT-EXIST", "human_label": "pass"}, ensure_ascii=False) + "\n"
                 + json.dumps({"id": "gc-002", "human_label": "maybe"}, ensure_ascii=False) + "\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="整批拒绝"):
        _import(tmp_path, p)
    assert not (tmp_path / "labeled.json").exists()


def test_blank_labels_are_skipped_not_guessed(tmp_path):
    _export(tmp_path)
    p = tmp_path / "partial.jsonl"
    p.write_text(json.dumps({"id": "gc-001", "human_label": "pass",
                             "annotator": "PIPELINE_SELFTEST_CONSTRUCTED(构造标签,非真人)"},
                            ensure_ascii=False) + "\n"
                 + json.dumps({"id": "gc-002", "human_label": "", "annotator": ""}, ensure_ascii=False) + "\n",
                 encoding="utf-8")
    imp = _import(tmp_path, p)
    assert imp["n_applied"] == 1 and imp["n_skipped_blank"] == 1
    labeled = json.loads((tmp_path / "labeled.json").read_text(encoding="utf-8"))
    by_id = {i["id"]: i for i in labeled["items"]}
    assert by_id["gc-001"]["status"] == "labeled"
    assert by_id["gc-002"]["status"] == "pending" and by_id["gc-002"]["human_label"] is None


# ------------------------------------------------------------------ 统计单元
def test_kappa_alpha_known_values_and_label_norm():
    perfect = hg.cohens_kappa([("pass", "pass")] * 10 + [("fail", "fail")] * 10)
    assert perfect["kappa"] == 1.0 and perfect["agreement"] == 1.0
    assert hg.krippendorff_alpha_binary([("pass", "pass")] * 10 + [("fail", "fail")] * 10)["alpha"] == 1.0
    chance = hg.cohens_kappa([("pass", "pass")] * 6 + [("pass", "fail")] * 4)
    assert chance["kappa"] == 0.0 and chance["agreement"] == 0.6
    assert hg.krippendorff_alpha_binary([("pass", "pass")] * 6 + [("pass", "fail")] * 4)["alpha"] < 0
    undefined = hg.cohens_kappa([("pass", "pass")] * 5)
    assert undefined["kappa"] is None and undefined["note"] == "UNDEFINED_PE_EQ_1"
    assert hg._norm_label(True) == "pass" and hg._norm_label("不通过") == "fail"
    assert hg._norm_label("garbage") is None


# ------------------------------------------------------------------ 指南含出处
def test_guide_documents_research_urls_and_honest_status():
    text = (hg.HERE / "HUMAN_GOLDEN_GUIDE.md").read_text(encoding="utf-8")
    assert "medium.com/airbnb-engineering" in text and "2026-07-28" in text
    assert "netflixtechblog.com" in text and "2026-10" in text
    assert "rss.com/podcasts/data-science-in-the-wild" in text and "2026-09-15" in text
    assert "没有任何真人标注" in text
    assert "hard stop" in text and "criteria drift" in text
    assert "50" in text and "100" in text
