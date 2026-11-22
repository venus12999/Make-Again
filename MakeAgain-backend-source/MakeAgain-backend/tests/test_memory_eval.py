"""E3沉淀 + E4调用 统一记忆评测的行为验收（全离线、不烧 key、不联网）。

验收项对应任务 E34-V1~V5：
- V1 统一入口跑出现有 Recall@K/MRR 真实数字：与固化基线
  outputs/golden-recall-v2.json 四配置逐项对齐（不依赖 git show）。
- V2 E3/E4 两段都有 case：批次 56 条（E3 抽取36+冲突2 / E4 召回18），
  入口输出同时含 E3_沉淀 与 E4_调用 两段。
- V3 负例不召回有真实数字：query_intent 关门注入率 1.0 → 开门注入率 0.0，
  q-16  retrieved=[] 且正例 Recall@K 不下降；E3 冲突修正真调产品代码。
- V4 ci_runner 兼容：构造临时 batches 目录（_index.json 登记本批次）后 9 项检查全 PASS，
  证明批次结构/source_id 可追溯；共享 _index.json 的登记由 lead 集成（见报告）。
- V5 本文件随全量 pytest 无回归。

口径：全部条目来自 eval/recall_corpus.json（synthetic-handwritten，非真实用户）；
E3 抽取指标是 golden 事实的"词面被支撑率"（自造口径，非产品抽取准确率），
paraphrase 有假阴性，逐条可追，不粉饰全绿。key 不进仓库、CLI 强制 MOCK_LLM。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "eval"))

import ci_runner  # noqa: E402
import memory_eval as me  # noqa: E402

BATCH = BACKEND / "eval" / "batches" / "E3_沉淀_E4_调用.json"
RUNNER = BACKEND / "eval" / "memory_eval.py"
BASELINE = REPO / "outputs" / "golden-recall-v2.json"


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=str(BACKEND), capture_output=True, text=True, encoding="utf-8", timeout=300,
    )


@pytest.fixture(scope="module")
def result(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("e34") / "result.json"
    proc = _run_cli("--out", str(out))
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr[-3000:]
    return json.loads(out.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ V1
def test_v1_unified_entry_reproduces_existing_recall_numbers(result):
    configs = result["E4_调用"]["configs"]
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    for key in ("dual0_sem0", "dual0_sem1", "dual1_sem0", "dual1_sem1"):
        got, exp = configs[key], baseline[key]
        assert got["status"] == "RUN"
        assert got["Recall@K"] == exp["Recall@K"], (key, got["Recall@K"], exp["Recall@K"])
        assert got["MRR"] == exp["MRR"], (key, got["MRR"], exp["MRR"])
    print("\n[V1真实数字] dual0_sem0 R@K=%.4f MRR=%.4f → dual1_sem1 R@K=%.4f MRR=%.4f；"
          "四配置与固化基线 outputs/golden-recall-v2.json 逐项相等" % (
              configs["dual0_sem0"]["Recall@K"], configs["dual0_sem0"]["MRR"],
              configs["dual1_sem1"]["Recall@K"], configs["dual1_sem1"]["MRR"]))


# ------------------------------------------------------------------ V2
def test_v2_both_stages_have_case_coverage(result):
    batch = me.load_batch(BATCH)
    stages = {it["stage"] for it in batch["items"]}
    kinds = {(it["stage"], it["kind"]) for it in batch["items"]}
    assert stages == {"E3", "E4"}
    assert ("E3", "extraction") in kinds and ("E3", "conflict") in kinds
    assert ("E4", "recall") in kinds
    assert set(result["stages"]) == {"E3", "E4"}
    e3, e4 = result["E3_沉淀"], result["E4_调用"]
    assert e3["n_cases"] >= 38 and e4["n_queries"] >= 18
    assert e3["extraction"]["n_cases"] >= 36 and e3["conflict_correction"]["n_cases"] >= 2
    print(f"\n[V2两段覆盖] E3={e3['n_cases']}例(抽取{e3['extraction']['n_cases']}+冲突"
          f"{e3['conflict_correction']['n_cases']}) / E4={e4['n_queries']}query；"
          f"批次n={batch['n']}，口径={batch['source']['dataset']}")


# ------------------------------------------------------------------ V3
def test_v3_negative_query_not_recalled_real_numbers(result):
    g = result["E4_调用"]["negative_gate"]
    off, on, contrast = g["query_intent_0"], g["query_intent_1"], g["contrast"]
    assert off["negative_query_injection_rate"] == 1.0   # 关门时 q-16 仍被注入记忆
    assert on["negative_query_injection_rate"] == 0.0    # 开门时 q-16 不召回任何记忆
    assert on["negative_retrieved"] == [[]], on["negative_retrieved"]
    assert contrast["recall_delta"] == 0.0               # 正例 Recall@K 不下降（不误伤）
    assert contrast["no_positive_harm"] is True
    assert contrast["guard_effective"] is True
    print("\n[V3负例真实数字] 关门注入=%.1f → 开门注入=%.1f；q-16 retrieved=%s；"
          "正例 Recall@K Δ=%.4f（守卫有效，产品「不主动翻旧事」）" % (
              off["negative_query_injection_rate"], on["negative_query_injection_rate"],
              on["negative_retrieved"], contrast["recall_delta"]))


def test_v3_conflict_correction_exercises_real_product_code(result):
    cf = result["E3_沉淀"]["conflict_correction"]
    assert cf["pass_rate"] == 1.0
    for case in cf["per_case"]:
        assert case["pass"] is True and case["chain_n"] >= 2, case
        assert case["checks"]["supersede_ok"] is True
        assert case["checks"]["old_absent_from_recall"] is True
        assert case["checks"]["new_present_in_recall"] is True
    print("\n[V3-E3冲突修正] 真调 correct_memory/memory_history/recall：%d/%d PASS，"
          "旧事实从召回消失、新事实排前；chain=%s" % (
              sum(1 for c in cf["per_case"] if c["pass"]), cf["n_cases"],
              [c["chain"] for c in cf["per_case"]]))


# ------------------------------------------------------------------ V4
def test_v4_ci_runner_passes_when_batch_registered(tmp_path):
    """构造性验证：把本批次登记进临时 _index.json 后，ci_runner 9 项检查全 PASS。"""
    bd = tmp_path / "batches"
    bd.mkdir()
    names = []
    for p in sorted(BATCH.parent.glob("*.json")):
        if p.name == "_index.json":
            continue
        shutil.copy(p, bd / p.name)
        names.append(p.stem)
    assert BATCH.stem in names
    (bd / "_index.json").write_text(
        json.dumps({"schema_version": "1.0", "batches": names}, ensure_ascii=False),
        encoding="utf-8",
    )
    gate = ci_runner.run(bd)
    assert gate["status"] == "PASS", json.dumps(gate["checks"], ensure_ascii=False)[:2000]
    assert gate["summary"]["checks_failed"] == 0
    print("\n[V4 CI兼容] 临时登记本批次后 %d/%d 检查 PASS（含 source_id_traceability/"
          "batch_schema）；共享 _index.json 的登记行由 lead 集成" % (
              gate["summary"]["checks_passed"], gate["summary"]["checks_total"]))


def test_v4_batch_schema_and_provenance_traceable():
    batch = me.load_batch(BATCH)
    assert batch["batch"] == BATCH.stem and batch["n"] == len(batch["items"])
    assert "synthetic-handwritten" in batch["source"]["note"]  # 非真实用户口径
    for it in batch["items"]:
        assert it["id"] and it["scene"] and it["language"] == "zh" and it["user_input"]
        assert it["source_id"].startswith("recall_corpus_v2:")
        assert it["source"]["dataset"] == "recall_corpus_v2"
        assert it["source"]["kind"] == "synthetic-handwritten"
        assert it["provenance"]["dataset"] == "recall_corpus_v2"
        assert isinstance(it["provenance"]["row_index"], int)
    print(f"\n[V4 数据口径] {batch['n']} 条全标 synthetic-handwritten + canonical source_id"
          f" + provenance 锚点，可逐条追回 recall_corpus.json")


# ------------------------------------------------------------------ 指标自身的可信度
def test_grounding_metric_discriminates_and_keeps_false_negatives_visible(result):
    assert me.fact_supported("外公已去世", "外公去世后，用户翻到旧眼镜。") is True
    assert me.fact_supported("奶奶的红烧肉配方", "用户养了一只猫，名字叫橘子。") is False
    ex = result["E3_沉淀"]["extraction"]
    assert ex["negative_control_reject_rate"] == 1.0        # 负例对照全拒绝
    assert ex["fact_support_rate_positive"] > 0.8           # 正例大多被支撑
    failed = [p["id"] for p in ex["per_case"] if not p["pass"]]
    assert failed  # 不粉饰全绿：paraphrase 假阴性逐条可追（scope 字段已说明）
    print("\n[指标可信度] 负例拒绝率=%.1f；正例事实被支撑率=%.4f；"
          "paraphrase 假阴性 %d 条已逐条留在 per_case（%s...）" % (
              ex["negative_control_reject_rate"], ex["fact_support_rate_positive"],
              len(failed), failed[:2]))
