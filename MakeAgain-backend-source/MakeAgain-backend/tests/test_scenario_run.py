"""V1+V2 三场景真 hy4 验证的行为验收（全离线、不烧 key、不联网）。

验收项对应 V12-V1~V5：
- V1 --mock 三场景跑通结构：3 persona × 5 检查点全 PASS，锚点门槛=3。
- V2/V3 真 hy4 loved_one / pet 报告质量：本文件对真跑产物做离线回放重算（存在才校验，
  不预设全绿，发现退化如实记录）；现场真跑由 CLI 手工完成，证据另存 outputs/scenarios/。
- V4 数字汇总：summary.scenarios 每场景含字段数/锚点/轮数/耗时。
- V5 本文件随全量 pytest 无回归。

口径：persona 与 mock trace 均 synthetic-handwritten 构造，非真人数据；真跑产物来自 hy4 现场生成，
key 不进仓库。判定函数复用 e2_interview_chain，唯一差异=锚点门槛 2→3。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "eval"))

import e2_interview_chain as e2  # noqa: E402
import scenario_run as sr  # noqa: E402

BATCH = BACKEND / "eval" / "batches" / "scenarios.json"
RUNNER = BACKEND / "eval" / "scenario_run.py"
REAL_OUT_DIR = REPO / "outputs" / "scenarios"


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=str(BACKEND), capture_output=True, text=True, encoding="utf-8", timeout=180,
    )


# ------------------------------------------------------------------ V1: mock 三场景结构
def test_v1_mock_runs_all_three_scenarios_with_five_checkpoints(tmp_path):
    out = tmp_path / "mock.json"
    proc = _run_cli("--mock", "--out", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mode"] == "mock" and data["summary"]["users"] == 3
    assert data["summary"]["exit_code"] == 0 and data["summary"]["overall_pass"] is True
    assert {r["scenario"] for r in data["results"]} == {"breakup", "loved_one", "pet"}
    assert {r["loss_type"] for r in data["results"]} == {"breakup", "loved_one", "pet"}
    assert all(set(r["checkpoints"]) == set(e2.CHECKPOINT_NAMES) for r in data["results"])
    assert all(r["evidence"]["personalization"]["min_required"] == 3 for r in data["results"])
    assert all(r["evidence"]["personalization"]["matched"] for r in data["results"])
    print("\n[V12-V1] 3 场景 × 5 检查点全 PASS；锚点门槛=3；口径=" + data["口径"])


def test_v1_mock_single_scenario_selection(tmp_path):
    out = tmp_path / "pet.json"
    proc = _run_cli("--mock", "--scenario", "pet", "--out", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["scenario"] == "pet" and data["summary"]["users"] == 1
    assert data["results"][0]["loss_type"] == "pet"
    print("\n[V12-V1] --scenario pet 单选生效，只跑 1 个 persona")


def test_batch_traceable_synthetic_three_scenes():
    obj = json.loads(BATCH.read_text(encoding="utf-8"))
    assert obj["batch"] == BATCH.stem and obj["n"] == len(obj["items"]) == 3
    assert {it["scenario"] for it in obj["items"]} == {"breakup", "loved_one", "pet"}
    for it in obj["items"]:
        assert it["id"] == it["source_id"]
        assert it["source"]["dataset"] == "synthetic-handwritten"
        assert it["source"].get("derived_from")
        p = it["persona"]
        assert len(p["answers"]) == 5 and len(p["anchors"]) >= 3  # 锚点门槛=3，至少要有3个可命中
    print("\n[数据口径] 3 persona 全标 synthetic-handwritten + source_id/derived_from 可追溯")


# ------------------------------------------------------------------ 门槛与复用一致性
def test_anchor_threshold_is_three_not_two_and_reuses_e2_matching():
    """核心证据：同样报告命中 2 锚点 → CP4 FAIL（E2 会放过），命中 3 → PASS。"""
    assert sr.MIN_ANCHORS == 3 and e2.PERSONALIZATION_MIN_ANCHORS == 2
    assert sr.MAX_TURNS == e2.REASONABLE_MAX_TURNS == 16
    # 判定函数确实复用 E2，不是另写一套
    assert sr.validate_report_fields is e2.validate_report_fields
    assert sr.check_turns is e2.check_turns
    assert sr.mock_trace is e2.mock_trace
    assert sr.real_trace is e2.real_trace

    item = sr.load_batch()[1]  # loved_one
    report = e2._mock_report(item["persona"], include_anchors=False)  # 字段齐全但无锚点
    report["summary"] = "他叫建国，教我骑自行车。"  # 只放 2 个锚点
    trace = {"report": report, "n_turns": 5, "gen_filler_turns": 0,
             "done": True, "reached_complete": True, "report_ready": True}
    scored2 = sr.evaluate_scenario(item, trace)
    assert scored2["checkpoints"]["CP4_personalization_match"] is False
    assert len(scored2["evidence"]["personalization"]["matched"]) == 2
    assert scored2["checkpoints"]["CP3_report_fields_complete"] is True  # 字段齐全≠个性化

    report["summary"] = "他叫建国，教我骑自行车，还说包饺子。"  # 加到 3 个
    scored3 = sr.evaluate_scenario(item, trace)
    assert scored3["checkpoints"]["CP4_personalization_match"] is True
    assert len(scored3["evidence"]["personalization"]["matched"]) == 3
    assert scored3["evidence"]["personalization"]["e2_min_required"] == 2  # 留痕
    print("\n[门槛守卫] 命中2锚点→CP4 FAIL / 命中3→PASS；匹配算法复用 e2，仅门槛 2→3")


def test_generic_report_fault_fails_personalization_guard(tmp_path):
    out = tmp_path / "fault_generic.json"
    proc = _run_cli("--mock", "--fault", "generic_report", "--out", str(out))
    assert proc.returncode == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["overall_pass"] is False
    assert all(r["checkpoints"]["CP4_personalization_match"] is False for r in data["results"])
    assert all(r["checkpoints"]["CP3_report_fields_complete"] is True for r in data["results"])
    print("\n[构造守卫] 通用模板报告 → CP4 FAIL（字段齐全仍判不个性化）")


# ------------------------------------------------------------------ V4: 数字汇总结构
def test_summary_has_numeric_table_per_scenario(tmp_path):
    out = tmp_path / "summary.json"
    proc = _run_cli("--mock", "--out", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    rows = data["summary"]["scenarios"]
    assert len(rows) == 3
    for r in rows:
        assert 0 <= r["report_field_count"] <= 9
        assert r["anchors_total"] >= 3 and 0 <= r["anchors_matched"] <= r["anchors_total"]
        assert isinstance(r["n_turns"], int) and isinstance(r["gen_filler_turns"], int)
        assert isinstance(r["elapsed_s"], (int, float))
        assert isinstance(r["pass"], bool)
    print("\n[V12-V4] summary.scenarios 每场景含 字段数/锚点命中/轮数/filler/耗时/pass")


# ------------------------------------------------------------------ V2/V3: 真跑产物离线回放
def test_real_hy4_outputs_offline_replay_if_present():
    """真跑产物存在时重算，验证记录数字可复现；不预设全绿——发现退化应如实暴露。"""
    targets = ("loved_one", "pet")
    present = [(name, REAL_OUT_DIR / f"{name}.json") for name in targets]
    present = [(name, p) for name, p in present if p.is_file()]
    if not present:
        pytest.skip("真跑产物 outputs/scenarios/{loved_one,pet}.json 尚未生成（现场真跑由 CLI 完成）")

    items = {it["scenario"]: it for it in sr.load_batch()}
    for name, path in present:
        obj = json.loads(path.read_text(encoding="utf-8"))
        assert obj["mode"] == "real" and obj["scenario"] == name
        item = items[name]
        for rec in obj["results"]:
            trace = dict(rec.get("trace") or {})
            trace["report"] = rec.get("report")
            scored = sr.evaluate_scenario(item, trace)
            ev = scored["evidence"]
            # 重算数字必须与真跑产物当场记录一致（可复现），不与期望值绑定
            assert ev["report_field_count"] == rec["evidence"]["report_field_count"]
            assert len(ev["personalization"]["matched"]) == len(rec["evidence"]["personalization"]["matched"])
            assert ev["turns"]["n_turns"] == rec["evidence"]["turns"]["n_turns"]
            assert scored["overall_pass"] == rec["overall_pass"]
            # 专属分析块口径：loved_one→grief_analysis / pet→pet_grief_analysis
            block = e2.ANALYSIS_BLOCK[name]
            assert isinstance((rec.get("report") or {}).get(block), dict), f"{name} 报告缺 {block}"
            print(f"\n[V12-V2/V3 真跑回放] {name}: 字段{ev['report_field_count']}/9 "
                  f"锚点{len(ev['personalization']['matched'])}/{ev['personalization']['total_anchors']} "
                  f"轮数{ev['turns']['n_turns']} filler{ev['turns']['gen_filler_turns']} "
                  f"耗时{ev['elapsed_s']}s pass={scored['overall_pass']}")
