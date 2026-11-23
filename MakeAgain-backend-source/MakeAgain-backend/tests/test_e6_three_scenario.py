"""F1 A/B 后三场景完整旅程真 hy4 验证的行为验收（离线、不烧 key、不联网）。

对应验收：
- F1-V1 --mock 三场景全链路结构 PASS（3×5CP）。
- F1-V2 真 hy4 三场景各 exit=0、5CP PASS（产物存在时校验，不存在明确 skip，不伪称跑过）。
- F1-V3 场景切换不误触发有数字：确定性检测器 + 真实接线 _maybe_transition_scene。
- F1-V4 画像注入正确有数字：画像按 companion_id 隔离 + 3 参兼容 + 真实 _context_block 注入。
- F1-V5 本文件随全量 pytest 无回归。

口径：persona/mock trace 均 synthetic-handwritten；A/B 探针是真实产品函数、零 LLM；
真 hy4 数字来自 outputs/scenarios/full_journey_*.json（真实 CLI 跑出，key 不入库）。
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

import e6_full_journey as e6  # noqa: E402
import e6_three_scenario as f1  # noqa: E402
from memory.db import _RawSessionLocal  # noqa: E402

RUNNER = BACKEND / "eval" / "e6_three_scenario.py"
BATCH = BACKEND / "eval" / "batches" / "E6_完整旅程.json"
REAL_DIR = REPO / "outputs" / "scenarios"


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=str(BACKEND), capture_output=True, text=True, encoding="utf-8", timeout=180,
    )


def _probe(item: dict) -> dict:
    """用真 DB（conftest 隔离 sqlite）+ mock trace 驱动 A/B 探针，零 LLM。"""
    trace = e6.mock_trace(item)
    with _RawSessionLocal() as db:
        return f1.ab_probes(item, trace, db)


# ------------------------------------------------------------------ F1-V1
def test_v1_mock_three_scenarios_full_chain(tmp_path):
    out = tmp_path / "mock.json"
    proc = _run_cli("--mock", "--out", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mode"] == "mock" and data["summary"]["users"] == 3
    assert {r["loss_type"] for r in data["results"]} == {"breakup", "loved_one", "pet"}
    assert data["summary"]["checkpoint_pass_counts"] == {n: 3 for n in e6.CHECKPOINT_NAMES}
    assert data["summary"]["journey_pass"] is True and data["summary"]["overall_pass"] is True
    for r in data["results"]:
        assert set(r["checkpoints"]) == set(e6.CHECKPOINT_NAMES)
        assert all(r["checkpoints"].values()), r["checkpoints"]
        assert r["report_field_count"] == 9, r["report_field_count"]
        ev = r["evidence"]
        assert len(ev["report_anchors_matched"]) >= 3
        assert ev["turns"]["n_turns"] <= ev["turns"]["max_allowed_turns"]
        assert r["ab_probes"]["status"] == "NOT_RUN"  # mock 不伪称跑 A/B 探针
    print("\n[F1-V1] 3 场景 × 5CP 全 PASS，报告 9/9、锚点 >=3、轮数 <=16，A/B 探针标 NOT_RUN")


def test_v1_fault_pass_through_guards_junction(tmp_path):
    """故障只打挂目标衔接点，证明复用来的 5CP 不是摆设。"""
    out = tmp_path / "fault.json"
    proc = _run_cli("--mock", "--fault", "missing_report_injection", "--out", str(out))
    assert proc.returncode == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["overall_pass"] is False
    assert all(r["checkpoints"]["CP2_report_to_dialogue"] is False for r in data["results"])
    others = [n for n in e6.CHECKPOINT_NAMES if n != "CP2_report_to_dialogue"]
    assert all(all(r["checkpoints"][n] for n in others) for r in data["results"])
    print("\n[F1-V1守卫] 注入 missing_report_injection → exit=1，仅 CP2 FAIL")


def test_batch_is_e6_reusable_and_traceable():
    obj = json.loads(BATCH.read_text(encoding="utf-8"))
    assert obj["n"] == len(obj["items"]) == 3
    for it in obj["items"]:
        assert it["source"]["dataset"] == "synthetic-handwritten"
        assert it["persona"]["loss_type"] in ("breakup", "loved_one", "pet")
        assert it["journey"]["dialogue_query"] and it["journey"]["recall_query"]
    assert set(f1.SCENARIOS) == {"breakup", "loved_one", "pet", "all"}
    print("\n[数据口径] 复用 E6_完整旅程.json 三场景 persona+journey，synthetic-handwritten 可追溯")


# ------------------------------------------------------------------ F1-V3 / F1-V4
@pytest.mark.parametrize("loss", ["breakup", "loved_one", "pet"])
def test_v3_scene_switch_probe_no_false_trigger(loss):
    item = next(it for it in e6.load_batch() if it["persona"]["loss_type"] == loss)
    pr = _probe(item)["scene"]
    assert pr["detector_on"] is True
    assert pr["dialogue_should_switch"] is False, f"{loss} 普通陪伴对话误触发切场景"
    assert pr["positive_control_should_switch"] is True, "正对照失效：检测器恒 False"
    assert pr["positive_target_scene"] == "friend"
    assert pr["third_person_should_switch"] is False, "第三人称('前任')误触发切场景"
    assert pr["companion_count_before"] == pr["companion_count_after"] == 1
    assert pr["loss_type_before"] == pr["loss_type_after"] == loss
    assert pr["pass"] is True
    print(f"\n[F1-V3 {loss}] 对话switch=False 正对照=True 第三人称=False 段数1->1 loss不变 -> PASS")


@pytest.mark.parametrize("loss", ["breakup", "loved_one", "pet"])
def test_v4_portrait_isolation_and_injection_probe(loss):
    item = next(it for it in e6.load_batch() if it["persona"]["loss_type"] == loss)
    pr = _probe(item)["portrait"]
    assert pr["written_with_companion_id"]
    assert pr["read_right_cid_nonempty"] is True, f"{loss} 正确 companion_id 读不到画像"
    assert pr["read_wrong_cid_empty"] is True, f"{loss} 错误 companion_id 竟读到画像（隔离失效）"
    assert pr["read_3arg_backcompat_nonempty"] is True, "3 参向后兼容被破坏"
    assert pr["subject_matches_anchor"] is True, f"{loss} 画像称呼不是本人: {pr['subject_name']}"
    assert pr["context_has_portrait"] is True, f"{loss} _context_block 注入不到画像"
    assert pr["context_anchor_hits"] >= 1
    assert pr["pass"] is True
    print(f"\n[F1-V4 {loss}] 右cid读=True 错cid空=True 3参兼容=True 称呼={pr['subject_name']} "
          f"注入锚点={pr['context_anchor_hits']} -> PASS")


# ------------------------------------------------------------------ F1-V2
@pytest.mark.parametrize("loss", ["breakup", "loved_one", "pet"])
def test_v2_real_hy4_artifact_if_present(loss):
    path = REAL_DIR / f"full_journey_{loss}.json"
    if not path.exists():
        pytest.skip(f"真 hy4 慢评产物不存在（需手工跑 CLI）：{path.name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["mode"] == "real" and data["scenario"] == loss
    assert data["summary"]["users"] == 1
    assert data["summary"]["exit_code"] == 0, f"{loss} 真跑 exit!=0"
    assert data["summary"]["checkpoint_pass_counts"] == {n: 1 for n in e6.CHECKPOINT_NAMES}
    r = data["results"][0]
    assert all(r["checkpoints"].values()), f"{loss} 5CP 未全 PASS: {r['checkpoints']}"
    assert r["report_field_count"] == 9, f"{loss} 报告字段 {r['report_field_count']}/9"
    assert len(r["evidence"]["report_anchors_matched"]) >= 3
    assert r["evidence"]["turns"]["n_turns"] <= r["evidence"]["turns"]["max_allowed_turns"]
    ab = r["ab_probes"]
    assert ab["pass"] is True, f"{loss} A/B 探针未过: {ab}"
    print(f"\n[F1-V2真实 {loss}] exit=0 5CP全PASS 报告9/9 锚点"
          f"{len(r['evidence']['report_anchors_matched'])} 轮数{r['evidence']['turns']['n_turns']} "
          f"注入锚点={ab['portrait']['journey_context_anchor_hits']} "
          f"切换不误触发={ab['scene']['pass']}")
