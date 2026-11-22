"""WP-04 结束仪式验证的行为验收（全离线，不烧 key，不联网）。

对应 D-V1..V5：
- V1  抠图本地 rembg 兜底真跑（有输出图，背景透明 + 透明边收紧）。
- V2  纪念文案个性化有数字（3 关系类型 → 3 段不同文案；prompt 路径不同对象 → 不同文案）。
- V3  模拟来电 AI 披露标记验证（2 端点 × 3 场景 = 6/6 带标记）。
- V4  归档只读不可逆 + 纪念物/记忆导出（写拒 2/2 ARCHIVED_READ_ONLY，导出有数字）。
- V5  CLI 离线模式 exit 0 + 证据结构；全量 pytest 无归并由命令行实跑记录在
      docs/ENDING_VALIDATION.md（不在 pytest 内递归跑 pytest）。

口径：合成测试图 synthetic-handwritten；rembg 用本机已缓存 u2netp 模型真跑（不发网络）；
文案走产品 mock 兜底 + 确定性桩（offline-placeholder token 不是 key，不打印不入库）。
真 hy4 属可选 --real，pytest 不跑。key 不进仓库。
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

import ending_validation as ev  # noqa: E402

RUNNER = BACKEND / "eval" / "ending_validation.py"
REPORT_DOC = REPO / "docs" / "ENDING_VALIDATION.md"


# ------------------------------------------------------------------ V1
def test_v1_cutout_local_rembg_fallback_real_offline(tmp_path, monkeypatch):
    """无 COS 凭据 + 合成图，本地 rembg 真跑，产出透明底 PNG（有输出图）。"""
    monkeypatch.setenv("USE_LOCAL_MATTE", "1")
    monkeypatch.setenv("LOCAL_MATTE_MODEL", "u2netp")
    result = ev.check_cutout(tmp_path)
    m = result["metrics"]

    assert result["pass"] is True, m
    assert m["backend"] == "local"                    # 无 COS 凭据
    assert m["model"] == "u2netp"                     # 许可安全白名单模型
    assert m["matte_png"]["mode"] == "RGBA"
    assert m["matte_png"]["pct_transparent"] > 5.0    # 背景真被抠掉
    assert m["trimmed_png"]["width"] < m["source_png"]["width"]  # 透明边收紧
    for rel in result["evidence"]:
        assert Path(rel).exists() and Path(rel).stat().st_size > 0


# ------------------------------------------------------------------ V2
def test_v2_memorial_copy_has_personalization_numbers():
    """兜底 3 类不同文案 + 真实 prompt 路径不同对象不同文案。"""
    result = ev.check_memorial_copy(uid_prefix="ending-test-dv2")
    m = result["metrics"]

    assert result["pass"] is True, m
    assert m["relationship_types_tested"] == 3
    assert m["distinct_fallback_texts"] == 3
    assert m["distinct_prompt_path_texts"] == 3
    assert m["subject_name_in_all_prompt_texts"] is True
    # 逐条看原文：文本确实带对象名与关系，不是平均分口径
    for row in m["prompt_path_rows"]:
        assert row["subject_name"] in row["text"]
        assert row["relationship_type"] in row["text"]


# ------------------------------------------------------------------ V3
def test_v3_disclosure_marker_on_both_endpoints():
    """纪念文案 + 模拟来电都带 AI 披露标记。"""
    result = ev.check_disclosure(uid_prefix="ending-test-dv3")
    m = result["metrics"]

    assert result["pass"] is True, m
    assert m["endpoints_same_marker"] is True
    assert m["disclosure_marker"]
    assert m["rows_marked"] == m["rows_total"] == 3
    for row in m["rows"]:
        assert row["memorial_disclosure"] == m["disclosure_marker"]
        assert row["call_disclosure"] == m["disclosure_marker"]


# ------------------------------------------------------------------ V4
def test_v4_archive_readonly_and_export(tmp_path):
    """归档幂等只读不可逆 + 导出有数字；纪念物不在导出范围是已知缺口。"""
    result = ev.check_archive_export(uid_prefix="ending-test-dv4", out_dir=tmp_path)
    m = result["metrics"]

    assert result["pass"] is True, m
    assert m["archived_at_set"] is True
    assert m["archive_idempotent"] is True
    assert m["ending_stage"] == "complete"
    assert m["read_still_allowed"] is True
    assert len(m["write_denials"]) == 2
    assert all(d["denied"] and d["code"] == "ARCHIVED_READ_ONLY" for d in m["write_denials"])
    assert m["export_source_entries"] >= 1
    assert m["export_portrait_entries"] >= 1
    assert "portrait" in m["export_categories"]
    # 已知缺口如实固化（不是把失败藏起来）：
    assert m["item_cutout_in_export"] is False   # GAP-E2
    assert m["item_original_key"] == ""          # GAP-E3
    assert Path(result["evidence"][0]).exists()


# ------------------------------------------------------------------ V5
def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=str(BACKEND), capture_output=True, text=True, encoding="utf-8", timeout=180,
    )


def test_v5_cli_mock_exit_zero_and_evidence_shape(tmp_path):
    """CLI --mock 全离线可复现，证据 JSON 结构齐全，不泄漏任何 key/占位符。"""
    out = tmp_path / "evidence.json"
    proc = _run_cli("--mock", "--out-dir", str(tmp_path / "media"), "--out", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))

    assert data["mode"] == "mock"
    assert data["baseline_commit"] == "6ab6f718"
    assert data["summary"]["all_pass"] is True
    assert data["summary"]["passed"] == data["summary"]["total"] == 4
    assert set(data["checks"]) == {
        "D-V1_cutout", "D-V2_memorial_copy", "D-V3_disclosure", "D-V4_archive_export",
    }
    assert all(c["pass"] for c in data["checks"].values())
    assert data["real_mode"]["status"] == "NOT_RUN"
    assert any("GAP-E2" in g for g in data["gaps"])
    assert any("GAP-E3" in g for g in data["gaps"])
    # 不打印、不写入任何凭据（含占位符 token）
    blob = out.read_text(encoding="utf-8") + proc.stdout + proc.stderr
    assert ev.OFFLINE_PLACEHOLDER_TOKEN not in blob
    assert "llm_api_key" not in blob


def test_v5_report_doc_records_gaps_and_real_commands():
    """报告文档必须存在且如实记录缺口、口径与真实命令（D-V5 的证据载体）。"""
    assert REPORT_DOC.exists()
    text = REPORT_DOC.read_text(encoding="utf-8")
    for token in ("D-V1", "D-V2", "D-V3", "D-V4", "D-V5", "GAP-E2", "GAP-E3",
                  "pytest", "rembg", "u2netp", "NOT_RUN"):
        assert token in text, token
