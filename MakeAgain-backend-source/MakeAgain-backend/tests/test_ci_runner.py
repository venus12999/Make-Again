"""H2 · CI 离线评测门禁的行为测试（全离线、不碰真 key、不联网）。

覆盖 H2-V1~V4：
- V1 workflow YAML 可被 python yaml.safe_load 解析，且含 checkout@v4/setup-python@v5、MOCK_LLM、路径过滤
- V2 ci_runner 对现有 eval/batches 输出真实 JSON + exit 0
- V3 构造缺 source_id 的坏数据 -> exit 1，且失败项明确指向 source_id_traceability
- V4 本文件与相关评测测试在 pytest 下无回归（由 workflow 的 pytest 步骤实际执行）

口径：本测试只用构造性坏数据 + 仓库内既有批次；不调用任何 LLM，不读取任何 key 文件。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml  # PyYAML 在 requirements-tested.txt 中已固定

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
RUNNER = BACKEND / "eval" / "ci_runner.py"
BATCHES = BACKEND / "eval" / "batches"
WORKFLOW = REPO / ".github" / "workflows" / "eval.yml"
ALL_CHECK_NAMES = {
    "mock_llm_forced", "batches_dir_exists", "index_valid", "batch_files_match_index",
    "json_parse", "batch_schema", "item_schema", "source_id_traceability",
    "taxonomy_coverage",
}


def _run_runner(batches_dir: Path, *extra: str):
    """真实子进程调用，拿 stdout(原始 JSON)、stderr 和退出码。"""
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--batches-dir", str(batches_dir), *extra],
        cwd=str(BACKEND), capture_output=True, text=True, encoding="utf-8", timeout=120,
    )
    return proc


def _as_json(proc) -> dict:
    return json.loads(proc.stdout)


def _copy_batches(tmp_path: Path) -> Path:
    dst = tmp_path / "batches"
    shutil.copytree(BATCHES, dst)
    return dst


def _mutate(path: Path, fn) -> None:
    obj = json.loads(path.read_text(encoding="utf-8"))
    fn(obj)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ H2-V1
def test_workflow_yaml_parses_with_safe_load_and_has_offline_gate():
    raw = WORKFLOW.read_text(encoding="utf-8")
    doc = yaml.safe_load(raw)  # 必须能被标准库解析，不靠 CI 专用 parser
    assert isinstance(doc, dict)
    # PyYAML 1.1 会把未加引号的 on 解析为布尔 True，两种都接受
    triggers = doc.get("on", doc.get(True))
    assert isinstance(triggers, dict)
    assert "push" in triggers and "pull_request" in triggers
    paths = triggers["pull_request"]["paths"]
    assert any("eval" in p for p in paths)
    assert any("agent" in p for p in paths)  # prompt 集中在 agent/*.py

    job = doc["jobs"]["eval-gate"]
    assert job["env"]["MOCK_LLM"] == "true"
    steps = job["steps"]
    assert any(s.get("uses") == "actions/checkout@v4" for s in steps)
    assert any(s.get("uses") == "actions/setup-python@v5" for s in steps)
    assert any("ci_runner.py" in s.get("run", "") for s in steps)
    assert any("pytest" in s.get("run", "") for s in steps)
    # 门禁绝不注入真 key
    assert "secrets." not in raw
    assert "OPENAI_API_KEY" not in raw and "LLM_API_KEY" not in raw


# ------------------------------------------------------------------ H2-V2
def test_ci_runner_on_real_batches_passes_with_structured_json():
    proc = _run_runner(BATCHES)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = _as_json(proc)
    assert result["status"] == "PASS" and result["exit_code"] == 0
    assert result["mock_llm_forced"] is True
    assert result["summary"]["checks_failed"] == 0
    assert result["summary"]["batches_checked"] >= 4
    assert result["summary"]["items_checked"] > 0
    assert {c["name"] for c in result["checks"]} == ALL_CHECK_NAMES
    assert all(c["status"] == "pass" for c in result["checks"])


def test_ci_runner_stdout_is_single_json_and_deterministic(tmp_path):
    a = _run_runner(BATCHES)
    b = _run_runner(BATCHES)
    assert a.returncode == b.returncode == 0
    assert a.stdout == b.stdout  # 无时间戳/临时路径，同输入字节一致
    assert a.stderr == ""
    _as_json(a)  # stdout 整体就是 JSON，没有人类日志混入
    out = tmp_path / "gate.json"
    c = _run_runner(BATCHES, "--out", str(out))
    assert c.returncode == 0
    assert out.read_text(encoding="utf-8") == a.stdout  # print 自带的 \n 与落盘一致


# ------------------------------------------------------------------ H2-V3
def test_missing_source_id_fails_gate_with_actionable_detail(tmp_path):
    bad = _copy_batches(tmp_path)

    def drop_source_id(obj):
        item = obj["items"][0]
        item.pop("source_id", None)
        if isinstance(item.get("source"), dict):
            item["source"].pop("id", None)

    _mutate(bad / "01_中文共情_CPsyCoun.json", drop_source_id)
    proc = _run_runner(bad)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    result = _as_json(proc)
    assert result["status"] == "FAIL" and result["exit_code"] == 1
    failed = {c["name"]: c for c in result["checks"] if c["status"] == "fail"}
    assert "source_id_traceability" in failed
    assert any("缺 source_id" in d for d in failed["source_id_traceability"]["details"])
    assert result["summary"]["checks_failed"] >= 1


def test_mismatched_source_id_also_fails(tmp_path):
    bad = _copy_batches(tmp_path)

    def break_link(obj):
        obj["items"][0]["source_id"] = "TOTALLY_UNRELATED:row:999"

    _mutate(bad / "01_中文共情_CPsyCoun.json", break_link)
    proc = _run_runner(bad)
    assert proc.returncode == 1
    result = _as_json(proc)
    failed = [c for c in result["checks"] if c["status"] == "fail"]
    assert any(c["name"] == "source_id_traceability" and
               any("无法与" in d for d in c["details"]) for c in failed)


def test_taxonomy_gap_and_count_mismatch_fail(tmp_path):
    bad = _copy_batches(tmp_path)

    def relabel(obj):
        for item in obj["items"]:
            if item.get("scene") == "分手失恋":
                item["scene"] = "情绪困扰"

    _mutate(bad / "01_中文共情_CPsyCoun.json", relabel)
    proc = _run_runner(bad)
    assert proc.returncode == 1
    failed = {c["name"] for c in _as_json(proc)["checks"] if c["status"] == "fail"}
    assert "taxonomy_coverage" in failed

    bad2 = _copy_batches(tmp_path / "second")
    _mutate(bad2 / "01_中文共情_CPsyCoun.json", lambda obj: obj.update(n=obj["n"] + 1))
    proc2 = _run_runner(bad2)
    assert proc2.returncode == 1
    failed2 = {c["name"] for c in _as_json(proc2)["checks"] if c["status"] == "fail"}
    assert "batch_schema" in failed2


# ------------------------------------------------------------------ H2-V4 边界
def test_missing_batches_dir_fails_without_crashing(tmp_path):
    proc = _run_runner(tmp_path / "does-not-exist")
    assert proc.returncode == 1
    result = _as_json(proc)  # 异常路径也必须输出结构化 JSON
    assert result["status"] == "FAIL"
    assert any(c["name"] == "batches_dir_exists" and c["status"] == "fail"
               for c in result["checks"])
