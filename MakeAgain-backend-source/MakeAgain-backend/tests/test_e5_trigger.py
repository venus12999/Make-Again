"""E5 行动段触发评测的行为验收（全离线、不烧 key、不联网）。

验收项对应任务 E5-V1~V5（每条都用真实子进程命令 + 真实输出，不靠“全绿”自证）：
- V1 时间×情绪触发矩阵跑出真实数字：该触发 8 / 不触发 4（另 weekly 触发1/抑制4），
  逐条看 rule_keys 与 candidates，不只看 summary。
- V2 打扰控制有数字且有反证：同一 rule_key 当天不重复、语录每天≤1、第三次拉取为空；
  另直接压产品代码，关掉 nudge_seen 去重后同一 rule 立即重复推——证明守卫非空转。
- V3 内容质量检查（具体非空话）有数字：10 条 applicable 全过、字长区间；
  构造性反例（空话/命令）必须被判 FAIL。
- V4 --mock 不烧 key：给 CLI 一个哨兵 key 文件（含明显标记），mock 模式传进去仍能 PASS，
  且 stdout/落盘 JSON 不含哨兵——证明 mock 分支没读 key。
- V5 本文件随全量 pytest 无回归（收集检查；全量命令在报告里实跑）。

口径：批次 case 全部 synthetic-handwritten；触发/去重/内容质量为确定性规则判定，
mock 文案走产品 _fallback_text（非真人偏好、非真模型输出）。key 明文不进测试、不进仓库。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "eval"))

import e5_trigger_eval as e5  # noqa: E402

BATCH = BACKEND / "eval" / "batches" / "E5_行动段触发.json"
RUNNER = BACKEND / "eval" / "e5_trigger_eval.py"
CI_RUNNER = BACKEND / "eval" / "ci_runner.py"


def _run_cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=str(BACKEND), capture_output=True, text=True, encoding="utf-8",
        timeout=180, env=(env or os.environ.copy()),
    )


@pytest.fixture(scope="module")
def mock_result(tmp_path_factory):
    out = tmp_path_factory.mktemp("e5-eval") / "mock.json"
    proc = _run_cli("--mock", "--out", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(out.read_text(encoding="utf-8")), proc


# ------------------------------------------------------------------ V1
def test_v1_mock_matrix_real_trigger_no_trigger_numbers(mock_result):
    out, _ = mock_result
    s = out["summary"]
    assert out["mode"] == "mock" and s["exit_code"] == 0
    assert s["cases"] == 19 and s["passed"] == 19 and s["failed"] == 0
    assert s["by_module"] == {"nudge": 12, "weekly": 5, "daily": 2}

    m = s["V1_trigger_matrix"]
    assert m["nudge_should_trigger"] >= 6, m  # 该触发：深夜/晚上/周六/凌晨/傍晚/频次聚合…
    assert m["nudge_should_not_trigger"] >= 4, m  # 不触发：静默/关闭/时间不匹配/freq<2 回落
    assert m["emotion_trigger_cases"] >= 6 and m["time_trigger_cases"] >= 2
    assert m["weekly_due"] >= 1 and m["weekly_suppressed"] >= 4

    # 逐条看 case 原文 + 实际 rule_keys，不盯平均分（AGENTS.md 约定）。
    nudge = [r for r in out["results"] if r["module"] == "nudge"]
    emotion_cases = [r for r in nudge if r["actual"]["rule_keys"][:1] == ["emotion:深夜"]]
    assert len(emotion_cases) >= 2
    for r in emotion_cases:
        assert r["late_night"] is True  # 深夜时间条件确实命中
        assert any(c["type"] == "emotion" for c in r["candidates"])
        assert all(c["rule_key"] != "late_night" for c in r["candidates"])  # 有夜间节点不重复提醒
    suppressed = [r for r in nudge if r["actual"]["rule_keys"] == []]
    assert {r["id"] for r in suppressed} >= {
        "e5-n07-quiet-hours-suppress", "e5-n08-disabled-suppress",
    }
    print(f"\n[V1真实数字] nudge 触发{m['nudge_should_trigger']}/不触发{m['nudge_should_not_trigger']}；"
          f"情绪节点{m['emotion_trigger_cases']}、时间触发{m['time_trigger_cases']}；"
          f"weekly due{m['weekly_due']}/抑制{m['weekly_suppressed']}；共{s['cases']}case")


def test_batch_is_traceable_and_ci_runner_compatible(tmp_path):
    obj = json.loads(BATCH.read_text(encoding="utf-8"))
    assert obj["batch"] == BATCH.stem and obj["n"] == len(obj["items"]) == 19
    idx = json.loads((BATCH.parent / "_index.json").read_text(encoding="utf-8"))
    assert BATCH.stem in idx["batches"]  # 共享文件仅追加一行，否则 batch_files_match_index 回归
    for it in obj["items"]:
        assert it["id"] == it["source_id"]
        assert it["source"]["dataset"] == "synthetic-handwritten"
        assert "agent/" in it["source"]["derived_from"]  # 每条追到产品实物代码
        assert datetime.fromisoformat(it["local_time"])  # 本地墙钟可解析

    # ci_runner 会扫整个 batches 目录并强制 index 与文件一一对应。工作区里并发队友的
    # E3/E6 批次尚未合入本分支基线，故固化「基线 5 批 + E5」跑门禁——只证明本分支自身兼容，
    # 不替队友未完成的批次背书（避免 git show / 依赖并发状态的脆弱断言）。
    baseline = [
        "01_中文共情_CPsyCoun", "02_支持策略_ESConv", "03_情绪识别_GoEmotions",
        "04_危机安全_CueSupport", "E2_理解段链路", "E5_行动段触发",
    ]
    gate_dir = tmp_path / "batches"
    gate_dir.mkdir()
    for name in baseline:
        shutil.copy(BATCH.parent / f"{name}.json", gate_dir / f"{name}.json")
    (gate_dir / "_index.json").write_text(
        json.dumps({"schema_version": "1.0", "batches": baseline}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(CI_RUNNER), "--batches-dir", str(gate_dir)],
        cwd=str(BACKEND), capture_output=True, text=True, encoding="utf-8", timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    gate = json.loads(proc.stdout)
    assert gate["status"] == "PASS" and gate["summary"]["checks_failed"] == 0
    assert gate["summary"]["batches_checked"] >= 5  # E5 批次已被门禁纳入
    print("\n[数据口径] 19条 synthetic-handwritten + source_id 可追溯 + 已登记 _index + ci_runner exit=0")


# ------------------------------------------------------------------ V2
def test_v2_anti_nag_numbers_and_pull_evidence(mock_result):
    out, _ = mock_result
    anti = out["summary"]["V2_anti_nag"]
    assert anti["nudge_cases_checked"] == 12
    assert anti["repeat_push_blocked"] == 12
    assert anti["quote_at_most_once_per_day"] == 12
    assert anti["third_pull_empty"] == 12
    assert anti["weekly_seen_blocked"] >= 1
    assert anti["daily_same_day_cached"] >= 2

    n01 = next(r for r in out["results"] if r["id"] == "e5-n01-deepnight-night-node")
    assert n01["actual"]["pull_sequence"] == [["emotion:深夜"], ["quote"], []]
    assert n01["actual"]["repeated_primary_rules"] == []
    assert n01["actual"]["quotes_in_3_pulls"] == 1
    w03 = next(r for r in out["results"] if r["id"] == "e5-w03-weekly-unseen-then-seen")
    assert w03["actual"]["due"] is True
    assert w03["actual"]["after_seen"] == {"due": False, "reason": "已查看"}
    print(f"\n[V2真实数字] 12/12 重复情绪推被拦、12/12 语录≤1/天、12/12 第三次拉取为空；"
          f"周报看后抑制={anti['weekly_seen_blocked']}，每日当日缓存={anti['daily_same_day_cached']}")


def test_v2_dedup_guard_is_not_vacuous(monkeypatch):
    """反证：把 nudge_seen 去重关掉，同一 rule_key 立刻重复推——证明守卫不是摆设。"""
    from agent import nudge
    from gateway.client import LLMClient
    from memory import store
    from memory.db import _RawSessionLocal
    from memory.models import EmotionNode
    from config import get_settings

    uid = "e5-guard-" + __import__("uuid").uuid4().hex[:8]
    now = datetime(2026, 3, 10, 23, 30) - timedelta(hours=get_settings().timezone_offset_hours)
    client = LLMClient()
    assert client.mock is True  # conftest 强制 MOCK_LLM=true
    with _RawSessionLocal() as db:
        store.get_or_create_user(db, uid)
        db.add(EmotionNode(user_id=uid, trigger="深夜独处", emotion="难过",
                           frequency=3, time_tag="深夜"))
        db.commit()
        r1 = nudge.get_nudges(db, uid, client, now=now)
        r2 = nudge.get_nudges(db, uid, client, now=now)
        assert r1["nudges"][0]["rule_key"] == "emotion:深夜"
        assert all(n["rule_key"] != "emotion:深夜" for n in r2["nudges"])  # 正常去重生效

        monkeypatch.setattr(nudge.store, "nudge_seen", lambda *a, **kw: False)
        r3 = nudge.get_nudges(db, uid, client, now=now)
        assert r3["nudges"][0]["rule_key"] == "emotion:深夜"  # 去掉去重 → 重复推，守卫有效
    print("\n[V2反证] 正常路径二次拉取无 emotion:深夜；monkeypatch nudge_seen=False 后同 rule 重现")


# ------------------------------------------------------------------ V3
def test_v3_content_quality_numbers_and_constructive_negative(mock_result):
    out, _ = mock_result
    q = out["summary"]["V3_content_quality"]
    assert q["checked"] >= 8 and q["passed"] == q["checked"] and q["failed"] == 0
    assert q["char_len_min"] >= 6 and q["char_len_max"] <= 50
    assert "好吗" in q["rules"]["suggestion_markers"]

    # 构造性反例：空话/命令/无具体动作必须被判 FAIL（全绿不是标准）。
    bad = e5.check_text_quality("加油，你一定会好起来的")
    assert bad["ok"] is False
    assert any(f.startswith("VAGUE_OR_COMMAND") for f in bad["failures"])
    assert "NO_SUGGESTION_MARKER" in bad["failures"]
    assert "NO_CONCRETE_ACTION" in bad["failures"]
    empty = e5.check_text_quality("")
    assert empty["ok"] is False and "EMPTY" in empty["failures"]
    good = e5.check_text_quality("到深夜了，出去走十分钟透透气，好吗?")
    assert good["ok"] is True
    print(f"\n[V3真实数字] applicable={q['checked']}，通过={q['passed']}，失败={q['failed']}，"
          f"字长[{q['char_len_min']},{q['char_len_max']}]；反例空话被判FAIL")


# ------------------------------------------------------------------ V4
def test_v4_mock_never_reads_key_file(tmp_path):
    sentinel = "E5_SENTINEL_KEY_MUST_NOT_BE_READ_9f3a"
    fake_key = tmp_path / "fake-key.txt"
    fake_key.write_text(f"BASE_URL=http://sentinel.invalid\nKEY={sentinel}\n", encoding="utf-8")
    out = tmp_path / "mock.json"
    env = os.environ.copy()
    env["E5_KEY_FILE"] = str(fake_key)
    env["LLM_API_KEY"] = sentinel
    proc = _run_cli("--mock", "--key-file", str(fake_key), "--out", str(out), env=env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = out.read_text(encoding="utf-8")
    result = json.loads(payload)
    assert result["mode"] == "mock" and result["summary"]["exit_code"] == 0
    assert "不读key" in result["口径"]
    assert sentinel not in proc.stdout and sentinel not in payload  # key 没被读也没被打印/落盘
    print("\n[V4] 哨兵 key 文件传给 --mock 仍 PASS，输出不含哨兵 → mock 分支不读 key")


# ------------------------------------------------------------------ V5
def test_v5_new_tests_are_collectable():
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/test_e5_trigger.py"],
        cwd=str(BACKEND), capture_output=True, text=True, encoding="utf-8", timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "test_e5" in proc.stdout
    print("\n[V5] test_e5_trigger.py 可被全量 pytest 正常收集；全量运行数字见交付报告")
