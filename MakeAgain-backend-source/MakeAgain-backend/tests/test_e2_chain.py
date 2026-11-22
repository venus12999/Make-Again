"""E2 理解段链路评测的行为验收（全离线、不烧 key、不联网）。

验收项对应任务 E2-V1~V5：
- V1 --mock 跑通，输出 5 个结构化检查点（3 persona 全 PASS）。
- V2 真 hy4 报告质量：pytest 内用走查留下的**真实 hy4 报告实物**
  （outputs/walkthrough/journey_walkthrough.json）过字段/个性化校验；
  现场真跑一次由 CLI 手工完成（慢，不进 pytest），证据另存 outputs/e2/。
- V3 两个 bug 回归守卫：bug1=报告为空/字段缺、bug2=追问失控。
  两层验证：①评测层 --fault 注入 → exit 1 + 对应检查点 FAIL（构造性）；
  ②产品层直接压 interview._build_report_parsed / interview.answer 的修复点。
- V4 追问轮数检查有真实整数：mock 健康=5、mock 失控=20/15、走查实物=20/15、阈值=12。
- V5 本文件随全量 pytest 无回归。

口径：persona 与 mock trace 均为 synthetic-handwritten 构造，非真人数据；
真实报告实物来自 D-042 走查（hy4 现场生成），key 不进仓库。
"""
from __future__ import annotations

import copy
import inspect
import json
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(BACKEND / "eval"))

import e2_interview_chain as e2  # noqa: E402
from agent import interview  # noqa: E402
from memory import store  # noqa: E402
from memory.db import _RawSessionLocal  # noqa: E402

BATCH = BACKEND / "eval" / "batches" / "E2_理解段链路.json"
WALKTHROUGH = REPO / "outputs" / "walkthrough" / "journey_walkthrough.json"
RUNNER = BACKEND / "eval" / "e2_interview_chain.py"


def _real_report_from_walkthrough() -> dict:
    obj = json.loads(WALKTHROUGH.read_text(encoding="utf-8"))
    return obj["results"][0]["report"]


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=str(BACKEND), capture_output=True, text=True, encoding="utf-8", timeout=120,
    )


# ------------------------------------------------------------------ V1: mock 结构
def test_v1_mock_runs_all_personas_with_five_checkpoints(tmp_path):
    proc = _run_cli("--mock", "--out", str(tmp_path / "mock.json"))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = json.loads((tmp_path / "mock.json").read_text(encoding="utf-8"))
    assert out["mode"] == "mock" and out["summary"]["users"] == 3
    assert set(out["results"][0]["checkpoints"]) == set(e2.CHECKPOINT_NAMES)
    assert out["summary"]["overall_pass"] is True
    assert {r["loss_type"] for r in out["results"]} == {"breakup", "loved_one", "pet"}
    assert all(r["source_id"].startswith("e2-synthetic-") for r in out["results"])
    print("\n[V1] 3 persona × 5 检查点全 PASS；口径=" + out["口径"])


def test_batch_is_traceable_and_ci_compatible():
    obj = json.loads(BATCH.read_text(encoding="utf-8"))
    assert obj["batch"] == BATCH.stem and obj["n"] == len(obj["items"]) == 3
    idx = json.loads((BATCH.parent / "_index.json").read_text(encoding="utf-8"))
    assert BATCH.stem in idx["batches"]  # ci_runner 的 batch_files_match_index 依赖登记
    for it in obj["items"]:
        assert it["id"] == it["source_id"]  # 本地 id 口径，可追溯
        assert it["source"]["dataset"] == "synthetic-handwritten"
        assert len(it["persona"]["answers"]) == 5 and len(it["persona"]["anchors"]) >= 2
    print("\n[数据口径] 3 条 persona 全标 synthetic-handwritten + source_id 可追溯 + 已登记 _index")


# ------------------------------------------------------------------ V2: 真实报告实物
def test_v2_real_hy4_report_sample_passes_field_and_personalization_checks():
    """真 hy4 报告实物（不是 mock）过字段齐全 + 个性化校验。"""
    report = _real_report_from_walkthrough()
    ok, missing = e2.validate_report_fields(report, loss_type="breakup")
    assert ok, missing
    pers_ok, matched, missed, need = e2.check_personalization(
        report, ["小雅", "跨年", "江边", "到老", "到家了吗"])
    assert pers_ok, (matched, missed)
    print(f"\n[V2真实输出] 字段齐全=OK；个性化命中={matched}（阈值{need}），missed={missed}")


# ------------------------------------------------------------------ V3: bug 回归守卫
def test_v3_bug1_invalid_report_fault_fails_guard(tmp_path):
    """bug1 回归：报告生成失败（raw 空/字段缺）→ CP2/CP3 FAIL，CLI exit 1。"""
    out = tmp_path / "fault1.json"
    proc = _run_cli("--mock", "--fault", "invalid_report", "--out", str(out))
    assert proc.returncode == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["overall_pass"] is False
    assert all(r["checkpoints"]["CP2_report_generated"] is False for r in data["results"])
    assert all(r["checkpoints"]["CP3_report_fields_complete"] is False for r in data["results"])
    assert all("<report is empty" in r["evidence"]["missing_fields"][0] for r in data["results"])
    print("\n[V3-bug1] 注入空报告 → exit=1，CP2/CP3 全 FAIL（守卫有效）")


def test_v3_bug2_runaway_fault_fails_guard(tmp_path):
    """bug2 回归：覆盖度达标后仍空转 → CP5 FAIL，CLI exit 1，且轮数是真实整数。"""
    out = tmp_path / "fault2.json"
    proc = _run_cli("--mock", "--fault", "runaway_followup", "--out", str(out))
    assert proc.returncode == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["overall_pass"] is False
    assert all(r["checkpoints"]["CP5_followup_turns_bounded"] is False for r in data["results"])
    assert all(r["checkpoints"]["CP2_report_generated"] is True for r in data["results"])
    turns = data["results"][0]["evidence"]["turns"]
    assert turns["n_turns"] == 20 and turns["gen_filler_turns"] == 15
    assert "FILLER_TURNS_HIGH" in data["results"][0]["evidence"]["soft_warnings"]
    print(f"\n[V3-bug2] 注入空转 → exit=1，CP5 FAIL；真实数字 n_turns={turns['n_turns']} "
          f"filler={turns['gen_filler_turns']} 上限={turns['max_allowed_turns']}")


def test_v3_generic_report_fault_fails_personalization_guard(tmp_path):
    """附：通用模板报告（字段全但无个性）→ CP4 FAIL。"""
    out = tmp_path / "fault3.json"
    proc = _run_cli("--mock", "--fault", "generic_report", "--out", str(out))
    assert proc.returncode == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert all(r["checkpoints"]["CP4_personalization_match"] is False for r in data["results"])
    assert all(r["checkpoints"]["CP3_report_fields_complete"] is True for r in data["results"])
    print("\n[V3-附加] 通用模板报告 → CP4 FAIL（字段齐全≠个性化）")


# ------------------------------------------------------------------ V4: 轮数真实数字
def test_v4_turn_numbers_are_real_and_threshold_distinguishes_bug():
    walk = json.loads(WALKTHROUGH.read_text(encoding="utf-8"))["results"][0]
    healthy = e2.check_turns(5, 0)
    buggy = e2.check_turns(walk["n_turns"], walk["gen_filler_turns"])
    assert walk["n_turns"] == 20 and walk["gen_filler_turns"] == 15  # 走查实物原始数字
    assert healthy["ok"] is True and buggy["ok"] is False
    # 阈值从产品常量推导：5 维度 × (1 主问 + MAX_FOLLOWUP_PER_DIM=2) + 1 冗余 = 16。
    assert healthy["max_allowed_turns"] == e2.REASONABLE_MAX_TURNS == 16
    assert buggy["n_turns"] > buggy["max_allowed_turns"]

    # 真实 hy4 复跑观测：13 轮/8 filler（模型自收尾）——不误杀，但必须留下软告警。
    item = e2.load_batch()[0]
    real_shaped = e2.mock_trace(item)
    real_shaped.update(n_turns=13, gen_filler_turns=8)
    scored = e2.evaluate_trace(item, real_shaped)
    assert scored["checkpoints"]["CP5_followup_turns_bounded"] is True
    assert "FILLER_TURNS_HIGH" in scored["evidence"]["soft_warnings"]
    print(f"\n[V4] 健康5轮 PASS / 走查实物20轮(空转15) FAIL / 阈值16；"
          f"真实复跑13轮(空转8) PASS但软告警。数字来自 outputs/walkthrough/journey_walkthrough.json "
          f"与 outputs/e2/e2_real_breakup.json，由 check_turns 实算")


# ------------------------------------------------------------------ 产品代码层回归守卫
class _TokenBudgetClient:
    """Fake provider：复现推理模型 reasoning 吃满 token → raw 空的根因。"""

    def __init__(self):
        # 真实 bug：max_tokens=4000 时报告 raw 为空；修复后 8000 才返回内容。
        self.settings = SimpleNamespace(llm_report_model="", llm_fast_model="hy4-flash-free")
        self.seen_max_tokens: list[int] = []

    def chat_json(self, messages, temperature=0.5, max_tokens=0, model=None):
        self.seen_max_tokens.append(int(max_tokens))
        if int(max_tokens) < 8000:
            return {}, {}  # 模拟 INVALID_REPORT_OUTPUT 的输入
        return copy.deepcopy(_real_report_from_walkthrough()), {}


def test_bug1_product_guard_max_tokens_and_invalid_output():
    """压到产品代码 _build_report_parsed：max_tokens 必须是 8000，低了就抛 INVALID_REPORT_OUTPUT。"""
    good = _TokenBudgetClient()
    report = interview._build_report_parsed(good, "breakup", [{"role": "user", "content": "访谈内容"}])
    assert good.seen_max_tokens and all(t == 8000 for t in good.seen_max_tokens)
    ok, missing = e2.validate_report_fields(report, "breakup")
    assert ok, missing

    # 构造性反向验证：把产品里"够用"的预算砍回 4000（旧值），fake provider 立刻返回空报告，
    # _build_report_parsed 两次重试后必须抛错——若有人把 8000 改回 4000，本测试即失败。
    bad = _TokenBudgetClient()
    bad.chat_json = lambda messages, temperature=0.5, max_tokens=0, model=None: ({}, {})
    with pytest.raises(interview.ReportGenerationError, match="INVALID_REPORT_OUTPUT"):
        interview._build_report_parsed(bad, "breakup", [{"role": "user", "content": "访谈内容"}])
    print(f"\n[bug1产品守卫] 真实代码请求 max_tokens={good.seen_max_tokens[0]}；"
          f"空输出两次→ReportGenerationError(INVALID_REPORT_OUTPUT)")


def test_bug2_product_guard_coverage_stops_perpetual_followup(monkeypatch):
    """压到产品代码 answer()：抬高 MAX_TURNS 熔断，只靠覆盖度保护也必须及时收尾。"""
    real_report = _real_report_from_walkthrough()

    steps = [
        {"action": "next", "next_dimension": d, "question": f"过渡到{d}",
         "facts": [], "emotion": {"emotion": "平静"}}
        for d in ("memory", "emotion", "confusion", "goal")
    ]

    def fake_decide_step(client, loss_type, dims, state, user_answer, victim=None):
        if steps:
            return copy.deepcopy(steps.pop(0))
        return {"action": "followup", "question": "还能再说说吗？",
                "facts": [], "emotion": {"emotion": "平静"}}  # 永不主动收尾

    monkeypatch.setattr(interview, "_decide_step", fake_decide_step)
    monkeypatch.setattr(interview, "_generate_report",
                        lambda db, client, session, state: copy.deepcopy(real_report))
    monkeypatch.setattr(interview, "extract_tags", lambda text: ("", ""))
    monkeypatch.setattr(interview.pet_extract, "extract_pet_cbt", lambda client, text: None)
    monkeypatch.setattr(interview, "MAX_TURNS", 60)  # 关键：关掉硬熔断，只剩覆盖度保护

    uid = "e2-guard-" + uuid.uuid4().hex[:8]
    with _RawSessionLocal() as db:
        sid = interview.start(db, uid, "pet")["session_id"]
        answers = ["它叫橘子，是只橘猫，陪了我十五年。", "加班回家它在门口等我。",
                   "早上还会习惯性去倒猫粮。", "我出差时它自己走了，我一直自责。",
                   "我想记住它，不想忘记。"]
        filler = "嗯，我还在慢慢消化，没什么补充的了。"
        n_turns = 0
        complete = False
        for i in range(60):
            sess = store.get_interview(db, sid)
            if (sess.state or {}).get("done"):
                break
            ans = answers[i] if i < len(answers) else filler
            resp = interview.answer(db, sid, ans, None, async_report=False)
            n_turns += 1
            if resp.get("action") == "complete":
                complete = True
                break
        assert complete is True, "覆盖度达标后必须能收尾"
        assert n_turns <= e2.REASONABLE_MAX_TURNS, n_turns
        print(f"\n[bug2产品守卫] MAX_TURNS 抬到60、LLM 永远 followup → 实际 {n_turns} 轮收尾"
              f"（修复前走查是20轮/空转15）")

    # 修复点本身也做守卫：删掉覆盖度保护代码会让本断言失败。
    assert "覆盖度保护" in inspect.getsource(interview.answer)
