"""U-002 · 30 轮 persona 漂移评测的行为验收（全离线、不烧 key、不联网）。

验收项对应任务 F2-V1~V5：
- V1 --mock 30 轮结构跑通：30 轮逐轮留痕 + 第 1/10/20/30 轮抽样 + 8 特征数字。
- V2 真 hy4 产物：若 outputs/persona/30turn_real.json 存在则离线回放校验其结构
  与真实数字（现场真跑由 CLI 完成，慢，不进 pytest）；不存在则 skip（NOT_RUN 不硬凑）。
- V3 第 1 vs 第 30 轮对比有真实数字（命中数 + 逐特征 + 分段命中率）。
- V4 漂移定位：--fault drift 从第 16 轮注入 → onset_turn=16 且 DRIFT。
- V5 本文件随全量 pytest 无回归。

口径：对话脚本/mock 回复均 synthetic-handwritten 构造，非真人数据；真实产物来自
hy4-flash-free 现场生成；特征关键词逐条来自 agent/prompts_registry.py 与 agent/persona.py 原文。
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

import persona_drift_eval as pd  # noqa: E402
from agent import prompts_registry as registry  # noqa: E402

RUNNER = BACKEND / "eval" / "persona_drift_eval.py"
REAL_OUT = REPO / "outputs" / "persona" / "30turn_real.json"


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=str(BACKEND), capture_output=True, text=True, encoding="utf-8", timeout=180,
    )


# ---------------------------------------------------------------- V0: 特征确实来自产品 prompt 原文
def test_features_are_grounded_in_product_prompt_text():
    """先读码纪律：每条规则都能追到产品 prompt 原文，不是自造评测。"""
    companion = registry.get("companion.system")
    breakup = registry.get("breakup.system")
    assert "我理解你" in breakup and "抱抱你" in breakup       # no_empty_platitude 垫场套话
    assert "不诊断" in breakup and "不开药" in breakup         # no_diagnosis
    assert "不评判" in companion and "不催促" in companion      # 不评判/不催促
    assert "(轻轻放下杯子)" in companion                        # no_bracketed_action 原文举例（半角括号）
    assert "不要主动翻出往事" in companion                      # no_unprompted_memory
    assert "不要以问句收尾" in breakup                          # no_question_spam
    assert "用 TA 自己的话或具体的事承接" in breakup            # gentle_specific
    assert "我懂你为什么会这么想" in breakup                    # 明确允许的话，不能当套话误杀
    print("\n[V0] 8 特征逐条可追到 companion.system / breakup.system 原文；"
          "并确认「我懂你为什么会这么想」是允许表述（规则只查开头套话）")


# ---------------------------------------------------------------- 规则单测（构造性）
def test_rule_checks_catch_each_violation():
    anchor = ["心里空落落"]
    good = "心里空落落，这份难受我接住了，你不用一个人扛着。"
    assert all(pd.check_features(good, "我今天又想起他了，心里空落落的。", anchor)[f]["hit"]
               for f in pd.FEATURES)

    cases = {
        "no_judgment": ("你这样太矫情了，活该。", "judgment"),
        "no_diagnosis": ("你可能有点抑郁，建议去看看心理医生，开点药。", "diagnosis"),
        "no_empty_platitude": ("我理解你，这很正常，慢慢来，会好起来的。", "opener"),
        "no_bracketed_action": ("（轻轻放下杯子）我在这里陪你。", "brackets"),
        "no_unprompted_memory": ("我记得你之前说过，那天厨房灯很暗。", "recall"),
        "no_question_spam": ("你现在好点了吗？是不是还想他？", "qmarks"),
        "gentle_specific": ("我在这儿，愿意听你说。", "anchors_hit"),
    }
    for feature, (reply, evidence_key) in cases.items():
        got = pd.check_features(reply, "我今天又想起他了，心里空落落的。", anchor)
        assert got[feature]["hit"] is False, feature
        assert evidence_key in got[feature]["evidence"], (feature, got[feature])

    # 明确允许的表述不能被误杀（D-030 教训：评测规则错会假报产品短板）
    allowed = "心里空落落，我懂你为什么会这么想，但事实可能还有另一面。"
    assert pd.check_features(allowed, "心里空落落", anchor)["no_empty_platitude"]["hit"] is True
    # 真实探针回归：hy4 首轮真回复里的"往前走"是 breakup rebuilding guidance 的正确用语，
    # 曾被误列进套话清单导致 7/8 假失败；此守卫防止规则回退。
    real_shaped = "今天又想起他，心里空落落的，它不代表你没往前走，我在这儿陪你。"
    assert pd.check_features(real_shaped, pd.PROBE, anchor)["no_empty_platitude"]["hit"] is True
    print("\n[规则单测] 7 种违规各被对应特征拦截；「我懂你为什么会这么想」「往前走」不误杀")


# ---------------------------------------------------------------- V1: mock 30 轮结构
def test_v1_mock_30_turns_structure(tmp_path):
    out = tmp_path / "mock30.json"
    proc = _run_cli("--mock", "--out", str(out))
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["mode"] == "mock" and data["scenario"] == "breakup"
    assert len(data["turns"]) == 30 and [t["turn"] for t in data["turns"]] == list(range(1, 31))
    assert data["sample_turns"] == [1, 10, 20, 30]
    assert all(set(t["features"]) == set(pd.FEATURES) for t in data["turns"])
    assert all(t["is_probe"] for t in data["turns"] if t["turn"] in (1, 10, 20, 30))
    assert "构造" in data["口径"] and "不读key" in data["口径"]
    assert data["summary"]["complete"] is True and data["summary"]["exit_code"] == 0
    assert data["analysis"]["drift_detection"]["verdict"] == "NO_DRIFT"
    assert "KEY" not in json.dumps(data, ensure_ascii=False)  # 结构产物不含任何 key 字段
    print(f"\n[V1] --mock 30 轮结构全通：30轮逐条 + 4抽样(同一探针) + 8特征；"
          f"exit={data['summary']['exit_code']}；口径={data['口径']}")


def test_mock_never_reads_key(monkeypatch):
    """mock 模式连 key 文件都不读（结构验收不烧 key 的硬证据）。"""
    def _boom(*a, **kw):
        raise AssertionError("mock 模式不应读取 key 文件")
    monkeypatch.setattr(pd, "load_conf", _boom)
    result = pd.run("mock")
    assert result["status"] == "OK" and result["summary"]["n_turns"] == 30
    assert result["key_source"] == "N/A(mock)"
    print("\n[V1-key] monkeypatch load_conf 抛错也不影响 mock → 确认零 key 读取")


# ---------------------------------------------------------------- V3: 第 1 vs 第 30 轮真实数字
def test_v3_first_vs_last_has_real_numbers(tmp_path):
    out = tmp_path / "mock30.json"
    assert _run_cli("--mock", "--out", str(out)).returncode == 0
    fvl = json.loads(out.read_text(encoding="utf-8"))["analysis"]["first_vs_last"]
    assert fvl["turn1_hits"] == len(pd.FEATURES)
    assert fvl["turn30_hits"] == len(pd.FEATURES)
    assert set(fvl["per_feature"]) == set(pd.FEATURES)
    assert all(v["turn1"] is True and v["turn30"] is True and v["regressed"] is False
               for v in fvl["per_feature"].values())
    print(f"\n[V3] 第1轮命中={fvl['turn1_hits']}/{fvl['total_features']}，"
          f"第30轮命中={fvl['turn30_hits']}/{fvl['total_features']}，逐特征全有数字")


def test_v3_segments_have_three_ten_turn_windows(tmp_path):
    out = tmp_path / "mock30.json"
    assert _run_cli("--mock", "--out", str(out)).returncode == 0
    segs = json.loads(out.read_text(encoding="utf-8"))["analysis"]["segments"]
    assert [s["range"] for s in segs] == ["1-10", "11-20", "21-30"]
    assert all(s["n"] == 10 for s in segs)
    assert all(0.0 <= s["mean_feature_hit_rate"] <= 1.0 for s in segs)
    print("\n[V3-分段] 三段各 10 轮，平均命中率均为真实计算值："
          + ", ".join(f"{s['range']}={s['mean_feature_hit_rate']}" for s in segs))


# ---------------------------------------------------------------- V4: 漂移定位
def test_v4_drift_fault_localizes_onset_turn(tmp_path):
    out = tmp_path / "drift.json"
    proc = _run_cli("--mock", "--fault", "drift", "--out", str(out))
    assert proc.returncode == 1  # 发现漂移 → exit 1（不粉饰）
    data = json.loads(out.read_text(encoding="utf-8"))
    dd = data["analysis"]["drift_detection"]
    assert dd["verdict"] == "DRIFT"
    assert dd["onset_turn"] == 16, dd
    assert dd["per_feature_onset"]["no_empty_platitude"] == 16
    fvl = data["analysis"]["first_vs_last"]
    assert fvl["turn1_hits"] > fvl["turn30_hits"]
    assert "no_empty_platitude" in fvl["regressed_features"]
    print(f"\n[V4] 第16轮注入套话漂移 → onset_turn={dd['onset_turn']}，"
          f"退化特征={fvl['regressed_features']}，第1={fvl['turn1_hits']}→第30={fvl['turn30_hits']}")


# ---------------------------------------------------------------- V2: 真跑产物离线回放（存在才测）
def test_v2_real_artifact_if_present():
    if not REAL_OUT.is_file():
        pytest.skip("真 hy4 产物未生成（NOT_RUN，不硬凑）")
    data = json.loads(REAL_OUT.read_text(encoding="utf-8"))
    assert data["mode"] == "real" and data["scenario"] == "breakup"
    assert "真实 hy4" in data["口径"] and "值不入库" in data["口径"]
    assert "KEY" not in json.dumps(data["turns"], ensure_ascii=False)
    assert len(data["turns"]) in (30, data["summary"]["expected_turns"])
    assert all(set(t["features"]) == set(pd.FEATURES) for t in data["turns"])
    assert set(data["analysis"].get("first_vs_last") or {}) >= {"turn1_hits", "turn30_hits",
                                                                "per_feature", "regressed_features"}
    status = data["summary"]["status"]
    print(f"\n[V2真跑产物] n={data['summary']['n_turns']} status={status} "
          f"exit={data['summary']['exit_code']} "
          f"第1={data['analysis']['first_vs_last']['turn1_hits']} "
          f"第30={data['analysis']['first_vs_last']['turn30_hits']} "
          f"verdict={data['analysis']['drift_detection']['verdict']}")


# ---------------------------------------------------------------- 环境失败降级守卫
def test_env_fail_keeps_partial_turns(monkeypatch):
    """端点挂起/异常 → ENV_FAIL 且保留已跑轮次，不硬等。"""
    import gateway.client as gw

    class _BoomClient:
        settings = type("S", (), {"llm_fast_model": "hy4-flash-free"})()

        def chat(self, *a, **kw):
            raise TimeoutError("endpoint hang (constructed)")

    monkeypatch.setattr(gw, "LLMClient", _BoomClient)
    monkeypatch.setattr(pd, "prepare_real_env", lambda key_file: {"model": "hy4-flash-free"})
    result = pd.run("real", key_file=Path("unused"), max_turns=3)
    assert result["status"] == "ENV_FAIL"
    assert result["summary"]["exit_code"] == 2
    assert result["turns"] == []  # 第 1 轮即挂，无可保留轮次
    assert "TimeoutError" in result["summary"]["env_fail_reason"]
    print("\n[ENV_FAIL守卫] 构造端点挂起 → status=ENV_FAIL, exit=2, reason 只留异常类型")


def test_empty_replies_are_env_fail_not_false_no_drift(monkeypatch):
    """真跑踩过的坑：reasoning 吃满 max_tokens → 29/30 轮正文为空。
    这种数据绝不能被判成 NO_DRIFT（全绿不是标准）。"""
    import gateway.client as gw

    class _EmptyClient:
        settings = type("S", (), {"llm_fast_model": "hy4-flash-free"})()

        def chat(self, *a, **kw):
            assert kw.get("max_tokens") == pd.CHAT_MAX_TOKENS
            return {"content": "", "usage": {"prompt_tokens": 10, "completion_tokens": 150},
                    "tool_calls": []}

    monkeypatch.setattr(gw, "LLMClient", _EmptyClient)
    monkeypatch.setattr(pd, "prepare_real_env", lambda key_file: {"model": "hy4-flash-free"})
    result = pd.run("real", key_file=Path("unused"), max_turns=3)
    assert result["status"] == "ENV_FAIL"
    assert result["summary"]["exit_code"] == 2
    assert "空正文 3/3" in result["summary"]["env_fail_reason"]
    assert result["turns"][0]["empty_reply"] is True
    print("\n[ENV_FAIL守卫-空正文] max_tokens 不足导致空回复 → ENV_FAIL，不谎报 NO_DRIFT")
