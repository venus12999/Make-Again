"""S2 judge 金标校准的行为测试（离线、确定性 fake judge，不碰真实模型/key）。

注意：conftest 把 MOCK_LLM=true 并封了外网，所以这里一律注入 FakeClient；
真实模型数字由 `python eval/judge_calibration.py` 独立验收（见 work 下证据输出）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "eval"))

import pytest  # noqa: E402

import judge_calibration as jc  # noqa: E402


class FakeClient:
    """按 payload 里的 user_input 找回金标条目，再按策略返回 judge JSON。"""

    def __init__(self, mode: str, bad_dim: str | None = None):
        assert mode in {"perfect", "position_bias", "always_pass", "one_dim_bad"}
        self.mode = mode
        self.bad_dim = bad_dim
        self.calls = 0
        self._by_input = {}
        for it in jc.load_golden()["items"]:
            self._by_input[it["user_input"]] = it

    def chat(self, messages, temperature=0.0, max_tokens=None, model=None):
        self.calls += 1
        payload = json.loads(messages[-1]["content"])
        item = self._by_input[payload["user_input"]]
        is_pair = "reply_A" in payload
        if is_pair:
            if self.mode == "position_bias":
                return {"content": '{"choice":"A","reason":"fake 总选第一个"}'}
            # 正确策略：好回复赢。A/B 身份由 swap 决定，payload 里 swap 不直接给出，
            # 用两个候选文本本身判断谁是好回复。
            expected = item["candidate_reply"] if item["human_label"] == "pass" else item["anchor_reply"]
            choice = "A" if payload["reply_A"] == expected else "B"
            return {"content": json.dumps(
                {"choice": choice, "reason": "fake 选更好的"}, ensure_ascii=False)}

        # 二元判定
        if self.mode in {"position_bias", "perfect"}:
            passed = item["human_label"] == "pass"
        elif self.mode == "always_pass":
            passed = True
        elif self.mode == "one_dim_bad":
            passed = True if item["dimension"] == self.bad_dim else (item["human_label"] == "pass")
        else:  # pragma: no cover
            passed = False
        return {"content": json.dumps({"pass": passed, "reason": "fake"}, ensure_ascii=False)}


# ------------------------------------------------------------------ S2-V1
def test_golden_shape_caliber_and_traceability():
    gold = jc.load_golden()
    items = gold["items"]
    assert len(items) == 30
    assert gold["caliber"] == "人工金标-非真实用户数据"
    from collections import Counter
    assert dict(Counter(i["batch"] for i in items)) == {
        "01_中文共情_CPsyCoun": 8, "02_支持策略_ESConv": 7,
        "03_情绪识别_GoEmotions": 7, "04_危机安全_CueSupport": 8,
    }
    assert set(Counter(i["dimension"] for i in items).values()) <= {7, 8}
    assert set(i["human_label"] for i in items) <= {"pass", "fail"}
    assert all(len(i["critique"]) >= 8 for i in items)
    assert all(i["caliber"] == "人工金标-非真实用户数据" for i in items)
    assert all(i["anchor_reply"] != i["candidate_reply"] for i in items)
    # source_id 必须能回溯到 eval/batches/ 里的真实条目
    batchdir = BACKEND / "eval" / "batches"
    for it in items:
        bname, sid = it["source_id"].split("#", 1)
        d = json.loads((batchdir / f"{bname}.json").read_text(encoding="utf-8"))
        assert sid in {x["id"] for x in d["items"]}, it["source_id"]


# ------------------------------------------------------------------ 统计正确性
def test_cohens_kappa_known_values():
    perfect = jc.cohens_kappa([(True, True)] * 10 + [(False, False)] * 10)
    assert perfect["kappa"] == 1.0 and perfect["agreement"] == 1.0
    # 6 pass/4 fail 的人工，judge 全 pass：po=0.6, pe=0.6 → kappa=0
    chance = jc.cohens_kappa([(True, True)] * 6 + [(False, True)] * 4)
    assert chance["kappa"] == 0.0 and chance["agreement"] == 0.6
    # 全正例 → pe==1，kappa 数学上无定义，不许编造
    undefined = jc.cohens_kappa([(True, True)] * 5)
    assert undefined["kappa"] is None and undefined["note"] == "UNDEFINED_PE_EQ_1"


def test_parse_rejects_garbage_never_invents_label():
    assert jc.parse_binary('{"pass": true}') is True
    assert jc.parse_binary('{"pass": "不达标"}') is False
    assert jc.parse_binary("garbage") is None
    assert jc.parse_binary('{"score": 5}') is None
    assert jc.parse_choice('{"choice":"A"}') == "A"
    assert jc.parse_choice('{"choice":"C"}') is None
    assert jc.pair_identity("A", swap=False) == "candidate"
    assert jc.pair_identity("A", swap=True) == "anchor"


# ------------------------------------------------------------------ S2-V4
def test_position_bias_detected_and_zero_for_stable_judge(tmp_path):
    stable = jc.run_calibration(FakeClient("perfect"), out_path=tmp_path / "stable.json",
                                curve_path=tmp_path / "stable.jsonl", show_progress=False)
    pos = stable["overall"]["position"]
    assert pos["position_bias_rate"] == 0.0
    assert pos["first_position_rate"] == 0.5
    assert pos["pair_agreement_before"] == 1.0 and pos["pair_agreement_after"] == 1.0

    biased = jc.run_calibration(FakeClient("position_bias"), out_path=tmp_path / "biased.json",
                                curve_path=tmp_path / "biased.jsonl", show_progress=False)
    bpos = biased["overall"]["position"]
    assert bpos["position_bias_rate"] == 1.0          # 换序后偏好全部翻转 = 纯位置偏差
    assert bpos["first_position_rate"] == 1.0


# ------------------------------------------------------------------ S2-V5
def test_per_dimension_consistency_is_separate_not_mixed(tmp_path):
    res = jc.run_calibration(FakeClient("one_dim_bad", bad_dim="crisis_safety"),
                             out_path=tmp_path / "separation.json",
                             curve_path=tmp_path / "separation.jsonl", show_progress=False)
    per = res["per_dimension"]
    assert set(per) == set(jc.DIMENSIONS)
    assert per["crisis_safety"]["kappa"] == 0.0       # 该维度全判 pass，4 pass/4 fail
    for d in ("empathy_specificity", "strategy_appropriateness", "emotion_accuracy"):
        assert per[d]["kappa"] == 1.0                  # 其余维度全对
    assert per["crisis_safety"]["position"]["n"] == 8  # 每维度单独统计，不是混在一起


# ------------------------------------------------------------------ S2-V3
def test_gate_blocks_low_consistency_and_passes_calibrated():
    low = jc.judge_consistency_gate(consistency={"overall": {"kappa": 0.3, "agreement": 0.5}})
    assert low["verdict"] == "JUDGE_NOT_CALIBRATED"
    assert low["calibrated"] is False
    assert low["merge_advice_allowed"] is False
    assert "judge未校准，结论不可信" in low["note"]

    ok = jc.judge_consistency_gate(consistency={"overall": {"kappa": 0.75, "agreement": 0.83}})
    assert ok["verdict"] == "JUDGE_CALIBRATED"
    assert ok["merge_advice_allowed"] is True

    missing = jc.judge_consistency_gate(path="/definitely/not/exist.json")
    assert missing["calibrated"] is False and "judge未校准，结论不可信" in missing["note"]
    incomplete = jc.judge_consistency_gate(consistency={"overall": {"kappa": None, "agreement": 0.9}})
    assert incomplete["calibrated"] is False


# ------------------------------------------------------------------ S2-V2 + V4 持久化
def test_run_calibration_writes_json_and_appends_curve(tmp_path):
    out, curve = tmp_path / "consistency.json", tmp_path / "curve.jsonl"
    jc.run_calibration(FakeClient("perfect"), out_path=out, curve_path=curve, show_progress=False)
    jc.run_calibration(FakeClient("perfect"), out_path=out, curve_path=curve, show_progress=False)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["caliber"] == "人工金标-非真实用户数据"
    assert data["n_items"] == 30
    assert data["overall"]["kappa"] == 1.0
    lines = curve.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2                              # 追加写，不覆盖历史
    assert all(json.loads(x)["kappa"] == 1.0 for x in lines)
    # 落盘后前置门能直接消费
    gate = jc.judge_consistency_gate(path=out)
    assert gate["merge_advice_allowed"] is True


def test_key_file_missing_is_not_run(tmp_path):
    _, status, reason = jc.read_key(tmp_path / "no-key.txt")
    assert status == "NOT_RUN" and "NOT_RUN" in reason and "不存在" in reason
