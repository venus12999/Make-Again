"""S3 HEART 5 维 rubric 离线验收（真实 hy4 数字见 eval/heart_rubric.py CLI 报告）。

这些测试只验证 rubric 定义、盲评构造、解析与对照字段的确定性逻辑，
不调用网络（tests/conftest.py 会阻断出站请求）。真实模型证据由 CLI 单独产出。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND / "eval"))
import heart_rubric as hr  # noqa: E402

BATCH = BACKEND / "eval" / "batches" / "01_中文共情_CPsyCoun.json"


# ---- S3-V1: 5 维定义完整、每维 0-5 anchors ----
def test_v1_five_dimensions_each_with_full_anchors():
    assert len(hr.HEART_DIMENSIONS) == 5
    assert hr.HEART_KEYS == (
        "human_alignment", "empathic_responsiveness", "attunement",
        "resonance", "task_following",
    )
    for d in hr.HEART_DIMENSIONS:
        assert d["name_zh"] and d["name_en"] and d["definition"], d["key"]
        assert set(d["anchors"]) == set(range(6)), d["key"]
        for level, text in d["anchors"].items():
            assert isinstance(text, str) and len(text) >= 8, (d["key"], level, text)


# ---- lead 集成接口 ----
def test_build_judge_rubric_interface_is_json_serializable():
    r = hr.build_judge_rubric()
    assert r["framework"] == "HEART"
    assert r["version"] == hr.HEART_VERSION
    assert r["scale"] == "0-5"
    assert [d["key"] for d in r["dimensions"]] == list(hr.HEART_KEYS)
    assert "control" not in r
    json.dumps(r, ensure_ascii=False)

    text = hr.rubric_to_prompt_text(r)
    for key in hr.HEART_KEYS:
        assert key in text
    assert "0 =" in text and "5 =" in text


def test_build_judge_rubric_with_control_for_ab():
    r = hr.build_judge_rubric(include_control=True)
    assert r["control"]["text"] == hr.LEGACY_RUBRIC_TEXT
    assert "对照字段" in hr.rubric_to_prompt_text(r)


# ---- 盲评：judge 只见 A/B，不知来源 ----
def test_pairwise_prompt_does_not_leak_source():
    prompt = hr.build_pairwise_prompt("我很累", "AAA唯一标记", "BBB唯一标记", "情绪困扰")
    assert "【回复 A】AAA唯一标记" in prompt
    assert "【回复 B】BBB唯一标记" in prompt
    for token in hr.BLIND_LEAK_TOKENS:
        assert token.lower() not in prompt.lower(), token
    for token in hr.BLIND_LEAK_TOKENS:
        assert token.lower() not in hr.PAIRWISE_JUDGE_SYSTEM.lower(), token


def test_blind_slots_swap_and_keep_mapping_only_caller_side():
    a = hr.assign_blind_slots("H", "M", human_on="A")
    assert (a["A"], a["B"], a["human_side"]) == ("H", "M", "A")
    b = hr.assign_blind_slots("H", "M", human_on="B")
    assert (b["A"], b["B"], b["human_side"]) == ("M", "H", "B")
    assert "human_side" not in hr.build_pairwise_prompt("x", b["A"], b["B"])
    try:
        hr.assign_blind_slots("H", "M", human_on="C")
    except ValueError:
        pass
    else:
        raise AssertionError("非法 human_on 应报错")


# ---- 解析：缺维/越界/垃圾一律 None，绝不编造 ----
def _scores(v):
    return {k: v for k in hr.HEART_KEYS}


def test_parse_single_judgment_strict():
    good = json.dumps({"scores": _scores(4), "reason": "具体承接"}, ensure_ascii=False)
    assert hr.parse_single_judgment(good)["scores"] == _scores(4)
    missing = json.dumps({"scores": {k: 4 for k in hr.HEART_KEYS[:-1]}}, ensure_ascii=False)
    assert hr.parse_single_judgment(missing) is None
    oob = json.dumps({"scores": _scores(4) | {"resonance": 9}}, ensure_ascii=False)
    assert hr.parse_single_judgment(oob) is None
    assert hr.parse_single_judgment("not json at all") is None


def test_parse_pairwise_judgment_strict():
    payload = {"A": _scores(5), "B": _scores(1), "prefer": "A", "reason": "A 更具体"}
    out = hr.parse_pairwise_judgment(json.dumps(payload, ensure_ascii=False))
    assert out["prefer"] == "A" and out["A"]["resonance"] == 5 and out["B"]["resonance"] == 1
    bad = dict(payload, prefer="HUMAN")  # 非法偏好
    assert hr.parse_pairwise_judgment(json.dumps(bad)) is None
    bad2 = dict(payload, B={k: 1 for k in hr.HEART_KEYS[:-1]})
    assert hr.parse_pairwise_judgment(json.dumps(bad2)) is None
    assert hr.parse_pairwise_judgment("") is None


def test_separation_report_needs_all_five_dims():
    sep = hr.separation_report(_scores(5), _scores(1))
    assert sep["all_separated"] is True and sep["n_separated"] == 5
    tied = hr.separation_report(_scores(5), _scores(1) | {"attunement": 5})
    assert tied["all_separated"] is False and tied["n_separated"] == 4


# ---- S3-V5: 01 批次维度字段改造 + items 逐字未动 ----
def test_batch_json_points_to_heart_and_keeps_legacy_control():
    d = json.loads(BATCH.read_text(encoding="utf-8"))
    assert "HEART" in d["eval"]
    for k in hr.HEART_KEYS:
        assert k in d["eval"]
    assert d["rubric"]["keys"] == list(hr.HEART_KEYS)
    assert d["rubric"]["spec"] == "eval/heart_rubric.py::build_judge_rubric"
    assert d["eval_control"]["text"] == hr.LEGACY_RUBRIC_TEXT
    assert d["n"] == 17 == len(d["items"])


# S3 只改维度字段、不动 items 数据。基线 items 的 sha256 固化在此（取自 main@7d77bb9 改前版本），
# 不依赖 git show —— 队友仓库同步后 git 历史里没有那个 commit，git show 会拿到改后版本导致误判。
_ITEMS_BASELINE_SHA256 = "c24cad2b3a444283a17448d67278bc73f4f809d661031db7d39f4e828e46f5eb"


def test_batch_items_byte_identical_to_baseline():
    import hashlib
    new = json.loads(BATCH.read_text(encoding="utf-8"))
    items_sha = hashlib.sha256(
        json.dumps(new["items"], ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert items_sha == _ITEMS_BASELINE_SHA256, "items 数据必须逐项不动（sha256 对照基线）"
    assert len(new["items"]) == 17
    assert new["rubric"]["keys"] == list(hr.HEART_KEYS), "维度字段应已切到 HEART 5维"


def test_redact_removes_key_and_bearer():
    fake = "sk-abcdefghijklmnop1234"
    out = hr.redact(f"err {fake} Bearer sk-abcdefghijklmnop1234 tail", [fake])
    assert fake not in out and "sk-abcdefghijklmnop1234" not in out
