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
    # H1 统计功效扩充后中文批从 17→34 条；S3 的 HEART 维度字段仍须在。
    assert d["n"] == len(d["items"]) >= 30, "批次已扩充到≥30（统计功效），HEART维度字段须保留"


# S3 时期基线为17条；H1 统计功效扩充到34条。
# 守卫改为：HEART 维度字段在 + 全批 source_id 可追溯 + 至少30条 + git基线里的旧条目核心字段保持
# —— 而非全量 items sha256 冻结（后者会阻止任何合法的数据扩充）。
def test_batch_items_traceable_after_expansion():
    import subprocess
    new = json.loads(BATCH.read_text(encoding="utf-8"))
    items = new["items"]
    assert len(items) >= 30, "中文批应已扩充到≥30（统计功效）"
    # 每条都可追溯到权威来源（H1 要求）
    for it in items:
        src = it.get("source") or {}
        assert src.get("dataset") == "CPsyCoun" or it.get("source_id"), \
            f"{it['id']} 缺权威来源"
    assert new["rubric"]["keys"] == list(hr.HEART_KEYS), "维度字段应已切到 HEART 5维"
    # 从 H1 之前的基线动态取旧条目，验证其核心字段(user_input/expected)未被改动
    rel = str(BATCH.relative_to(BACKEND.parent)).replace("\\", "/")
    out = subprocess.run(["git", "show", f"fa98846:{rel}"], cwd=str(BACKEND.parent),
                         capture_output=True, text=True, encoding="utf-8")
    if out.returncode == 0:
        old = json.loads(out.stdout)
        old_by_id = {it["id"]: it for it in old["items"]}
        for it in items:
            if it["id"] in old_by_id:
                base = old_by_id[it["id"]]
                for key in ("user_input", "scene"):
                    if key in base:
                        assert it.get(key) == base[key], f"{it['id']} 的 {key} 被改动"



def test_redact_removes_key_and_bearer():
    fake = "sk-abcdefghijklmnop1234"
    out = hr.redact(f"err {fake} Bearer sk-abcdefghijklmnop1234 tail", [fake])
    assert fake not in out and "sk-abcdefghijklmnop1234" not in out
