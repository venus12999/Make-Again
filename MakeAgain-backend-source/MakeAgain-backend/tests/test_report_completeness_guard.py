"""U-017 回归守卫：报告完整性校验，残缺/截断报告绝不落库。

2026-11-23 E6 真跑暴露：hy4 输出被截断，parse_json 容错解析出"部分合法 JSON"，
旧 _valid_report 用 get(k, default) 让缺失字段拿默认值过关 → 残缺报告落库。
本测试固化守卫：9 个必需字段必须存在且非空。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
from agent.interview import _valid_report, _REPORT_REQUIRED_STR, _REPORT_REQUIRED_DICT  # noqa: E402


def _full_report() -> dict:
    return {
        "title": "带着裂缝往前走",
        "keywords": ["付出型", "被背叛", "慢慢消化"],
        "summary": "这段关系里你投入得很认真，值得被看见。",
        "quote": "不是你不值得，是承诺被辜负。",
        "user_portrait": {"特质": "付出型"},
        "object_portrait": {"称呼": "小雅", "关系": "前任"},
        "goal": {"type": "carry_on", "label": "带着记忆走下去", "reason": "想往前走"},
        "heal_plan": {"summary": "慢慢重建生活", "stages": [{"title": "先照顾自己", "desc": "吃饭睡觉", "time": "本周"}]},
        "relationship_analysis": {"模式": "付出与失衡"},
    }


def test_v1_full_report_passes():
    assert _valid_report(_full_report()) is True


def test_v2_e6_truncated_real_artifact_rejected():
    """E6 真跑那份残缺报告（只有4字段）必须被拒——这是 U-017 的直接回归守卫。"""
    p = BACKEND.parent / "outputs" / "e6" / "e6_real_breakup.json"
    if not p.exists():
        pytest.skip("缺 E6 真跑产物")
    data = json.loads(p.read_text(encoding="utf-8"))
    report = data["results"][0].get("report")
    assert report is not None
    assert _valid_report(report) is False, "截断报告必须被 _valid_report 拒绝"


def test_v3_each_required_field_missing_rejected():
    """逐个删必需字段：每个都必须导致校验失败（防 get-default 蒙混）。"""
    full = _full_report()
    for key in _REPORT_REQUIRED_STR + ("keywords",) + _REPORT_REQUIRED_DICT:
        broken = dict(full)
        broken.pop(key)
        assert _valid_report(broken) is False, f"缺 {key} 应被拒"


def test_v4_each_required_field_empty_rejected():
    """逐个把必需字段设空：字符串空/dict 空/keywords 空列表都必须被拒。"""
    full = _full_report()
    for key in _REPORT_REQUIRED_STR:
        broken = dict(full); broken[key] = "   "
        assert _valid_report(broken) is False, f"{key} 空白应被拒"
    for key in _REPORT_REQUIRED_DICT:
        broken = dict(full); broken[key] = {}
        assert _valid_report(broken) is False, f"{key} 空 dict 应被拒"
    broken = dict(full); broken["keywords"] = []
    assert _valid_report(broken) is False, "keywords 空列表应被拒"


def test_v5_report_token_budget_raised():
    """源码守卫：max_tokens 已提到 12000（reasoning 留足余量）。"""
    src = (BACKEND / "agent" / "interview.py").read_text(encoding="utf-8")
    assert "max_tokens=12000" in src
    assert "max_tokens=8000" not in src.split("def _build_report_parsed")[1].split("def ")[0]
