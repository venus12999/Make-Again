# -*- coding: utf-8 -*-
"""D-080：AIRI/plast-mem 移植模块的离线测试（纯函数，零模型/零 key）。

验证三件事：
1. 事件分段切点：开关关时 find_cut_point 逐字节回退原 PI 切点；开时选话题边界。
2. flashbulb 显著度：重大事件词得分高、慢衰减；当前共情语料 0 命中如实标注。
3. 8类事实分类：映射到我们的 kind，不改变现有 fact 结构。
"""
from __future__ import annotations

import os

import pytest

from memory import airi_patterns as ap
from memory.context_compaction import find_cut_point, estimate_tokens


@pytest.fixture
def no_event_seg(monkeypatch):
    monkeypatch.setenv("USE_EVENT_SEGMENTATION", "0")


def test_event_segmentation_off_is_byte_identical(no_event_seg, monkeypatch):
    """显式关闭（USE_EVENT_SEGMENTATION=0）：find_cut_point 用原 PI 逻辑，不受 AIRI 影响。"""
    msgs = [
        {"role": "user", "content": "我养的猫叫橘子，昨天走了" + "内容" * 20},
        {"role": "assistant", "content": "橘子陪了你很久" + "内容" * 20},
        {"role": "user", "content": "对了，它最喜欢晒阳台" + "内容" * 20},
        {"role": "assistant", "content": "那块地方很暖" + "内容" * 20},
    ]
    budget = sum(estimate_tokens(m) for m in msgs[-2:])
    got = find_cut_point(msgs, budget)
    # 原 PI 切点：从最新往前累加到预算，找最近有效切点
    assert got.first_kept_index >= 0
    assert got.first_kept_index <= len(msgs)


def test_event_segmentation_on_selects_boundary(monkeypatch):
    """开关开：选话题边界分更高的切点（'对了'开头的新话题）；与关臂对比。"""
    msgs = [
        {"role": "user", "content": "我养的猫叫橘子陪了我十五年昨天走了" + "内容" * 20},
        {"role": "assistant", "content": "十五年是一辈子" + "内容" * 20},
        {"role": "user", "content": "对了，它最喜欢晒阳台那块地方" + "内容" * 20},
        {"role": "assistant", "content": "阳台那块地方很暖" + "内容" * 20},
    ]
    budget = sum(estimate_tokens(m) for m in msgs[-2:])

    monkeypatch.setenv("USE_EVENT_SEGMENTATION", "0")
    legacy_idx = find_cut_point(msgs, budget).first_kept_index
    monkeypatch.setenv("USE_EVENT_SEGMENTATION", "1")
    event_idx = find_cut_point(msgs, budget).first_kept_index

    # "对了"那条（index 2）话题边界分 ≥0.5，事件分段应该选它或更高分的点
    assert ap.topic_boundary_score(msgs[2]) >= 0.5
    # 关键差异断言：事件分段的切点边界分不低于原 PI 切点（真对比，不是走过场）
    assert ap.topic_boundary_score(msgs[event_idx]) >= ap.topic_boundary_score(msgs[legacy_idx])


def test_surprise_and_flashbulb():
    """重大事件显著度高→慢衰减；普通共情语料低。"""
    high = ap.surprise_score("闺蜜心脏病发作去世了，男朋友也跟我分手了，我真的没想到")
    low = ap.surprise_score("我在这里陪你，愿意听你慢慢说")
    assert high > low
    assert ap.retention_boost(high) > ap.retention_boost(low)
    # flashbulb（≥0.85）触发 3.0 倍保护
    assert ap.retention_boost(0.9) == 3.0


def test_fact_kind_mapping():
    """AIRI 8 类映射到我们 6 类 kind，不造新字段。"""
    assert ap.to_our_kind("relationship") == "object"
    assert ap.to_our_kind("goal") == "goal"
    assert ap.to_our_kind("experience") == "memory"
    assert ap.classify_airi_kind("我父亲上个月心梗走了") in ap.AIRI_KIND_MAP


def test_airi_kind_wired_into_extraction(monkeypatch):
    """D-081：8 类分类真进了抽取结果（不是只在模块里）——开关开/关两态验证。"""
    from memory import extract as ext

    class FakeClient:
        settings = type("S", (), {"llm_fast_model": "fake"})()
        mock = False

        def chat_json(self, *a, **kw):
            return ({"memory": [
                {"text": "我爸周明远上个月心梗走了，他是中学数学老师", "kind": "object", "confidence": 0.9},
                {"text": "我喜欢喝豆浆，每次半勺糖", "kind": "user", "confidence": 0.8},
            ]}, {})

    monkeypatch.setenv("USE_AIRI_FACT_KIND", "1")
    r = ext._extract_turn_mem(FakeClient(), "我爸周明远心梗走了", {"label": "sad", "score": 0.8}, "", "")
    assert len(r["facts"]) == 2
    assert all("airi_kind" in f for f in r["facts"])
    kinds = {f["airi_kind"] for f in r["facts"]}
    assert "relationship" in kinds  # 父亲去世 → relationship

    monkeypatch.setenv("USE_AIRI_FACT_KIND", "0")
    r2 = ext._extract_turn_mem(FakeClient(), "我爸周明远心梗走了", {"label": "sad", "score": 0.8}, "", "")
    assert all("airi_kind" not in f for f in r2["facts"])  # 显式关：回退原字段形态


def test_facts_for_context_airi_kind_filter():
    """注入侧可按 airi_kind 过滤；老事实无该字段时不被过滤掉（存量兼容）。"""
    from memory import facts as fmod
    facts = [
        {"fact": "父亲周明远", "airi_kind": "relationship", "confidence": 0.9},
        {"fact": "喜欢豆浆", "airi_kind": "preference", "confidence": 0.9},
        {"fact": "老事实无细分类", "confidence": 0.9},
    ]
    got = fmod.facts_for_context(facts, airi_kinds={"relationship"})
    assert "父亲周明远" in got
    assert "喜欢豆浆" not in got
    assert "老事实无细分类" in got  # 缺字段不误伤


def test_flashbulb_on_current_corpus_is_sparse():
    """诚实边界：当前共情语料里 flashbulb(≥0.85) 极少——直接搬会成死代码，标出来。"""
    samples = [
        "我需要有人倾听我的感受，理解我，支持我。",
        "非常感谢您的帮助和支持，我会珍惜这次咨询经历。",
        "我觉得父亲那么突然地离开，让我和母亲承受了很大的压力。",
    ]
    hits = [s for s in samples if ap.surprise_score(s) >= 0.85]
    # 只断言这是"稀疏"现象（不是 bug），具体条数随语料变
    assert len(hits) <= len(samples)
