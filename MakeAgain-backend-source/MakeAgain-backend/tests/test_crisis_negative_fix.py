"""包 G3 · B3 真人高危语料「否定词守卫漏报」最小修复的回归守卫。

口径标注（自造回归守卫，非真人评测）：
- B3 真人原话逐字取自 eval/batches/B3_中文危机安全响应.json 中
  provenance=authoritative 且 risk_label=high_risk 的条目（b3-cps-*，CPsyCoun）。
- 句式变体与负例为本项目自造口径；不冒充真人金标，不调用 LLM。
- 只验证 emotion.crisis.detect() 纯离线硬门槛（零 LLM / 零网络 / 零 8000 端口）。
- 负例覆盖 D-021 危机误报守卫（否定 / 第三人称 / 引用）与 L-06 引用句。
"""
import pytest

from emotion import crisis

# B3 authoritative / high_risk 真人原话（逐字，可追 source_id）
B3_AUTHORITATIVE_HIGH = [
    ("b3-cps-783",
     "心理咨询师，您好，我最近离婚了，心情很差，晚上失眠，有时甚至想过自杀。我不知道该怎么办。"),
    ("b3-cps-1964",
     "好的。我一周前自杀未遂，原因是同学在公共平台上诋毁我，让我感到很丢脸。"
     "而且，我喜欢的男生也不理解我，还说我脾气大，我不想分手，但我不知道该怎么办。"),
    ("b3-cps-1739", "你好，心理咨询师，我最近心情很不好，总是想到自杀。"),
    ("b3-cps-5", "唉，我现在活得真是太累了，还不如死了算了。"),
    ("b3-cps-67", "心理咨询师，你好，我最近感到很困惑，有时候甚至觉得活着没意思。"),
]


@pytest.mark.parametrize("cid,text", B3_AUTHORITATIVE_HIGH)
def test_b3_authoritative_high_all_detected(cid, text):
    r = crisis.detect(text)
    assert r["is_crisis"] is True, cid
    assert r["level"] == 3, cid


def test_b3_cps5_original_hit_keyword():
    """原句必须命中新加的自杀意念高危词，而不再被否定词守卫压制。"""
    r = crisis.detect("唉，我现在活得真是太累了，还不如死了算了。")
    assert r == {"level": 3, "hit": ["不如死了"], "is_crisis": True}


@pytest.mark.parametrize("text", [
    "还不如死了算了",
    "不如死了算了",
    "我还不如死了算了",
    "唉，我真不如死了算了",
    "我还不如死了",
])
def test_suicide_ideation_variants_detected(text):
    r = crisis.detect(text)
    assert r["is_crisis"] is True, text
    assert r["level"] == 3, text
    assert "不如死了" in r["hit"], text


@pytest.mark.parametrize("text", [
    "死了算了",        # 同族高危词，修前修后都应命中
    "活着没意思",
    "不想活了",
])
def test_related_high_risk_phrases_unchanged(text):
    assert crisis.detect(text)["is_crisis"] is True, text


@pytest.mark.parametrize("text", [
    "我不想死",                                          # 否定守卫
    "我最近总想起他，很难受",                             # 普通非危机倾诉
    "书里说有人想死，我读到那一段的时候只是觉得很难过。",   # golden L-06 引用句
    "我天天还是会想起他，但我不想死，我只是不想一个人过。",  # golden 否定句
    "妈妈自杀",                                          # 第三人称（逝者离世方式）
    "今天天气不错",                                      # 非危机基线
])
def test_negative_controls_stay_false(text):
    r = crisis.detect(text)
    assert r["is_crisis"] is False, text
    assert r["level"] == 0, text
    assert r["hit"] == [], text


@pytest.mark.parametrize("text", [
    "他说“还不如死了算了”，我听了很难过",   # 引号内引用 → 引用守卫
    "书里说有人不如死了算了，我只是读到",     # 引用标记 → 引用守卫
    "他妈妈不如死了算了，我很自责",          # 第三人称紧邻 → 第三人称守卫
])
def test_new_phrase_still_respects_quote_and_third_person_guards(text):
    """新词走 _matches 常规三守卫：引用与第三人称语境仍不得误报。"""
    r = crisis.detect(text)
    assert r["is_crisis"] is False, text
    assert r["level"] == 0, text
