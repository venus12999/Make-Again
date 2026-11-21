"""U-013 召回前的 query-intent 判定：这个 query 在谈论过去的经历/关系/记忆吗？

背景（为什么不是分数阈值）：
    V2 golden 的 q-16「今天晚饭吃什么？」负例注入率 1.0。
    上一次修复尝试（已回退，见 PROGRESS.md U-013）在 recall 层用
    「bm25=0 且 relevance=0 则过滤」，结果误伤纯语义召回
    「失眠难过 → 彻夜睡不着」：负例 vec 分数 0.478 与正例 0.503 区分度不足，
    分数层无法兼顾。本模块因此不在分数层动手，而在**召回前**用确定性特征
    判断 query 意图，判为「与记忆无关」才短路返回空。

设计纪律：
    1. 零 LLM / 零阈值 / 零模型：全部是封闭词表 + 子串匹配，离线可复现。
    2. 保守优先（宁可漏拦，不可误伤）：只在「命中现世杂务词 AND 零记忆线索」
       时判无关；其余一律放行。上一版失败的根因就是误伤，所以这里默认放行。
    3. 纯函数：只依赖 query 文本，不读 DB、不读 user_id，天然不越权、
       不受调用方数据影响。
    4. 开关 USE_QUERY_INTENT 默认关（需 golden 验证）；关闭时 recall() 不调用
       本模块，逐字节走原路径。

边界（诚实标注）：
    - 词表是封闭的手工枚举，不是通用语言理解。新出现的专名若同时命中杂务词
      （例如「晚饭吃什么？豆豆呢」）会漏拦；但不会误伤（记忆线索会直接放行）。
    - 只覆盖中文日常口语 + 少量英文杂务词；方言/拼音/错别字不在范围内。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntentDecision:
    """判定结果。evidence / mundane_hits 保留命中词，便于验收与调试。"""

    memory_related: bool
    evidence: tuple[str, ...] = ()
    mundane_hits: tuple[str, ...] = ()
    reason: str = ""


# ---- 正特征：出现过这些词 = 在谈论经历/关系/记忆（任一命中即放行） ----

_FIRST_PERSON = ("我", "咱", "俺", "本人", "自己")

_RELATION_ENTITY = (
    "父亲", "爸爸", "母亲", "妈妈", "爷爷", "奶奶", "外公", "外婆", "姥姥", "姥爷",
    "爸妈", "父母", "家人", "亲人", "孩子", "儿子", "女儿",
    "前任", "前男友", "前女友", "对象", "伴侣", "老公", "老婆", "丈夫", "妻子",
    "男朋友", "女朋友", "男友", "女友", "朋友", "闺蜜", "兄弟",
    "宠物", "小猫", "小狗", "猫", "狗", "猫毛", "狗毛",
)

_REMINISCENCE = (
    "想起", "记起", "记得", "回忆", "回想", "经历过", "经历", "养过", "养了", "养的",
    "留了", "留下", "留给", "埋在", "埋葬", "安葬", "分手", "去世", "离世",
    "度过", "走过", "有过", "见过", "告别", "重逢", "释怀", "放下",
)

_PAST_TIME = (
    "去年", "前年", "从前", "以前", "之前", "曾经", "当年", "当初", "那年",
    "小时候", "过去", "周年", "那天", "昨天", "昨晚", "前天",
)

_EMOTION_STATE = (
    "难过", "失眠", "伤心", "痛苦", "想念", "思念", "孤独", "寂寞", "焦虑",
    "委屈", "心碎", "崩溃", "压抑", "遗憾", "释怀", "想哭", "哭了",
    "开心", "高兴", "幸福", "释然",
)

# 产品/golden 语料里出现的具体专名。刻意小而有界：新专名走「无杂务词→放行」的
# 保守默认路径，不会误伤；只是漏拦由后续 LLM 意图识别升级（D-015 spike 后再定）。
_NAMED_ENTITY = ("豆豆", "橘子", "汤圆", "桂花糕")

_POSITIVE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("first_person", _FIRST_PERSON),
    ("relation_entity", _RELATION_ENTITY),
    ("reminiscence", _REMINISCENCE),
    ("past_time", _PAST_TIME),
    ("emotion_state", _EMOTION_STATE),
    ("named_entity", _NAMED_ENTITY),
)

# ---- 负特征：现世杂务的「明确请求句式」，只在零正特征时才判无关 ----
# 刻意不放裸主题词（"天气"/"晚饭"/"几点"）：它们也可能是回忆里的名词
# （"那天的天气怎么样"）。只收「吃什么 / 天气怎么样 / 现在几点」这类
# 现在时的请求短语，再配合上面的过去时间词守住误伤防线。

_MUNDANE = (
    "吃什么", "吃啥", "今晚吃", "点外卖", "外卖", "点餐",
    "今天天气", "明天天气", "天气怎么样", "天气预报",
    "现在几点", "几点了", "闹钟",
)

_MUNDANE_EN = (
    "dinner", "lunch", "breakfast", "what to eat", "weather", "what time",
)


def _matched_words(text: str, table: tuple[str, ...]) -> list[str]:
    return [w for w in table if w in text]


def evaluate_query_intent(query: str | None) -> IntentDecision:
    """判定 query 是否在询问/谈论过去的经历、关系或记忆。

    规则（不是阈值）：
        memory_related = 命中任一正特征 or 未命中任何杂务词
    即：只有「杂务词命中 AND 正特征全空」才判为与记忆无关。
    """
    raw = (query or "").strip()
    text = raw.lower()
    if not raw:
        return IntentDecision(memory_related=True, reason="empty_query_default_pass")

    evidence: list[str] = []
    for group_name, table in _POSITIVE_GROUPS:
        hits = _matched_words(text, table)
        if hits:
            evidence.append(f"{group_name}:{'+'.join(hits)}")

    mundane = _matched_words(text, _MUNDANE)
    mundane_en = _matched_words(text, _MUNDANE_EN)
    mundane_hits = mundane + mundane_en

    if evidence:
        return IntentDecision(True, tuple(evidence), tuple(mundane_hits), "positive_memory_cue")
    if mundane_hits:
        return IntentDecision(False, (), tuple(mundane_hits), "mundane_topic_without_memory_cue")
    # 保守默认：看不出是杂务就当作可能在聊过去，放行（避免误伤）
    return IntentDecision(True, (), (), "no_mundane_topic_default_pass")


def is_memory_irrelevant(query: str | None) -> bool:
    """召回门禁用：True 表示该 query 与记忆无关，调用方可直接短路返回空。"""
    return not evaluate_query_intent(query).memory_related
