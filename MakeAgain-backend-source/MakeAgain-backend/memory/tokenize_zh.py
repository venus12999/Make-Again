"""中文分词 / 词形归一（纯函数，零重依赖；Mem0 官方确认有可测收益，见 D-063）。

设计口径（context engineering research P1 / D-063）：
- 零重依赖优先：只用标准库 re / unicodedata + 一张保守的繁简映射表；jieba 为可选增强，
  缺失时自动退回「单字 + 相邻二元字」切分，绝不因为没装 jieba 而报错。
- 纯函数：normalize / tokenize / signature 都不读写 DB、不调 LLM、不依赖环境开关，
  同一输入同一输出（signature 刻意不依赖 jieba，保证跨环境可复现）。
- 归一项：NFKC 全角→半角、繁体→简体（保守子集）、ASCII 小写、零宽字符剔除、空白折叠。
- 本模块只提供工具函数，**未接任何主链路**；接线与收益评测由 lead 决定（未接线=未验证）。

用法：
    normalize("ＩＴ＇Ｓ　離別療愈")  -> "it's 离别疗愈"
    tokenize("我最近很难过")        -> ["最近", "难过"]（jieba 可用时；否则含二元字）
    signature("難過") == signature("难过")  -> True
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# 保守繁→简单字表：只收录陪伴语料高频字，不是完整 OpenCC 转换。
# 不认识的字原样保留（宁可漏转，不可错转导致语义漂移）。
_TRAD_TO_SIMP: dict[str, str] = {
    "離": "离", "別": "别", "彆": "别", "療": "疗", "癒": "愈", "這": "这",
    "個": "个", "們": "们", "時": "时", "間": "间", "記": "记", "憶": "忆",
    "愛": "爱", "親": "亲", "貓": "猫", "歲": "岁", "聽": "听", "說": "说",
    "話": "话", "東": "东", "後": "后", "裡": "里", "裏": "里", "過": "过",
    "還": "还", "學": "学", "會": "会", "難": "难", "憂": "忧", "傷": "伤",
    "開": "开", "關": "关", "為": "为", "與": "与", "從": "从", "對": "对",
    "買": "买", "賣": "卖", "進": "进", "遠": "远", "長": "长", "斷": "断",
    "覺": "觉", "溫": "温", "風": "风", "陽": "阳", "陰": "阴", "淚": "泪",
    "實": "实", "準": "准", "備": "备", "幫": "帮", "讓": "让", "給": "给",
    "變": "变", "處": "处", "緒": "绪", "續": "续", "聯": "联", "繫": "系",
    "係": "系", "鍵": "键", "盤": "盘", "電": "电", "題": "题", "評": "评",
    "價": "价", "屬": "属", "導": "导", "遊": "游", "戲": "戏", "藝": "艺",
    "術": "术", "應": "应", "該": "该", "當": "当", "總": "总", "結": "结",
    "終": "终", "於": "于", "並": "并", "現": "现",
    "見": "见", "樣": "样", "麼": "么", "種": "种", "類": "类", "護": "护",
    "麗": "丽", "華": "华", "語": "语", "詞": "词", "讀": "读", "寫": "写",
    "數": "数", "據": "据", "檢": "检", "測": "测", "構": "构", "運": "运",
    "態": "态", "強": "强", "調": "调", "節": "节", "奏": "奏", "壓": "压",
    "鬱": "郁", "慮": "虑", "剛": "刚", "靈": "灵",
    "喪": "丧", "殯": "殡", "墳": "坟", "禱": "祷", "靜": "静",
    "獨": "独", "瞭": "了", "幾": "几", "點": "点", "鐘": "钟",
}
_TRAD_TABLE = str.maketrans(_TRAD_TO_SIMP)

_ZERO_WIDTH = frozenset("\u200b\u200c\u200d\u2060\ufeff\u00ad")

# 停用词：只放明确的虚词/语气词，不放语义线索词（如「已经」「但是」保留给下游判据）。
_ZH_STOP = frozenset({
    "的", "了", "是", "在", "和", "与", "也", "都", "而", "及", "我", "你",
    "他", "她", "它", "这", "那", "有", "就", "很", "被", "把", "吗", "呢",
    "吧", "啊", "哦", "嗯", "着", "得", "一个", "一下", "真的", "其实",
    "然后", "所以", "因为", "如果", "虽然", "但是",
})
_EN_STOP = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "and", "or",
    "in", "on", "for", "it", "i", "you", "my", "me", "this", "that", "so",
    "very", "really", "just", "then",
})
STOPWORDS = _ZH_STOP | _EN_STOP

# 只有「双虚词」组合才当作无信息二元字丢弃；「用户」「现在」这类保留。
_FUNCTION_CHARS = frozenset("的了是在和与也都而及我很被把你他她它这那有就还再又从对吗呢吧啊哦嗯")

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_WORD_RE = re.compile(r"[a-z0-9]+")

_jieba_module: object | None = None
_jieba_checked = False


def _get_jieba():
    """惰性可选导入；首次检查后缓存结果，缺失时返回 None。"""
    global _jieba_module, _jieba_checked
    if not _jieba_checked:
        _jieba_checked = True
        try:
            import jieba  # type: ignore
            _jieba_module = jieba
        except Exception:
            _jieba_module = None
    return _jieba_module


def normalize(text: object) -> str:
    """词形归一：NFKC（全角→半角）+ 保守繁→简 + ASCII 小写 + 零宽剔除 + 空白折叠。

    保留标点（由 tokenize 决定如何处理）；None / 非字符串安全转空串。
    """
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    s = s.translate(_TRAD_TABLE)
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _ZERO_WIDTH)
    return re.sub(r"\s+", " ", s).strip()


def _clean_tokens(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        t = str(raw or "").strip()
        if not t or t in STOPWORDS or t in seen:
            continue
        if all(not ch.isalnum() for ch in t):  # 纯标点/符号
            continue
        seen.add(t)
        out.append(t)
    return out


def _simple_cut(norm: str) -> list[str]:
    """无 jieba 时的确定性兜底：ASCII 词 + 中文单字/相邻二元字。"""
    out: list[str] = []
    for m in _WORD_RE.finditer(norm):
        word = m.group(0)
        if word not in _EN_STOP:
            out.append(word)
    for run in _CJK_RE.findall(norm):
        for i, ch in enumerate(run):
            if ch not in _ZH_STOP:
                out.append(ch)
            if i + 1 < len(run):
                bigram = run[i:i + 2]
                if bigram not in _ZH_STOP:
                    out.append(bigram)
    return _clean_tokens(out)


def tokenize(text: object, *, use_jieba: bool | None = None) -> list[str]:
    """分词并返回去停用词/去标点的 token 列表（保序去重）。

    use_jieba=None（默认）：有 jieba 就用，没有就走确定性兜底；
    use_jieba=False：强制兜底，便于测试跨环境一致性；
    use_jieba=True：尝试 jieba，缺失时仍安全退回兜底。
    """
    norm = normalize(text)
    if not norm:
        return []
    jieba = _get_jieba() if use_jieba is not False else None
    if jieba is not None:
        try:
            return _clean_tokens(jieba.lcut(norm))
        except Exception:
            pass  # jieba 异常一律回退，绝不让记忆链路因为分词崩掉
    return _simple_cut(norm)


def signature(text: object) -> frozenset[str]:
    """稳定的词形签名：ASCII 词 + 中文相邻二元字（过滤双虚词组合）。

    刻意不调用 jieba：fact_check 等下游需要跨环境可复现的重叠判据。
    繁简/全半角差异在 normalize 阶段已抹平，故 signature("難過") == signature("难过")。
    """
    norm = normalize(text)
    sig: set[str] = set()
    for word in _WORD_RE.findall(norm):
        if word not in _EN_STOP:
            sig.add(word)
    for run in _CJK_RE.findall(norm):
        if len(run) == 1:
            if run[0] not in _ZH_STOP:
                sig.add(run)
            continue
        for i in range(len(run) - 1):
            bg = run[i:i + 2]
            if not (bg[0] in _FUNCTION_CHARS and bg[1] in _FUNCTION_CHARS):
                sig.add(bg)
    return frozenset(sig)


def overlap(left: object, right: object) -> frozenset[str]:
    """两段文本的共有词形签名（纯函数，供 Fact 校验做确定性重叠判据）。"""
    return signature(left) & signature(right)
