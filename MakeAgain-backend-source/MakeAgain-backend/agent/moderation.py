"""内容安全:检测真实定位/联系方式/身份信息,阻止其进入树洞信或回信。

三层检测(按开销从低到高):
1. 正则硬匹配(零成本快筛,不可绕过):手机号、座机、邮箱、身份证、社交账号、URL、地址。
2. Microsoft Presidio(可选,USE_PRESIDIO=1):中文 PatternRecognizer 包装现有正则,
   AnalyzerEngine 统一识别,结果映射回现有 flag 名(contact/location)。
3. LLM 语义检测(兜底):真实人名、具体工作单位/学校、具体住址(正则难覆盖的语义信息)。

任一层命中即视为"含敏感信息",调用方据此拒绝(让用户去掉后再发),而不是静默改写。
"""
from __future__ import annotations

import logging
import os
import re
from functools import lru_cache

from gateway.client import LLMClient

logger = logging.getLogger(__name__)

_MOBILE = re.compile(r"1[3-9]\d{9}")
_LANDLINE = re.compile(r"(?<!\d)0\d{2,3}-?\d{7,8}(?!\d)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_IDCARD = re.compile(r"\d{17}[\dXx]")
_SOCIAL = re.compile(r"(微信|vx|VX|qq|QQ|抖音|微博|钉钉|电话|手机)\s*[:：]?\s*[a-zA-Z0-9_-]{5,}")
_URL = re.compile(r"https?://\S+")
_ADDR = re.compile(r"(省|市|区|县|镇|乡|村)[^\n,。;]{0,8}(路|街|号|栋|单元|室|小区|大厦|广场|花园|苑)")
_ME_CONTACT = re.compile(r"(加我|联系我|找我|搜我)\s*[^\n,。;]{0,10}\d{5,}")

_REGEX_RULES: list[tuple[str, re.Pattern]] = [
    ("contact", _MOBILE), ("contact", _LANDLINE), ("contact", _EMAIL),
    ("contact", _IDCARD), ("contact", _SOCIAL), ("contact", _URL),
    ("location", _ADDR), ("contact", _ME_CONTACT),
]

# ---------------------------------------------------------------------------
# Presidio(可选层)—— 默认关闭,可随时回滚到"正则 + LLM"原方案。
# ---------------------------------------------------------------------------
_USE_PRESIDIO_VALUES = {"1", "true", "yes", "on"}


def _presidio_enabled() -> bool:
    """读取开关:os.getenv("USE_PRESIDIO", "0"),默认关闭。"""
    return os.getenv("USE_PRESIDIO", "0").strip().lower() in _USE_PRESIDIO_VALUES


# Presidio 实体 -> 现有 flag.type,保持 treehole.py 调用方格式不变。
_PRESIDIO_ENTITY_TO_FLAG = {
    "CONTACT": "contact",
    "LOCATION": "location",
}


def _build_presidio_recognizers() -> list:
    """把现有中文正则包装成 Presidio 的中文 PatternRecognizer。"""
    from presidio_analyzer import Pattern, PatternRecognizer

    contact_patterns = [
        Pattern(name=name, regex=pat.pattern, score=1.0)
        for name, pat in (
            ("mobile", _MOBILE),
            ("landline", _LANDLINE),
            ("email", _EMAIL),
            ("idcard", _IDCARD),
            ("social", _SOCIAL),
            ("url", _URL),
            ("me_contact", _ME_CONTACT),
        )
    ]
    location_patterns = [Pattern(name="address", regex=_ADDR.pattern, score=1.0)]
    return [
        PatternRecognizer(
            supported_entity="CONTACT", patterns=contact_patterns,
            supported_language="zh", name="makeagain_zh_contact",
        ),
        PatternRecognizer(
            supported_entity="LOCATION", patterns=location_patterns,
            supported_language="zh", name="makeagain_zh_location",
        ),
    ]


@lru_cache(maxsize=1)
def _get_presidio_analyzer():
    """懒加载 AnalyzerEngine:只有正则未命中且开关打开时才付出 spaCy 初始化开销。"""
    import spacy
    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import SpacyNlpEngine

    registry = RecognizerRegistry(
        recognizers=_build_presidio_recognizers(), supported_languages=["zh"]
    )
    # 使用空白中文 pipeline,避免下载/加载大型统计模型(如 zh_core_web_sm)。
    nlp_engine = SpacyNlpEngine(
        models=[{"lang_code": "zh", "model_name": "zh_core_web_sm"}]
    )
    nlp_engine.nlp = {"zh": spacy.blank("zh")}
    return AnalyzerEngine(
        registry=registry, nlp_engine=nlp_engine, supported_languages=["zh"]
    )


def _presidio_flags(text: str) -> list[dict]:
    """调用 Presidio 并映射回现有 flag 结构;异常时静默降级,不影响主流程。"""
    try:
        results = _get_presidio_analyzer().analyze(text=text, language="zh")
    except Exception as exc:  # pragma: no cover - 依赖缺失/模型异常时降级
        logger.warning("Presidio scan failed, falling back: %s", exc)
        return []
    return [
        {"type": _PRESIDIO_ENTITY_TO_FLAG.get(r.entity_type, "other"),
         "text": text[r.start:r.end]}
        for r in results
    ]


def _unique_result(flags: list[dict]) -> dict:
    """按片段文本去重,返回调用方期望的 {"clean","flags"} 结构。"""
    seen: set[str] = set()
    uniq: list[dict] = []
    for f in flags:
        if f["text"] not in seen:
            seen.add(f["text"])
            uniq.append(f)
    return {"clean": len(uniq) == 0, "flags": uniq}


_MODERATION_SYSTEM = "你是内容安全审核助手,只输出 JSON,不要多余文字。"


def scan_pii(text: str, client: LLMClient | None = None) -> dict:
    """返回 {"clean": bool, "flags": [{"type","text"}]}。clean=True 表示未检出敏感信息。"""
    text = text or ""
    flags: list[dict] = []

    # 1) 零成本快筛:正则命中立即返回,避免 spaCy / Presidio 的加载开销。
    for typ, pat in _REGEX_RULES:
        for m in pat.finditer(text):
            flags.append({"type": typ, "text": m.group(0)})
    if flags:
        return _unique_result(flags)

    # 2) Presidio(可选):开关打开时用中文 PatternRecognizer 再扫一遍。
    if _presidio_enabled():
        flags.extend(_presidio_flags(text))
        if flags:
            return _unique_result(flags)

    # 3) LLM 语义兜底:真实人名、具体工作单位/学校、具体住址等。
    if client is not None and not client.mock:
        parsed, _ = client.chat_json(
            [
                {"role": "system", "content": _MODERATION_SYSTEM},
                {"role": "user", "content": (
                    "检查下面这段文字是否包含任何可能暴露真实身份的信息,包括:\n"
                    "- 真实人名(全名或可定位到具体人的称呼)\n"
                    "- 具体地址(省市区县 + 路/街/小区/楼栋/门牌)\n"
                    "- 具体工作单位或学校名称\n"
                    "- 联系方式(电话/微信/QQ/邮箱等账号)\n"
                    "只输出 JSON:{\"pii\":[{\"type\":\"name|location|workplace|contact\",\"text\":\"片段\"}]}\n"
                    "没有则 pii 为空数组。\n\n文字:\n" + text
                )},
            ],
            temperature=0.1, model=client.settings.llm_fast_model,
        )
        if isinstance(parsed, dict):
            for it in parsed.get("pii") or []:
                if isinstance(it, dict) and it.get("text"):
                    flags.append({"type": it.get("type", "other"), "text": str(it["text"])})

    return _unique_result(flags)
