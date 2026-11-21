"""写入流程:从一轮倾诉抽取事实 + 情绪 + 显著性,供记忆流落库。"""
from __future__ import annotations

import json
import logging
import os

from emotion.classifier import classify
from gateway.client import LLMClient

logger = logging.getLogger(__name__)

_TIME_TAGS = ["凌晨", "深夜", "夜里", "晚上", "傍晚", "下午", "中午", "上午", "早上", "周末", "周六", "周日"]
_PLACE_TAGS = ["学校", "教室", "家里", "她家", "教堂", "寺庙", "医院", "公司", "公园", "海边", "咖啡店", "拜佛"]

_USE_MEM_PROMPTS_ENV = "USE_MEM_PROMPTS"


def use_mem_prompts() -> bool:
    """USE_MEM_PROMPTS 开关:默认关闭,关闭时 extract_turn 走基线手写 prompt 逐字节不变。"""
    return os.getenv(_USE_MEM_PROMPTS_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


def _first_tag(text: str, tags: list[str]) -> str | None:
    for t in tags:
        if t in text:
            return t
    return None


def extract_tags(text: str) -> tuple[str | None, str | None]:
    """轻量提取地点/时间标签(供访谈等场景复用,不依赖 LLM)。"""
    return _first_tag(text, _PLACE_TAGS), _first_tag(text, _TIME_TAGS)


def extract_turn(client: LLMClient, user_text: str, *, object_hint: str | None = None) -> dict:
    """返回 {facts, emotion, summary, importance, place_tag, time_tag}。

    object_hint: 可选的「思念对象提示」,注入到抽取 prompt 里,约束 summary/facts 对
    「TA」的称呼(如丧亲场景传「逝者是亲人,别写前任/前女友」),避免 LLM 默认套失恋用词。

    USE_MEM_PROMPTS=1 时改用复制的 mem0/graphiti prompt（见 memory/mem_prompts.py），
    新路径任何异常/空结果都回退下面这段基线实现;签名与返回 dict 形态不变。
    """
    emotion = classify(user_text, client)
    place_tag = _first_tag(user_text, _PLACE_TAGS)
    time_tag = _first_tag(user_text, _TIME_TAGS)

    if client.mock:
        return {
            "facts": [],
            "emotion": emotion,
            "summary": user_text[:120],
            "importance": _importance_from_emotion(emotion),
            "place_tag": place_tag,
            "time_tag": time_tag,
        }

    if use_mem_prompts():
        try:
            return _extract_turn_mem(client, user_text, emotion, place_tag, time_tag, object_hint)
        except Exception:  # noqa: BLE001 - 开关可回滚:新路径任何失败一律回退基线
            pass
    return _extract_turn_legacy(client, user_text, emotion, place_tag, time_tag, object_hint)


# ---- WP-1：抽取 → 实体双向链接的接线点 ----
def link_facts_for_memory(facts, *, memory_id, user_id="", entities=None, entity_extractor=None):
    """把抽取产出的 facts 接进实体图，回填每条 fact 的 linked_memory_ids。

    这是 WP-C 遗留接口 ``entity_graph.link_facts`` 真正接到抽取链路的那一跳
    （此前 agent 只提供接口，抽取链路上 link_facts 引用 0 次，导致恒为空）。

    回退纪律（与画像开关 USE_ENTITY_PORTRAIT 同一口径）：
    - ``USE_ENTITY_PORTRAIT`` 关闭：**原样返回同一个 facts 对象**，实体图零调用，
      因此落库字节与基线逐字节一致；
    - 无 memory_id（confirm 流 pending 阶段）：同样原样返回，交由 confirm 路由在
      confirm_candidate 之后补做（接口见 memory.async_write.link_facts_for_entry）；
    - 任何失败一律 swallow（debug/warning 级），绝不破坏抽取/主对话链路。

    entities / entity_extractor 透传给 entity_graph.link_facts：前者显式指定实体，
    后者注入真 NER；缺省用实体图的确定性兜底抽取器。
    """
    if not memory_id:
        return facts
    from memory import entity_graph

    if not entity_graph.enabled():
        return facts
    try:
        kwargs = {"memory_id": str(memory_id), "user_id": str(user_id or "")}
        if entities is not None:
            kwargs["entities"] = entities
        if entity_extractor is not None:
            kwargs["entity_extractor"] = entity_extractor
        linked = entity_graph.link_facts(facts, **kwargs)
        return linked if isinstance(linked, list) else facts
    except Exception:  # noqa: BLE001 - 实体写失败不破坏抽取主链路
        logger.warning("link_facts_for_memory failed; fallback to raw facts", exc_info=True)
        return facts


def _extract_turn_legacy(client, user_text, emotion, place_tag, time_tag, object_hint=None) -> dict:
    """基线原样实现（原 extract_turn 第 43-84 行逐字保留），作为开关关闭/新路径失败时的回退。"""
    hint = f"[场景提示] {object_hint}\n\n" if object_hint else ""
    messages = [
        {"role": "system", "content": "你是记忆抽取助手,只输出 JSON,不要多余文字。"},
        {"role": "user", "content": (
            "从下面的倾诉中抽取结构化信息,输出 JSON,字段:\n"
            "{\"facts\":[{\"kind\":\"user|object|memory|emotion_trigger|goal|time\",\"fact\":\"...\",\"confidence\":0.8}],\n"
            " \"summary\":\"一句话概括\",\n"
            " \"importance\":0到10(对疗愈进程的重要程度),\n"
            f" \"place_tag\":\"地点,只能从这些词里选:{'/'.join(_PLACE_TAGS)},没有则为 null\",\n"
            f" \"time_tag\":\"时间,只能从这些词里选:{'/'.join(_TIME_TAGS)},没有则为 null\"}}\n\n"
            "facts 的 kind 含义:user=用户自身信息,object=思念对象信息,memory=共同记忆细节,"
            "emotion_trigger=情绪触发点,goal=疗愈目标,time=事件发生的真实时间(如「去年生日」「三年前」「纪念日」,"
            "指事件发生的时间,不是聊天时刻;没有则省略)。\n\n"
            + hint
            + f"文本:\n{user_text}"
        )},
    ]
    parsed, _ = client.chat_json(messages, temperature=0.2, model=client.settings.llm_fast_model)
    if not parsed:
        return {
            "facts": [], "emotion": emotion, "summary": user_text[:120],
            "importance": _importance_from_emotion(emotion),
            "place_tag": place_tag, "time_tag": time_tag,
        }

    facts = parsed.get("facts")
    if not isinstance(facts, list):
        facts = []
    importance = float(parsed.get("importance", _importance_from_emotion(emotion)))

    # 硬校验:标签必须落在标准词表内,否则回退到词典提取结果(标准词表)或 None,
    # 避免 LLM 造出"今天"这类非标准标签污染情绪节点。
    llm_place = parsed.get("place_tag")
    llm_time = parsed.get("time_tag")
    return {
        "facts": facts,
        "emotion": emotion,
        "summary": parsed.get("summary") or user_text[:120],
        "importance": max(0.0, min(10.0, importance)),
        "place_tag": llm_place if llm_place in _PLACE_TAGS else place_tag,
        "time_tag": llm_time if llm_time in _TIME_TAGS else time_tag,
    }


# --- USE_MEM_PROMPTS=1 的新路径：复制资产走现有 instructor/chat_json 链路 ---

_FACT_KINDS = {"user", "object", "memory", "emotion_trigger", "goal", "time"}


def _extract_turn_mem(client, user_text, emotion, place_tag, time_tag, object_hint=None) -> dict:
    """mem0 ADDITIVE_EXTRACTION_PROMPT + graphiti 实体负面清单的抽取路径。

    中文适配：few-shot 保留上游英文原文不改；通过 generate_additive_extraction_prompt
    的 use_input_language=True 注入 Language Requirement，再用领域提示（含丧亲语境红线、
    标准词表、事件真实时间定义）约束输出。输出经 _map_mem_extraction 转回原 dict 形态。
    """
    from memory import mem_prompts

    entity_prompt = mem_prompts.ChatPrompt.from_messages(
        mem_prompts.extract_message({
            "entity_types": "人物(person)/地点(place)/物品(object)/事件(event)",
            "previous_episodes": [],
            "episode_content": user_text,
            "custom_extraction_instructions": (
                "中文语料：亲属/宠物裸词必须带所有者(如「奶奶的猫」而不是「猫」)；"
                "抽取说话人；不抽抽象概念。"
            ),
        })
    )
    domain_rules = (
        "【本项目字段契约，最高优先级，覆盖上文的 JSON 输出结构】\n"
        "除 memory 数组外，顶层还要输出 summary(一句话中文概括)、importance(0-10 数字)、"
        "place_tag/time_tag(只能从给定标准词表选，没有则 null)。\n"
        "memory 每一项除 id/text/attributed_to/linked_memory_ids 外，还要输出："
        "kind(只能取 user/object/memory/emotion_trigger/goal/time)、confidence(0-1 数字)。\n"
        "kind 含义：user=用户自身信息，object=思念对象信息，memory=共同记忆细节，"
        "emotion_trigger=情绪触发点，goal=疗愈目标，time=事件发生的真实时间(如「去年生日」「三年前」，"
        "指事件发生的时间而不是聊天时刻；没有则省略)。\n"
        "【领域红线】丧亲/离别疗愈语境：逝者称为「亲人/TA」，不要默认套失恋用词(前任/前女友)；"
        "「猫去世」这类冲突事实必须作为独立 memory 保留，不得因为已有「养猫」记忆而省略。\n"
        f"地点标准词表：{'/'.join(_PLACE_TAGS)}；时间标准词表：{'/'.join(_TIME_TAGS)}。\n"
        "【graphiti 实体负面清单】\n" + entity_prompt.user.content + "\n"
        + mem_prompts.DO_NOT_ESCAPE_UNICODE
    )
    hint = f"[场景提示] {object_hint}\n\n" if object_hint else ""
    user_prompt = hint + mem_prompts.generate_additive_extraction_prompt(
        summary=None,
        recently_extracted_memories=[],
        existing_memories=[],
        new_messages=[{"role": "user", "content": user_text}],
        custom_instructions=domain_rules,
        use_input_language=True,
    )
    messages = [
        {"role": "system", "content": (
            mem_prompts.ADDITIVE_EXTRACTION_PROMPT + mem_prompts.DO_NOT_ESCAPE_UNICODE
        )},
        {"role": "user", "content": user_prompt},
    ]
    parsed, _ = client.chat_json(messages, temperature=0.2, model=client.settings.llm_fast_model)
    result = _map_mem_extraction(parsed, user_text, emotion, place_tag, time_tag)
    if not result["facts"] and len(user_text) >= 30:
        # 长文本空抽取：视为新路径失败，回退基线 prompt，宁可重复也不丢事实。
        raise ValueError("mem0 additive extraction returned no facts for non-trivial text")
    return result


def _map_mem_extraction(parsed, user_text, emotion, place_tag, time_tag) -> dict:
    """把 mem0 `{"memory":[{id,text,attributed_to,linked_memory_ids,...}]}` 映射回原 dict 形态。

    新字段 attributed_to / linked_memory_ids 放在每条 fact 内，顶层键与基线完全一致，调用方零改动。
    """
    if not isinstance(parsed, dict):
        raise ValueError("mem0 extraction output is not a JSON object")
    memories = parsed.get("memory")
    if not isinstance(memories, list):
        raise ValueError("mem0 extraction output has no memory array")

    facts = []
    for item in memories:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        kind = item.get("kind")
        if kind not in _FACT_KINDS:
            kind = "memory"
        try:
            confidence = float(item.get("confidence", 0.8))
        except (TypeError, ValueError):
            confidence = 0.8
        facts.append({
            "kind": kind,
            "fact": text,
            "confidence": confidence,
            "attributed_to": item.get("attributed_to") or "user",
            "linked_memory_ids": item.get("linked_memory_ids") or [],
        })

    importance = float(parsed.get("importance", _importance_from_emotion(emotion)))
    llm_place = parsed.get("place_tag")
    llm_time = parsed.get("time_tag")
    return {
        "facts": facts,
        "emotion": emotion,
        "summary": parsed.get("summary") or user_text[:120],
        "importance": max(0.0, min(10.0, importance)),
        "place_tag": llm_place if llm_place in _PLACE_TAGS else place_tag,
        "time_tag": llm_time if llm_time in _TIME_TAGS else time_tag,
    }


def resolve_memory_conflicts(client, old_memory: list[dict], retrieved_facts: list[str]) -> list[dict]:
    """备用冲突判决:复制的 mem0 DEFAULT_UPDATE_MEMORY_PROMPT 输出 ADD/UPDATE/DELETE/NONE。

    extract_turn 不调用它（基线没有此能力），因此开关关闭时对现有行为零影响。
    """
    from memory import mem_prompts

    messages = [
        {"role": "system", "content": mem_prompts.DEFAULT_UPDATE_MEMORY_PROMPT},
        {"role": "user", "content": (
            "Old Memory:\n" + json.dumps(old_memory, ensure_ascii=False)
            + "\n\nRetrieved facts:\n" + json.dumps(retrieved_facts, ensure_ascii=False)
        )},
    ]
    parsed, _ = client.chat_json(messages, temperature=0.0, model=client.settings.llm_fast_model)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("memory"), list):
        raise ValueError("conflict decision output has no memory array")
    return parsed["memory"]


def validate_extracted_edges(payload: dict) -> list[dict]:
    """用复制的 graphiti Edge/ExtractedEdges schema 校验事实三元组与 valid_at/invalid_at 溯源。"""
    from memory import mem_prompts

    return [edge.model_dump() for edge in mem_prompts.ExtractedEdges.model_validate(payload).edges]


def _importance_from_emotion(emotion: dict) -> float:
    # 越消极、强度越高,越值得记住
    s = abs(emotion.get("valence", 0)) * 5 + emotion.get("arousal", 0.5) * 5
    return max(3.0, min(10.0, s))
