"""失恋陪伴 Agent:CBT 四阶段(评估→教育→干预→作业)的会话内微推进 + 跨天会话开场重估。

与通用陪伴的差异:
1. 每次会话开场用「四维信号 + 开场白」重估当前疗愈状态(不沿用上次硬阶段,跨天可回退)。
2. 失恋专属 SYSTEM_PROMPT:强化认知重构(自动思维/认知扭曲)、平衡归因、受害者保护。
3. 会话内门控(User Gate + 冷却)以 prompt 规则落地:推进跟着用户走,不硬推。
4. 回写走 CBT 抽取(自动思维/认知扭曲/反思就绪/功能恢复信号)。
"""
from __future__ import annotations
from agent import prompts_registry as registry

from datetime import datetime, timedelta

from agent import item as item_mod
from agent import note as note_mod
from agent import photo as photo_mod
from agent import todo as todo_mod
from agent.companion import _context_block, _detect_item_mention, _detect_photo_mention, _detect_self_note_mention, _detect_todo_mention, _execute_tools, _victim_context
from config import get_settings
from emotion.classifier import classify
from emotion.crisis import CRISIS_MESSAGE, detect
from emotion.tone import SHARED_RULE
from gateway.client import LLMClient
from memory import session_context, store, context_compaction as cc
from memory.async_write import schedule_memory_write
from memory.breakup import extract as breakup_extract
from memory.breakup import state as breakup_state
from memory.breakup.recall import recall as breakup_recall

BREAKUP_SYSTEM_PROMPT = registry.get('breakup.system')

# 会话内门控(User Gate + 冷却):推进跟着用户走,不硬推。
MICRO_GATE_RULE = registry.get('breakup.micro_gate')


def _build_prompt(stage_info: dict, context_block: str) -> str:
    return BREAKUP_SYSTEM_PROMPT.format(
        rule=SHARED_RULE,
        stage_guidance=stage_info["guidance"],
        microrule=MICRO_GATE_RULE,
        context=context_block,
    )


def breakup_chat(db, user_id: str, message: str, client: LLMClient, session_id: str | None = None, no_tools: bool = False) -> dict:
    # 1. 危机前置检测(硬门槛,不可绕过)
    crisis = detect(message)
    if crisis["is_crisis"]:
        session_context.append_turn(user_id, "user", message, session_id)
        session_context.append_turn(user_id, "assistant", CRISIS_MESSAGE, session_id)
        store.save_chat_turn(db, user_id, session_id, message, CRISIS_MESSAGE)
        return {"reply": CRISIS_MESSAGE, "crisis": crisis, "emotion": None, "recalled": 0, "stage": "acute", "memory_context": ""}

    # 2. 会话开场重估当前疗愈状态(四维信号 + 开场白;不沿用上次硬阶段)
    stage_info = breakup_state.locate_stage(db, user_id, message)

    # 3. 召回记忆 + 画像
    portraits = {
        "user": store.get_portrait(db, user_id, "user"),
        "object": store.get_portrait(db, user_id, "object"),
    }
    turns = session_context.get_turns(user_id, session_id)
    # D-070: 长对话压缩（开关默认关，失败逐字节回退；摘要作为 system 插在历史前）
    summary_block = None
    turns, summary_block = cc.maybe_compact(db, user_id, session_id, turns, client)
    # U-013/D-064：与 companion.py 统一，「正在聊」判据只用最近 1 轮，
    # 不用全部 turns（否则跨天/重开的历史会让记忆被误判为刚聊过而永久召回不到）。
    recent_text = " ".join(t["content"] for t in turns[-1:])
    memories = breakup_recall(db, user_id, message, recent_text=recent_text)

    offset = get_settings().timezone_offset_hours
    date_key = (datetime.now() + timedelta(hours=offset)).strftime("%Y-%m-%d")
    daily_pick = store.get_daily_pick(db, user_id, date_key)

    # 4. 构建提示并生成(热路径:非推理模型,保证语音场景首字延迟)
    context_block = _context_block(db, user_id, message, portraits, memories, daily_pick)
    prompt = _build_prompt(stage_info, context_block)
    victim = _victim_context(db, user_id, message)
    if victim:
        prompt += (
            f"\n\n[安全边界] 检测到用户在被伤害情境({', '.join(victim)}),"
            "本次不做任何归咎于用户的表述,转为共情 + 识别不健康关系 + 重建自我。"
        )
    msgs = [{"role": "system", "content": prompt}]
    if summary_block:
        msgs.append({"role": "system", "content": "[此前对话摘要]\n" + summary_block})
    msgs += turns + [{"role": "user", "content": message}]
    tools = None if no_tools else [item_mod.ITEM_RITUAL_TOOL, photo_mod.PHOTO_UPLOAD_TOOL,
                                   todo_mod.TODO_LIST_TOOL, note_mod.SELF_NOTE_TOOL]
    result = client.chat(
        msgs, temperature=0.7, model=client.settings.llm_fast_model,
        max_tokens=150,  # 硬封顶(~250字):大多数回复在120字内,给偶尔稍长留点余量
        tools=tools,
    )

    reply = result.get("content") or ""
    reply, tool = _execute_tools(db, user_id, client, msgs, reply, result.get("tool_calls") or [])

    # 规则兜底:模型漏调照片工具时,只要用户明确提到照片就弹出上传卡片。
    if not no_tools:
        if tool is None and _detect_photo_mention(message):
            tool = photo_mod.tool_payload([{"surface": True, "photo_title": "", "scene_description": ""}], reply)
        if tool is None and _detect_self_note_mention(message):
            tool = note_mod.tool_payload([{"surface": True, "target": "self", "opening": ""}], reply)
        if tool is None and _detect_todo_mention(message):
            tool = todo_mod.fallback_todo(db, user_id, client, message, reply)
        if tool is None and _detect_item_mention(message):
            tool = item_mod.tool_payload([{"surface": True, "intent": "keep", "item_name": "", "item_description": ""}], reply)

    # 5. 异步回写记忆:CBT 抽取(自动思维/认知扭曲/反思就绪/功能信号)
    schedule_memory_write(user_id, message, extractor=breakup_extract.extract_breakup_turn)
    session_context.append_turn(user_id, "user", message, session_id)
    session_context.append_turn(user_id, "assistant", reply, session_id)
    store.save_chat_turn(db, user_id, session_id, message, reply)

    return {
        "reply": reply,
        "crisis": crisis,
        "emotion": classify(message),
        "recalled": len(memories),
        "stage": stage_info["stage"],
        "tool": tool,
        "usage": result.get("usage"),
        "memory_context": context_block,
    }
