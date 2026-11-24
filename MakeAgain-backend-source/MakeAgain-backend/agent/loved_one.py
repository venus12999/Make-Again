"""亲友离世陪伴 Agent:哀伤双过程(DPM)的会话内微推进 + 跨天会话开场重估。

与失恋引擎的差异(同构,内容不同):
1. 每次会话开场用「六维信号 + 开场白」重估当前哀伤状态(不沿用上次硬阶段,跨天可回退)。
2. 丧亲专属 SYSTEM_PROMPT:目标不是「放下/向前看」,而是「带着记忆继续生活 + 找到与逝者健康的持续联结」;
   核心认知工作是「意义重建 + 内疚/未竟之事的处理 + 支持纪念」,而非失恋的「平衡归因/复盘关系」。
3. 会话内门控(User Gate + 冷却)同构:推进跟着用户走,不硬推。
4. 回写走哀伤抽取(自动思维/认知扭曲/意义重建/死亡接受/持续联结/功能恢复 + 渴望/内疚/否认)。
"""
from __future__ import annotations
from agent import prompts_registry as registry

import re
from datetime import datetime, timedelta

from agent import item as item_mod
from agent import note as note_mod
from agent import photo as photo_mod
from agent import todo as todo_mod
from agent.companion import _context_block, _detect_item_mention, _detect_photo_mention, _detect_self_note_mention, _detect_todo_mention, _execute_tools
from config import get_settings
from emotion.classifier import classify
from emotion.crisis import CRISIS_MESSAGE, detect
from emotion.tone import SHARED_RULE
from gateway.client import LLMClient
from memory import session_context, store, context_compaction as cc
from memory.async_write import schedule_memory_write
from memory.loved_one import extract as loved_one_extract
from memory.loved_one import state as loved_one_state
from memory.loved_one.recall import recall as loved_one_recall

LOVED_ONE_SYSTEM_PROMPT = registry.get('loved_one.system')

# 会话内门控(User Gate + 冷却):推进跟着用户走,不硬推(与失恋引擎同构)。
MICRO_GATE_RULE = registry.get('loved_one.micro_gate')

# 丧亲特有危机:想随逝者而去(在丧亲语境下是自伤风险)。通用危机检测不覆盖这类「去陪TA」「随TA走」的隐晦表达。
# 关键词表只覆盖「他/她」代词与少数裸短语;亲属词(妈妈/爸爸/奶奶…)与「随…走」句式用正则补全,避免漏检。
_REUNION_CRISIS_HINTS = (
    "去陪他", "去陪她", "跟他走", "跟她走", "跟他一起走", "跟她一起走",
    "去天堂陪", "去那边陪", "去那边找", "想去找他", "想去找她",
)

# 逝者指代:代词 + 常见亲属称谓(丧亲语境下这些指代对象是已故亲人)。
_REUNION_TARGET = (
    r"他|她|他们|她们|妈妈|爸爸|母亲|父亲|奶奶|爷爷|姥姥|姥爷|外公|外婆|亲人|家人|"
    r"孩子|儿子|女儿|老公|老婆|妻子|丈夫|爱人|弟弟|妹妹|哥哥|姐姐|大爷|叔叔|阿姨"
)

# 自成一体的「去陪着逝者」信号:去陪/去找/去天堂/去那边 + 指代,无需方向词。
_REUNION_SELF_RE = re.compile(
    rf"(?:去陪|去找|去天堂|去那边)[^\s，。.!！?？]{{0,4}}?(?:{_REUNION_TARGET})"
)

# 「随/跟/陪」需带方向词(走/而去/一起/团聚/离开)才构成「想随逝者而去」,
# 避免「陪妈妈吃饭」「跟妈妈视频」这类与逝者无关的日常表达误报。
_REUNION_MOVE_RE = re.compile(
    rf"(?:随|跟|陪|想随|想跟|想陪)[^\s，。.!！?？]{{0,4}}?"
    rf"(?:{_REUNION_TARGET})[^\s，。.!！?？]{{0,4}}?"
    rf"(?:走|而去|一起|团聚|离开)"
)


def _is_reunion_hint(message: str) -> bool:
    """想随逝者而去:关键词(代词)或亲属词/「随…走」句式命中。"""
    m = message or ""
    if any(k in m for k in _REUNION_CRISIS_HINTS):
        return True
    return bool(_REUNION_SELF_RE.search(m)) or bool(_REUNION_MOVE_RE.search(m))


def _collapse_paragraphs(reply: str) -> str:
    """收敛换行:最多保留两段,防模型偶发跑成多段论述;顺带把「您」统一成「你」。"""
    r = (reply or "").strip()
    parts = [p.strip() for p in r.split("\n") if p.strip()]
    if len(parts) > 2:
        r = "\n".join(parts[:2])
    return r.replace("您", "你")


# 反套话硬兜底:这些词句是中文安慰语境下的强先验,即便进了规则黑名单仍偶发;出现则用更硬指令重生成一次。
_CLICHE_HARD_BAN = (
    "慢慢来", "一点一点", "这很正常", "没关系", "会好起来",
    "坚强一点", "向前看", "深呼吸",
    "你不是一个人", "我会一直在这", "我在这里陪着你", "我很在意你", "我在这儿",
    "陪你一步步",
)


def _contains_cliche(reply: str) -> bool:
    return any(k in (reply or "") for k in _CLICHE_HARD_BAN)


def _build_prompt(stage_info: dict, context_block: str) -> str:
    return LOVED_ONE_SYSTEM_PROMPT.format(
        rule=SHARED_RULE,
        stage_guidance=stage_info["guidance"],
        microrule=MICRO_GATE_RULE,
        context=context_block,
    )


def loved_one_chat(db, user_id: str, message: str, client: LLMClient, session_id: str | None = None, no_tools: bool = False) -> dict:
    # 1. 危机前置检测(硬门槛,不可绕过):通用自伤 + 丧亲特有「想随逝者而去」
    crisis = detect(message)
    reunion = _is_reunion_hint(message)
    if crisis["is_crisis"] or reunion:
        session_context.append_turn(user_id, "user", message, session_id)
        session_context.append_turn(user_id, "assistant", CRISIS_MESSAGE, session_id)
        store.save_chat_turn(db, user_id, session_id, message, CRISIS_MESSAGE)
        return {"reply": CRISIS_MESSAGE, "crisis": crisis, "emotion": None,
                "recalled": 0, "stage": "acute", "memory_context": ""}

    # 2. 会话开场重估当前哀伤状态(六维信号 + 开场白;不沿用上次硬阶段)
    stage_info = loved_one_state.locate_stage(db, user_id, message)

    # 3. 召回记忆 + 画像
    portraits = {
        "user": store.get_portrait(db, user_id, "user"),
        "object": store.get_portrait(db, user_id, "object"),
    }
    turns = session_context.get_turns(user_id, session_id)
    # D-070: 长对话压缩（开关默认关，失败逐字节回退；摘要作为 system 插在历史前）
    summary_block = None
    turns, summary_block = cc.maybe_compact(db, user_id, session_id, turns, client)
    # U-013: 「正在聊」判据用最近 1 轮，跟 companion/breakup 统一（用全量 turns 会把跨天旧记忆误判为刚聊过）
    recent_text = " ".join(t["content"] for t in turns[-1:])
    memories = loved_one_recall(db, user_id, message, recent_text=recent_text)

    offset = get_settings().timezone_offset_hours
    date_key = (datetime.now() + timedelta(hours=offset)).strftime("%Y-%m-%d")
    daily_pick = store.get_daily_pick(db, user_id, date_key)

    # 4. 构建提示并生成(热路径:非推理模型,保证语音场景首字延迟)
    context_block = _context_block(db, user_id, message, portraits, memories, daily_pick)
    prompt = _build_prompt(stage_info, context_block)
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

    reply = _collapse_paragraphs(result.get("content") or "")

    # 反套话兜底:强先验词句(慢慢来/一点一点/这很正常…)即便进了规则黑名单仍偶发,出现则用更硬指令重生成一次。
    if _contains_cliche(reply):
        hard_system = prompt + (
            "\n\n【硬性要求,本轮必须遵守】不要出现任何固定套话或口头禅(「慢慢来」「一点一点」「这很正常」"
            "「没关系」「会好起来」「坚强一点」「向前看」「深呼吸」「你不是一个人」「我会一直在这里」"
            "「我在这里陪着你」「我很在意你」),换成一句针对 TA 当下那句话的具体回应。"
        )
        hard_msgs = [{"role": "system", "content": hard_system}]
        if summary_block:
            hard_msgs.append({"role": "system", "content": "[此前对话摘要]\n" + summary_block})
        hard_msgs += turns + [{"role": "user", "content": message}]
        result2 = client.chat(
            hard_msgs, temperature=0.4, model=client.settings.llm_fast_model,
            max_tokens=150, tools=tools,
        )
        reply2 = _collapse_paragraphs(result2.get("content") or "")
        if reply2 and not _contains_cliche(reply2):
            reply, result = reply2, result2

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

    # 单段兜底:工具二次生成(或直接生成)仍可能两段,最终再收一次
    reply = _collapse_paragraphs(reply)

    # 5. 异步回写记忆:哀伤抽取(自动思维/认知扭曲/意义重建/死亡接受/持续联结/功能 + 渴望/内疚/否认)
    schedule_memory_write(user_id, message, extractor=loved_one_extract.extract_loved_one_turn)
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
