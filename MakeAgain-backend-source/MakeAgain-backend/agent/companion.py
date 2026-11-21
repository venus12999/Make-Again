"""陪伴 Agent:危机检测 → 召回记忆 → 构建提示 → 生成回复 → 回写记忆。"""
from __future__ import annotations
from agent import prompts_registry as registry

import json
import os
from datetime import datetime, timedelta

from agent import item as item_mod
from agent import note as note_mod
from agent import photo as photo_mod
from agent import todo as todo_mod
from config import get_settings
from emotion.classifier import classify
from emotion.crisis import CRISIS_MESSAGE, detect
from emotion.tone import ANALYZE, SHARED_RULE, detect_victim_signals, pick_tone
from gateway.client import LLMClient
from memory import recall, session_context, store, temporal
from memory.async_write import schedule_memory_write
from memory.facts import facts_for_context

SYSTEM_PROMPT = registry.get('companion.system')


def _use_pydantic_ai() -> bool:
    """USE_PYDANTIC_AI 开关：默认关闭，逐字读取环境变量，可随时回滚。"""
    return os.getenv("USE_PYDANTIC_AI", "0").strip().lower() in {"1", "true", "yes", "on"}


def _dict_text(d: dict) -> str:
    if not d:
        return "暂无"
    return "\n".join(f"- {k}: {v}" for k, v in d.items() if v)


# 照片提及关键词:模型在自然语境下(尤其温度较高时)容易漏调照片工具,
# 这里用零 LLM 的关键词兜底,保证「提到照片 → 上传卡片」稳定弹出。
_PHOTO_MENTION_HINTS = ("照片", "合照", "合影", "相片", "相册", "拍立得")


def _detect_photo_mention(message: str) -> bool:
    return any(k in (message or "") for k in _PHOTO_MENTION_HINTS)


# 「给自己写的话」提及关键词:模型漏调 note 工具时兜底弹出日记卡片(卡片零内容也能用,日记面即开)。
_NOTE_MENTION_HINTS = (
    "没说出口", "说不出口", "没来得及", "有好多话", "憋在心里", "想对你说", "想对你说的话",
    "想告诉", "好想告诉", "写给他", "写给她", "写给它", "写下来", "记下来", "不寄出的信", "想说的话",
)


def _detect_self_note_mention(message: str) -> bool:
    return any(k in (message or "") for k in _NOTE_MENTION_HINTS)


# 「待办」提及关键词:求助/功能卡住。模型漏调 todo 时,走 todo.fallback_todo 二次生成小任务(不是空卡片)。
_TODO_MENTION_HINTS = (
    "睡不着", "睡不好", "吃不下", "没胃口", "提不起劲", "空落落", "怎么才能", "走不出来",
)


def _detect_todo_mention(message: str) -> bool:
    return any(k in (message or "") for k in _TODO_MENTION_HINTS)


# 「物品」提及关键词:模型漏调物品工具时兜底弹出物品卡(纪念·keep 为主,拿不准就 keep)。
# 只收「有情感意义的实体物」信号,避开照片/写给自己/待办的重叠词。
# 注意:热路径模型(deepseek-chat)对 function calling 不稳定,兜底是物品卡触发的实际主路径,
# 故要覆盖常见纪念物名词 + 赠送/遗物/亲手做等强信号,否则「毛衣/手链」这类提法会一直漏掉。
_ITEM_MENTION_HINTS = (
    # 赠送 / 遗物 / 信物(强信号)
    "送的", "送我的", "送给我的", "给我的", "送你的", "礼物", "信物", "纪念品", "遗物", "旧物",
    # 亲手做 / 舍不得扔(强信号)
    "织的", "缝的", "留下的", "收起来", "还挂着", "还留着", "舍不得扔", "珍藏",
    # 常见纪念物(具体名词)
    "毛衣", "围巾", "手链", "项链", "戒指", "耳环", "镯子", "手表", "挂坠", "胸针",
    "玩偶", "娃娃", "毛绒", "杯子", "眼镜", "香水", "吉他", "相机", "相框", "日记本",
    "衣服", "外套", "衬衫", "帽子", "鞋子",
)


def _detect_item_mention(message: str) -> bool:
    return any(k in (message or "") for k in _ITEM_MENTION_HINTS)


def _context_block(db, user_id: str, message: str, portraits: dict, memories: list[dict], daily_pick=None) -> str:
    parts: list[str] = []
    if portraits.get("user"):
        parts.append(f"[用户画像]\n{_dict_text(portraits['user'])}")
    if portraits.get("object"):
        parts.append(f"[思念对象画像]\n{_dict_text(portraits['object'])}")
    if memories:
        lines = []
        for m in memories[:5]:
            e = m["entry"]
            text = (e.summary or e.content or "").strip()[:80]
            if not text:
                continue
            facts = facts_for_context(e.facts, min_conf=0.6, max_n=2)
            if facts:
                text += f" 【要点:{'、'.join(facts)}】"
            when = ""
            ts = getattr(e, "ts", None)
            if ts is not None:
                when = f"{ts.month}月{ts.day}日 "
            lines.append(f"- {when}TA曾说过:{text}")
        if lines:
            parts.append("[相关记忆(均为TA过去的倾诉,不是此刻正在发生)]\n" + "\n".join(lines))
    if daily_pick and daily_pick.opening:
        parts.append(f"[今日启发] {daily_pick.opening}")
    # 时间依据:仅在用户回想过去状态 / 讲现在状态·变化 / 带时间参照时按需注入(不常驻、不灌水)
    skip_ids = {m["entry"].id for m in memories}
    trajectory = temporal.recall_daily_summaries(db, user_id, message)
    if trajectory:
        parts.append(trajectory)
    past_event = temporal.recall_by_time(db, user_id, message, skip_ids=skip_ids)
    if past_event:
        parts.append(past_event)
    return "\n\n".join(parts) if parts else "(暂无记忆)"


def _victim_context(db, user_id: str, message: str) -> list[str]:
    """聚合当前消息 + 近期记忆里的受害者信号(零 LLM 安全网)。"""
    hits = list(detect_victim_signals(message))
    for m in store.list_memories(db, user_id)[:10]:
        hits += detect_victim_signals((m.content or "") + (m.summary or ""))
    return hits


def _execute_tools(db, user_id: str, client: LLMClient, msgs: list[dict],
                   reply: str, tool_calls: list[dict]) -> tuple[str, dict | None]:
    """执行工具(本地去重判断)→ 回传结果 → 模型结合语气/画像写个性化文案。

    返回 (最终回复, 要展示的工具 payload 或 None)。供通用与失恋两条链路复用。
    """
    tool = None
    if not tool_calls:
        return reply, tool
    results = []
    for tc in tool_calls:
        name = (tc.get("function") or {}).get("name")
        if name == "suggest_photo_upload":
            r = photo_mod.execute_tool(db, user_id, tc)
        elif name == "suggest_todo_list":
            r = todo_mod.execute_tool(db, user_id, tc)
        elif name == "suggest_self_note":
            r = note_mod.execute_tool(db, user_id, tc)
        else:
            r = item_mod.execute_tool(db, user_id, tc)
        r["_tool"] = name
        results.append(r)
    msgs.append({"role": "assistant", "content": reply, "tool_calls": tool_calls})
    for i, (tc, r) in enumerate(zip(tool_calls, results)):
        # 剥离内部标记,保持回喂给模型的 tool 消息干净
        content = {k: v for k, v in r.items() if k != "_tool"}
        msgs.append({
            "role": "tool",
            "tool_call_id": tc.get("id") or f"call_{i}",
            "content": json.dumps(content, ensure_ascii=False),
        })
    final = client.chat(msgs, temperature=0.7, model=client.settings.llm_fast_model)
    reply = final.get("content") or reply
    # 取第一个要展示的动作(照片拍立得 / 物品纪念·寄存 / 待办 / 写给自己)
    for r in results:
        if not r.get("surface"):
            continue
        _t = r.get("_tool")
        if _t == "suggest_photo_upload":
            tool = photo_mod.tool_payload([r], reply)
        elif _t == "suggest_todo_list":
            tool = todo_mod.tool_payload([r], reply)
        elif _t == "suggest_self_note":
            tool = note_mod.tool_payload([r], reply)
        else:
            tool = item_mod.tool_payload([r], reply)
        if tool:
            break
    return reply, tool


def chat(db, user_id: str, message: str, client: LLMClient, session_id: str | None = None, no_tools: bool = False) -> dict:
    """陪伴入口:按丧失类型路由到对应场景引擎。

    - breakup(分手):走失恋 CBT 引擎(会话开场重估 + 认知重构 + 会话内门控)。
    - loved_one(亲友离世):走哀伤双过程引擎(六维信号重估 + 意义重建 + 内疚/纪念)。
    - pet(宠物离世):走宠物哀伤引擎(六维信号重估 + 给哀伤正名 + 安乐内疚/纪念)。
    - 其余:走通用陪伴引擎(保持原有行为)。

    no_tools=True 表示本轮是工具动作的收尾(前端「确定/取消」后补的那句),
    不再触发工具卡片,只生成一句自然的回应。

    USE_PYDANTIC_AI=1 时路由到 agent_v2 的 pydantic-ai typed-tools 编排;
    默认关闭,关闭时下面的原有逻辑逐字节不变。
    """
    if _use_pydantic_ai():
        from agent.agent_v2 import chat_v2
        return chat_v2(db, user_id, message, client, session_id, no_tools=no_tools)
    store.get_or_create_user(db, user_id)
    lt = store.effective_loss_type(db, user_id)
    if lt == "breakup":
        from agent.breakup import breakup_chat
        return breakup_chat(db, user_id, message, client, session_id, no_tools=no_tools)
    if lt == "loved_one":
        from agent.loved_one import loved_one_chat
        return loved_one_chat(db, user_id, message, client, session_id, no_tools=no_tools)
    if lt == "pet":
        from agent.pet import pet_chat
        return pet_chat(db, user_id, message, client, session_id, no_tools=no_tools)
    return _chat_generic(db, user_id, message, client, session_id, no_tools=no_tools)


def _chat_generic(db, user_id: str, message: str, client: LLMClient, session_id: str | None = None, no_tools: bool = False) -> dict:
    # 1. 危机前置检测(硬门槛,不可绕过)
    crisis = detect(message)
    if crisis["is_crisis"]:
        session_context.append_turn(user_id, "user", message, session_id)
        session_context.append_turn(user_id, "assistant", CRISIS_MESSAGE, session_id)
        store.save_chat_turn(db, user_id, session_id, message, CRISIS_MESSAGE)
        return {"reply": CRISIS_MESSAGE, "crisis": crisis, "emotion": None, "recalled": 0}

    # 2. 召回记忆 + 画像
    portraits = {
        "user": store.get_portrait(db, user_id, "user"),
        "object": store.get_portrait(db, user_id, "object"),
    }
    # 最近几轮对话作为「已在聊」参照:已在其中的记忆不再重复注入(克制重复引用)
    turns = session_context.get_turns(user_id, session_id)
    # 「正在聊」的判据用最近 1 轮,而不是整个 session 历史:
    # 用全部 6 轮会让跨天/重新打开 App 后的回忆被误判为"刚聊过"而永久召回不到。
    recent_text = " ".join(t["content"] for t in turns[-1:])
    memories = recall.recall(db, user_id, message, recent_text=recent_text)

    # 今日主题(若当天已选):注入上下文,让对话延续当天主题
    offset = get_settings().timezone_offset_hours
    date_key = (datetime.now() + timedelta(hours=offset)).strftime("%Y-%m-%d")
    daily_pick = store.get_daily_pick(db, user_id, date_key)

    # 3. 构建提示并生成(热路径:非推理模型,保证语音场景首字延迟)
    tone = pick_tone(db, user_id, message=message)
    context_block = _context_block(db, user_id, message, portraits, memories, daily_pick)
    prompt = SYSTEM_PROMPT.format(
        tone=tone["prompt"], rule=SHARED_RULE, context=context_block,
    )
    if tone["tone"] == ANALYZE:
        victim = _victim_context(db, user_id, message)
        if victim:
            prompt += (
                f"\n\n[安全边界] 检测到用户在被伤害情境({', '.join(victim)}),"
                "本次分析不做任何归咎于用户的表述,转为共情 + 识别不健康关系 + 重建自我。"
            )
    msgs = [{"role": "system", "content": prompt}] + turns + [{"role": "user", "content": message}]
    tools = None if no_tools else [item_mod.ITEM_RITUAL_TOOL, photo_mod.PHOTO_UPLOAD_TOOL,
                                   todo_mod.TODO_LIST_TOOL, note_mod.SELF_NOTE_TOOL]
    result = client.chat(
        msgs, temperature=0.7, model=client.settings.llm_fast_model, tools=tools,
    )

    reply = result.get("content") or ""
    reply, tool = _execute_tools(db, user_id, client, msgs, reply, result.get("tool_calls") or [])

    # 规则兜底:模型漏调照片工具时(自然语境 + 高温度下常见),只要用户明确提到照片,
    # 就直接弹出上传卡片,不依赖模型触发。标题留空,前端/后端各自有默认兜底文案。
    if not no_tools:
        if tool is None and _detect_photo_mention(message):
            tool = photo_mod.tool_payload([{"surface": True, "photo_title": "", "scene_description": ""}], reply)
        if tool is None and _detect_self_note_mention(message):
            tool = note_mod.tool_payload([{"surface": True, "target": "self", "opening": ""}], reply)
        if tool is None and _detect_todo_mention(message):
            tool = todo_mod.fallback_todo(db, user_id, client, message, reply)
        if tool is None and _detect_item_mention(message):
            tool = item_mod.tool_payload([{"surface": True, "intent": "keep", "item_name": "", "item_description": ""}], reply)

    # 4. 异步回写记忆:抽取事实/情绪 + 落库放到后台线程,不阻塞本轮回复。
    #    情绪仍用轻量词典法即时返回给前端展示,完整抽取走后台。
    schedule_memory_write(user_id, message)
    # 同步写临时会话上下文(纯内存,下一轮立即可见,不等异步回写)
    session_context.append_turn(user_id, "user", message, session_id)
    session_context.append_turn(user_id, "assistant", reply, session_id)
    # 同步落库聊天历史(user + assistant 各一条,精确到分钟,保证历史立即可见)
    store.save_chat_turn(db, user_id, session_id, message, reply)

    # 情绪分类：golden set 实测 hy4 准确率 0.963 vs 词典 0.444（+51.9pp），故默认用 LLM 分类。
    # 显式设 USE_LLM_EMOTION=0/off 可关闭回退词典（如离线/省成本场景）；异常也自动回退。
    import os as _os
    _emo_flag = _os.getenv("USE_LLM_EMOTION", "1").strip().lower()
    _emo_enabled = _emo_flag not in ("0", "false", "no", "off")
    _emo_client = client if _emo_enabled else None
    try:
        emotion_result = classify(message, _emo_client)
    except Exception:
        emotion_result = classify(message)
    return {
        "reply": reply,
        "crisis": crisis,
        "emotion": emotion_result,
        "recalled": len(memories),
        "tone": tone["tone"],
        "tool": tool,
        "usage": result.get("usage"),
        "memory_context": context_block,
    }
