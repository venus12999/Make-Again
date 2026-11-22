"""访谈 Agent:AI 提问 → 用户回答 → AI 追问,最终生成用户画像 + 对象画像 + 疗愈计划。

对齐需求 F1-2(初始问卷轮询,含轻量心理测评)、F1-3(初始报告 + 写入 memory + 人机协同重生成)。
设计要点:
- 问题清单由若干"维度"构成,AI 在维度内动态追问,而非机械读题。
- 访谈状态持久化在 interview_sessions.state,可中断恢复。
- 每个回答经一次 LLM 调用同时产出:下一步动作 + 追问问题 + 事实 + 情绪,并写回记忆流。
- 结束时生成反馈报告,把两份画像写入 memory,并确认疗愈目标。
"""
from __future__ import annotations
from agent import prompts_registry as registry

import copy

from agent import async_report as async_report_mod, onboarding
from agent.companion import _victim_context
from emotion.crisis import CRISIS_MESSAGE, detect
from gateway.client import LLMClient
from memory import store
from memory.breakup import extract as breakup_extract
from memory.loved_one import extract as loved_one_extract
from memory.pet import extract as pet_extract
from memory.extract import extract_tags

MAX_FOLLOWUP_PER_DIM = 2
MAX_TURNS = 20

# 亲友离世 = 亲人 + 挚友(同龄人)离世;"family" 为早期旧值,保留映射以兼容存量数据
LOSS_TYPE_LABELS = {"breakup": "分手", "loved_one": "亲友离世", "family": "亲友离世", "pet": "宠物离世"}

# 访谈维度(问题清单)
DIMENSIONS = [
    {
        "key": "object",
        "title": "聊聊 TA",
        "question": "先跟我聊聊 TA 吧——TA 是你什么人,叫什么(昵称也可以)?",
    },
    {
        "key": "memory",
        "title": "共同的记忆",
        "question": "你们之间,有哪些让你反复想起的时刻或地方?挑一件最清晰的说说。",
    },
    {
        "key": "emotion",
        "title": "现在的情绪",
        "question": "最近哪些时刻、哪些事,最容易让你突然难过或走神?",
    },
    {
        "key": "confusion",
        "title": "放不下的事",
        "question": "现在最让你放不下、或者一直想不通的是什么?",
    },
    {
        "key": "goal",
        "title": "你的期望",
        "question": "如果让你选,你更希望的是「彻底忘记、翻篇」,还是「带着这段记忆,继续往前走」?",
    },
]

# 分手专属访谈维度(问题清单,对齐 LEKIA 2.0 CBT 认知层 + 四维疗愈信号)。
# 顺序即一条情绪疗愈弧线:温暖记忆 → 分手事实 → 当下情绪 → 反刍 → 归因 → 相处模式 → 功能 → 目标。
# 每维产出标注在旁,访谈结束后直接初始化 gather_signals 的四维信号 + 两份画像 + 安全边界。
BREAKUP_DIMENSIONS = [
    {
        "key": "object",
        "title": "聊聊 TA",
        "question": "先跟我聊聊 TA 吧——TA 是你什么人,叫什么(昵称也可以),你们在一起多久了?",
        # 产出:对象画像(称呼/关系/时长)、关系背景
    },
    {
        "key": "memory",
        "title": "最难忘的记忆",
        "question": "你们之间,有没有一个让你一想到就心头一暖的画面、地方或瞬间?挑一个最清晰的说说。",
        # 产出:记忆点(报告题眼的情绪锚)、共同记忆
    },
    {
        "key": "event",
        "title": "走到这一步",
        "question": "这段感情是怎么一步步走到分开的?是谁先提的、大概多久以前?",
        # 产出:分手时间线 → 急性期判断(分手多久)、事件经过
    },
    {
        "key": "emotion",
        "title": "现在的你",
        "question": "这几天,你的心情大概是什么样的?哪些时刻最难受?",
        # 产出:情绪基线(baseline)、情绪波动点
    },
    {
        "key": "rumination",
        "title": "反复想起",
        "question": "有哪些念头、画面或想不通的问题,会反复在你脑子里打转、放不下?",
        # 产出:反刍强度(rumination)、困惑点
    },
    {
        "key": "attribution",
        "title": "你怎么想这件事",
        "question": "分开之后,你心里是怎么跟自己解释这件事的?会不会觉得有自己的责任?",
        # 产出:自动思维 + 认知扭曲 + 反思就绪度(reflective_readiness)、受害者识别
    },
    {
        "key": "attachment",
        "title": "相处里的模式",
        "question": "回想你们相处的时候,闹矛盾、有分歧通常是怎么处理的?有没有一种反复出现的老样子?",
        # 产出:依恋线索(attachment)、相处模式、反思就绪度
    },
    {
        "key": "functional",
        "title": "现在的日常",
        "question": "最近你的作息、吃饭、工作学习、还有跟朋友来往,大概是什么状态?有没有在为自己做点什么?",
        # 产出:功能恢复信号(functional_signal)
    },
    {
        "key": "goal",
        "title": "你想要的",
        "question": "如果让你选,你更希望的是「彻底忘记、翻篇」,还是「带着这段记忆,继续往前走」?",
        # 产出:疗愈目标(goal)
    },
]


# 亲友离世专属访谈维度(问题清单,对齐哀伤双过程模型 DPM + 六维疗愈信号)。
# 顺序即一条哀伤疗愈弧线:温暖记忆 → 去世经过 → 当下情绪/思念 → 内疚未竟之事 → 意义重建 → 持续联结/纪念 → 功能 → 目标。
# 每维产出标注在旁,访谈结束后直接初始化 gather_signals 的六维信号 + 两份画像。
LOVED_ONE_DIMENSIONS = [
    {
        "key": "object",
        "title": "聊聊 TA",
        "question": "先跟我聊聊 TA 吧——TA 是你什么人,叫什么(昵称也可以)?",
        # 产出:对象画像(称呼/关系)、关系背景
    },
    {
        "key": "memory",
        "title": "最难忘的记忆",
        "question": "TA 身上有没有一个画面、一句话或一个地方,让你一想到心里就软一下?挑一个最清晰的说说。",
        # 产出:记忆点(报告题眼的情绪锚)、持续联结素材
    },
    {
        "key": "event",
        "title": "TA 是怎么走的",
        "question": "能跟我说说 TA 是什么时候、怎么离开的吗?大概多久以前?",
        # 产出:去世时间线 → 急性期判断、死亡接受起点
    },
    {
        "key": "emotion",
        "title": "现在的你",
        "question": "这几天,你心里大概是什么滋味?哪些时刻最想念、最难受?",
        # 产出:情绪基线(baseline)、渴望思念(yearning)
    },
    {
        "key": "guilt",
        "title": "放不下的自责",
        "question": "有没有什么事,让你一直觉得「要是当时……就好了」,或者总觉得有自己的责任?",
        # 产出:内疚与未竟之事(guilt)、自动思维 + 认知扭曲
    },
    {
        "key": "meaning",
        "title": "你怎么想这件事",
        "question": "TA 的离开,对你意味着什么?TA 有没有教会你、或者留给你什么?",
        # 产出:意义重建(meaning_reconstruction)、死亡接受(death_acceptance)
    },
    {
        "key": "bond",
        "title": "纪念 TA 的方式",
        "question": "你现在会用什么方式想 TA、纪念 TA?有没有想一直为 TA 留着的东西或习惯?",
        # 产出:持续联结质量(continuing_bond)、纪念方式
    },
    {
        "key": "functional",
        "title": "现在的日常",
        "question": "最近你的作息、吃饭、工作学习、还有跟家人朋友来往,大概是什么状态?有没有在为自己做点什么?",
        # 产出:功能恢复信号(functional_signal)
    },
    {
        "key": "goal",
        "title": "你想要的",
        "question": "如果让你选,你更希望的是「把 TA 彻底放下」,还是「带着 TA 的记忆,继续过好自己的生活」?",
        # 产出:疗愈目标(goal)
    },
]


# 宠物离世专属访谈维度(问题清单,对齐哀伤双过程模型 DPM + 六维疗愈信号)。
# 顺序即一条哀伤疗愈弧线:温暖记忆 → 离世经过(含安乐决定) → 当下情绪/思念 → 内疚(安乐+疏忽) → 意义重建 → 持续联结/纪念 → 功能 → 目标。
# 每维产出标注在旁,访谈结束后直接初始化 gather_signals 的六维信号 + 两份画像。
PET_DIMENSIONS = [
    {
        "key": "object",
        "title": "聊聊它",
        "question": "先跟我聊聊它吧——它叫什么名字、是什么(猫/狗/别的)、陪你多久了?",
        # 产出:对象画像(名字/品种/陪伴时长)、关系背景
    },
    {
        "key": "memory",
        "title": "最难忘的记忆",
        "question": "它身上有没有一个画面、一个动作或一个地方,让你一想到心里就软一下?挑一个最清晰的说说。",
        # 产出:记忆点(报告题眼的情绪锚)、持续联结素材
    },
    {
        "key": "event",
        "title": "它是怎么走的",
        "question": "能跟我说说它是什么时候、怎么离开的吗?如果做了安乐,也愿意的话可以说说当时的决定。",
        # 产出:离世时间线 → 急性期判断、死亡接受起点、安乐决定背景
    },
    {
        "key": "emotion",
        "title": "现在的你",
        "question": "这几天,你心里大概是什么滋味?哪些时刻最想它、最难受?",
        # 产出:情绪基线(baseline)、渴望思念(yearning)
    },
    {
        "key": "guilt",
        "title": "放不下的自责",
        "question": "有没有什么事,让你一直觉得「要是当时……就好了」?比如没早点发现它不舒服、那阵子没多陪它;如果当时做了安乐的决定,也愿意的话可以说说你的纠结。",
        # 产出:安乐决定内疚 + 疏忽内疚(guilt)、自动思维 + 认知扭曲
    },
    {
        "key": "meaning",
        "title": "你怎么想这件事",
        "question": "它的离开,对你意味着什么?它有没有教会你、或者留给你什么?",
        # 产出:意义重建(meaning_reconstruction)、死亡接受(death_acceptance)
    },
    {
        "key": "bond",
        "title": "纪念它的方式",
        "question": "你现在会用什么方式想它、纪念它?有没有想一直为它留着的东西(项圈/玩具/照片)或习惯?",
        # 产出:持续联结质量(continuing_bond)、纪念方式
    },
    {
        "key": "functional",
        "title": "现在的日常",
        "question": "最近你的作息、吃饭、工作学习、还有跟家人朋友来往,大概是什么状态?有没有在为自己做点什么?",
        # 产出:功能恢复信号(functional_signal)
    },
    {
        "key": "goal",
        "title": "你想要的",
        "question": "如果让你选,你更希望的是「把它彻底放下」,还是「带着它的记忆,继续过好自己的生活」?",
        # 产出:疗愈目标(goal)
    },
]


def _dims_for(loss_type: str | None) -> list[dict]:
    """按 loss_type 决定访谈维度:分手用分手九维、亲友离世用丧亲九维、宠物离世用宠物九维、其余用通用五维。"""
    if loss_type == "breakup":
        return list(BREAKUP_DIMENSIONS)
    if loss_type == "loved_one":
        return list(LOVED_ONE_DIMENSIONS)
    if loss_type == "pet":
        return list(PET_DIMENSIONS)
    return list(DIMENSIONS)


STEP_SYSTEM = registry.get('interview.step')


BRIDGE_SYSTEM = registry.get('interview.bridge')


REPORT_SYSTEM = registry.get('interview.report')


REPORT_SYSTEM_BREAKUP = registry.get('interview.report_breakup')


REPORT_SYSTEM_LOVED_ONE = registry.get('interview.report_loved_one')


REPORT_SYSTEM_PET = registry.get('interview.report_pet')


def _report_system(loss_type: str | None) -> str:
    if loss_type == "breakup":
        return REPORT_SYSTEM_BREAKUP
    if loss_type == "loved_one":
        return REPORT_SYSTEM_LOVED_ONE
    if loss_type == "pet":
        return REPORT_SYSTEM_PET
    return REPORT_SYSTEM


def _fmt_relationship_analysis(ra: dict) -> str:
    """把关系复盘压成一段可召回/可展示的文本:优先用 LLM 写好的温暖 narrative,兜底用模板串成连贯段落(不做分列罗列)。"""
    narrative = (ra.get("narrative") or "").strip()
    if narrative:
        return narrative

    parts: list[str] = []
    att = ra.get("attachment") or {}
    user_att, other_att = att.get("user"), att.get("other")
    if user_att and user_att != "暂未提及" and other_att and other_att != "暂未提及":
        parts.append(f"你更偏「{user_att}」的相处方式,TA 更偏「{other_att}」")
    elif user_att and user_att != "暂未提及":
        parts.append(f"你的依恋倾向更偏「{user_att}」")

    cause_txt = "、".join(
        f"{c.get('factor')}" + (f"({c.get('explain')})" if c.get("explain") else "")
        for c in (ra.get("causes") or [])
        if isinstance(c, dict) and c.get("factor") and c.get("factor") != "暂未提及"
    )
    if cause_txt:
        parts.append(f"走到分手,更多在于 {cause_txt}")

    if ra.get("patterns") and ra["patterns"] != "暂未提及":
        parts.append(f"你们之间是「{ra['patterns']}」的相处模式")

    sugg = [str(s) for s in (ra.get("suggestions") or []) if s and str(s) != "暂未提及"]
    if sugg:
        parts.append("往后可以试着" + "、".join(sugg))

    if not parts:
        return ""
    return "。".join(parts) + "。"


def _fmt_grief_analysis(ga: dict) -> str:
    """把哀伤叙事压成一段可召回/可展示的文本:优先用 LLM 写好的温暖 narrative,兜底用模板串成连贯段落。"""
    narrative = (ga.get("narrative") or "").strip()
    if narrative:
        return narrative

    parts: list[str] = []
    meaning = (ga.get("meaning") or "").strip()
    if meaning and meaning != "暂未提及":
        parts.append(f"TA 的离开让你慢慢懂得了 {meaning}")

    guilt = ga.get("guilt") or {}
    still = (guilt.get("still_guilty") or "").strip()
    reframe = (guilt.get("reframe") or "").strip()
    if still and reframe:
        parts.append(f"你一直放不下 {still},但 {reframe}")

    cb = ga.get("continuing_bond") or {}
    sugg = [str(s) for s in (cb.get("suggestions") or []) if s and str(s) != "暂未提及"]
    if sugg:
        parts.append("往后可以" + "、".join(sugg) + "来纪念 TA")

    if not parts:
        return ""
    return "。".join(parts) + "。"


def _fmt_pet_analysis(pa: dict) -> str:
    """把宠物哀伤叙事压成一段可召回/可展示的文本:优先用 LLM 写好的温暖 narrative,兜底用模板串成连贯段落。"""
    narrative = (pa.get("narrative") or "").strip()
    if narrative:
        return narrative

    parts: list[str] = []
    meaning = (pa.get("meaning") or "").strip()
    if meaning and meaning != "暂未提及":
        parts.append(f"它走后你慢慢懂得了 {meaning}")

    guilt = pa.get("guilt") or {}
    still = (guilt.get("still_guilty") or "").strip()
    reframe = (guilt.get("reframe") or "").strip()
    if still and reframe:
        parts.append(f"你一直放不下 {still},但 {reframe}")

    cb = pa.get("continuing_bond") or {}
    sugg = [str(s) for s in (cb.get("suggestions") or []) if s and str(s) != "暂未提及"]
    if sugg:
        parts.append("往后可以" + "、".join(sugg) + "来纪念它")

    if not parts:
        return ""
    return "。".join(parts) + "。"


def questions() -> list[dict]:
    return [{"key": d["key"], "title": d["title"], "question": d["question"]} for d in DIMENSIONS]


def start(db, user_id: str, loss_type: str, question_key: str | None = None) -> dict:
    store.get_or_create_user(db, user_id, loss_type=loss_type)
    # 首次进入才推进到「访谈中」;已进入过(中断恢复/已完成)不再回退阶段。
    if onboarding.get_phase(db, user_id) == "new":
        onboarding.set_phase(db, user_id, "interview")

    dims = _dims_for(loss_type)
    dim_idx = 0
    if question_key:
        for i, d in enumerate(dims):
            if d["key"] == question_key:
                dim_idx = i
                break

    first_q = dims[dim_idx]["question"]
    state = {
        "dimension_idx": dim_idx,
        "followup_count": 0,
        "asked": [first_q],
        "covered": [],
        "facts": [],
        "history": [{"role": "assistant", "content": first_q}],
        "done": False,
        "report": None,
    }
    session = store.create_interview(db, user_id, loss_type, state)
    return {"session_id": session.id, "question": first_q, "loss_type": loss_type}


def answer(db, session_id: str, user_answer: str, client: LLMClient, async_report: bool = False) -> dict:
    session = store.get_interview(db, session_id)
    if session is None:
        raise ValueError("session not found")
    state = copy.deepcopy(session.state or {})
    if state.get("done"):
        return {"action": "done", "report": state.get("report")}

    state.setdefault("history", []).append({"role": "user", "content": user_answer})

    # 危机门控(硬门槛,不可绕过):访谈中流露出自伤倾向时,立即暂停访谈、给危机支持,不追问、不推进。
    crisis = detect(user_answer)
    if crisis["is_crisis"]:
        state["history"].append({"role": "assistant", "content": CRISIS_MESSAGE})
        store.add_memory(
            db, session.user_id, type="interview", content=user_answer,
            summary=user_answer[:120], facts=[], emotion=None, importance=8.0,
        )
        store.update_interview(db, session, state=state)
        return {"action": "crisis", "crisis": crisis, "message": CRISIS_MESSAGE}

    dims = _dims_for(session.loss_type)
    victim = _victim_context(db, session.user_id, user_answer) if session.loss_type == "breakup" else None
    step = _decide_step(client, session.loss_type, dims, state, user_answer, victim=victim)

    action = step.get("action")
    if action not in ("followup", "next", "complete"):
        action = "followup"

    question = step.get("question") or ""
    facts = step.get("facts") if isinstance(step.get("facts"), list) else []
    emotion = _normalize_emotion(step.get("emotion"))

    # 写回记忆流(访谈本身也是一种倾诉)
    place_tag, time_tag = extract_tags(user_answer)
    # 分手/亲友离世场景:额外抽 CBT/哀伤过程字段,让访谈直接初始化 gather_signals 的疗愈信号
    # (否则访谈结束后信号仍为空)。分手抽自动思维/认知扭曲/反刍/功能,丧亲抽渴望/内疚/意义/死亡接受/持续联结。
    if session.loss_type == "breakup":
        cbt = breakup_extract.extract_cbt(client, user_answer)
    elif session.loss_type == "loved_one":
        cbt = loved_one_extract.extract_grief_cbt(client, user_answer)
    elif session.loss_type == "pet":
        cbt = pet_extract.extract_pet_cbt(client, user_answer)
    else:
        cbt = None
    store.add_memory(
        db, session.user_id, type="interview", content=user_answer,
        summary=user_answer[:120], facts=facts, emotion=emotion, importance=6.0,
        place_tag=place_tag, time_tag=time_tag, cbt=cbt,
    )
    for f in facts:
        if isinstance(f, dict):
            state.setdefault("facts", []).append(f)

    # 覆盖追踪:当前维度问过即算覆盖,再合并 LLM 判定本次回答顺带聊到的维度,
    # 避免后面再重复问用户已经说过的事。
    covered = list(dict.fromkeys(state.get("covered", [])))
    cur_key = dims[state["dimension_idx"]]["key"]
    if cur_key not in covered:
        covered.append(cur_key)
    for k in (step.get("covered_dimensions") or []):
        if isinstance(k, str) and k not in covered:
            covered.append(k)
    state["covered"] = covered

    goal_idx = len(dims) - 1
    goal_key = dims[goal_idx]["key"]

    # 目标早期识别:用户已明确表达疗愈目标倾向(goal_signal=true)且目标维度还没聊过时,
    # 直接跳到目标维度做一次确认,不再把剩余维度走完。
    jumped_to_goal = False
    if bool(step.get("goal_signal")) and state["dimension_idx"] < goal_idx and goal_key not in covered:
        state["dimension_idx"] = goal_idx
        state["followup_count"] = 0
        action = "next"
        question = _bridge_question(client, session.loss_type, dims, state, goal_idx)
        jumped_to_goal = True

    # 追问次数保护
    if action == "followup" and state["followup_count"] >= MAX_FOLLOWUP_PER_DIM:
        action = "next"

    # 最后一个维度(目标)不再追问:用户确认疗愈目标后立即收尾生成报告,
    # 避免在"要不要写点纪念"这类延伸话题上继续拖延,影响用户获得反馈的节奏。
    if action == "followup" and state["dimension_idx"] == goal_idx:
        action = "complete"

    # 覆盖度保护:所有维度都聊过之后,LLM 仍可能返回 followup 在旧话题上打转
    # (实测走查出现答完5题后再追问15轮"能松一点/先不用急着下结论"的空转)。
    # 全部维度 covered 且已到目标维度时,不再追问,直接收尾——真实用户没有那么多话可反复讲。
    if action == "followup" and len(covered) >= len(dims) and state["dimension_idx"] >= goal_idx:
        action = "complete"

    turns = len([h for h in state["history"] if h["role"] == "user"])
    if turns >= MAX_TURNS:
        action = "complete"

    if action == "followup":
        state["followup_count"] += 1
    elif jumped_to_goal:
        # 刚跳到目标维度:本轮先问出目标确认问题,下一轮用户回答后再收尾。
        pass
    elif action == "next":
        # 进入下一个还没被覆盖的维度,用 LLM 承接上下文写好的过渡问题(缺省才退回问卷原题)
        next_idx, next_q = _resolve_next(client, session.loss_type, dims, covered, step, state, state["dimension_idx"])
        if next_idx is not None:
            state["dimension_idx"] = next_idx
            state["followup_count"] = 0
            question = next_q
        else:
            action = "complete"

    if action == "complete":
        if async_report:
            # 报告生成放到后台:立即标记完成、推进阶段并返回,前端轮询 report_ready。
            state["done"] = True
            state["generating"] = True
            state["report"] = None
            state["report_ready"] = False
            store.update_interview(db, session, state=state, status="completed")
            onboarding.set_phase(db, session.user_id, "report")
            async_report_mod.schedule_report(
                session.user_id, session.id, session.loss_type, copy.deepcopy(state["history"])
            )
            return {"action": "complete", "done": True, "generating": True}

        report = _generate_report(db, client, session, state)
        state["done"] = True
        state["report"] = report
        state["report_ready"] = True
        store.update_interview(db, session, state=state, status="completed")
        onboarding.set_phase(db, session.user_id, "report")
        return {"action": "complete", "report": report}

    state["asked"].append(question)
    state["history"].append({"role": "assistant", "content": question})
    store.update_interview(db, session, state=state)
    return {
        "action": action,
        "question": question,
        "dimension": dims[state["dimension_idx"]]["title"],
        "emotion": emotion,
    }


def revise(db, session_id: str, supplement: str, client: LLMClient) -> dict:
    """人机协同:用户补充叙述后重新生成报告。"""
    session = store.get_interview(db, session_id)
    if session is None:
        raise ValueError("session not found")
    state = copy.deepcopy(session.state or {})
    state.setdefault("history", []).append({"role": "user", "content": f"[补充] {supplement}"})
    report = _generate_report(db, client, session, state)
    state["report"] = report
    store.update_interview(db, session, state=state)
    return report


def _facts_block(state: dict) -> str:
    """把访谈里已收集的事实压成一段,供引导员自然回调(让用户感到被记住)。"""
    lines = []
    for f in (state.get("facts") or [])[-12:]:
        if isinstance(f, dict):
            fact = (f.get("fact") or "").strip()
            if fact:
                lines.append(f"- {fact}")
    return "\n".join(lines) if lines else "(暂无)"


def _decide_step(client: LLMClient, loss_type: str | None, dims: list[dict], state: dict,
                 user_answer: str, victim: list[str] | None = None) -> dict:
    dim = dims[state["dimension_idx"]]
    recent = "\n".join(f"{h['role']}: {h['content'][:100]}" for h in state.get("history", [])[-6:])
    facts_block = _facts_block(state)
    dim_lines = "\n".join(f"- {d['key']}({d['title']}): {d['question']}" for d in dims)
    victim_note = ""
    if victim:
        victim_note = f"\n[安全边界] 检测到用户在被伤害情境({', '.join(victim)}):归因维度不要追问'你哪里做错了',转为共情 + 识别不健康关系。\n"
    messages = [
        {"role": "system", "content": STEP_SYSTEM},
        {"role": "user", "content": (
            f"失去类型:{loss_type or '未知'}\n"
            f"访谈维度清单(按顺序):\n{dim_lines}\n"
            f"当前维度:{dim['title']}(key={dim['key']})\n"
            f"本维度已追问次数:{state.get('followup_count', 0)}(最多 {MAX_FOLLOWUP_PER_DIM} 次)\n"
            f"已覆盖维度:{state.get('covered', []) or '无'}\n"
            f"已问过的问题:{state.get('asked', [])}\n"
            f"已收集到的关键信息(供自然回调,别复述原话):\n{facts_block}\n\n"
            f"最近对话:\n{recent}\n\n"
            f"用户最新回答:\n{user_answer}"
            f"{victim_note}"
        )},
    ]
    parsed, _ = client.chat_json(messages, temperature=0.6, model=client.settings.llm_fast_model)
    if not parsed:
        return {"action": "followup", "question": "", "facts": [], "emotion": {}, "note": "解析失败兜底"}
    return parsed


def _bridge_question(client: LLMClient, loss_type: str | None, dims: list[dict], state: dict, target_idx: int) -> str:
    """为切换到下一维度写一句自然过渡的话(承接上下文),避免空降问卷原题。"""
    target = dims[target_idx]
    recent = "\n".join(f"{h['role']}: {h['content'][:120]}" for h in state.get("history", [])[-4:])
    messages = [
        {"role": "system", "content": BRIDGE_SYSTEM},
        {"role": "user", "content": (
            f"失去类型:{loss_type or '未知'}\n"
            f"要过渡到的新话题:{target['title']}(参考问题「{target['question']}」,但不要照搬)\n"
            f"最近对话:\n{recent}\n\n"
            f"请写一句自然过渡的话。"
        )},
    ]
    try:
        result = client.chat(messages, temperature=0.7, max_tokens=120, model=client.settings.llm_fast_model)
        text = (result.get("content") if isinstance(result, dict) else "").strip()
    except Exception:
        text = ""
    return text or target["question"]


def _resolve_next(client: LLMClient, loss_type: str | None, dims: list[dict], covered: list[str],
                  step: dict, state: dict, current_idx: int) -> tuple[int | None, str]:
    """决定下一个要进入的维度并给出问题文案。

    优先采用 LLM 建议的、未覆盖的维度 + 它已写好的过渡句;
    否则顺序找 current_idx 之后第一个未覆盖维度,并临时让 LLM 写一句自然过渡(承接上下文),
    只有 LLM 失效时才退回问卷原题(兜底,仍有信息量)。
    """
    suggested = step.get("next_dimension")
    llm_q = (step.get("question") or "").strip()
    if isinstance(suggested, str) and suggested:
        for i, d in enumerate(dims):
            if d["key"] == suggested and d["key"] not in covered:
                return i, (llm_q or _bridge_question(client, loss_type, dims, state, i))
    for i in range(current_idx + 1, len(dims)):
        if dims[i]["key"] not in covered:
            return i, _bridge_question(client, loss_type, dims, state, i)
    return None, ""


class ReportGenerationError(RuntimeError):
    """The provider did not return a valid displayable report."""


def _valid_report(parsed):
    return (isinstance(parsed,dict)
            and all(isinstance(parsed.get(k),str) and parsed[k].strip() for k in ("title","summary"))
            and isinstance(parsed.get("quote",""),str)
            and isinstance(parsed.get("keywords",[]),list)
            and all(isinstance(k,str) for k in parsed.get("keywords",[]))
            and all(isinstance(parsed.get(k,{}),dict) for k in ("user_portrait","object_portrait")))


def _build_report_parsed(client: LLMClient, loss_type: str | None, history: list[dict],
                         subject_name: str | None = None) -> dict:
    """生成报告 JSON(纯 LLM 调用,无 DB 副作用),供同步 / 后台两路复用。"""
    conv = "\n".join(f"{h['role']}: {h['content']}" for h in (history or []))
    user_content = f"失去类型:{LOSS_TYPE_LABELS.get(loss_type, loss_type or '未知')}\n\n访谈对话:\n{conv}"
    # 用户在访谈里已确认过 TA 的称呼,优先用它,避免 LLM 从语音转写里又抽出旧名字。
    if subject_name:
        user_content += f"\n\n对象称呼(用户已确认):{subject_name}。请全文统一使用这个称呼,不要改成别的名字。"
    messages = [
        {"role": "system", "content": _report_system(loss_type)},
        {"role": "user", "content": user_content},
    ]
    # 报告走热路径快模型(默认),时延优先;要更高分析质量时,在 .env 把 llm_report_model 设为 llm_model。
    model = client.settings.llm_report_model or client.settings.llm_fast_model
    # 推理模型(hy4 等)的 reasoning 会先吃 token:实测 4000 时 completion 用满 4000 而 raw 为空,
    # 报告 JSON 实际需 ~6800 tokens。给足 8000;非推理模型(deepseek-chat)只用其中一部分,无副作用。
    parsed, meta = client.chat_json(messages, temperature=0.5, max_tokens=8000, model=model)
    if not _valid_report(parsed):
        # 输出可能被截断或格式错误,再试一次
        messages.append({"role": "user", "content": "请只输出一个完整、合法的 JSON 对象,不要任何其他文字。"})
        parsed, _ = client.chat_json(messages, temperature=0.3, max_tokens=8000, model=model)
    if not _valid_report(parsed):
        raise ReportGenerationError("INVALID_REPORT_OUTPUT")
    return parsed


def _persist_report(db, user_id: str, loss_type: str | None, parsed: dict) -> None:
    """报告落库:画像 + 记忆流 + 看板(reports 表)。"""
    # 兜底:即便 LLM 没听进上面的称呼提示,也强制用用户确认过的名字覆盖对象画像的「称呼」。
    companion = store.get_current_companion(db, user_id)
    confirmed_name = (companion.subject_name or "").strip() if companion else ""
    if confirmed_name:
        obj = parsed.get("object_portrait")
        if not isinstance(obj, dict):
            obj = {}
        obj["称呼"] = confirmed_name
        parsed["object_portrait"] = obj

    user_portrait = parsed.get("user_portrait") or {}
    object_portrait = parsed.get("object_portrait") or {}

    # 分手:把关系复盘压成温暖连贯的一段,同时进画像(供跟踪报告/后续展示)与记忆流(供聊天召回)
    ra_text = ""
    if loss_type == "breakup" and parsed.get("relationship_analysis"):
        ra_text = _fmt_relationship_analysis(parsed["relationship_analysis"])
        if ra_text:
            user_portrait["关系复盘"] = ra_text

    # 亲友离世:把哀伤叙事压成温暖连贯的一段,同时进画像与记忆流
    ga_text = ""
    if loss_type == "loved_one" and parsed.get("grief_analysis"):
        ga_text = _fmt_grief_analysis(parsed["grief_analysis"])
        if ga_text:
            user_portrait["哀伤叙事"] = ga_text

    # 宠物离世:把哀伤叙事压成温暖连贯的一段,同时进画像与记忆流
    pa_text = ""
    if loss_type == "pet" and parsed.get("pet_grief_analysis"):
        pa_text = _fmt_pet_analysis(parsed["pet_grief_analysis"])
        if pa_text:
            user_portrait["哀伤叙事"] = pa_text

    store.upsert_portrait(db, user_id, "user", user_portrait, status="draft")
    store.upsert_portrait(db, user_id, "object", object_portrait, status="draft")
    store.add_memory(
        db, user_id, type="report", content=parsed.get("summary", ""),
        summary=(parsed.get("summary") or "")[:120], facts=[], emotion=None, importance=9.0,
    )
    if ra_text:
        store.add_memory(
            db, user_id, type="report", content=ra_text, summary=ra_text[:120],
            facts=[], emotion=None, importance=8.0,
        )
    if ga_text:
        store.add_memory(
            db, user_id, type="report", content=ga_text, summary=ga_text[:120],
            facts=[], emotion=None, importance=8.0,
        )
    if pa_text:
        store.add_memory(
            db, user_id, type="report", content=pa_text, summary=pa_text[:120],
            facts=[], emotion=None, importance=8.0,
        )
    store.save_report(db, user_id, "initial", parsed)


def _generate_report(db, client: LLMClient, session, state: dict) -> dict:
    companion = store.get_current_companion(db, session.user_id)
    subject_name = (companion.subject_name or "").strip() if companion else ""
    parsed = _build_report_parsed(client, session.loss_type, state.get("history", []),
                                  subject_name=subject_name or None)
    _persist_report(db, session.user_id, session.loss_type, parsed)
    return parsed


def progress(session) -> dict:
    """访谈对外状态视图:进度 / 当前维度 / 报告是否就绪,供前端轮询展示。"""
    state = session.state or {}
    dims = _dims_for(session.loss_type)
    idx = state.get("dimension_idx", 0)
    cur = dims[idx] if 0 <= idx < len(dims) else {}
    history = state.get("history", [])
    question = next(
        (h.get("content", "") for h in reversed(history) if h.get("role") == "assistant"), ""
    )
    report_ready = bool(state.get("report_ready"))
    done = bool(state.get("done")) or session.status == "completed"
    return {
        "session_id": session.id, "companion_id":session.companion_id, "phase":session.status,
        "user_id": session.user_id,
        "status": session.status,
        "done": done,
        "generating": bool(state.get("generating")) and not report_ready,
        "report_ready": report_ready,
        "report_status": "ready" if report_ready else "failed" if state.get("report_failed") else "generating" if state.get("generating") else "not_started",
        "question": question,
        "dimension": cur.get("title"),
        "dimension_key": cur.get("key"),
        "progress": {"covered": len(state.get("covered", [])), "total": len(dims)},
    }


def _normalize_emotion(e: dict | None) -> dict:
    e = e or {}
    emo = e.get("emotion") or "其他"
    try:
        valence = max(-1.0, min(1.0, float(e.get("valence", 0.0))))
        arousal = max(0.0, min(1.0, float(e.get("arousal", 0.5))))
        score = max(0, min(100, int(e.get("score", 50))))
    except (TypeError, ValueError):
        valence, arousal, score = 0.0, 0.5, 50
    return {"emotion": emo, "valence": valence, "arousal": arousal, "score": score}


def _fallback_report(state: dict, loss_type: str | None = None) -> dict:
    report = {
        "title": "谢谢你愿意说出来",
        "keywords": ["温柔", "诚实"],
        "summary": "谢谢你愿意把这些告诉我。我们已经一起梳理了这段关系的脉络,接下来会陪你慢慢往前走。",
        "quote": "慢慢来,也算在往前走",
        "user_portrait": {"失去类型": "暂未提及", "关系与背景": "暂未提及", "当前情绪状态": "暂未提及",
                          "情绪波动点": "暂未提及", "困惑点": "暂未提及", "未说出口的话": "暂未提及"},
        "object_portrait": {"称呼": "暂未提及", "关系": "暂未提及", "性格": "暂未提及",
                            "共同记忆": "暂未提及", "重要地点与时间": "暂未提及"},
        "goal": {"type": "uncertain", "label": "还没想清楚", "reason": "需要进一步对话"},
        "heal_plan": {"summary": "先稳定情绪,再逐步明确方向",
                      "stages": [{"title": "照顾好当下", "desc": "先保证睡眠和饮食", "time": "本周"}]},
    }
    if loss_type == "breakup":
        report["user_portrait"].update({
            "依恋类型": "暂未提及", "关系中的模式": "暂未提及",
            "自身盲区": "暂未提及", "可成长方向": "暂未提及", "关系症结": "暂未提及",
        })
        report["object_portrait"].update({
            "TA的行为模式": "暂未提及", "TA的依恋倾向": "暂未提及", "TA在关系中的问题": "暂未提及",
        })
        report["relationship_analysis"] = {
            "attachment": {"user": "暂未提及", "basis": "", "other": "暂未提及", "other_basis": ""},
            "causes": [], "patterns": "暂未提及", "blind_spots": [], "suggestions": [],
        }
    if loss_type == "loved_one":
        report["user_portrait"].update({
            "内疚与未竟之事": "暂未提及", "死亡接受状态": "暂未提及",
            "意义重建": "暂未提及", "现在的日常状态": "暂未提及",
        })
        report["object_portrait"].update({
            "TA留给你的": "暂未提及", "你心里的TA": "暂未提及",
        })
        report["grief_analysis"] = {
            "meaning": "暂未提及",
            "guilt": {"still_guilty": "", "reframe": ""},
            "continuing_bond": {"current": "暂未提及", "suggestions": []},
            "death_acceptance": "暂未提及",
        }
    if loss_type == "pet":
        report["user_portrait"].update({
            "内疚与未竟之事": "暂未提及", "死亡接受状态": "暂未提及",
            "意义重建": "暂未提及", "现在的日常状态": "暂未提及", "哀伤正名": "暂未提及",
        })
        report["object_portrait"].update({
            "它留给你的": "暂未提及", "你心里的它": "暂未提及",
        })
        report["pet_grief_analysis"] = {
            "meaning": "暂未提及",
            "guilt": {"still_guilty": "", "reframe": ""},
            "continuing_bond": {"current": "暂未提及", "suggestions": []},
            "death_acceptance": "暂未提及",
            "disenfranchised": "暂未提及",
        }
    return report
