"""模拟来电/语音陪伴(F2-7):根据用户留下的记忆与陪伴类型,生成一段适合朗读的短语音文案。

设计要点:
- 一次 fast 模型调用,把「对象名 + 关系类型 + 近期记忆摘要」织成一段 40~80 字的温柔话语,像一通简短的电话。
- 不煽情、不空喊加油、不编造细节;落点在「我在听、我陪着你」。
- 无记忆数据或 mock 时,按关系类型回落固定温和文案,保证流程不中断。
- 仅生成文本与披露信息;音频由前端调用 /speech/tts 生成并播放,不在服务端保存音频资产。
"""
from __future__ import annotations
from agent import prompts_registry as registry

from gateway.client import LLMClient
from memory import store

_SYSTEM = registry.get('call.system')

_FALLBACK = {
    "pet": "我还在你身边,只是换了一种方式陪着你。慢慢来,不急。",
    "relative": "我听见你了,也记得你。好好休息,我会一直陪着你。",
    "breakup": "我听见你了。难过的时候,允许自己慢慢来,我在。",
}

_DISCLOSURE = "以下文字由 AI 根据你留下的故事整理"


def call_script(db, user_id: str, client: LLMClient) -> dict:
    """生成模拟来电文案,返回 {text, subject_name, relationship_type, disclosure}。"""
    store.get_or_create_user(db, user_id)
    companion = store.get_current_companion(db, user_id)
    state = store.get_user_state(db, user_id)
    relationship_type = companion.relationship_type if companion is not None else None
    if relationship_type is None:
        relationship_type = state.relationship_type if state is not None else None
    rt = (relationship_type or "").strip().lower()
    subject_name = companion.subject_name if companion is not None else None
    if subject_name is None:
        subject_name = state.subject_name if state is not None else None
    subject_name = (subject_name or "").strip()
    fallback = _FALLBACK.get(rt, _FALLBACK["breakup"])

    if client.mock:
        return {"text": fallback, "subject_name": subject_name or None,
                "relationship_type": rt or None, "disclosure": _DISCLOSURE}

    mem_txt = "\n".join(
        f"- {m.summary or m.content}" for m in store.list_memories(db, user_id)[:8]
    ) or "(暂无)"
    prompt = (
        f"关系类型:{rt or '未知'};对象名:{subject_name or '未具名'}。\n"
        f"这段陪伴里 TA 说过/经历过的片段(按时间,均为过去):\n{mem_txt}\n\n"
        "请写一段 40~80 字的来电话语,像一通简短的电话:温柔、克制、不煽情、不空喊加油、不编造细节,"
        "点到一两个真实片段,落点在「我在听、我陪着你」。只输出这段文字。"
    )
    try:
        result = client.chat(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0.6, model=client.settings.llm_fast_model,
        )
        text = (result.get("content") or "").strip()
    except Exception:
        text = ""
    if text and len(text) > 120:
        text = text[:119] + "…"
    return {"text": text or fallback, "subject_name": subject_name or None,
            "relationship_type": rt or None, "disclosure": _DISCLOSURE}