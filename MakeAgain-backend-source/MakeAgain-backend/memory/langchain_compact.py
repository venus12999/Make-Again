"""直接复用 LangChain SummarizationMiddleware 的压缩核心（本地已装依赖，零新增包）。

设计见 docs/LANGCHAIN_COMPACT_REUSE_PLAN.md。
- 只复用 before_model 的"触发判定/安全切点/摘要生成/重建"逻辑，不接 create_agent 整套。
- 摘要用我们的 LLMClient（RunnableLambda 适配），中文 prompt，key 不打印。
- 输出转成我们的 turns 结构；失败返回 None 让调用方逐字节回退自研路径。
"""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain.agents.middleware import SummarizationMiddleware

logger = logging.getLogger(__name__)

SUMMARY_BLOCK_MARK = "[此前对话摘要]"


class _StubChatModel(BaseChatModel):
    """最小 BaseChatModel：仅为构造中间件提供 model 对象；用 tokens/messages 触发，不依赖 profile。"""

    @property
    def _llm_type(self) -> str:
        return "makeagain-stub"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])


def to_lc_messages(turns: list[dict]):
    out = []
    for t in turns:
        role, content = t.get("role", "user"), t.get("content", "")
        if role == "assistant":
            out.append(AIMessage(content=content))
        elif role == "system":
            out.append(SystemMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


def from_lc_messages(messages) -> list[dict]:
    """把中间件的返回（RemoveMessage/摘要 HumanMessage/保留原文）转成我们的 turns。"""
    turns: list[dict] = []
    for m in messages:
        # RemoveMessage(id=REMOVE_ALL_MESSAGES) 是 LangGraph reducer 的删除标记，
        # 我们自己解读返回值，必须跳过它（否则会变成一条空 user 消息）
        if isinstance(m, RemoveMessage):
            continue
        extra = getattr(m, "additional_kwargs", {}) or {}
        content = getattr(m, "content", "")
        if not isinstance(content, str):
            content = str(content)
        if extra.get("lc_source") == "summarization":
            # 摘要是一等消息，转成 system 注入（保持 D-070 的 system 摘要语义）
            # LangChain 硬编码英文前缀 "Here is a summary of the conversation to date:"，
            # 中文陪伴产品不需要——去掉，只保留我们的中文结构化摘要正文
            en_prefix = "Here is a summary of the conversation to date:\n\n"
            if content.startswith(en_prefix):
                content = content[len(en_prefix):]
            turns.append({"role": "system", "content": SUMMARY_BLOCK_MARK + "\n" + content})
        elif isinstance(m, AIMessage):
            turns.append({"role": "assistant", "content": content})
        elif isinstance(m, SystemMessage):
            turns.append({"role": "system", "content": content})
        else:
            turns.append({"role": "user", "content": content})
    return turns


def compact_via_langchain(turns: list[dict], client, *, trigger_tokens: int, keep_messages: int,
                          summarizer_fn=None) -> list[dict] | None:
    """返回压缩后的 turns（含摘要 system 消息）；失败/未触发返回 None 让调用方回退。

    summarizer_fn: callable(messages)->str；不传则用 client + 中文摘要 prompt。
    """
    try:
        lc_msgs = to_lc_messages(turns)

        if summarizer_fn is not None:
            summarizer = RunnableLambda(lambda msgs: summarizer_fn(msgs))
        else:
            def summarizer_fn_default(msgs):
                from memory.context_compaction import SUMMARY_SYSTEM, SUMMARY_PROMPT
                conv = "\n".join(
                    f"{'用户' if isinstance(m, HumanMessage) else '陪伴者'}: {m.content}" for m in msgs
                )
                resp = client.chat(
                    [{"role": "system", "content": SUMMARY_SYSTEM},
                     {"role": "user", "content": f"<conversation>\n{conv}\n</conversation>\n\n{SUMMARY_PROMPT}"}],
                    temperature=0.0, model=client.settings.llm_fast_model,
                    max_tokens=8000,  # D-073 R1 实测：本机 hy4 长摘要需要 8000
                )
                return (resp.get("content") or "").strip()
            summarizer = RunnableLambda(lambda msgs: summarizer_fn_default(msgs))

        mw = SummarizationMiddleware(
            model=_StubChatModel(),
            trigger=("tokens", trigger_tokens),
            keep=("messages", keep_messages),
            summarizer=summarizer,
        )
        out = mw.before_model({"messages": lc_msgs}, runtime=None)
        if not out or "messages" not in out:
            return None
        result = from_lc_messages(out["messages"])
        # 只有真的产出了摘要才算压缩成功（否则是未触发或空摘要）
        if not any(t.get("role") == "system" and SUMMARY_BLOCK_MARK in t.get("content", "") for t in result):
            return None
        return result
    except Exception as exc:
        logger.warning("langchain compaction failed (fallback to legacy): %s", exc)
        return None
