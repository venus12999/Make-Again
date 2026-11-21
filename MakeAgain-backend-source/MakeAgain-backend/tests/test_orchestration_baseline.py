"""C4 编排层对照臂 spike 的证据测试（基线 main 59e4719，分支 spike/C4-baseline）。

判据 J1-J5 见 docs/ORCHESTRATION_SELECTION_PLAN.md；裁决见 docs/ORCHESTRATION_RESEARCH.md。

口径（务必区分）：
- 本文件所有用例都是离线确定性验证（FunctionModel / 本地 SQLite / 内存 OTel exporter），
  证明的是"协议链路 + 上下文组装"通不通，**不是**真实模型的工具调用成功率。
- 真实 hy4 命中率在 work/c4_j2_real_probe.py（未提交 scratch）单独跑，数字写进报告；
  无 key/网关不可用时标 NOT_RUN / ENV_FAIL，绝不把离线结果冒充真实成功率。
- 语料来自 eval/golden_corpus.json（P-05/P-06 冲突记忆、手写模拟语料，非采集的真实用户数据），
  记忆 fixture 严格照抄 eval/CORPUS_README.md §4。
"""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from agent import agent_v2 as v2  # noqa: E402  安装 opentelemetry._events shim（J5 见下）
from pydantic_ai.messages import (  # noqa: E402
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel  # noqa: E402

from agent import companion  # noqa: E402
from gateway.client import LLMClient  # noqa: E402
from memory import recall as recall_mod  # noqa: E402
from memory.models import MemoryEntry  # noqa: E402
from memory.vector_index import _reset_semantic_state, reset_index  # noqa: E402

CORPUS = __import__("pathlib").Path(__file__).resolve().parents[1] / "eval" / "golden_corpus.json"
EVAL_USER = "c4-baseline-user"

MEMORY_FIXTURES = [
    {"id": "mem-pet-cat-owned", "content": "用户养了一只猫，名字叫橘子，陪了用户八年。",
     "summary": "用户养猫橘子", "facts": ["用户养猫", "猫名叫橘子", "陪伴八年"],
     "importance": 0.6, "ts": "2025-03-01T10:00:00Z"},
    {"id": "mem-pet-cat-deceased", "content": "用户养的猫橘子已于 2026-11-20 去世，被埋在楼下那棵树下。",
     "summary": "用户的猫橘子已去世", "facts": ["用户的猫橘子已去世", "橘子被埋在楼下树下"],
     "importance": 0.9, "ts": "2026-11-20T10:00:00Z"},
]


def _load_corpus() -> dict[str, dict]:
    return {item["id"]: item for item in json.loads(CORPUS.read_text(encoding="utf-8"))}


@pytest.fixture
def db():
    from memory.db import _RawSessionLocal
    from memory.scope import current_db
    session = _RawSessionLocal()
    session.info.update(managed=True, borrowed=True)
    token = current_db.set(session)
    yield session
    current_db.reset(token)
    session.finish()
    session.close()


@pytest.fixture
def conflict_db(db, monkeypatch):
    """写入 P-05/P-06 冲突记忆并建索引，固定 now 保证排序可复现。"""
    monkeypatch.setenv("USE_DUAL_RECALL", "1")
    monkeypatch.setenv("USE_SEMANTIC_EMBEDDING", "1")
    monkeypatch.setenv("USE_QUERY_INTENT", "1")
    for f in MEMORY_FIXTURES:
        db.add(MemoryEntry(
            id=f["id"], user_id=EVAL_USER, type="chat", content=f["content"],
            summary=f["summary"], facts=f["facts"], importance=f["importance"],
            ts=datetime.fromisoformat(f["ts"].replace("Z", "+00:00")).replace(tzinfo=None),
        ))
    db.commit()

    class _FixedDatetime:
        @staticmethod
        def now(tz=None):
            return datetime(2026, 11, 21, 12, 0, 0)

    monkeypatch.setattr(recall_mod, "datetime", _FixedDatetime)
    _reset_semantic_state()
    reset_index()
    for f in MEMORY_FIXTURES:
        assert recall_mod.index_memory(db.get(MemoryEntry, f["id"]), db=db)
    return db


def _text_model(text: str = "（mock）我在这里，愿意听你慢慢说。") -> FunctionModel:
    return FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart(text)]))


# ---------------------------------------------------------------------------
# J1 · 上下文注入：新事实（去世）排前，旧事实（养猫）降级为过去倾诉
# ---------------------------------------------------------------------------
def test_j1_conflict_memory_new_fact_first(conflict_db, capsys):
    corpus = _load_corpus()
    p06 = corpus["P-06"]
    assert p06["memory_expect"][0] == "mem-pet-cat-deceased"
    out = v2.chat_v2(conflict_db, EVAL_USER, p06["memory_query"], LLMClient())
    ctx = out["memory_context"]
    print("\n[J1 context_block 原文]\n" + ctx)  # pytest -s 可见，报告逐字引用
    assert out["recalled"] >= 2
    assert "去世" in ctx and "养猫" in ctx
    assert ctx.index("去世") < ctx.index("养猫"), "新事实必须排在旧事实之前"
    assert "TA过去的倾诉,不是此刻正在发生" in ctx, "旧事实必须被标为过去倾诉"
    assert "还在养" not in ctx, "不得把旧事实留作当前结论"


# ---------------------------------------------------------------------------
# J2 · 离线协议验证：4 张工具卡 × 3 场景 = 12 组合（非真实成功率）
# ---------------------------------------------------------------------------
_TOOL_ARGS = {
    "suggest_item_ritual": {"item_name": "围巾", "intent": "keep", "item_description": ""},
    "suggest_photo_upload": {"photo_title": "老照片", "scene_description": ""},
    "suggest_todo_list": {"tasks": ["下楼走十分钟"]},
    "suggest_self_note": {"target": "self", "opening": ""},
}
_EXPECTED_TYPE = {
    "suggest_item_ritual": ("item_keep", "item_let_go"),
    "suggest_photo_upload": ("photo_upload",),
    "suggest_todo_list": ("todo_list",),
    "suggest_self_note": ("self_note",),
}
_SCENES = {
    "breakup": ["他送我的围巾还在", "翻到一张老照片", "我最近睡不好", "有好多话没来得及跟他说"],
    "loved_one": ["妈妈留下的手表我一直收着", "我们那张合照我舍不得删", "吃不下饭，提不起劲", "好想告诉她我有多后悔"],
    "pet": ["橘子留下来的项圈我还留着", "手机里全是它的照片和视频", "走不出来，家里空落落", "那些没说出口的话"],
}


def _typed_tool_model(name: str, args: dict) -> FunctionModel:
    """第一次发指定 ToolCallPart；工具结果回来后吐正文收尾。"""
    def fn(messages, info):
        if messages and any(isinstance(p, ToolReturnPart) for p in getattr(messages[-1], "parts", [])):
            return ModelResponse(parts=[TextPart("好，我陪你一步步来。")])
        return ModelResponse(parts=[ToolCallPart(name, args)])
    return FunctionModel(fn)


def test_j2_offline_tool_card_matrix(db, monkeypatch):
    """12 组合全部走 typed tool，不落关键词兜底（离线协议链路，不是模型能力）。"""
    monkeypatch.setenv("WP1_USE_STRUCTURED_ACTION", "0")
    orig = v2._run_tool
    calls: list[str] = []
    monkeypatch.setattr(v2, "_run_tool", lambda deps, name, args: (calls.append(name), orig(deps, name, args))[1])
    tools = list(_TOOL_ARGS)
    rows = []
    n = 0
    for scene, texts in _SCENES.items():
        for tool, text in zip(tools, texts):
            n += 1
            calls.clear()
            out = v2.chat_v2(db, "c4-j2-%s-%d" % (scene, n), text, LLMClient(),
                             model=_typed_tool_model(tool, _TOOL_ARGS[tool]))
            card_type = (out.get("tool") or {}).get("type")
            rows.append((scene, tool, bool(calls), card_type))
            assert calls == [tool], "%s/%s 未走 typed tool" % (scene, tool)
            assert card_type in _EXPECTED_TYPE[tool]
    assert n == 12
    print("\n[J2 offline matrix] n=%d typed=%d fallback=0" % (n, len(rows)))


def test_j2_offline_validation_retry(db, monkeypatch):
    """todo.tasks=5 触发 Pydantic 校验重试 → 第二次 3 条才执行工具。"""
    monkeypatch.setenv("WP1_USE_STRUCTURED_ACTION", "0")
    state = {"n": 0}

    def fn(messages, info):
        if messages and any(isinstance(p, ToolReturnPart) for p in getattr(messages[-1], "parts", [])):
            return ModelResponse(parts=[TextPart("好，我陪你从一小步开始。")])
        state["n"] += 1
        tasks = ["1", "2", "3", "4", "5"] if state["n"] == 1 else ["下楼走十分钟", "喝一杯温水", "给朋友发条消息"]
        return ModelResponse(parts=[ToolCallPart("suggest_todo_list", {"tasks": tasks})])

    out = v2.chat_v2(db, "c4-j2-retry", "我最近睡不好", LLMClient(), model=FunctionModel(fn))
    assert state["n"] == 2, "第一次非法参数应触发一次重试"
    assert out["tool"]["type"] == "todo_list" and len(out["tool"]["tasks"]) == 3
    print("\n[J2 retry] model_calls=%d tasks=%d" % (state["n"], len(out["tool"]["tasks"])))


def test_j2_offline_keyword_fallback_when_no_tool_call(db, monkeypatch):
    """模型只吐正文（tool_calls 为空）且备选方案关闭时，落零 LLM 关键词兜底。"""
    monkeypatch.setenv("WP1_USE_STRUCTURED_ACTION", "0")
    out = v2.chat_v2(db, "c4-j2-fallback", "翻到一张老照片，突然很想他", LLMClient(),
                     model=_text_model())
    assert out["tool"] is not None
    assert out["tool"]["type"] == "photo_upload"
    print("\n[J2 fallback] card=%s (typed tool 未调用)" % out["tool"]["type"])


# ---------------------------------------------------------------------------
# J3 · 多轮：P-05 → P-06，带 history 时旧事实仍不升格为现状
# ---------------------------------------------------------------------------
def test_j3_history_keeps_old_fact_as_past(conflict_db, capsys):
    corpus = _load_corpus()
    p05, p06 = corpus["P-05"], corpus["P-06"]
    seen: list = []

    def capture(messages, info):
        seen.extend(messages)
        return ModelResponse(parts=[TextPart("（mock）我在这里，愿意听你慢慢说。")])

    u = "c4-j3-user"
    # 用同一 user 的冲突记忆（fixture 以 EVAL_USER 写入），这里直接复用 EVAL_USER
    v2.chat_v2(conflict_db, EVAL_USER, p05["text"], LLMClient(), session_id="c4-s1")
    seen.clear()
    out = v2.chat_v2(conflict_db, EVAL_USER, p06["text"], LLMClient(),
                     session_id="c4-s1", model=FunctionModel(capture))
    ctx = out["memory_context"]
    print("\n[J3 with-history context_block 原文]\n" + ctx)
    assert "去世" in ctx and ctx.index("去世") < ctx.index("养猫")
    assert "还在养" not in out["reply"] and "还在养" not in ctx
    user_texts = [p.content for m in seen for p in getattr(m, "parts", []) if isinstance(p, UserPromptPart)]
    assert any(p05["text"] in t for t in user_texts), "P-05 应作为历史轮注入"
    assert any(p06["text"] in t for t in user_texts), "P-06 应作为当前轮注入"

    # 不带 history 的对照：同一用户、全新 session_id（记忆按 user 共享，会话轮次按 session 隔离）
    out2 = v2.chat_v2(conflict_db, EVAL_USER, p06["text"], LLMClient(), session_id="c4-s2-fresh")
    assert "去世" in out2["memory_context"]
    print("[J3 no-history context_block 原文]\n" + out2["memory_context"])


# ---------------------------------------------------------------------------
# J4 · 可观测 / 可评测
# ---------------------------------------------------------------------------
def _otel_is_real() -> bool:
    """全局 opentelemetry>=1.44 时 _events 是 shim，Event 无 .attributes，instrument=True 会崩。"""
    from opentelemetry._events import Event
    return hasattr(Event, "attributes")


def test_j4_trace_spans_with_in_memory_exporter():
    """pin 环境（opentelemetry-api<1.44）下 instrument=True 应产出 OTel span。"""
    if not _otel_is_real():
        pytest.skip("全局 opentelemetry-api>=1.44：shim 的 Event 无 .attributes，instrument=True 崩溃（见报告 J4）；需 pin<1.44 后重跑")
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from pydantic_ai import Agent

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)

    class Deps:
        prompt = "system: 陪伴助手"

    def fn(messages, info):
        if messages and any(isinstance(p, ToolReturnPart) for p in getattr(messages[-1], "parts", [])):
            return ModelResponse(parts=[TextPart("好，我陪你。")])
        return ModelResponse(parts=[ToolCallPart("suggest_todo_list", {"tasks": ["下楼走十分钟"]})])

    agent = Agent(None, instructions=lambda ctx: ctx.deps.prompt, deps_type=Deps,
                  output_type=str, instrument=True, end_strategy="early")
    agent.run_sync("我最近睡不好", deps=Deps(), model=FunctionModel(fn))
    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    print("\n[J4 spans] count=%d names=%s" % (len(spans), names))
    assert len(spans) >= 4
    assert any("agent run" in n for n in names)


def test_j4_pydantic_evals_adapter_shape():
    """golden_corpus.json 不能被 pydantic_evals.Dataset 直接吃，需 20 行适配。"""
    data = json.loads(CORPUS.read_text(encoding="utf-8"))
    assert len(data) == 27
    # 适配后每条必映射的字段（与 work/c4_j4_evals_probe.py 的 20 行 adapter 一致）
    required = {"id", "text", "scene", "source", "emotion_expect", "crisis_expect"}
    assert all(required <= set(item) for item in data)
    try:
        import pydantic_evals  # noqa: F401
    except ImportError:
        pytest.skip("pydantic_evals 未安装（不在 requirements）；隔离 env 实测 27→27 cases、evaluate_sync OK，见报告 J4")
    print("\n[J4 eval] corpus=%d items, adapter_mapping_fields=%s" % (len(data), sorted(required)))


# ---------------------------------------------------------------------------
# J5 · 工程债：pin 后裸 import；shim 在 pin 环境是 no-op
# ---------------------------------------------------------------------------
def test_j5_shim_is_noop_after_agent_v2_import():
    from opentelemetry import _events  # noqa: F401
    assert v2._ensure_otel_events_shim() is None
    print("\n[J5] agent_v2 import 后 opentelemetry._events 可用；shim 调用为 no-op")


def test_j5_bare_import_requires_pin():
    import opentelemetry
    from importlib.metadata import version
    v = version("opentelemetry-api")
    if tuple(int(x) for x in v.split(".")[:2]) >= (1, 44):
        pytest.skip("全局 opentelemetry-api=%s 未 pin；裸 import pydantic_ai 失败，隔离 env 用 1.43.0 实测 OK（见报告 J5）" % v)
    import pydantic_ai  # noqa: F401
    assert pydantic_ai.__version__ == "0.8.1"
