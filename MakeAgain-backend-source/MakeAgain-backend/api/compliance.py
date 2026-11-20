"""WP7 · 合规基线路由与事件契约（AI 标识 / 数据删除 / 敏感数据标记）。

设计约束（来自 MASTER_PLAN_PARALLEL.md 第 5 节 WP7 行为验收表）：

- 独立路由文件：本模块自带 ``router``，不碰 api/routes.py；W0 在 main.py 统一
  ``app.include_router(compliance.router)`` 即可挂载（本工作包无权改 main.py）。
- 开关 ``USE_COMPLIANCE_GATE`` 用 os.getenv 读取，默认关闭。关闭时所有判定函数
  退化为"不落事件 / 不写审计"的向后兼容路径，可随时回滚；开启时才
  真正执行事件记录与严格访问日志。
- 不实际删数据：data-delete 只提供契约 + 事件表（进程内内存表 + threading.Lock），
  物理清除在 W0 集成时接入 store / 对象存储（见 docs/COMPLIANCE.md）。
- 不可改文件：store.py / models.py / auth.py / routes.py / conftest.py 均不属于本包，
  所以事件表、敏感标记都以本模块内的纯函数 + 内存表实现，不写主链路。
- 不设未成年人门槛：本产品是失恋/丧亲/宠物离世后的疗愈陪伴，不是虚拟亲密关系/AI 伴侣；
  未成年人同样是目标用户，不做年龄拦截。若将来被监管归为"虚拟亲密关系"，由产品方决策，
  不在代码里写死 18 岁门槛。
- 敏感标记复用 emotion.crisis.detect（只 import，不修改硬门槛），叠加哀伤关键词，
  命中后写"严格访问审计"事件，供后续权限层消费。
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

# ---- 开关 -------------------------------------------------------------------

_SWITCH = "USE_COMPLIANCE_GATE"


def gate_enabled() -> bool:
    """每次调用都读环境变量，便于运行时/测试切换；默认关闭。"""
    return os.getenv(_SWITCH, "0").strip().lower() in ("1", "true", "yes", "on")


# ---- 常量 --------------------------------------------------------------------

# AI 生成披露字段模板（结构化，替代现有单一 disclosure 字符串）。
# 对标《人工智能生成合成内容标识办法》的"显式标识 + 隐式标识"双轨要求。
DISCLOSURE_TEMPLATE: dict[str, Any] = {
    "is_ai_generated": True,
    "ai_disclosure": "本内容由人工智能生成，非真人实时对话或真人创作",
    "generated_by": "MakeAgain-AI",
    "content_types": ["simulated_call", "memorial_text", "chat_reply", "tts_audio"],
    "explicit_label": "AI 生成 / 虚拟陪伴",
    "implicit_marker": "ai_generated_content_v1",  # 供内容元数据/水印层落地
    "disclosure_required": True,
    "standard": "人工智能生成合成内容标识办法（显式+隐式标识）",
}

# 敏感内容分类关键词（自伤复用危机硬门槛，哀伤为本合规模块补充）。
_GRIEF_KEYWORDS = ("去世", "离世", "走了", "失去", "想念", "难过", "哀伤", "葬礼", "遗物", "自杀")


# ---- 进程内事件表（W0 集成时替换为 store 持久化） ----------------------------

_LOCK = threading.Lock()
_SEQ = 0
_DELETION_REQUESTS: list[dict[str, Any]] = []  # 数据删除请求事件表
_SENSITIVE_AUDIT: list[dict[str, Any]] = []     # 敏感数据严格访问审计日志


def reset_compliance_state() -> None:
    """清空内存表（测试/重置用）。"""
    global _SEQ
    with _LOCK:
        _SEQ = 0
        _DELETION_REQUESTS.clear()
        _SENSITIVE_AUDIT.clear()


def get_deletion_requests(user_id: str | None = None) -> list[dict[str, Any]]:
    with _LOCK:
        rows = _DELETION_REQUESTS if user_id is None else [
            e for e in _DELETION_REQUESTS if e["user_id"] == user_id
        ]
        return [dict(e) for e in rows]


def get_sensitive_audit(user_id: str | None = None) -> list[dict[str, Any]]:
    with _LOCK:
        rows = _SENSITIVE_AUDIT if user_id is None else [
            e for e in _SENSITIVE_AUDIT if e["user_id"] == user_id
        ]
        return [dict(e) for e in rows]


def _now() -> float:
    return time.time()


def _iso(ts: float) -> str:
    """把 epoch 秒格式化成 UTC ISO-8601（Z 结尾）。"""
    return datetime.fromtimestamp(ts, timezone.utc).replace(tzinfo=None).isoformat() + "Z"


# ---- 1. AI 标识 --------------------------------------------------------------

def ai_disclosure(*, content_type: str, enforced: bool | None = None) -> dict[str, Any]:
    """返回结构化 AI 披露字段；叠加内容类型，供模拟来电/纪念文案等响应消费。"""
    effective = gate_enabled() if enforced is None else enforced
    out = dict(DISCLOSURE_TEMPLATE)
    out["content_type"] = content_type
    out["enforced"] = effective
    out["legacy_disclosure"] = "以下文字由 AI 根据你留下的故事整理"
    return out


def audit_existing_disclosure() -> dict[str, Any]:
    """检查现有 agent 层 AI 披露字段完整性，输出结构化缺口清单（不做修改）。

    现状（读码核对）：agent/ending.py 与 agent/call.py 仅有单字符串 disclosure，
    缺少结构化字段；/api/chat 响应无披露字段（见 docs/COMPLIANCE.md）。
    """
    fields = ["is_ai_generated", "ai_disclosure", "generated_by", "content_type",
              "explicit_label", "implicit_marker"]
    found = {"ending.py": ["disclosure"], "call.py": ["disclosure"], "routes.py(chat)": []}
    gaps = []
    for module, present in found.items():
        missing = [f for f in fields if f not in present]
        if missing:
            gaps.append({"location": module, "missing_fields": missing})
    return {
        "required_fields": fields,
        "current_fields": found,
        "gaps": gaps,
        "note": "本模块提供结构化模板；把字段接入 chat/ending/call 响应体由 W0 集成完成（不在此包改主链路文件）",
    }


# ---- 2. 数据删除请求（只标记，不实际删） -------------------------------------

_VALID_SCOPES = {"memories", "photos", "items", "letters", "chat", "portrait", "all"}


def request_data_deletion(
    *,
    user_id: str,
    scope: list[str] | None = None,
    reason: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """标记数据删除请求并落事件表；不物理删除任何数据。

    开关关闭时只返回契约（enforced=False，不落事件），供回滚。
    """
    scopes = scope or ["all"]
    unknown = [s for s in scopes if s not in _VALID_SCOPES]
    if unknown:
        return {"ok": False, "reason": "invalid_scope", "unknown": unknown}

    if not gate_enabled():
        return {
            "ok": True, "deletion_id": None, "user_id": user_id,
            "status": "not_enforced", "scope": scopes, "enforced": False,
            "note": "USE_COMPLIANCE_GATE 关闭：仅返回契约，不记录事件、不删数据",
        }

    global _SEQ
    with _LOCK:
        _SEQ += 1
        ts = _now()
        event: dict[str, Any] = {
            "id": f"del-{_SEQ}",
            "user_id": user_id,
            "request_id": request_id,
            "scope": list(scopes),
            "reason": reason,
            "status": "pending_erasure",
            "physical_delete": False,
            "ts": ts,
            "iso": _iso(ts),
            "audit": "deletion_request_recorded",
        }
        _DELETION_REQUESTS.append(event)
    return {
        "ok": True,
        "deletion_id": event["id"],
        "user_id": user_id,
        "status": "pending_erasure",
        "scope": scopes,
        "enforced": True,
        "physical_delete": False,
        "event": dict(event),
        "note": "仅记录请求与事件；物理清除由 W0 接入 store / 对象存储后执行",
    }


# ---- 3. 敏感数据标记（自伤 / 哀伤 + 严格访问审计） ---------------------------

def mark_sensitive(*, user_id: str, text: str, source: str = "chat") -> dict[str, Any]:
    """对自伤/哀伤内容打敏感标记，写严格访问审计日志（开关开启时）。"""
    try:
        from emotion.crisis import detect  # 复用硬门槛，不修改它
        crisis = detect(text)
    except Exception:
        crisis = {"is_crisis": False, "level": 0, "hit": []}

    categories: list[str] = []
    if crisis.get("is_crisis"):
        categories.append("self_harm")
    if any(k in (text or "") for k in _GRIEF_KEYWORDS):
        categories.append("grief")
    sensitive = bool(categories)
    level = "high" if "self_harm" in categories else ("medium" if sensitive else "none")

    out: dict[str, Any] = {
        "user_id": user_id,
        "source": source,
        "sensitive": sensitive,
        "categories": categories,
        "level": level,
        "access_policy": "restricted" if sensitive else "normal",
        "strict_audit": sensitive,
        "enforced": False,
        "audit_id": None,
    }
    if not gate_enabled() or not sensitive:
        return out

    global _SEQ
    with _LOCK:
        _SEQ += 1
        ts = _now()
        audit: dict[str, Any] = {
            "id": f"sens-{_SEQ}",
            "user_id": user_id,
            "source": source,
            "categories": categories,
            "level": level,
            "crisis_level": crisis.get("level"),
            "access_policy": "restricted",
            "ts": ts,
            "iso": _iso(ts),
            "access_log": "strict",
        }
        _SENSITIVE_AUDIT.append(audit)
    out["enforced"] = True
    out["audit_id"] = audit["id"]
    out["audit"] = dict(audit)
    return out


# ---- FastAPI 路由（独立文件；W0 统一 include_router） ------------------------

router = APIRouter(prefix="/api/compliance")


class DataDeleteReq(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    scope: list[str] | None = None
    reason: str | None = None
    request_id: str | None = None


class SensitiveReq(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    text: str = ""
    source: str = "chat"


@router.get("/disclosure")
def compliance_disclosure(content_type: str = "simulated_call"):
    return ai_disclosure(content_type=content_type)


@router.get("/disclosure/audit")
def disclosure_audit():
    return audit_existing_disclosure()


@router.post("/data-delete")
def data_delete(req: DataDeleteReq):
    return request_data_deletion(
        user_id=req.user_id, scope=req.scope,
        reason=req.reason, request_id=req.request_id,
    )


@router.get("/data-delete/requests")
def deletion_requests(user_id: str | None = None):
    return {"requests": get_deletion_requests(user_id)}


@router.post("/sensitive/mark")
def sensitive_mark(req: SensitiveReq):
    return mark_sensitive(user_id=req.user_id, text=req.text, source=req.source)


@router.get("/sensitive/audit")
def sensitive_audit(user_id: str | None = None):
    return {"audit": get_sensitive_audit(user_id)}


@router.get("/status")
def compliance_status():
    return {"switch": _SWITCH, "enabled": gate_enabled()}
