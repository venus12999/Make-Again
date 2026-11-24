"""WP4 · 危机触达闭环（构建在 emotion.crisis 硬门槛之上）。

设计约束（来自 MASTER_PLAN_PARALLEL.md 第 5 节 WP4 行为验收表）：

- 不改硬门槛：本模块只 import crisis.detect / CRISIS_MESSAGE，绝不修改关键词表、
  否定词守卫、第三人称守卫；detect 返回结构（level/hit/is_crisis）保持逐字不变。
- 开关USE_CRISIS_FLOW：用 os.getenv 读取，默认关闭；关闭时 handle_crisis 退化成旧路径
  （只返回 detect 字段 + CRISIS_MESSAGE），不落事件、不分级、不确认、不冷却，可随时回滚。
- 分级话术（C-SSRS / SAFE-T 思路）：level 3 = 明确自伤意念/方式 → 强干预话术 + 热线 +
  人工复核入口；level 2 = 被动意念/痛苦表达 → 支持性话术 + 热线，但不上人工复核。
- 确认轮：高危命中后 need_confirm=True 且给出 confirm_prompt「你现在安全吗」，不是直接结束；
  confirm_crisis() 处理用户回答，回答"不安全"则升级人工复核。
- 冷却期：同用户 1 小时内重复高危，不再重复弹同样话术（改成 COOLDOWN_MESSAGE，避免骚扰），
  但每条都落 crisis event 记录。
- 记录：store.py 不属于本工作包所有权，故用进程内内存表 + threading.Lock；对外暴露
  get_events() / get_pending_confirm() / reset_crisis_state() 供验证与测试。
- W0 集成入口：crisis.detect_with_flow(text, user_id, **kw) 会委托给本模块 handle_crisis。
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

from emotion.crisis import CRISIS_MESSAGE, detect

logger = logging.getLogger(__name__)

# ---- 开关 -------------------------------------------------------------------

_SWITCH = "USE_CRISIS_FLOW"


def _flow_enabled() -> bool:
    """每次调用都读环境变量，便于运行时/测试切换；默认关闭。"""
    return os.getenv(_SWITCH, "0").strip().lower() in ("1", "true", "yes", "on")


# ---- 常量：热线 / 分级话术 / 确认轮 / 冷却 ----------------------------------

HOTLINE = "全国 24 小时心理援助热线 400-161-9995，紧急情况请拨打 120 或 110。"

# level 3（明确自伤意念/方式，C-SSRS 高风险）：强干预 + 热线 + 复核入口
CRISIS_MESSAGE_L3 = (
    "我听见了，你现在真的处在一个很危险的时刻，这不是你一个人在扛。\n"
    + HOTLINE + "\n"
    "如果你身边有人，请现在就告诉他们；也可以直接联系上面的热线。"
    "我会陪着你，但此刻专业的人能给你更实在的帮助。"
)

# level 2（被动意念/痛苦表达，SAFE-T 中风险）：支持性 + 热线，不弹人工复核
CRISIS_MESSAGE_L2 = (
    "听起来你正在经历很难熬的一段时间，你的感受我收到了，也不用一个人硬撑。\n"
    + HOTLINE + "\n"
    "如果痛苦在加重，请一定联系上面能真正陪你聊的人；我在这里陪着你。"
)

# 冷却期内不再重复弹完整话术（避免骚扰），但仍落记录
COOLDOWN_MESSAGE = (
    "我还在，也没有忘记你刚才说的。你不用重复告诉我——"
    "如果现在有危险，请直接拨打 400-161-9995 或 120。"
)

CONFIRM_PROMPT = (
    "你现在安全吗？如果此刻有伤害自己的念头或冲动，请直接告诉我，"
    "并马上联系 400-161-9995 或 120。"
)

COOLDOWN_SECONDS = 3600  # 1 小时

_GRADED = {2: ("level2", "中风险", CRISIS_MESSAGE_L2), 3: ("level3", "高风险", CRISIS_MESSAGE_L3)}


def _review_entry(level: int, forced: bool = False) -> dict:
    """level 3 的人工复核入口标记（结构化，便于 W0/前端消费）。"""
    return {
        "queue": "human_review",
        "priority": "high" if level >= 3 else "normal",
        "reason": "escalated_by_user_answer" if forced else "level3_suicidal_ideation",
        "level": level,
    }


# ---- 进程内事件表 ------------------------------------------------------------

_LOCK = threading.Lock()
_SEQ = 0
_EVENTS: list[dict[str, Any]] = []                 # 全部危机事件（含冷却期内的重复事件）
_LAST_SCRIPT_TS: dict[str, float] = {}            # user_id -> 上次弹出完整话术的时间
_PENDING_CONFIRM: dict[str, dict[str, Any]] = {}  # user_id -> 待确认事件


def _now(now: float | None) -> float:
    return time.time() if now is None else now


def reset_crisis_state() -> None:
    """清空内存表（测试/重置用）。"""
    global _SEQ
    with _LOCK:
        _SEQ = 0
        _EVENTS.clear()
        _LAST_SCRIPT_TS.clear()
        _PENDING_CONFIRM.clear()


def get_events(user_id: str | None = None) -> list[dict[str, Any]]:
    """读取已记录的危机事件（副本，避免外部修改内部状态）。"""
    with _LOCK:
        rows = _EVENTS if user_id is None else [e for e in _EVENTS if e["user_id"] == user_id]
        return [dict(e) for e in rows]


def get_pending_confirm(user_id: str) -> dict[str, Any] | None:
    with _LOCK:
        ev = _PENDING_CONFIRM.get(user_id)
        return dict(ev) if ev is not None else None


# ---- 主入口 ------------------------------------------------------------------

def handle_crisis(text: str, user_id: str, session_id: str | None = None, now: float | None = None) -> dict:
    """在硬门槛之上返回带闭环字段的危机结果。

    返回结构在 detect 的 level/hit/is_crisis 基础上**增量**扩展，不删改任何旧字段：
      关闭开关：额外只带 reply（== 旧 CRISIS_MESSAGE），完全等价旧路径；
      开启开关：额外带 reply / need_confirm / confirm_prompt / review_required /
                review_entry / graded_script / severity / hotline / cooldown_active /
                script_repeated / event_id / event。
    """
    base = dict(detect(text))  # 硬门槛原样调用，零修改
    if not _flow_enabled():
        # 回滚路径：等价于 agent 里旧的 detect + CRISIS_MESSAGE，不产生任何新字段/记录
        base["reply"] = CRISIS_MESSAGE if base["is_crisis"] else None
        return base

    ts = _now(now)
    if not base["is_crisis"]:
        base.update({
            "reply": None,
            "need_confirm": False,
            "confirm_prompt": None,
            "review_required": False,
            "review_entry": None,
            "graded_script": None,
            "severity": None,
            "hotline": None,
            "cooldown_active": False,
            "script_repeated": False,
            "event_id": None,
            "event": None,
        })
        return base

    level = base["level"]
    graded_script, severity, graded_message = _GRADED.get(level, _GRADED[2])

    global _SEQ
    with _LOCK:
        last = _LAST_SCRIPT_TS.get(user_id)
        cooldown_active = last is not None and (ts - last) < COOLDOWN_SECONDS
        script_repeated = not cooldown_active
        reply = COOLDOWN_MESSAGE if cooldown_active else graded_message
        if script_repeated:
            _LAST_SCRIPT_TS[user_id] = ts

        review_required = level >= 3
        _SEQ += 1
        event: dict[str, Any] = {
            "id": f"crisis-{_SEQ}",
            "user_id": user_id,
            "session_id": session_id,
            "ts": ts,
            "text": text,
            "level": level,
            "severity": severity,
            "hit": list(base["hit"]),
            "need_confirm": True,
            "confirm_prompt": CONFIRM_PROMPT,
            "review_required": review_required,
            "review_entry": _review_entry(level) if review_required else None,
            "cooldown_active": cooldown_active,
            "script_repeated": script_repeated,
            "reply": reply,
            "state": "pending_confirm",
        }
        _EVENTS.append(event)
        _PENDING_CONFIRM[user_id] = event

    base.update({
        "reply": reply,
        "need_confirm": True,
        "confirm_prompt": CONFIRM_PROMPT,
        "review_required": review_required,
        "review_entry": dict(event["review_entry"]) if event["review_entry"] else None,
        "graded_script": graded_script,
        "severity": severity,
        "hotline": HOTLINE,
        "cooldown_active": cooldown_active,
        "script_repeated": script_repeated,
        "event_id": event["id"],
        "event": dict(event),
    })
    return base


def confirm_crisis(user_id: str, safe: bool, now: float | None = None, writeback=None) -> dict:
    """确认轮的用户回答：safe=True 记 confirmed_safe；safe=False 升级人工复核。

    R3（U-004 最后一公里）：更新内存事件**之后**额外做一次持久化回写。
      - ``writeback`` 是可选回调（签名同 memory.crisis_store.update_event_state）；
        不传时按开关懒加载默认实现，避免 emotion 模块 import 期依赖 memory。
      - **仅 USE_CRISIS_FLOW 开时才调用**；关时零调用，返回 dict 与旧版逐字节一致
        （H1 的默认关语义不变）。
      - 回写失败 swallow：只记 warning，不影响确认轮话术/返回结构。
      - ``no_pending_confirm`` 早退路径在回写之前，行为也不变。
    """
    ts = _now(now)
    with _LOCK:
        ev = _PENDING_CONFIRM.pop(user_id, None)
        if ev is None:
            return {"ok": False, "reason": "no_pending_confirm", "user_id": user_id}
        ev["state"] = "confirmed_safe" if safe else "escalated"
        ev["confirmed_at"] = ts
        if not safe:
            ev["review_required"] = True
            ev["review_entry"] = _review_entry(ev["level"], forced=True)
        result = {"ok": True, "event": dict(ev)}

    if _flow_enabled():
        snapshot = result["event"]  # 锁内已复制，锁外只读快照，不碰共享 ev
        hook = writeback
        if hook is None:
            try:
                from memory.crisis_store import update_event_state as hook  # 懒加载，避免 import 期循环依赖
            except Exception as exc:
                logger.warning("crisis confirm writeback unavailable: %s", exc)
                hook = None
        if hook is not None:
            try:
                hook(
                    user_id=user_id,
                    event_ref=snapshot.get("id"),
                    state=snapshot["state"],
                    extra={
                        "review_required": snapshot.get("review_required"),
                        "review_entry": snapshot.get("review_entry"),
                        "confirmed_at": snapshot.get("confirmed_at"),
                    },
                )
            except Exception as exc:  # 回写是增强，绝不能破坏确认轮话术
                logger.warning("crisis confirm state writeback failed (non-blocking): %s", exc)
    return result
