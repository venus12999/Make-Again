"""状态数据化:把记忆流里的情绪信号,聚合为可解释的"状态"与"记忆点"数据。

纯本地聚合,零 LLM。产出两类:
- 状态指标:baseline / trend / volatility / acute_ratio / calm_ratio / reconcile / risk / stage
- 记忆点数据:active_days / longest_streak / first_reconcile / top_time_tag / top_place_tag / mention_count ...

供报告选卡、回信资格、语气路由复用(报告只把它当内部信号,不直接给用户看冷分数)。
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta

from config import get_settings
from emotion.tone import ACUTE_EMOTIONS
from memory import store

CALM_EMOTIONS = {"平静", "释怀"}
STAGES = ["低谷期", "波动期", "趋稳期", "平静期", "和解期"]


def _local_date(ts) -> str:
    offset = get_settings().timezone_offset_hours
    return (ts + timedelta(hours=offset)).strftime("%Y-%m-%d")


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 50.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_state(db, user_id: str, since=None) -> dict:
    """聚合情绪信号,返回状态指标 + 阶段(0~4 对应 STAGES)。since 为时间下界时只统计该时间之后。"""
    day_scores: dict[str, list[float]] = {}
    day_emotions: dict[str, list[str]] = {}
    for m in store.list_memories(db, user_id, since=since):
        if not (m.emotion and "score" in m.emotion):
            continue
        d = _local_date(m.ts)
        day_scores.setdefault(d, []).append(float(m.emotion["score"]))
        day_emotions.setdefault(d, []).append(m.emotion.get("emotion", "其他"))

    days = sorted(day_scores)
    n_days = len(days)

    last7 = days[-7:]
    scores_7 = [s for d in last7 for s in day_scores[d]]
    baseline = _mean(scores_7)
    volatility = _std(scores_7)

    # 趋势:后半段均值 − 前半段均值
    trend = 0.0
    if n_days >= 2:
        half = max(1, n_days // 2)
        prev_avg = _mean([s for d in days[:half] for s in day_scores[d]])
        recent_avg = _mean([s for d in days[half:] for s in day_scores[d]])
        trend = recent_avg - prev_avg

    last14 = days[-14:]
    emos_14 = [e for d in last14 for e in day_emotions[d]]
    acute_ratio = sum(1 for e in emos_14 if e in ACUTE_EMOTIONS) / len(emos_14) if emos_14 else 0.0
    calm_ratio = sum(1 for e in emos_14 if e in CALM_EMOTIONS) / len(emos_14) if emos_14 else 0.0

    # 风险度:急性占比 + 低基线 + 谷底
    valley = min(scores_7) if scores_7 else 50.0
    risk = _clamp(acute_ratio * 0.6 + (1 - baseline / 100.0) * 0.2 + (0.3 if valley < 30 else 0.0), 0.0, 1.0)

    # 和解度(0~100)
    goal = store.get_or_create_user(db, user_id).goal
    trend_score = _clamp((trend + 10.0) / 20.0, 0.0, 1.0)
    goal_bonus = 1.0 if goal else 0.5
    reconcile = (
        0.35 * calm_ratio + 0.20 * trend_score + 0.20 * (1 - acute_ratio)
        + 0.15 * (baseline / 100.0) + 0.10 * goal_bonus
    ) * 100.0

    stage = _stage(baseline, trend, acute_ratio, calm_ratio, reconcile)
    return {
        "baseline": round(baseline, 1),
        "trend": round(trend, 1),
        "volatility": round(volatility, 1),
        "acute_ratio": round(acute_ratio, 2),
        "calm_ratio": round(calm_ratio, 2),
        "risk": round(risk, 2),
        "reconcile": round(reconcile, 1),
        "stage": stage,
        "stage_label": STAGES[stage],
        "n_days": n_days,
        "n_memories": sum(len(v) for v in day_scores.values()),
    }


def _stage(baseline: float, trend: float, acute_ratio: float, calm_ratio: float, reconcile: float) -> int:
    # 阶段判定(阈值可调,先按经验值)
    if reconcile >= 70 and calm_ratio >= 0.5:
        return 4  # 和解期
    if calm_ratio >= 0.4 and baseline >= 55 and acute_ratio <= 0.3:
        return 3  # 平静期
    if trend > 3 and baseline >= 50:
        return 2  # 趋稳期
    if acute_ratio >= 0.5 or baseline < 45:
        return 0  # 低谷期
    return 1  # 波动期


# ---- 记忆点数据 ----

def active_days(db, user_id: str, since=None) -> int:
    return len({_local_date(m.ts) for m in store.list_memories(db, user_id, since=since)})


def longest_streak(db, user_id: str) -> int:
    """最长连续有倾诉记录的天数。"""
    days = sorted({_local_date(m.ts) for m in store.list_memories(db, user_id)})
    if not days:
        return 0
    dates = [datetime.strptime(d, "%Y-%m-%d").date() for d in days]
    best = cur = 1
    for i in range(1, len(dates)):
        if (dates[i] - dates[i - 1]).days == 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def first_reconcile(db, user_id: str, since=None) -> str | None:
    """第一条体现释怀/放下的记忆的日期(按时间正序),没有则 None。"""
    hits = [k for k in ("释怀", "放下", "祝福", "走出来")]
    mems = sorted(store.list_memories(db, user_id, since=since), key=lambda m: m.ts)
    for m in mems:
        if (m.emotion and m.emotion.get("emotion") == "释怀") or any(k in (m.content or "") for k in hits):
            return _local_date(m.ts)
    return None


def top_tags(db, user_id: str, since=None) -> tuple[str | None, str | None]:
    mems = store.list_memories(db, user_id, since=since)
    time_c = Counter(m.time_tag for m in mems if m.time_tag)
    place_c = Counter(m.place_tag for m in mems if m.place_tag)
    top_time = time_c.most_common(1)[0][0] if time_c else None
    top_place = place_c.most_common(1)[0][0] if place_c else None
    return top_time, top_place


def night_count(db, user_id: str) -> int:
    night = {"凌晨", "深夜", "夜里", "晚上"}
    return sum(1 for m in store.list_memories(db, user_id) if m.time_tag in night)


def mention_count(db, user_id: str) -> int | None:
    """对象「称呼」在倾诉里出现的次数;称呼缺省或为单字代词时返回 None(卡省略)。"""
    obj = store.get_portrait(db, user_id, "object")
    name = (obj or {}).get("称呼") or (obj or {}).get("姓名")
    if not name or len(str(name)) < 2:
        return None
    name = str(name)
    return sum((m.content or "").count(name) + (m.summary or "").count(name)
               for m in store.list_memories(db, user_id))


def saddest_day(db, user_id: str) -> str | None:
    """平均情绪分最低的一天(本地日期),无数据则 None。"""
    day_scores: dict[str, list[float]] = {}
    for m in store.list_memories(db, user_id):
        if m.emotion and "score" in m.emotion:
            day_scores.setdefault(_local_date(m.ts), []).append(float(m.emotion["score"]))
    if not day_scores:
        return None
    return min(day_scores, key=lambda d: _mean(day_scores[d]))


# ============================================================================
# WP5 · 显式场景状态机（开关 USE_PERSONA，默认关闭、可回滚）
#
# 设计约束：
# - 不新增表 / 不改 companions 结构；场景真源就是 companions.relationship_type，
#   `store.effective_loss_type()` 已经是全站消费方（chat 路由 / 语气 / 树洞 / 主题）。
# - 「进入新关系」= 归档旧 companion + 新建新关系 companion 并置为当前，
#   这样历史陪伴完整保留，后续对话自然按新场景 prompt 路由。
# - 状态机显式化：命名状态 / 命名动作 / 决策记录 / 触发源，全部可观测、可测试。
# ============================================================================

SCENE_BREAKUP = "breakup"
SCENE_LOVED_ONE = "loved_one"
SCENE_PET = "pet"
SCENE_FRIEND = "friend"
VALID_SCENES = frozenset({SCENE_BREAKUP, SCENE_LOVED_ONE, SCENE_PET, SCENE_FRIEND})

TRIGGER_INTERVIEW = "interview"            # 访谈后进入新关系
TRIGGER_EXPLICIT = "explicit"              # 用户显式设置
TRIGGER_RELATIONSHIP = "relationship"      # 关系类型变化驱动
VALID_TRIGGERS = frozenset({TRIGGER_INTERVIEW, TRIGGER_EXPLICIT, TRIGGER_RELATIONSHIP})

ACTION_NOOP = "noop"
ACTION_ARCHIVE_AND_CREATE = "archive_and_create"

# 新场景 prompt 提示段：供 get_persona / 后续对话使用；与 agent 各场景引擎的设定保持一致。
SCENE_PROMPT_HINT = {
    SCENE_BREAKUP: "分手场景：陪用户带着这段关系继续生活；想弄明白「为什么会分手」时可做去指责化的平衡复盘。",
    SCENE_LOVED_ONE: "亲友离世场景：哀伤双过程，陪伴接纳丧失、意义重建与纪念，不做归咎。",
    SCENE_PET: "宠物离世场景：给哀伤正名，温柔接住「安乐内疚」，一起纪念这段陪伴。",
    SCENE_FRIEND: "其它关系场景：通用温柔陪伴，不主动翻旧事，稳稳接住当下情绪。",
}


def persona_enabled() -> bool:
    """WP5 总开关 USE_PERSONA，默认关闭。W0 验证通过后统一打开；关闭时所有写操作为 no-op。"""
    return os.getenv("USE_PERSONA", "0").strip().lower() in {"1", "true", "yes", "on"}


def scene_prompt_hint(loss_type: str | None) -> str:
    """给定 loss_type 返回该场景的 prompt 提示段；未知类型回落通用陪伴。"""
    return SCENE_PROMPT_HINT.get(loss_type, SCENE_PROMPT_HINT[SCENE_FRIEND])


def current_scene_prompt(db, user_id: str) -> str:
    """从 companions 真源解析当前场景，供 persona / prompt 拼装使用。"""
    return scene_prompt_hint(store.effective_loss_type(db, user_id))


@dataclass(frozen=True)
class SceneDecision:
    """一次场景切换的显式决策记录（不可变，便于审计与回放）。"""

    current_scene: str | None
    target_scene: str
    trigger: str
    action: str
    relationship_type: str
    reason: str


class SceneStateMachine:
    """场景切换状态机。

    状态集：breakup / loved_one / pet / friend（由 companions.relationship_type 映射而来）。
    转换规则（显式）：
      - 目标 == 当前              → NOOP，不写库。
      - 目标 != 当前（均为合法态） → ARCHIVE_AND_CREATE：归档旧陪伴、新建新关系陪伴并置为当前。
    触发源：interview / explicit / relationship，仅作审计标记，不改变转换规则。
    """

    def decide(self, current_scene: str | None, target_scene: str,
               trigger: str = TRIGGER_EXPLICIT) -> SceneDecision:
        if target_scene not in VALID_SCENES:
            raise ValueError(f"未知目标场景: {target_scene!r}")
        if trigger not in VALID_TRIGGERS:
            raise ValueError(f"未知触发源: {trigger!r}")
        relationship_type = store.normalize_relationship_type(target_scene)
        if relationship_type is None:
            raise ValueError(f"场景无法映射到关系类型: {target_scene!r}")
        if current_scene == target_scene:
            return SceneDecision(current_scene, target_scene, trigger, ACTION_NOOP,
                                 relationship_type, "目标场景与当前场景一致，无需切换")
        return SceneDecision(
            current_scene, target_scene, trigger, ACTION_ARCHIVE_AND_CREATE,
            relationship_type,
            f"场景变化 {current_scene or '(无)'} → {target_scene}：归档旧陪伴并新建新关系陪伴",
        )

    def apply(self, db, user_id: str, target_scene: str,
              trigger: str = TRIGGER_EXPLICIT) -> dict:
        """执行状态机；返回含决策记录与前后 loss_type 的可观测结果。开关关闭时不写库。"""
        if not persona_enabled():
            return {
                "ok": True, "enabled": False, "changed": False, "decision": None,
                "previous_loss_type": None, "loss_type": None,
                "reason": "USE_PERSONA 未开启（默认关闭），未改动任何数据",
            }
        current_scene = store.effective_loss_type(db, user_id)
        decision = self.decide(current_scene, target_scene, trigger)
        result = {
            "ok": True,
            "enabled": True,
            "decision": decision,
            "changed": decision.action != ACTION_NOOP,
            "trigger": trigger,
            "previous_loss_type": current_scene,
            "loss_type": target_scene if decision.action != ACTION_NOOP else current_scene,
            "scene_prompt": scene_prompt_hint(
                target_scene if decision.action != ACTION_NOOP else current_scene
            ),
        }
        if decision.action == ACTION_NOOP:
            current = store.get_current_companion(db, user_id)
            result["companion_id"] = current.id if current else None
            result["reason"] = decision.reason
            return result

        previous = store.get_current_companion(db, user_id)
        previous_id = previous.id if previous else None
        if previous is not None and previous.archived_at is None:
            store.archive_companion(db, previous.id, ritual="scene_transition")
        new_companion = store.create_companion(
            db, user_id, relationship_type=decision.relationship_type, make_current=True
        )
        result.update({
            "previous_companion_id": previous_id,
            "companion_id": new_companion.id,
            "relationship_type": new_companion.relationship_type,
            # 真源回读：确认新的 loss_type 已由 companions 表生效
            "loss_type": store.effective_loss_type(db, user_id),
            "reason": decision.reason,
        })
        return result


_SCENE_MACHINE = SceneStateMachine()


def transition_scene(db, user_id: str, target_loss_type: str,
                     trigger: str = TRIGGER_EXPLICIT) -> dict:
    """便捷入口：访谈 / 显式设置后调用，识别并落场景变化。"""
    return _SCENE_MACHINE.apply(db, user_id, target_loss_type, trigger)


def on_relationship_change(db, user_id: str, relationship_type: str,
                           trigger: str = TRIGGER_RELATIONSHIP) -> dict:
    """由关系类型变化反向驱动状态机（relationship 变化 → loss_type 变化）。"""
    normalized = store.normalize_relationship_type(relationship_type)
    loss_type = store.relationship_to_loss_type(normalized)
    if loss_type is None:
        return {"ok": False, "enabled": persona_enabled(), "changed": False,
                "error": "invalid_relationship_type", "relationship_type": relationship_type}
    return transition_scene(db, user_id, loss_type, trigger)


def scene_decision_for(current_scene: str | None, target_scene: str,
                       trigger: str = TRIGGER_EXPLICIT) -> SceneDecision:
    """纯函数：只产出决策记录，不碰 DB（供路由/前端预判与测试断言）。"""
    return _SCENE_MACHINE.decide(current_scene, target_scene, trigger)
