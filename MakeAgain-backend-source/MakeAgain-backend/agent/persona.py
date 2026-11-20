"""WP5 · persona 卡：不可变身份/语气底线 与 可变用户画像 分离。

- 不可变部分（陪伴者身份、语气底线）是代码级常量，任何 update_user_profile 都改写不了；
  get_persona 每次都从常量读取，DB 里即使被塞了同名字段也一律忽略。
- 可变画像落在既有 portraits 表（kind="persona_profile"，复用 JSON content 字段，
  不新增表/列），摘要是它的派生只读字段（"_summary"），画像一变就刷新，
  prompt 段不会出现「旧身份残留」。
- 总开关 USE_PERSONA 默认关闭：关闭时所有写接口是 no-op，行为与改造前一致。

对外接口：get_persona / update_user_profile / refresh_summary / refresh_if_due /
         build_persona_prompt。
"""
from __future__ import annotations

import os
from types import MappingProxyType

from memory import store
from memory.state import current_scene_prompt, persona_enabled

# 复用既有 portraits 表，不碰表结构
PERSONA_PROFILE_KIND = "persona_profile"

# ---- 不可变部分：代码常量，用户画像更新永远改不到 ----
IMMUTABLE_CORE = MappingProxyType({
    "identity": (
        "你是「重逢」，一位温柔、耐心、不评判的 AI 陪伴者，"
        "陪伴经历过失去的人（分手/亲友离世/宠物离世）带着记忆继续生活。"
    ),
    "tone_bottom_line": (
        "温暖、具体、有共情；不评判任何情绪，不空喊「加油/振作」，不催促「快好起来」；"
        "不是心理医生，不诊断、不开药；对方有自伤倾向时立即引导寻求专业帮助；"
        "只引用用户说过的内容，绝不编造细节；不用括号写动作神态，方便语音朗读。"
    ),
})
IMMUTABLE_FIELDS = frozenset(IMMUTABLE_CORE.keys())

# ---- 派生/内部字段：用户不可写，由本模块维护 ----
SUMMARY_KEY = "_summary"
SUMMARY_TURN_KEY = "_summary_at_turn"
INTERNAL_PREFIX = "_"
DEFAULT_SUMMARY = "（画像为空，尚未生成摘要）"


def _summary_every() -> int:
    """定期摘要刷新的轮数阈值，可用 PERSONA_SUMMARY_EVERY 覆盖，默认 30。"""
    try:
        return max(1, int(os.getenv("PERSONA_SUMMARY_EVERY", "30")))
    except ValueError:
        return 30


def _read_profile(db, user_id: str) -> dict:
    return dict(store.get_portrait(db, user_id, PERSONA_PROFILE_KIND) or {})


def _write_profile(db, user_id: str, profile: dict) -> None:
    store.upsert_portrait(db, user_id, PERSONA_PROFILE_KIND, profile, status="draft")


def _public_profile(profile: dict) -> dict:
    """剥掉内部派生字段，只暴露可变画像本身。"""
    return {k: v for k, v in profile.items() if not k.startswith(INTERNAL_PREFIX)}


def _profile_text(profile: dict) -> str:
    if not profile:
        return "（暂未设置可变画像）"
    lines = []
    for k, v in profile.items():
        if isinstance(v, (list, tuple)):
            v = "、".join(str(x) for x in v) if v else "无"
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def _local_summary(profile: dict) -> str:
    """零 LLM 的确定性摘要：完全由当前画像推导，天然无旧身份残留。"""
    if not profile:
        return DEFAULT_SUMMARY
    parts = []
    for k, v in profile.items():
        if isinstance(v, (list, tuple)):
            v = "、".join(str(x) for x in v) if v else "无"
        parts.append(f"{k}：{v}")
    return "当前画像 — " + "；".join(parts) + "。"


def _llm_summary(profile: dict, client) -> str:
    """真实模型可用时走一次受约束的摘要；mock/无 client 一律走确定性摘要。"""
    if client is None or getattr(client, "mock", False):
        return _local_summary(profile)
    system = (
        "你是「重逢」的画像摘要器。只根据可变用户画像输出一句中文摘要。"
        "不得输出、改写或复述陪伴者身份与语气底线，不得编造画像里没有的内容。"
    )
    user = (
        f"不可变身份与底线（仅用于边界校验，禁止写进摘要）：{dict(IMMUTABLE_CORE)}\n"
        f"可变用户画像：{profile}\n只输出摘要本身。"
    )
    try:
        result = client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2, model=client.settings.llm_fast_model,
        )
    except Exception:  # noqa: BLE001 摘要失败回落确定性结果，不影响主链路
        return _local_summary(profile)
    return (result.get("content") or "").strip() or _local_summary(profile)


def build_persona_prompt(persona: dict) -> str:
    """把 persona 卡拼成可直接注入 system prompt 的文本段。"""
    immutable = persona.get("immutable") or {}
    parts = []
    if persona.get("scene_prompt"):
        parts.append(f"[场景] {persona['scene_prompt']}")
    parts += [
        "[Persona·不可变身份与底线]",
        f"身份：{immutable.get('identity', IMMUTABLE_CORE['identity'])}",
        f"语气底线：{immutable.get('tone_bottom_line', IMMUTABLE_CORE['tone_bottom_line'])}",
        "[Persona·可变用户画像]",
        _profile_text(persona.get("profile") or {}),
        "[Persona·当前画像摘要]",
        persona.get("summary") or DEFAULT_SUMMARY,
    ]
    return "\n".join(parts)


def get_persona(db, user_id: str, *, include_prompt: bool = True) -> dict:
    """读取 persona 卡：不可变常量 + 当前可变画像 + 派生摘要（+ 场景 prompt 段）。纯读。"""
    profile = _read_profile(db, user_id)
    public = _public_profile(profile)
    persona = {
        "enabled": persona_enabled(),
        # 每次都从代码常量取，DB 里被塞进来的同名字段一律不生效
        "immutable": dict(IMMUTABLE_CORE),
        "profile": public,
        "summary": profile.get(SUMMARY_KEY) or _local_summary(public),
    }
    if include_prompt:
        persona["scene_prompt"] = current_scene_prompt(db, user_id)
        persona["prompt_section"] = build_persona_prompt(persona)
    return persona


def refresh_summary(db, user_id: str, client=None, *, turns: int | None = None,
                    force: bool = False) -> dict:
    """从最新可变画像重算摘要并落库；返回最新 persona。开关关闭时不写库。"""
    if not persona_enabled():
        return get_persona(db, user_id)
    profile = _read_profile(db, user_id)
    current_turn = turns if turns is not None else (
        store.count_completed_chat_turns(db, user_id)
    )
    if not force and current_turn is not None:
        last = profile.get(SUMMARY_TURN_KEY)
        if last is not None and current_turn - int(last) < _summary_every():
            return get_persona(db, user_id)  # 未到阈值，不改摘要
    public = _public_profile(profile)
    profile[SUMMARY_KEY] = _llm_summary(public, client)
    profile[SUMMARY_TURN_KEY] = current_turn
    _write_profile(db, user_id, profile)
    return get_persona(db, user_id)


def refresh_if_due(db, user_id: str, client=None) -> dict:
    """定期刷新入口：距上次刷新达到 PERSONA_SUMMARY_EVERY 轮才真正重算。"""
    return refresh_summary(db, user_id, client=client, force=False)


def update_user_profile(db, user_id: str, updates: dict, *, client=None) -> dict:
    """只更新可变画像；不可变字段/内部字段被拒并列入 rejected，绝不落库。

    画像一变就强制刷新摘要，保证 prompt 段与最新画像一致。开关关闭时整体 no-op。
    """
    if not isinstance(updates, dict):
        raise TypeError("updates 必须是 dict")
    rejected = sorted(
        k for k in updates
        if k in IMMUTABLE_FIELDS or (isinstance(k, str) and k.startswith(INTERNAL_PREFIX))
    )
    accepted = {k: v for k, v in updates.items() if k not in rejected}
    if not persona_enabled():
        return {
            "enabled": False, "changed": False,
            "accepted": [], "rejected": rejected,
            "reason": "USE_PERSONA 未开启（默认关闭），未改动任何数据",
            "persona": get_persona(db, user_id),
        }
    profile = _read_profile(db, user_id)
    profile.update(accepted)
    _write_profile(db, user_id, profile)
    persona = refresh_summary(db, user_id, client=client, force=True)
    return {
        "enabled": True,
        "changed": bool(accepted),
        "accepted": sorted(accepted),
        "rejected": rejected,
        "persona": persona,
    }
