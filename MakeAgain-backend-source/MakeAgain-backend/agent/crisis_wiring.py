"""共享危机检测入口（D-073 包 R2 · H1 尾巴）。

把 H1 在 ``agent/companion.py::_detect_crisis`` 里实现的开关语义抽成独立模块，
供三个场景专属引擎（breakup / loved_one / pet）与 agent_v2 共用，避免复制四份。

开关语义（与 companion.py::_detect_crisis 完全一致）：
  * ``USE_CRISIS_FLOW`` 关（默认）→ 直接返回 ``emotion.crisis.detect(message)`` 原值，
    连 dict 对象都不复制，逐字节等价旧路径；
  * 开 → 只在硬门槛命中（base["is_crisis"]）时叠加增量：调 detect_with_flow 取
    闭环字段 → 去掉 level/hit/is_crisis 后挂到 ``crisis["flow"]`` → 用 crisis_store
    独立会话真提交落一条事件（用户可见 reply 仍是 CRISIS_MESSAGE，硬门槛不变）；
  * 任何失败（闭环探测 / 落库）一律 swallow，绝不破坏危机话术返回。

诚实边界（本次范围）：
  * 本模块是 companion.py::_detect_crisis 的**第二处实现**（同语义、非同一份代码）。
    companion.py 在 H1（D-070/D-072 链）已改且属于别人的写作用域，本包不碰它；
    两处实现未来可合并为一份，本次不合并。
  * 只对 detect() 判定的 is_crisis 叠加闭环；loved_one/pet 的「想随逝者/宠物而去」
    隐晦暗示仍是各引擎自己的 reunion 守卫（原先就没走 detect 的 is_crisis），
    本包不改变该语义。
"""
from __future__ import annotations

import logging
import os

from emotion.crisis import detect

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}


def crisis_flow_enabled() -> bool:
    """逐字读取开关，默认关；每次调用都读，便于运行时/测试切换。"""
    return os.getenv("USE_CRISIS_FLOW", "0").strip().lower() in _TRUTHY


def detect_crisis(db, user_id: str, message: str, session_id: str | None) -> dict:
    """硬门槛检测；开关开启时叠加危机闭环（分级/确认/冷却/事件持久化）。

    与 companion.py::_detect_crisis 的入参、返回结构、失败处理逐字一致。
    """
    base = detect(message)  # 原硬门槛永远先算：非危机 / 开关关闭逐字节走旧路径
    if not base["is_crisis"] or not crisis_flow_enabled():
        return base
    try:
        from emotion.crisis import detect_with_flow
        from memory import crisis_store

        flow = detect_with_flow(message, user_id, session_id=session_id)
        # 只增量扩展字段：level/hit/is_crisis 不变，也不替换用户可见的 CRISIS_MESSAGE。
        base = dict(base)
        base["flow"] = {k: v for k, v in flow.items()
                        if k not in ("level", "hit", "is_crisis")}
        event = flow.get("event")
        if event:
            # crisis_store 内部用 _RawSessionLocal 独立会话真提交，绕开 managed
            # session 只 flush 不 commit 的坑（D-070 同类坑）；其自身已 swallow 异常。
            # 这里再包一层：落库失败只丢持久化，不影响已算好的内存闭环字段与话术。
            try:
                crisis_store.save_crisis_event(event, db=db)
            except Exception as exc:
                logger.warning("crisis event persistence failed (non-blocking): %s", exc)
    except Exception as exc:
        logger.warning("crisis flow wiring failed (non-blocking): %s", exc)
    return base
