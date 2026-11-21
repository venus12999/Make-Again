"""后台记忆回写:把"抽取事实/情绪 + 落库"移到后台线程,让主对话先返回。

设计要点:
- 主对话(陪伴 Agent)只同步做 危机检测 → 召回 → 生成回复,这三步是用户等待的关键路径。
- 记忆抽取(2 次 LLM 调用)与落库放到单 worker 后台线程,不阻塞回复。
- 单 worker 串行化写库,避免 SQLite 并发写锁竞争;后台任务使用独立的 DB 会话与 LLM 客户端。
"""
from __future__ import annotations

import logging

from gateway.client import LLMClient
from memory.db import SessionLocal

logger = logging.getLogger(__name__)


def schedule_memory_write(user_id: str, message: str, *, type: str = "chat", extractor=None) -> None:
    """提交一次后台回写,立即返回。抽取失败不影响主流程。

    extractor 为可选的自定义抽取函数 `f(client, text) -> turn dict`,
    用于按场景替换抽取逻辑(如失恋场景的 CBT 抽取);缺省走通用抽取。
    """
    from memory.jobs import enqueue
    enqueue("memory",user_id,{"message":message,"type":type,"extractor":extractor.__module__ if extractor else None})


def schedule_trial_echo(user_id: str, message: str, *, fail_times: int = 0) -> None:
    """Huey 试点的可选入口：在当前事务内投一个 trial_echo intent。

    开关 USE_HUEY_TRIAL 关闭时是纯 no-op；现有 schedule_memory_write 链路不变。
    """
    from memory.jobs import enqueue, huey_trial_enabled
    if not huey_trial_enabled():
        return
    enqueue("trial_echo", user_id, {"message": message, "fail_times": fail_times})


def _do_write(user_id: str, message: str, type: str, extractor=None) -> None:
    from memory import extract, portrait, reflect, store

    db = SessionLocal()
    try:
        client = LLMClient()  # 独立客户端,避免与请求线程共享 httpx 连接
        if extractor is not None:
            turn = extractor(client, message)
        else:
            turn = extract.extract_turn(client, message)
        # 记忆用户控制权：普通聊天抽取(extractor=None)走 Candidate 确认流；
        # 场景提取器(breakup/loved_one/pet)携带 companion scope，直接落 Fact——
        # Candidate 登记册不带 scope，走确认会绕过 UC-4 跨 Journey 隔离。
        from memory import confirm as confirm_mod
        if extractor is not None:
            entry = store.add_memory(
                db, user_id, type=type, content=message, summary=turn["summary"],
                facts=turn["facts"], emotion=turn["emotion"], importance=turn["importance"],
                place_tag=turn["place_tag"], time_tag=turn["time_tag"], cbt=turn.get("cbt"),
            )
            result = {"mode": "baseline", "memory_id": str(entry.id)}
        else:
            result = confirm_mod.submit_candidate(
                db, user_id, type=type, content=message, summary=turn["summary"],
                facts=turn["facts"], emotion=turn["emotion"], importance=turn["importance"],
                place_tag=turn["place_tag"], time_tag=turn["time_tag"], cbt=turn.get("cbt"),
            )
        if result.get("mode") == "confirm":
            # 待用户确认：此时无 entry，跳过依赖 entry 的情绪节点更新；
            # 回声壁/画像仍可做（基于消息与情绪，不依赖落库 entry）。
            logger.info("memory candidate pending confirmation: %s", result.get("candidate_id"))
        else:
            # 基线路径：entry 已写入，继续下游增强。
            entry = db.get(store.MemoryEntry, result["memory_id"])
            if entry is not None:
                reflect.upsert_emotion_node(db, entry)
        # 回声壁:随机捕获富有情感的句子,随机延时后回显(零 LLM、纯采样,不阻塞主流程)
        from agent import echo as echo_mod
        echo_mod.maybe_capture_echo(db, user_id, message, turn.get("emotion"))
        # 积累足够新事实后,后台增量合并进画像(不进热路径)
        from memory.jobs import enqueue, flush_intents
        enqueue("portrait",user_id,{})
        try:
            flush_intents(db)
        except Exception:  # noqa: BLE001 flush 失败不阻塞主流程
            logger.exception("flush_intents failed in _do_write")
    except Exception:  # noqa: BLE001 后台任务必须兜底,不能把异常抛回请求
        raise
    finally:
        db.close()


def flush_memory_writes() -> None:
    """兼容入口:记忆回写已改为持久化 Job,由 worker 异步完成;此处不再阻塞。

    脚本/测试需要等待落库时,应轮询 durable job 状态或直接运行 worker。
    """
