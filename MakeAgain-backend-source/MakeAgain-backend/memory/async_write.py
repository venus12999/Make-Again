"""后台记忆回写:把"抽取事实/情绪 + 落库"移到后台线程,让主对话先返回。

设计要点:
- 主对话(陪伴 Agent)只同步做 危机检测 → 召回 → 生成回复,这三步是用户等待的关键路径。
- 记忆抽取(2 次 LLM 调用)与落库放到单 worker 后台线程,不阻塞回复。
- 单 worker 串行化写库,避免 SQLite 并发写锁竞争;后台任务使用独立的 DB 会话与 LLM 客户端。
"""
from __future__ import annotations

import logging
import os

from gateway.client import LLMClient
from memory.db import SessionLocal

logger = logging.getLogger(__name__)

# ---- Session checkpoint 接线（USE_CHECKPOINT，默认关；关闭时零调用逐字节回退） ----
# D-073：把 checkpoint 的 save 挂到「后台写入链」_do_write（memory job），不进 chat 热路径。
# 默认关：_maybe_save_checkpoint 立即返回——不 import checkpoint、不建表、不写库、不 commit，
# 后台写入链路与接线前逐字节一致（可回退）。开时懒建 session_checkpoints 表（不改 models.py）。
USE_CHECKPOINT_ENV = "USE_CHECKPOINT"
_CHECKPOINT_TRUTHY = frozenset(("1", "true", "yes", "on"))


def checkpoint_enabled() -> bool:
    """运行期读取开关（os.getenv，测试可 monkeypatch）；默认关。"""
    return os.getenv(USE_CHECKPOINT_ENV, "").strip().lower() in _CHECKPOINT_TRUTHY


def _maybe_save_checkpoint(db, user_id: str, message: str, type: str,
                           turn: dict | None, result: dict, session_id: str | None) -> None:
    """后台写入成功后落一份 checkpoint（最近主题 + 阶段 + 关键事实 id）。

    开关关闭时**零调用**直接返回；开启时表由 checkpoint.ensure_checkpoint_table 懒建。
    旁路增强：任何失败都 swallow，绝不破坏已完成的记忆落库（store.add_memory 已先 commit）。
    session_id 缺省用 "default"，保证显式 session 之间的隔离可测且向后兼容。
    """
    if not checkpoint_enabled():
        return
    try:
        from memory import checkpoint as ckpt

        sid = str(session_id or "").strip() or "default"
        summary = str((turn or {}).get("summary") or "")[:200]
        payload = {
            "stage": type,
            "topics": [t for t in ((turn or {}).get("place_tag"), (turn or {}).get("time_tag")) if t],
            "summary": summary,
            "key_fact_ids": [str(result["memory_id"])] if result.get("memory_id") else [],
            "last_message": str(message or "")[:200],
        }
        ckpt.ensure_checkpoint_table(db)
        ckpt.save_checkpoint(db, user_id, sid, payload, summary=summary)
        db.commit()
    except Exception:  # noqa: BLE001 - checkpoint 是旁路，失败不能破坏记忆写入
        logger.warning("checkpoint save failed in _do_write", exc_info=True)
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass


def link_facts_for_entry(db, memory_id, user_id: str, facts, *,
                         entities=None, entity_extractor=None) -> list:
    """WP-1 接线接口：已落库记忆的 facts → 实体图回填 → 写回业务库（持久化）。

    给 confirm 路由在 ``confirm_candidate`` 之后补做（pending 阶段没有 memory_id，
    extract 侧无法链接，跳过）：拿到 candidate 的 memory_id 与其 facts 调用本接口即可。

    - ``USE_ENTITY_PORTRAIT`` 关闭：extract 层原样返回，本函数零写库 → 逐字节回退；
    - 失败 swallow：返回原 facts，不抛异常、不破坏主链路。
    """
    if not memory_id:
        return facts
    from memory import extract, store

    linked = extract.link_facts_for_memory(
        facts, memory_id=memory_id, user_id=user_id,
        entities=entities, entity_extractor=entity_extractor,
    )
    if linked is facts:
        return facts
    try:
        entry = db.get(store.MemoryEntry, str(memory_id))
        if entry is not None:
            entry.facts = linked
            db.commit()
    except Exception:  # noqa: BLE001 - 回填失败不能破坏已落库的记忆
        logger.warning("persist linked facts failed for memory_id=%s", memory_id, exc_info=True)
    return linked


def schedule_memory_write(user_id: str, message: str, *, type: str = "chat", extractor=None,
                          session_id: str | None = None) -> None:
    """提交一次后台回写,立即返回。抽取失败不影响主流程。

    extractor 为可选的自定义抽取函数 `f(client, text) -> turn dict`,
    用于按场景替换抽取逻辑(如失恋场景的 CBT 抽取);缺省走通用抽取。

    session_id 是可选的关键字参数（D-073 checkpoint 接线）：不传时 payload 与接线前
    **逐字节一致**，现有 chat 热路径（agent/*）零改动；传入后随 job payload 透传到
    后台 _do_write，用于 checkpoint 的 session 隔离。
    """
    from memory.jobs import enqueue
    payload = {"message": message, "type": type, "extractor": extractor.__module__ if extractor else None}
    if session_id is not None:
        payload["session_id"] = str(session_id)
    enqueue("memory", user_id, payload)


def schedule_trial_echo(user_id: str, message: str, *, fail_times: int = 0) -> None:
    """Huey 试点的可选入口：在当前事务内投一个 trial_echo intent。

    开关 USE_HUEY_TRIAL 关闭时是纯 no-op；现有 schedule_memory_write 链路不变。
    """
    from memory.jobs import enqueue, huey_trial_enabled
    if not huey_trial_enabled():
        return
    enqueue("trial_echo", user_id, {"message": message, "fail_times": fail_times})


def _do_write(user_id: str, message: str, type: str, extractor=None, *,
              session_id: str | None = None) -> None:
    """后台 memory job 的实际写入体（chat 热路径之外）。

    session_id 可选（D-073 checkpoint 接线）：不传则与接线前调用方式完全兼容；
    传入后在本次写库成功后由 _maybe_save_checkpoint 落一份 checkpoint 快照。
    """
    from memory import extract, portrait, reflect, store

    db = SessionLocal()
    # WP-2：memory job worker 的 scope 守卫接主链路。
    # companion scope 来自当前 job：admin_cli.run_one 从 job.companion_id 注入
    # db.info['companion_id']，SessionLocal() 返回的就是该受管会话。用 isolation
    # 的库层能力先算 scope（companion 归属 / consent / 归档只读），任一不满足即
    # ScopeDenied 拒绝本次后台写入，绝不落库——UC-4 从"库层能力"变"端到端生效"。
    # USE_SCOPE_GUARD 关闭时整段跳过，写入链路与接线前逐字节一致（安全地基可回退）。
    from memory.jobs import scope_guard_enabled
    # 只对 worker 受管会话生效：admin_cli 与 worker 测试都设 db.info['borrowed']=True。
    # 直接调用 _do_write（测试/脚本/确认流补做）无此标记，跳过守卫，避免误伤。
    _is_worker = bool(db.info.get('borrowed'))
    if scope_guard_enabled() and _is_worker:
        from memory import isolation, confirm as confirm_mod
        cid = db.info.get('companion_id')
        # 场景提取器(extractor!=None)或确认流关闭 → 本次会直接落 Fact，必须校验 scope；
        # 确认流开启的普通聊天 → 只落 Candidate(pending，不带 scope、不进业务库)，无需 scope，放行。
        will_write_fact = extractor is not None or not confirm_mod.confirmation_enabled()
        if not cid:
            if will_write_fact:
                raise isolation.ScopeDenied('SCOPE_DENIED', 'memory job 缺少 companion scope，拒绝后台写入')
        else:
            _scope = isolation.build_scope(db, user_id, cid)
            if not _scope.consent:
                raise isolation.ScopeDenied('CONSENT_REQUIRED', '用户未授权记忆使用，拒绝后台写入')
            if _scope.lifecycle == 'archived':
                raise isolation.ScopeDenied('ARCHIVED_READ_ONLY', '归档 Journey 只读，拒绝后台写入')
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
            # 待用户确认：此时无 entry/memory_id，实体链接无法做，跳过；
            # confirm 之后由路由调用 link_facts_for_entry(db, memory_id, user_id, facts) 补做。
            # 回声壁/画像仍可做（基于消息与情绪，不依赖落库 entry）。
            logger.info("memory candidate pending confirmation: %s", result.get("candidate_id"))
        else:
            # 基线路径：entry 已写入，继续下游增强。
            entry = db.get(store.MemoryEntry, result["memory_id"])
            if entry is not None:
                reflect.upsert_emotion_node(db, entry)
                # WP-1 接线：抽取后的 facts 经实体图回填 linked_memory_ids 并持久化。
                # 开关关闭 / 无 id / 任何失败都原样返回，主链路不受影响。
                try:
                    link_facts_for_entry(db, str(entry.id), user_id, turn.get("facts") or [])
                except Exception:  # noqa: BLE001 - 双保险：链接步骤永不破坏已落库记忆
                    logger.warning("link_facts step failed in _do_write", exc_info=True)
        # D-073：后台写入成功后落 checkpoint（USE_CHECKPOINT 默认关→零调用；开→懒建表+save）。
        # 放在事实落库之后、portrait 入队之前；失败 swallow，不影响已落库记忆。
        _maybe_save_checkpoint(db, user_id, message, type, turn, result,
                               session_id or db.info.get('checkpoint_session_id'))
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
