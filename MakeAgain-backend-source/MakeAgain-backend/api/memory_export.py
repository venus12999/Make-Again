"""UC-3 · 记忆删除与导出路由（独立文件，不碰 api/routes.py）。

由 lead 在 main.py 里 include_router 注册：
    from api.memory_export import router as memory_export_router
    app.include_router(memory_export_router)

所有端点走 SecureRoute：鉴权、owner 校验、幂等回执、审计信封均由统一边界提供；
本模块只负责「user_id = 当前会话用户」的强绑定与业务错误映射。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.security import SecureRoute, fail
from memory import export as memory_export
from memory.db import get_session

router = APIRouter(prefix="/api", route_class=SecureRoute)


@router.get("/memory/export")
def memory_export_all(db=Depends(get_session)):
    """GET /api/memory/export —— 导出本人全部记忆（含 provenance）。"""
    return memory_export.export_user_memory(db, db.info["user_id"])


@router.delete("/memory/all")
def memory_delete_all(db=Depends(get_session)):
    """DELETE /api/memory/all —— 全量删除本人记忆，返回类别/数量执行回执。"""
    user_id = db.info["user_id"]
    return memory_export.delete_all_memories(db, user_id, actor_id=user_id)


@router.delete("/memory/{memory_id}")
def memory_delete_one(memory_id: str, db=Depends(get_session)):
    """DELETE /api/memory/{id} —— 级联删除单条；越权/不存在一律 404（不泄露他人数据是否存在）。"""
    user_id = db.info["user_id"]
    result = memory_export.delete_memory(db, user_id, memory_id, actor_id=user_id)
    if not result.get("deleted"):
        fail(404, "NOT_FOUND", "记录不存在")
    return result
