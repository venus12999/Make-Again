"""UC-2 记忆纠错 HTTP 端点（独立路由文件，由 lead 在 main.py 注册）。

不修改 api/routes.py；route_class=SecureRoute 以复用认证、归属校验与幂等回执。
资源归属在 memory.correct 内按认证用户二次校验（path 里的 memory_id 不走框架自动校验）。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.security import SecureRoute, fail
from memory import correct as correct_mod
from memory.db import get_session

router = APIRouter(prefix="/api", route_class=SecureRoute)


class CorrectionReq(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    reason: str = Field(default="", max_length=500)


@router.post("/memory/{memory_id}/correct")
def correct(memory_id: str, req: CorrectionReq, db: Session = Depends(get_session)):
    """用户说"记错了"：提交正确内容 + 理由，旧事实失效、新事实生效、变更链留痕。"""
    uid = db.info.get("user_id")
    if not uid:
        fail(401, "UNAUTHENTICATED", "请先登录")
    try:
        return correct_mod.correct_memory(db, memory_id, req.content, req.reason, user_id=uid)
    except correct_mod.CorrectionError as exc:
        fail(exc.status, exc.code, exc.message)


@router.get("/memory/{memory_id}/history")
def history(memory_id: str, db: Session = Depends(get_session)):
    """查看某条记忆的完整 supersede 链（每版正文/有效期/后继/理由/trace）。"""
    uid = db.info.get("user_id")
    if not uid:
        fail(401, "UNAUTHENTICATED", "请先登录")
    try:
        return correct_mod.memory_history(db, memory_id, user_id=uid)
    except correct_mod.CorrectionError as exc:
        fail(exc.status, exc.code, exc.message)
