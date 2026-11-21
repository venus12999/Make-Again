"""UC-1 记忆确认流路由（独立文件，由 lead 在 api/routes.py / main.py 集成注册）。

USE_MEMORY_CONFIRM 默认开：抽取先落 Candidate(pending)，用户逐条 confirm/reject/edit。
端点：
  POST /api/memory/candidates                 抽取落候选（开关关时逐字节走基线 add_memory）
  GET  /api/memory/candidates                 列待确认（?status= 可看全部）
  POST /api/memory/candidates/{id}/confirm    确认成 Fact
  POST /api/memory/candidates/{id}/reject     拒绝（永不召回）
  POST /api/memory/candidates/{id}/edit       纠正候选（已确认的走 supersede 旧事实失效）

owner 校验双层：SecureRoute 拒 body/path 里的 user_id 不一致；记忆层再按 owner 过滤候选。
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.security import SecureRoute, fail
from memory import confirm as _confirm
from memory.db import get_session

router = APIRouter(prefix="/api/memory", route_class=SecureRoute)


def _uid(db) -> str:
    return str(db.info["user_id"])


class CandidateIn(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    type: str = Field(default="chat", max_length=16)
    user_id: str | None = None  # SecureRoute 校验：非本人直接 403
    request_id: str | None = None


class DecisionIn(BaseModel):
    user_id: str | None = None
    reason: str | None = Field(default=None, max_length=500)
    request_id: str | None = None


class EditIn(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    user_id: str | None = None
    reason: str | None = Field(default=None, max_length=500)
    request_id: str | None = None


def _guard(error: Exception):
    if isinstance(error, _confirm.PermissionDenied):
        fail(404, "NOT_FOUND", "记忆候选不存在")
    if isinstance(error, _confirm.CandidateError):
        fail(409, "CANDIDATE_STATE", str(error))
    raise error


@router.post("/candidates")
def candidate_create(req: CandidateIn, db=Depends(get_session)):
    result = _confirm.submit_candidate(
        db, _uid(db), type=req.type, content=req.content,
    )
    return {"ok": True, **result}


@router.get("/candidates")
def candidate_list(status: str | None = None, limit: int = 50, db=Depends(get_session)):
    if status is not None and status not in _confirm._VALID_STATUSES:
        fail(422, "INVALID_STATUS", "无效的审核状态")
    candidates = _confirm.list_candidates(_uid(db), status=status, limit=limit)
    return {
        "ok": True,
        "candidates": candidates,
        "count": len(candidates),
        "confirmation_enabled": _confirm.confirmation_enabled(),
    }


@router.post("/candidates/{candidate_id}/confirm")
def candidate_confirm(candidate_id: str, req: DecisionIn, db=Depends(get_session)):
    try:
        result = _confirm.confirm_candidate(db, candidate_id, _uid(db))
    except (_confirm.PermissionDenied, _confirm.CandidateError) as exc:
        _guard(exc)
    return {"ok": True, "candidate": result, "memory_id": result.get("memory_id")}


@router.post("/candidates/{candidate_id}/reject")
def candidate_reject(candidate_id: str, req: DecisionIn, db=Depends(get_session)):
    try:
        result = _confirm.reject_candidate(candidate_id, _uid(db), reason=req.reason or "")
    except (_confirm.PermissionDenied, _confirm.CandidateError) as exc:
        _guard(exc)
    return {"ok": True, "candidate": result}


@router.post("/candidates/{candidate_id}/edit")
def candidate_edit(candidate_id: str, req: EditIn, db=Depends(get_session)):
    try:
        result = _confirm.edit_candidate(
            db, candidate_id, _uid(db), content=req.content, reason=req.reason or "",
        )
    except (_confirm.PermissionDenied, _confirm.CandidateError) as exc:
        _guard(exc)
    return {"ok": True, "candidate": result, "superseded_by": result.get("successor_id")}
