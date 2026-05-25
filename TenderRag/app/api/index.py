from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import require_role
from app.db.database import get_db
from app.db.models import User, IndexTask, IntentLog
from app.schemas.index import IndexBuildRequest, IndexTaskResponse

router = APIRouter(tags=["index"])
ADMIN = require_role("admin")


@router.post("/api/index/build")
async def trigger_build(
    body: IndexBuildRequest,
    user: User = Depends(ADMIN),
):
    from arq.connections import create_pool
    from app.task.arq_config import redis_settings

    pool = await create_pool(redis_settings)
    try:
        job = await pool.enqueue_job(
            "build_index", body.task_type, body.category, body.document_ids
        )
        return {"job_id": job.job_id}
    finally:
        await pool.close()


@router.get("/api/index/tasks")
async def list_tasks(
    page: int = 1,
    page_size: int = 20,
    user: User = Depends(ADMIN),
    db: AsyncSession = Depends(get_db),
):
    query = (
        select(IndexTask)
        .order_by(desc(IndexTask.created_at))
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    result = await db.execute(query)
    tasks = result.scalars().all()
    return [
        IndexTaskResponse(
            id=t.id,
            task_type=t.task_type,
            category=t.category,
            status=t.status or "queued",
            result_msg=t.result_msg,
            created_at=t.created_at.isoformat() if t.created_at else None,
        )
        for t in tasks
    ]


@router.get("/api/index/tasks/{task_id}")
async def get_task(
    task_id: int,
    user: User = Depends(ADMIN),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(IndexTask).where(IndexTask.id == task_id))
    t = result.scalar_one_or_none()
    if not t:
        raise HTTPException(404, "任务不存在")
    return IndexTaskResponse(
        id=t.id,
        task_type=t.task_type,
        category=t.category,
        status=t.status or "queued",
        result_msg=t.result_msg,
        created_at=t.created_at.isoformat() if t.created_at else None,
    )


@router.get("/api/intent-logs")
async def list_intent_logs(
    failed_level: str | None = None,
    page: int = 1,
    page_size: int = 50,
    user: User = Depends(ADMIN),
    db: AsyncSession = Depends(get_db),
):
    query = select(IntentLog).order_by(desc(IntentLog.created_at))
    if failed_level:
        query = query.where(IntentLog.failed_level == failed_level)
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    logs = result.scalars().all()
    return [
        {
            "id": l.id,
            "session_id": l.session_id,
            "question": l.question,
            "failed_level": l.failed_level,
            "scores": l.scores,
            "final_intent": l.final_intent,
            "created_at": l.created_at.isoformat() if l.created_at else None,
        }
        for l in logs
    ]