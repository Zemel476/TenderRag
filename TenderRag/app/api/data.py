from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.dependencies import require_role
from app.data.repository import list_records, get_record

router = APIRouter(prefix="/api/data", tags=["data"])
ADMIN = require_role("admin")


@router.get("/{category}")
async def browse_data(
    category: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    user=Depends(ADMIN),
):
    if category not in ("legal", "tender", "product"):
        raise HTTPException(400, "category 必须为 legal/tender/product")
    try:
        return await list_records(category, page, page_size)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.get("/{category}/{record_id}")
async def get_data_detail(
    category: str,
    record_id: int,
    user=Depends(ADMIN),
):
    if category not in ("legal", "tender", "product"):
        raise HTTPException(400, "category 必须为 legal/tender/product")
    record = await get_record(category, record_id)
    if not record:
        raise HTTPException(404, "记录不存在")
    return record