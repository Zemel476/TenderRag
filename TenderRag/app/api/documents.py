import os
import uuid
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import require_role
from app.db.database import get_db
from app.db.models import User, Document
from app.file.minio_client import upload_file, ensure_bucket
from app.schemas.document import DocumentResponse, DocumentUpdate

router = APIRouter(prefix="/api/documents", tags=["documents"])

SUPPORTED_EXTENSIONS = {".pdf": "pdf", ".md": "md", ".txt": "txt"}
ADMIN = require_role("admin")


@router.post("/upload")
async def upload_single(
    category: str = "legal",
    file: UploadFile = File(...),
    user: User = Depends(ADMIN),
    db: AsyncSession = Depends(get_db),
):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件类型: {ext}，仅支持 PDF/MD/TXT")
    file_type = SUPPORTED_EXTENSIONS[ext]

    ensure_bucket()
    today = date.today().isoformat()
    object_name = f"{category}/{today}/{uuid.uuid4().hex}{ext}"

    content = await file.read()
    file_size = len(content)

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        upload_file(tmp_path, object_name)
    finally:
        os.unlink(tmp_path)

    doc = Document(
        filename=file.filename,
        category=category,
        file_path=object_name,
        file_size=file_size,
        status="pending",
        uploaded_by=user.id,
        created_by=str(user.id),
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    # Enqueue ARQ job
    from arq.connections import create_pool
    from app.task.arq_config import redis_settings

    pool = await create_pool(redis_settings)
    try:
        await pool.enqueue_job(
            "process_document", doc.id, file.filename, file_type, category, object_name
        )
    finally:
        await pool.close()

    return {"id": doc.id, "status": doc.status, "filename": file.filename}


@router.get("")
async def list_documents(
    category: str | None = None,
    page: int = 1,
    page_size: int = 20,
    user: User = Depends(ADMIN),
    db: AsyncSession = Depends(get_db),
):
    query = select(Document).where(Document.is_deleted == False)
    if category:
        query = query.where(Document.category == category)
    query = (
        query.order_by(desc(Document.created_at))
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    result = await db.execute(query)
    docs = result.scalars().all()
    return [
        DocumentResponse(
            id=d.id,
            filename=d.filename,
            category=d.category,
            file_size=d.file_size,
            status=d.status or "pending",
            created_at=d.created_at.isoformat() if d.created_at else None,
        )
        for d in docs
    ]


@router.put("/{doc_id}")
async def update_document(
    doc_id: int,
    body: DocumentUpdate,
    user: User = Depends(ADMIN),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.is_deleted == False)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "文件不存在")
    if body.filename is not None:
        doc.filename = body.filename
    if body.category is not None:
        doc.category = body.category
    doc.updated_by = str(user.id)
    await db.commit()
    return {"ok": True}


@router.delete("/{doc_id}")
async def delete_document(
    doc_id: int,
    user: User = Depends(ADMIN),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Document).where(Document.id == doc_id, Document.is_deleted == False)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "文件不存在")
    doc.is_deleted = True
    doc.deleted_by = str(user.id)
    doc.deleted_at = datetime.now(timezone.utc)
    await db.commit()
    return {"ok": True}