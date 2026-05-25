import os
import logging
import tempfile
from datetime import datetime, timezone

import pymysql
from llama_index.core.schema import Document as LlamaDocument
from sqlalchemy import select

from app.config import settings
from app.db.database import async_session
from app.db.models import Document, IndexTask
from app.file.minio_client import download_file
from app.file.parser import parse_file
from app.file.chunker import chunk_text
from app.rag.indexing import MilvusIndexWriter

logger = logging.getLogger(__name__)


async def process_document(ctx, doc_id: int, filename: str, file_type: str, category: str, minio_key: str):
    """ARQ job: parse uploaded file → chunk → embed → Milvus."""
    # Mark as processing
    async with async_session() as db:
        result = await db.execute(select(Document).where(Document.id == doc_id))
        doc = result.scalar_one_or_none()
        if doc:
            doc.status = "processing"
            doc.updated_at = datetime.now(timezone.utc)
            await db.commit()

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, filename)
            download_file(minio_key, local_path)

            text = parse_file(local_path, file_type)
            chunks = chunk_text(text, file_type, metadata={
                "source": filename,
                "category": category,
                "doc_id": str(doc_id),
            })

            if chunks:
                docs = [LlamaDocument(text=c["content"], metadata=c["metadata"]) for c in chunks]
                writer = MilvusIndexWriter(collection_name=f"ml_{category}")
                writer.write_documents(docs)

        # Mark as done
        async with async_session() as db:
            result = await db.execute(select(Document).where(Document.id == doc_id))
            doc = result.scalar_one_or_none()
            if doc:
                doc.status = "done"
                doc.updated_at = datetime.now(timezone.utc)
                await db.commit()

        logger.info("Document processed doc_id=%s chunks=%d", doc_id, len(chunks))
        return {"status": "done", "chunks": len(chunks)}

    except Exception as e:
        async with async_session() as db:
            result = await db.execute(select(Document).where(Document.id == doc_id))
            doc = result.scalar_one_or_none()
            if doc:
                doc.status = "failed"
                doc.updated_at = datetime.now(timezone.utc)
                await db.commit()
        logger.exception("Document processing failed doc_id=%s", doc_id)
        return {"status": "failed", "error": str(e)}


async def build_index(ctx, task_type: str, category: str, document_ids: list[int] | None = None):
    """ARQ job: vectorize structured data from MySQL business tables."""
    # Create task record
    async with async_session() as db:
        task = IndexTask(
            task_type=task_type,
            category=category,
            document_ids=document_ids,
            status="running",
            created_by="system",
        )
        db.add(task)
        await db.commit()
        await db.refresh(task)
        task_id = task.id

    try:
        conn = pymysql.connect(
            host=settings.database_url,
            user=settings.database_user,
            password=settings.database_password,
            database=settings.database_db_name,
            charset="utf8mb4",
        )
        table_map = {"legal": "法律法规", "tender": "政府招标信息", "product": "市场产品信息"}
        table = table_map.get(category)
        if not table:
            raise ValueError(f"Unknown category: {category}")

        cursor = conn.cursor()
        if document_ids:
            placeholders = ",".join(["%s"] * len(document_ids))
            cursor.execute(f"SELECT * FROM `{table}` WHERE id IN ({placeholders})", document_ids)
        else:
            cursor.execute(f"SELECT * FROM `{table}`")
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()

        # Build all chunks first, then write once
        all_chunks = []
        for row in rows:
            data = dict(zip(columns, [str(v) if v else "" for v in row]))
            content = " ".join(data.values())
            if len(content) < 10:
                continue
            chunks = chunk_text(content, "txt", metadata={"source": table, "category": category, **data})
            all_chunks.extend(chunks)

        processed = len(all_chunks)
        if all_chunks:
            docs = [LlamaDocument(text=c["content"], metadata=c["metadata"]) for c in all_chunks]
            writer = MilvusIndexWriter(collection_name=f"ml_{category}")
            writer.write_documents(docs)
            processed = len(all_chunks)

        # Mark done
        async with async_session() as db:
            result = await db.execute(select(IndexTask).where(IndexTask.id == task_id))
            t = result.scalar_one_or_none()
            if t:
                t.status = "done"
                t.result_msg = f"成功处理 {processed} 条记录"
                await db.commit()

        return {"status": "done", "processed": processed}

    except Exception as e:
        async with async_session() as db:
            result = await db.execute(select(IndexTask).where(IndexTask.id == task_id))
            t = result.scalar_one_or_none()
            if t:
                t.status = "failed"
                t.result_msg = str(e)
                await db.commit()
        logger.exception("Index build failed")
        return {"status": "failed", "error": str(e)}