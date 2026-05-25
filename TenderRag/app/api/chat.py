import asyncio
import json
import queue
import threading
from asyncio import get_running_loop

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import build_graph
from app.agents.nodes import set_stream_queue
from app.auth.dependencies import get_current_user
from app.chat.session import SessionManager
from app.db.database import get_db
from app.db.models import User
from app.schemas.chat import ChatRequest
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["chat"])
graph = build_graph()


def sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/api/chat")
async def chat(
    req: ChatRequest,
    user: User = Depends(get_current_user),
):
    """SSE streaming chat endpoint."""
    logger.info(
        "Chat request session=%s user=%s len=%d",
        req.session_id, user.username, len(req.message),
    )

    async def generate():
        stream_queue: queue.Queue = queue.Queue()
        loop = get_running_loop()

        def run_graph():
            set_stream_queue(stream_queue)
            try:
                graph.invoke({
                    "session_id": req.session_id,
                    "question": req.message,
                })
            except Exception:
                logger.exception("Graph invoke failed session=%s", req.session_id)
            finally:
                set_stream_queue(None)
                stream_queue.put(("done", None))

        thread = threading.Thread(target=run_graph, daemon=True)
        thread.start()

        while True:
            try:
                msg_type, content = await loop.run_in_executor(
                    None, lambda: stream_queue.get(timeout=0.1),
                )
                if msg_type == "message":
                    yield sse_event({"type": "message", "content": content})
                elif msg_type == "done":
                    break
            except queue.Empty:
                if not thread.is_alive():
                    break
                continue

        thread.join(timeout=1)
        yield sse_event({"type": "done"})
        logger.info("Chat done session=%s user=%s", req.session_id, user.username)

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/api/chat/non-stream")
async def chat_non_stream(
    req: ChatRequest,
    user: User = Depends(get_current_user),
):
    """Non-streaming chat endpoint."""
    logger.info("Non-stream chat request session=%s user=%s", req.session_id, user.username)
    loop = get_running_loop()

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: graph.invoke({
                "session_id": req.session_id,
                "question": req.message,
            })),
            timeout=300,
        )
    except Exception:
        logger.exception("Non-stream chat failed session=%s", req.session_id)
        return {
            "session_id": req.session_id,
            "content": "抱歉，处理请求时出现异常，请稍后再试。",
            "intent": "",
        }

    answer = result.get("answer", "").strip() if result else ""
    return {
        "session_id": req.session_id,
        "content": answer or "抱歉，暂时无法回答您的问题。",
        "intent": result.get("intent", "") if result else "",
    }


class CreateSessionRequest(BaseModel):
    title: str = "新对话"


@router.get("/api/sessions")
async def list_sessions(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List sessions for current user."""
    sm = SessionManager(db)
    return await sm.list_sessions(user.id)


@router.post("/api/sessions")
async def create_session(
    body: CreateSessionRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new session."""
    sm = SessionManager(db)
    session = await sm.create_session(user.id, title=body.title)
    return {"session_id": session.id, "title": session.title}


@router.get("/api/sessions/{session_id}/messages")
async def get_messages(
    session_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get messages for a session."""
    sm = SessionManager(db)
    return await sm.get_messages(session_id)


@router.delete("/api/sessions/{session_id}")
async def delete_session(
    session_id: int,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Soft-delete a session."""
    sm = SessionManager(db)
    await sm.soft_delete_session(session_id, deleted_by=str(user.id))
    return {"ok": True}