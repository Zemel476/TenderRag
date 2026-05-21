import json
import asyncio
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.agents.graph import build_graph
from app.agents.nodes import set_stream_queue
from app.schemas.chat import ChatRequest
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()
graph = build_graph()
_executor = ThreadPoolExecutor(max_workers=4)


def sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/api/chat")
async def chat(req: ChatRequest):
    """SSE 流式聊天端点。"""
    logger.info("收到请求 session=%s len=%d", req.session_id, len(req.message))

    async def generate():
        start_time = time.time()
        yield sse_event({"type": "thinking", "content": "thinking..."})

        answer = ""
        stream_queue: queue.Queue = queue.Queue()

        def run_graph():
            set_stream_queue(stream_queue)
            try:
                return graph.invoke({
                    "session_id": req.session_id,
                    "question": req.message,
                })
            finally:
                set_stream_queue(None)

        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(_executor, run_graph)

        while True:
            try:
                msg_type, content = stream_queue.get(timeout=0.05)
                if msg_type == "message":
                    answer += content
                    yield sse_event({"type": "message", "content": content})
                elif msg_type == "done":
                    break
            except queue.Empty:
                if future.done():
                    break

        await future

        if not answer or not answer.strip():
            answer = "抱歉，暂时无法回答您的问题。"
            yield sse_event({"type": "message", "content": answer})

        yield sse_event({"type": "done"})
        logger.info("会话结束 session=%s elapsed=%.2fs", req.session_id, time.time() - start_time)

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/api/chat/non-stream")
async def chat_non_stream(req: ChatRequest):
    """非流式聊天端点"""
    logger.info("非流式请求 session=%s", req.session_id)
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_executor, lambda: graph.invoke({
                "session_id": req.session_id,
                "question": req.message,
            })),
            timeout=300,
        )
    except Exception:
        logger.exception("非流式请求异常 session=%s", req.session_id)
        return {
            "session_id": req.session_id,
            "content": "抱歉，处理请求时出现异常，请稍后再试。",
            "intent": "",
        }
    answer = result.get("answer", "").strip()
    return {
        "session_id": req.session_id,
        "content": answer or "抱歉，暂时无法回答您的问题。",
        "intent": result.get("intent", ""),
    }
