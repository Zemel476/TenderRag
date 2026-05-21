import queue
import threading
import time
import uuid
from copy import deepcopy

import gradio as gr

from app.agents.graph import build_graph
from app.agents.nodes import set_stream_queue
from app.memory.store import memory_store
from app.utils.logger import get_logger

logger = get_logger(__name__)

graph = build_graph()


def _make_session(title: str | None = None) -> dict:
    session_id = uuid.uuid4().hex
    return {
        "id": session_id,
        "title": title or "新对话",
        "created_at": time.time(),
    }


def _ensure_conversations(conversations: dict | None) -> dict:
    if conversations:
        return conversations

    session = _make_session("新对话")
    return {session["id"]: session}


def _session_choices(conversations: dict) -> list[tuple[str, str]]:
    ordered = sorted(
        conversations.values(),
        key=lambda item: item.get("created_at", 0),
        reverse=True,
    )
    return [(session["title"], session["id"]) for session in ordered]


def _get_session(conversations: dict, session_id: str | None) -> tuple[dict, str]:
    conversations = _ensure_conversations(conversations)
    if session_id not in conversations:
        session_id = next(iter(conversations))
    return conversations[session_id], session_id


def _title_from_message(message: str) -> str:
    title = message.strip().replace("\n", " ")
    return title[:4] + ("..." if len(title) > 4 else "")


def _display_history(session_id: str) -> list[dict]:
    messages = memory_store.get_messages(session_id)
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def init_state():
    conversations = _ensure_conversations(None)
    session_id = next(iter(conversations))
    return (
        conversations,
        gr.update(choices=_session_choices(conversations), value=session_id),
        _display_history(session_id),
    )


def new_chat(conversations: dict):
    conversations = deepcopy(_ensure_conversations(conversations))
    session = _make_session(f"新对话 {len(conversations) + 1}")
    conversations[session["id"]] = session
    logger.info("新建对话 session=%s total=%d", session["id"], len(conversations))
    return (
        conversations,
        gr.update(choices=_session_choices(conversations), value=session["id"]),
        [],
        "",
    )


def switch_chat(session_id: str, conversations: dict):
    conversations = _ensure_conversations(conversations)
    _get_session(conversations, session_id)
    logger.debug("切换对话 session=%s", session_id)
    return _display_history(session_id), ""


def clear_current_chat(session_id: str, conversations: dict):
    conversations = deepcopy(_ensure_conversations(conversations))
    session, session_id = _get_session(conversations, session_id)
    session["title"] = "新对话"
    memory_store.clear(session_id)
    logger.info("清空对话 session=%s", session_id)
    return (
        conversations,
        gr.update(choices=_session_choices(conversations), value=session_id),
        [],
        "",
    )


def delete_chat(session_id: str, conversations: dict):
    conversations = deepcopy(_ensure_conversations(conversations))
    if len(conversations) <= 1:
        gr.Warning("至少保留一个对话。")
        return (
            conversations,
            gr.update(choices=_session_choices(conversations), value=session_id),
            _display_history(session_id),
            "",
        )
    memory_store.clear(session_id)
    del conversations[session_id]
    new_id = next(iter(conversations))
    logger.info("删除对话 old=%s new=%s remaining=%d", session_id, new_id, len(conversations))
    return (
        conversations,
        gr.update(choices=_session_choices(conversations), value=new_id),
        _display_history(new_id),
        "",
    )


def submit_message(message: str, session_id: str, conversations: dict):
    conversations = deepcopy(_ensure_conversations(conversations))
    session, session_id = _get_session(conversations, session_id)
    cleaned = (message or "").strip()

    if not cleaned:
        gr.Warning("请输入问题后再发送。")
        yield (
            conversations,
            gr.update(choices=_session_choices(conversations), value=session_id),
            _display_history(session_id),
            "",
        )
        return

    if not memory_store.get_messages(session_id) or session["title"].startswith("新对话"):
        session["title"] = _title_from_message(cleaned)

    logger.info("用户提问 session=%s question=%s", session_id, cleaned[:80])

    memory_store.add_message(session_id, "user", cleaned)

    display = _display_history(session_id)
    display.append({"role": "assistant", "content": "thinking..."})
    yield (
        conversations,
        gr.update(choices=_session_choices(conversations), value=session_id),
        display,
        "",
    )

    stream_queue: queue.Queue = queue.Queue()
    result_holder: dict = {}

    def run_graph():
        set_stream_queue(stream_queue)
        try:
            result_holder["result"] = graph.invoke({
                "session_id": session_id,
                "question": cleaned,
            })
        except Exception as exc:
            logger.exception("graph 执行异常 session=%s", session_id)
            result_holder["error"] = exc
        finally:
            set_stream_queue(None)
            stream_queue.put(("done", None))

    worker = threading.Thread(target=run_graph, daemon=True)
    worker.start()

    answer = ""
    while True:
        try:
            msg_type, content = stream_queue.get(timeout=0.1)
        except queue.Empty:
            if not worker.is_alive():
                break
            continue

        if msg_type == "message" and content:
            answer += content
            display[-1]["content"] = answer
            yield (
                conversations,
                gr.update(choices=_session_choices(conversations), value=session_id),
                display,
                "",
            )
        elif msg_type == "done":
            break

    worker.join(timeout=1)

    if result_holder.get("error"):
        error_msg = f"抱歉，处理请求时出错：{result_holder['error']}"
        display[-1]["content"] = error_msg
    elif not answer.strip():
        result = result_holder.get("result") or {}
        fallback = result.get("answer", "").strip() or "抱歉，暂时无法回答您的问题。"
        display[-1]["content"] = fallback

    logger.info("回答完成 session=%s answer_len=%d", session_id, len(answer))

    yield (
        conversations,
        gr.update(choices=_session_choices(conversations), value=session_id),
        display,
        "",
    )


with gr.Blocks(title="TenderRag", fill_height=True) as demo:
    conversations_state = gr.State()

    gr.Markdown("# TenderRag")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=200):
            with gr.Row():
                new_chat_btn = gr.Button("新建", variant="primary", size="md")
                clear_chat_btn = gr.Button("清空", size="md")
                delete_chat_btn = gr.Button("删除", variant="stop", size="md")
            session_radio = gr.Radio(
                label="对话列表",
                choices=[],
                interactive=True,
                scale=5,
            )

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="聊天",
                height=380,
                buttons=["copy"],
            )
            message_box = gr.Textbox(
                label="输入问题",
                placeholder="请输入招标、投标、商品或法律相关问题...",
                lines=3,
                max_lines=8,
            )
            send_btn = gr.Button("发送", variant="primary")

    demo.load(
        init_state,
        outputs=[conversations_state, session_radio, chatbot],
    )
    new_chat_btn.click(
        new_chat,
        inputs=[conversations_state],
        outputs=[conversations_state, session_radio, chatbot, message_box],
    )
    clear_chat_btn.click(
        clear_current_chat,
        inputs=[session_radio, conversations_state],
        outputs=[conversations_state, session_radio, chatbot, message_box],
    )
    delete_chat_btn.click(
        delete_chat,
        inputs=[session_radio, conversations_state],
        outputs=[conversations_state, session_radio, chatbot, message_box],
    )
    session_radio.change(
        switch_chat,
        inputs=[session_radio, conversations_state],
        outputs=[chatbot, message_box],
    )
    submit_inputs = [message_box, session_radio, conversations_state]
    submit_outputs = [conversations_state, session_radio, chatbot, message_box]
    send_btn.click(
        submit_message,
        inputs=submit_inputs,
        outputs=submit_outputs,
        concurrency_limit=1,
    )
    message_box.submit(
        submit_message,
        inputs=submit_inputs,
        outputs=submit_outputs,
        concurrency_limit=1,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
