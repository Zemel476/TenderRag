import logging
import queue
import threading
import time

from app.agents.prompts import INTENT_CLASSIFY_PROMPT, DOMAIN_PROMPTS
from app.models.llm import get_llm
from app.memory.store import memory_store as _store
from app.utils.logger import get_logger

logger = get_logger(__name__)

_tls = threading.local()


def set_stream_queue(q: queue.Queue | None) -> None:
    _tls.stream_queue = q


def _stream_send(msg_type: str, content: str | None) -> None:
    q = getattr(_tls, "stream_queue", None)
    if q is not None:
        q.put((msg_type, content))


def memory_retrieve(state: dict) -> dict:
    session_id = state.get("session_id", "default")
    messages = _store.get_messages(session_id)
    state["history"] = messages
    logger.debug("记忆检索完成 session=%s history_len=%d", session_id, len(messages))
    return state


def classify_intent(state: dict) -> dict:
    start_time = time.time()
    question = state["question"]
    llm = get_llm()
    prompt = INTENT_CLASSIFY_PROMPT.format(question=question)
    response = llm.complete(prompt)
    intent = str(response).strip().lower()

    for domain in ("legal", "tender", "bidding", "product", "other"):
        if domain in intent:
            state["intent"] = domain
            break
    else:
        state["intent"] = "other"

    logger.info("意图分类 intent=%s elapsed=%.2fs question=%s", state["intent"], time.time() - start_time, question[:50])
    return state


def _format_context(results: list[dict]) -> str:
    if not results:
        return "暂无相关信息。"

    parts = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        meta_str = ", ".join(f"{k}={v}" for k, v in meta.items() if k not in ("_node_type", "document_id", "doc_id", "ref_doc_id"))
        parts.append(f"[片段{i}] [{meta_str}]\n{r['content']}")

    return "\n\n---\n\n".join(parts)


# ============================================================
# Legal Agent
# ============================================================
def legal_agent(state: dict) -> dict:
    start_time = time.time()
    state["intent"] = "legal"
    try:
        from app.rag.legal import get_nodes
        results = get_nodes(state["question"], 5)
        state["context"] = _format_context(results)
        logger.info("法律检索完成 count=%d elapsed=%.2fs", len(results), time.time() - start_time)
    except Exception:
        logger.exception("法律索引检索失败")
        state["context"] = "[提示] 法律索引检索失败，请稍后再试。"

    state["synthesize_prompt"] = DOMAIN_PROMPTS["legal"]
    return state


# ============================================================
# Tender Agent
# ============================================================
def tender_agent(state: dict) -> dict:
    start_time = time.time()
    state["intent"] = "tender"
    try:
        from app.rag.tender import get_nodes
        results = get_nodes(state["question"])
        state["context"] = _format_context(results)
        logger.info("招标检索完成 count=%d elapsed=%.2fs", len(results), time.time() - start_time)
    except Exception:
        logger.exception("招标索引检索失败")
        state["context"] = "[提示] 招标索引检索失败，请稍后再试。"

    state["synthesize_prompt"] = DOMAIN_PROMPTS["tender"]
    return state


# ============================================================
# Bidding Agent
# ============================================================
def bidding_agent(state: dict) -> dict:
    state["intent"] = "bidding"
    try:
        from app.rag.bidding import get_nodes
        results = get_nodes(state["question"])
        state["context"] = _format_context(results)
        logger.info("投标检索完成 count=%d", len(results))
    except Exception:
        logger.exception("投标索引检索失败")
        state["context"] = "[提示] 投标索引检索失败，请稍后再试。"

    state["synthesize_prompt"] = DOMAIN_PROMPTS["bidding"]
    return state


# ============================================================
# Product Agent
# ============================================================
def product_agent(state: dict) -> dict:
    start_time = time.time()
    state["intent"] = "product"
    try:
        from app.rag.product import get_nodes
        results = get_nodes(state["question"])
        state["context"] = _format_context(results)
        logger.info("商品检索完成 count=%d elapsed=%.2fs", len(results), time.time() - start_time)
    except Exception:
        logger.exception("商品索引检索失败")
        state["context"] = "[提示] 商品索引检索失败，请稍后再试。"

    state["synthesize_prompt"] = DOMAIN_PROMPTS["product"]
    return state


# ============================================================
# Other / Synthesize / Memory
# ============================================================
def other_node(state: dict) -> dict:
    state["context"] = ""
    state["intent"] = "other"
    state["synthesize_prompt"] = DOMAIN_PROMPTS["other"]
    logger.info("意图为 other，跳过检索")
    return state


def synthesize(state: dict):
    start_time = time.time()
    intent = state.get("intent", "other")

    history = state.get("history", [])
    history_lines = []
    for msg in history:
        history_lines.append(f"{msg['role']}: {msg['content']}")
    history_text = "\n".join(history_lines)

    prompt_template = state.get("synthesize_prompt", DOMAIN_PROMPTS["other"])
    prompt = prompt_template.format(
        context=state.get("context", "暂无背景信息。"),
        history=history_text,
        question=state["question"],
    )

    llm = get_llm()
    answer = ""
    for resp in llm.stream_complete(prompt):
        if resp.delta:
            answer += resp.delta
            _stream_send("message", resp.delta)

    state["answer"] = answer
    _stream_send("done", None)
    logger.info("合成完成 intent=%s answer_len=%d elapsed=%.2fs", intent, len(answer), time.time() - start_time)
    return state


def store_memory(state: dict) -> dict:
    session_id = state.get("session_id", "default")
    _store.add_message(session_id, "user", state["question"])
    if "answer" in state and state["answer"]:
        _store.add_message(session_id, "assistant", state["answer"])
    logger.debug("记忆存储完成 session=%s", session_id)
    return state
