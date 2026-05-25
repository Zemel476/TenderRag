import asyncio
import logging
import queue
import threading
import time
from datetime import datetime, timezone

from app.agents.prompts import DOMAIN_PROMPTS
from app.models.llm import get_llm
from app.intent.pipeline import IntentPipeline
from app.chat.session import SessionManager
from app.db.database import async_session
from app.utils.logger import get_logger

logger = get_logger(__name__)

_tls = threading.local()

DOMAIN_NAMES = {
    "legal": "法律",
    "tender": "招标",
    "bidding": "投标",
    "product": "产品",
}


def set_stream_queue(q: queue.Queue | None) -> None:
    _tls.stream_queue = q


def _stream_send(msg_type: str, content: str | None) -> None:
    q = getattr(_tls, "stream_queue", None)
    if q is not None:
        q.put((msg_type, content))


async def _log_intent_to_db(question: str, result):
    try:
        async with async_session() as db:
            from app.db.models import IntentLog
            log = IntentLog(
                question=question,
                failed_level=result.level,
                scores=result.scores,
                final_intent=",".join(result.intents) if result.intents else "other",
                created_by="system",
            )
            db.add(log)
            await db.commit()
    except Exception:
        logger.exception("Failed to log intent")


def _log_intent_sync(question: str, result):
    """Sync wrapper for intent logging — runs in graph thread without event loop."""
    try:
        asyncio.run(_log_intent_to_db(question, result))
    except Exception:
        logger.exception("Failed to log intent")


intent_pipeline = IntentPipeline(log_callback=_log_intent_sync)


def memory_retrieve(state: dict) -> dict:
    session_id = state.get("session_id", "default")
    try:
        sid = int(session_id)
    except (ValueError, TypeError):
        state["history"] = []
        return state

    async def _fetch():
        async with async_session() as db:
            sm = SessionManager(db)
            return await sm.get_messages(sid)

    try:
        history = asyncio.run(_fetch())
    except Exception:
        logger.exception("memory_retrieve failed session_id=%s", session_id)
        history = []

    state["history"] = history
    logger.debug("记忆检索完成 session=%s history_len=%d", session_id, len(history))
    return state


def classify_intent(state: dict) -> dict:
    start_time = time.time()
    question = state["question"]
    intents = intent_pipeline.classify(question)

    logger.info(
        "意图分类 intents=%s elapsed=%.2fs question=%s",
        intents, time.time() - start_time, question[:50],
    )
    return {**state, "intents": intents}


def _format_context(results: list[dict]) -> str:
    if not results:
        return "暂无相关信息。"

    parts = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        meta_str = ", ".join(
            f"{k}={v}"
            for k, v in meta.items()
            if k not in ("_node_type", "document_id", "doc_id", "ref_doc_id")
        )
        parts.append(f"[片段{i}] [{meta_str}]\n{r['content']}")

    return "\n\n---\n\n".join(parts)


def _search_domain(domain: str, question: str, top_k: int = 5) -> str:
    try:
        if domain == "legal":
            from app.rag.legal import get_nodes
        elif domain == "tender":
            from app.rag.tender import get_nodes
        elif domain == "bidding":
            from app.rag.bidding import get_nodes
        elif domain == "product":
            from app.rag.product import get_nodes
        else:
            return f"[提示] 未知领域: {domain}"

        results = get_nodes(question, top_k)
        context = _format_context(results)
        logger.info(
            "%s 检索完成 count=%d", DOMAIN_NAMES.get(domain, domain), len(results),
        )
        return context
    except Exception:
        logger.exception("%s 索引检索失败", DOMAIN_NAMES.get(domain, domain))
        return f"[提示] {DOMAIN_NAMES.get(domain, domain)} 索引检索失败，请稍后再试。"


def legal_agent(state: dict) -> dict:
    start_time = time.time()
    context = _search_domain("legal", state["question"])
    logger.info("法律检索完成 elapsed=%.2fs", time.time() - start_time)
    return {
        "contexts": {"legal": context},
        "synthesize_prompts": {"legal": DOMAIN_PROMPTS["legal"]},
    }


def tender_agent(state: dict) -> dict:
    start_time = time.time()
    context = _search_domain("tender", state["question"])
    logger.info("招标检索完成 elapsed=%.2fs", time.time() - start_time)
    return {
        "contexts": {"tender": context},
        "synthesize_prompts": {"tender": DOMAIN_PROMPTS["tender"]},
    }


def bidding_agent(state: dict) -> dict:
    start_time = time.time()
    context = _search_domain("bidding", state["question"])
    logger.info("投标检索完成 elapsed=%.2fs", time.time() - start_time)
    return {
        "contexts": {"bidding": context},
        "synthesize_prompts": {"bidding": DOMAIN_PROMPTS["bidding"]},
    }


def product_agent(state: dict) -> dict:
    start_time = time.time()
    context = _search_domain("product", state["question"])
    logger.info("商品检索完成 elapsed=%.2fs", time.time() - start_time)
    return {
        "contexts": {"product": context},
        "synthesize_prompts": {"product": DOMAIN_PROMPTS["product"]},
    }


def merge_contexts(state: dict) -> dict:
    intents = state.get("intents", ["other"])
    contexts = state.get("contexts", {})
    prompts = state.get("synthesize_prompts", {})

    if intents == ["other"] or not contexts:
        return {
            "context": "",
            "synthesize_prompt": DOMAIN_PROMPTS["other"],
            "intent": "other",
        }

    parts = []
    for domain in intents:
        ctx = contexts.get(domain, "")
        if not ctx or ctx == "暂无相关信息。" or ctx.startswith("[提示]"):
            continue
        label = DOMAIN_NAMES.get(domain, domain)
        parts.append(f"【{label}领域】\n{ctx}")

    if parts:
        combined = "\n\n---\n\n".join(parts)
    else:
        merged = [c for c in contexts.values() if c.strip()]
        combined = merged[0] if merged else "暂无相关信息。"

    if len(intents) == 1:
        prompt = prompts.get(intents[0], DOMAIN_PROMPTS["other"])
    else:
        prompt = DOMAIN_PROMPTS["multi"]

    logger.info(
        "上下文合并完成 intents=%s domains=%d context_len=%d",
        intents, len(contexts), len(combined),
    )
    return {
        "context": combined,
        "synthesize_prompt": prompt,
        "intent": ",".join(intents),
    }


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
    logger.info(
        "合成完成 intent=%s answer_len=%d elapsed=%.2fs",
        intent, len(answer), time.time() - start_time,
    )
    return state


def store_memory(state: dict) -> dict:
    session_id = state.get("session_id", "default")
    intents = state.get("intents", ["other"])
    try:
        sid = int(session_id)
    except (ValueError, TypeError):
        return state

    async def _store():
        async with async_session() as db:
            sm = SessionManager(db)
            await sm.add_message(sid, "user", state["question"], intents)
            if state.get("answer"):
                await sm.add_message(sid, "assistant", state["answer"], None)

    try:
        asyncio.run(_store())
    except Exception:
        logger.exception("store_memory failed session_id=%s", session_id)

    return state