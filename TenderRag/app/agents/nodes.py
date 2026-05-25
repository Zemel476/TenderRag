import json
import queue
import threading
import time
from datetime import datetime, timezone

from sqlalchemy import select

from app.agents.prompts import DOMAIN_PROMPTS
from app.models.llm import get_llm
from app.intent.pipeline import IntentPipeline
from app.chat.session import sync_redis_client
from app.db.database import SyncSessionLocal
from app.db.models import IntentLog, Message, Session as DBSession
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


def _log_intent_sync(question: str, result):
    """Log intent classification result using sync DB (runs in graph thread)."""
    try:
        with SyncSessionLocal() as db:
            log = IntentLog(
                question=question,
                failed_level=result.level,
                scores=result.scores,
                final_intent=",".join(result.intents) if result.intents else "other",
                created_by="system",
            )
            db.add(log)
            db.commit()
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

    try:
        cache_key = f"session:{sid}:messages"
        cached = sync_redis_client.lrange(cache_key, 0, -1)
        if cached:
            history = [json.loads(m) for m in cached]
        else:
            with SyncSessionLocal() as db:
                rows = db.execute(
                    select(Message)
                    .where(Message.session_id == sid, Message.is_deleted == False)
                    .order_by(Message.created_at)
                    .limit(20)
                ).scalars().all()
                history = [
                    {"role": m.role, "content": m.content, "intents": m.intents}
                    for m in rows
                ]
                for item in history:
                    sync_redis_client.rpush(cache_key, json.dumps(item, ensure_ascii=False))
                sync_redis_client.expire(cache_key, 3600)
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


def _add_message_sync(db, session_id: int, role: str, content: str, intents: list[str] | None = None):
    """Add a message using sync DB + sync Redis (graph thread safe)."""
    msg = Message(
        session_id=session_id,
        role=role,
        content=content,
        intents=intents,
        created_by="system",
    )
    db.add(msg)
    db.commit()

    cache_key = f"session:{session_id}:messages"
    item = {
        "role": role,
        "content": content,
        "intents": intents,
    }
    sync_redis_client.rpush(cache_key, json.dumps(item, ensure_ascii=False))
    sync_redis_client.expire(cache_key, 3600)

    session = db.execute(
        select(DBSession).where(DBSession.id == session_id)
    ).scalar_one_or_none()
    if session:
        session.updated_at = datetime.now(timezone.utc)
        db.commit()


def store_memory(state: dict) -> dict:
    session_id = state.get("session_id", "default")
    intents = state.get("intents", ["other"])
    try:
        sid = int(session_id)
    except (ValueError, TypeError):
        return state

    try:
        with SyncSessionLocal() as db:
            _add_message_sync(db, sid, "user", state["question"], intents)
            if state.get("answer"):
                _add_message_sync(db, sid, "assistant", state["answer"], None)
    except Exception:
        logger.exception("store_memory failed session_id=%s", session_id)

    return state