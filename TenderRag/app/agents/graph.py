import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.types import Send

from app.agents.nodes import (
    memory_retrieve,
    classify_intent,
    legal_agent,
    tender_agent,
    bidding_agent,
    product_agent,
    merge_contexts,
    synthesize,
    store_memory,
)


class AgentState(TypedDict, total=False):
    session_id: str
    question: str
    intents: Annotated[list[str], operator.add]
    history: list[dict]
    contexts: Annotated[dict[str, str], operator.or_]
    synthesize_prompts: Annotated[dict[str, str], operator.or_]
    context: str
    synthesize_prompt: str
    intent: str
    answer: str


def route_intent(state: AgentState) -> list[Send]:
    """根据多个意图返回 Send 列表，实现并行扇出。"""
    intents = state.get("intents", ["other"])
    domains = [i for i in intents if i != "other"]

    # 纯 "other" 意图：跳过所有 Agent，直接到 merge_contexts
    if not domains:
        return [Send("merge_contexts", state)]

    base = {
        "question": state["question"],
        "session_id": state.get("session_id", "default"),
        "history": state.get("history", []),
    }

    return [Send(f"{d}_agent", base) for d in domains]


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # 前处理
    graph.add_node("memory_retrieve", memory_retrieve)
    graph.add_node("classify_intent", classify_intent)

    # 领域 Agent（由 Send 并行调用）
    graph.add_node("legal_agent", legal_agent)
    graph.add_node("tender_agent", tender_agent)
    graph.add_node("bidding_agent", bidding_agent)
    graph.add_node("product_agent", product_agent)

    # 扇入 & 后处理
    graph.add_node("merge_contexts", merge_contexts)
    graph.add_node("synthesize", synthesize)
    graph.add_node("store_memory", store_memory)

    # 边
    graph.set_entry_point("memory_retrieve")
    graph.add_edge("memory_retrieve", "classify_intent")
    graph.add_conditional_edges("classify_intent", route_intent)

    for node in ("legal_agent", "tender_agent", "bidding_agent", "product_agent"):
        graph.add_edge(node, "merge_contexts")

    graph.add_edge("merge_contexts", "synthesize")
    graph.add_edge("synthesize", "store_memory")
    graph.add_edge("store_memory", END)

    return graph.compile()
