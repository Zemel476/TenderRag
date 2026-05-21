from typing import TypedDict
from langgraph.graph import StateGraph, END
from app.agents.nodes import (
    memory_retrieve,
    classify_intent,
    legal_agent,
    tender_agent,
    bidding_agent,
    product_agent,
    other_node,
    synthesize,
    store_memory,
)


class AgentState(TypedDict, total=False):
    session_id: str
    question: str
    intent: str
    history: list[dict]
    context: str
    synthesize_prompt: str
    answer: str


def route_intent(state: dict) -> str:
    """路由函数：根据 intent 分发到对应节点。"""
    intent = state.get("intent", "other")
    return {
        "legal": "legal_agent",
        "tender": "tender_agent",
        "bidding": "bidding_agent",
        "product": "product_agent",
    }.get(intent, "other_node")


def build_graph() -> StateGraph:
    """构建 LangGraph StateGraph。"""
    graph = StateGraph(AgentState)

    # 前处理
    graph.add_node("memory_retrieve", memory_retrieve)
    graph.add_node("classify_intent", classify_intent)

    graph.add_node("legal_agent", legal_agent)
    graph.add_node("tender_agent", tender_agent)
    graph.add_node("bidding_agent", bidding_agent)
    graph.add_node("product_agent", product_agent)
    graph.add_node("other_node", other_node)

    # 后处理
    graph.add_node("synthesize", synthesize)
    graph.add_node("store_memory", store_memory)

    # 边：START → memory_retrieve → classify_intent → 路由
    graph.set_entry_point("memory_retrieve")
    graph.add_edge("memory_retrieve", "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "legal_agent": "legal_agent",
            "tender_agent": "tender_agent",
            "bidding_agent": "bidding_agent",
            "product_agent": "product_agent",
            "other_node": "other_node",
        },
    )

    # 所有 Agent 节点 → synthesize → memory_store → END
    for node in ("legal_agent", "tender_agent", "bidding_agent", "product_agent", "other_node"):
        graph.add_edge(node, "synthesize")

    graph.add_edge("synthesize", "store_memory")
    graph.add_edge("store_memory", END)

    return graph.compile()
