from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import build_nodes
from source.rag_core.retriever import S3Retriever
from source.rag_core.generator import GeminiGenerator


def route_after_guardrails(state: AgentState) -> str:
    if state.get("is_malicious") or state.get("is_nonsense"):
        return "generator"
    return "analyzer"


def route_after_analyzer(state: AgentState) -> str:
    if state.get("is_greeting") or state.get("is_out_of_scope"):
        return "generator"
    return "retriever"


def build_graph(retriever: S3Retriever, generator: GeminiGenerator):
    nodes = build_nodes(retriever, generator)

    workflow = StateGraph(AgentState)

    workflow.add_node("guardrails", nodes["guardrails"])
    workflow.add_node("analyzer", nodes["analyzer"])
    workflow.add_node("retriever", nodes["retriever"])
    workflow.add_node("generator", nodes["generator"])

    workflow.set_entry_point("guardrails")

    workflow.add_conditional_edges(
        "guardrails",
        route_after_guardrails,
        {"generator": "generator", "analyzer": "analyzer"},
    )

    workflow.add_conditional_edges(
        "analyzer",
        route_after_analyzer,
        {"generator": "generator", "retriever": "retriever"},
    )

    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", END)

    return workflow.compile()