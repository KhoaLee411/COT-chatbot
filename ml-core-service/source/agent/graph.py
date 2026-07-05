from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import build_nodes
from source.rag_core.retriever import S3Retriever
from source.rag_core.generator import GeminiGenerator


def route_after_classify(state: AgentState) -> str:
    if (
        state.get("is_malicious")
        or state.get("is_nonsense")
        or state.get("is_greeting")
        or state.get("is_out_of_scope")
    ):
        return "generator"
    return "retriever"


def build_graph(retriever: S3Retriever, generator: GeminiGenerator):
    nodes = build_nodes(retriever, generator)

    workflow = StateGraph(AgentState)

    workflow.add_node("classify", nodes["classify"])
    workflow.add_node("retriever", nodes["retriever"])
    workflow.add_node("generator", nodes["generator"])

    workflow.set_entry_point("classify")

    workflow.add_conditional_edges(
        "classify",
        route_after_classify,
        {"generator": "generator", "retriever": "retriever"},
    )

    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", END)

    return workflow.compile()