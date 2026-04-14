"""
graph.py — LangGraph graph definition for the Q&A Chat Agent
"""

from typing import Generator
from langgraph.graph import StateGraph, END

from models import GraphState, AgentInput, AgentOutput
from nodes import (
    router_node,
    extractor_node,
    education_node,
    completion_node,
    file_extraction_node,
    document_upload_node,
    assembler_node,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL EDGE — routes after the router node
# ─────────────────────────────────────────────────────────────────────────────

def route_decision(state: GraphState) -> str:
    """Map the route flag to the next node name."""
    mapping = {
        "extract":         "extractor",
        "educate":         "educator",
        "complete":        "completion",
        "file_extraction": "file_extractor",
        "document_upload": "document_uploader",
    }
    return mapping.get(state.route, "extractor")


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    # ── nodes ─────────────────────────────────────────────────────────────────
    graph.add_node("router",         router_node)
    graph.add_node("extractor",      extractor_node)
    graph.add_node("educator",       education_node)
    graph.add_node("completion",     completion_node)
    graph.add_node("file_extractor",    file_extraction_node)
    graph.add_node("document_uploader", document_upload_node)
    graph.add_node("assembler",          assembler_node)

    # ── entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("router")

    # ── conditional routing from router ───────────────────────────────────────
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "extractor":         "extractor",
            "educator":          "educator",
            "completion":        "completion",
            "file_extractor":    "file_extractor",
            "document_uploader": "document_uploader",
        },
    )

    # ── all processing nodes → assembler → END ────────────────────────────────
    for node in ("extractor", "educator", "completion", "file_extractor", "document_uploader"):
        graph.add_edge(node, "assembler")

    graph.add_edge("assembler", END)

    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

# Singleton compiled graph (compile once, reuse across requests)
_compiled_graph = build_graph()


def run_agent(agent_input: AgentInput) -> AgentOutput:
    """
    Main entry point — accepts AgentInput, returns AgentOutput.

    Usage:
        from graph import run_agent
        from models import AgentInput, QAPair

        output = run_agent(AgentInput(
            product_profile_id="uuid-123",
            current_qa=[...],
            user_message="We plan to deploy in US East using GPT-4.",
        ))
        print(output.agent_message)
        print(output.quality_score)
    """
    initial_state = GraphState(agent_input=agent_input)
    final_state   = _compiled_graph.invoke(initial_state)

    # langgraph returns a dict when using pydantic state
    if isinstance(final_state, dict):
        final_state = GraphState(**final_state)

    return AgentOutput(
        extracted_answers=final_state.extracted_answers,
        agent_message=final_state.agent_message,
        interactive_elements=final_state.interactive_elements,
        all_questions_answered=final_state.all_questions_answered,
        quality_score=final_state.quality_score,
    )


def stream_agent(agent_input: AgentInput) -> Generator[dict, None, None]:
    """
    Streaming entry point — yields node-level progress updates as dicts.

    Each yielded dict has:
        {"node": "<node_name>", "state": <partial GraphState dict>}

    The final yield contains the complete output.
    """
    initial_state = GraphState(agent_input=agent_input)

    for event in _compiled_graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            yield {"node": node_name, "state": node_output}
