from langgraph.graph import StateGraph, END
from app.core.logging_config import get_logger
from app.workflows.state import PAReviewState
from app.workflows.nodes import (
    intake_node,
    retrieve_policy_node,
    summarize_case_node,
    extract_evidence_node,
    compare_criteria_node,
    recommend_decision_node,
    generate_reviewer_note_node,
    audit_log_node,
)

logger = get_logger(__name__)


def build_pa_review_graph() -> StateGraph:
    """
    Build and compile the prior authorization review workflow.

    Workflow order:
    intake → retrieve_policy → summarize_case → extract_evidence
    → compare_criteria → recommend_decision
    → generate_reviewer_note → audit_log → END

    Returns:
        Compiled LangGraph StateGraph ready to invoke.
    """
    graph = StateGraph(PAReviewState)

    graph.add_node("intake", intake_node)
    graph.add_node("retrieve_policy", retrieve_policy_node)
    graph.add_node("summarize_case", summarize_case_node)
    graph.add_node("extract_evidence", extract_evidence_node)
    graph.add_node("compare_criteria", compare_criteria_node)
    graph.add_node("recommend_decision", recommend_decision_node)
    graph.add_node("generate_reviewer_note", generate_reviewer_note_node)
    graph.add_node("audit_log", audit_log_node)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "retrieve_policy")
    graph.add_edge("retrieve_policy", "summarize_case")
    graph.add_edge("summarize_case", "extract_evidence")
    graph.add_edge("extract_evidence", "compare_criteria")
    graph.add_edge("compare_criteria", "recommend_decision")
    graph.add_edge("recommend_decision", "generate_reviewer_note")
    graph.add_edge("generate_reviewer_note", "audit_log")
    graph.add_edge("audit_log", END)

    compiled = graph.compile()
    logger.info("PA review workflow graph compiled successfully")
    return compiled


# Module-level compiled graph — import this in services
pa_review_graph = build_pa_review_graph()
