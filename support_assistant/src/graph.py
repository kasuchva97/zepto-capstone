"""LangGraph StateGraph: classify_intent -> (retrieve_and_answer | direct_answer).

Routing (the conditional edge) never depends on MOCK_LLM -- only each
node's own generation step does. retrieve_and_answer always performs real
embedding + ChromaDB retrieval in both modes (no API key needed for that
part); only the final answer text is templated (mock) vs. LLM-generated
(MOCK_LLM=0).
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.ingest import retrieve_top_k
from src.llm import (
    classify_intent_llm,
    classify_intent_mock,
    generate_direct_answer_llm,
    generate_direct_answer_mock,
    generate_policy_answer_mock,
    generate_structured_policy_answer_llm,
    mock_llm_enabled,
)
from src.prompts import build_policy_answer_prompt


class SupportState(TypedDict):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float


def classify_intent_node(state: SupportState) -> SupportState:
    intent = classify_intent_mock(state["query"]) if mock_llm_enabled() else classify_intent_llm(state["query"])
    return {**state, "intent": intent}


def retrieve_and_answer_node(state: SupportState, collection) -> SupportState:
    hits = retrieve_top_k(collection, state["query"], k=3)
    sources = [h["id"] for h in hits]

    if mock_llm_enabled():
        top_chunk_text = hits[0]["text"] if hits else ""
        answer = generate_policy_answer_mock(top_chunk_text)
        confidence = 1.0
    else:
        context = "\n\n".join(f"[{h['id']}] {h['text']}" for h in hits)
        prompt = build_policy_answer_prompt(query=state["query"], context=context)
        result = generate_structured_policy_answer_llm(prompt)
        answer = result["answer"]
        confidence = result["confidence"]

    return {**state, "answer": answer, "sources": sources, "confidence": confidence}


def direct_answer_node(state: SupportState) -> SupportState:
    if mock_llm_enabled():
        answer = generate_direct_answer_mock()
    else:
        answer = generate_direct_answer_llm(state["query"])
    return {**state, "answer": answer, "sources": [], "confidence": 1.0}


def _route_on_intent(state: SupportState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def build_support_graph(collection):
    """collection: an already-ingested ChromaDB collection (see src/ingest.py)."""
    graph = StateGraph(SupportState)

    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("retrieve_and_answer", lambda state: retrieve_and_answer_node(state, collection))
    graph.add_node("direct_answer", direct_answer_node)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        _route_on_intent,
        {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"},
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


def run_support_graph(compiled_graph, query: str) -> SupportState:
    initial_state: SupportState = {"query": query, "intent": "", "answer": "", "sources": [], "confidence": 0.0}
    return compiled_graph.invoke(initial_state)
