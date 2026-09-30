from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app.agents import (
    RetrieverAgent,
    SummarizerAgent,
    CriticAgent,
)
from app.ingest import ingest_pdf
from app.config import MAX_REVISIONS


class ResearchState(TypedDict, total=False):
    question: str
    retrieved_documents: list
    answer: str
    decision: str
    feedback: str
    revisions: int


# Create shared components
vector_store = ingest_pdf("paper.pdf")

retriever = RetrieverAgent(vector_store)
summarizer = SummarizerAgent()
critic = CriticAgent()


def retrieve_node(state: ResearchState):
    documents = retriever.run(state["question"])

    return {
        "retrieved_documents": documents
    }


def summarize_node(state: ResearchState):
    answer = summarizer.run(
        state["question"],
        state.get("retrieved_documents", [])
    )

    return {
        "answer": answer
    }


def critic_node(state: ResearchState):
    result = critic.run(
        state["question"],
        state.get("answer", ""),
        state.get("retrieved_documents", [])
    )

    return {
        "decision": result["decision"],
        "feedback": result["feedback"]
    }


def route_after_critic(state: ResearchState):
    if state.get("decision") == "PASS":
        return "end"

    revisions = state.get("revisions", 0)

    if revisions >= MAX_REVISIONS:
        return "end"

    return "revise"


def revise_node(state: ResearchState):
    revisions = state.get("revisions", 0)

    return {
        "revisions": revisions + 1
    }


# Build graph
builder = StateGraph(ResearchState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("summarize", summarize_node)
builder.add_node("critic", critic_node)
builder.add_node("revise", revise_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "summarize")
builder.add_edge("summarize", "critic")

builder.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "end": END,
        "revise": "revise",
    }
)

builder.add_edge("revise", "summarize")

graph = builder.compile()