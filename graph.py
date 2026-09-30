from typing import TypedDict

from langgraph.graph import StateGraph, END

from agents import RetrieverAgent, SummarizerAgent, CriticAgent


# ============================================================
# 1. State
# ============================================================

class RAGState(TypedDict):
    question: str
    documents: list
    answer: str
    verdict: str
    feedback: str
    revision_count: int


# ============================================================
# 2. Build LangGraph
# ============================================================

def build_graph(model, index, chunks):

    # Create agents
    retriever = RetrieverAgent(
        model=model,
        index=index,
        chunks=chunks,
        top_k=3
    )

    summarizer = SummarizerAgent()
    critic = CriticAgent()

    # ========================================================
    # Retriever Node
    # ========================================================

    def retrieve_node(state):

        print("\n" + "=" * 60)
        print("[Retriever Agent]")
        print("=" * 60)

        print("Searching the document...")

        documents = retriever.retrieve(
            state["question"]
        )

        print(f"Retrieved {len(documents)} relevant documents.")

        return {
            "documents": documents
        }

    # ========================================================
    # Summarizer Node
    # ========================================================

    def summarize_node(state):

        print("\n" + "=" * 60)
        print("[Summarizer Agent]")
        print("=" * 60)

        print("Creating draft answer...")

        answer = summarizer.summarize(
            state["question"],
            state["documents"]
        )

        print("\nDraft Answer:")
        print("-" * 60)
        print(answer)

        return {
            "answer": answer
        }

    # ========================================================
    # Critic Node
    # ========================================================

    def critic_node(state):

        print("\n" + "=" * 60)
        print("[Critic Agent]")
        print("=" * 60)

        print("Checking the draft answer...")

        result = critic.critique(
            state["question"],
            state["answer"],
            state["documents"]
        )

        revision_count = state.get("revision_count", 0)

        if result["verdict"] == "revise":
            revision_count += 1

        print("\nCritic Verdict:", result["verdict"])
        print("Critic Feedback:", result["feedback"])
        print("Revision Count:", revision_count)

        return {
            "verdict": result["verdict"],
            "feedback": result["feedback"],
            "revision_count": revision_count
        }

    # ========================================================
    # Decide what happens after Critic
    # ========================================================

    def critic_router(state):

        # Prevent infinite loops
        if state["revision_count"] >= 2:
            print("\nMaximum revisions reached.")
            return "pass"

        if state["verdict"] == "pass":
            return "pass"

        return "revise"

    # ========================================================
    # Create StateGraph
    # ========================================================

    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node(
        "retriever",
        retrieve_node
    )

    workflow.add_node(
        "summarizer",
        summarize_node
    )

    workflow.add_node(
        "critic",
        critic_node
    )

    # ========================================================
    # Starting point
    # ========================================================

    workflow.set_entry_point("retriever")

    # ========================================================
    # Normal flow
    # ========================================================

    workflow.add_edge(
        "retriever",
        "summarizer"
    )

    workflow.add_edge(
        "summarizer",
        "critic"
    )

    # ========================================================
    # Critic decision
    # ========================================================

    workflow.add_conditional_edges(
        "critic",
        critic_router,
        {
            "pass": END,
            "revise": "summarizer"
        }
    )

    # ========================================================
    # Compile graph
    # ========================================================

    return workflow.compile()