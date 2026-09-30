from app.graph import graph


def run_question(question):
    result = graph.invoke({
        "question": question,
        "revisions": 0
    })

    return result


def test_rag_question():
    result = run_question(
        "What is RAG and why is it useful?"
    )

    assert result["answer"]
    assert result["decision"] == "PASS"


def test_retriever_agent():
    result = run_question(
        "What is the role of the retriever agent?"
    )

    answer = result["answer"].lower()

    assert "retriever" in answer
    assert "document" in answer
    assert result["decision"] == "PASS"


def test_summarizer_agent():
    result = run_question(
        "What is the role of the summarizer agent?"
    )

    answer = result["answer"].lower()

    assert "summarizer" in answer
    assert "answer" in answer
    assert result["decision"] == "PASS"


def test_critic_agent():
    result = run_question(
        "What is the role of the critic agent?"
    )

    answer = result["answer"].lower()

    assert "critic" in answer
    assert (
        "revision" in answer
        or "revise" in answer
        or "check" in answer
    )

    assert result["decision"] == "PASS"


def test_langgraph():
    result = run_question(
        "Why is LangGraph used in this project?"
    )

    answer = result["answer"].lower()

    assert "langgraph" in answer
    assert (
        "state" in answer
        or "routing" in answer
        or "loop" in answer
    )

    assert result["decision"] == "PASS"


def test_revision_budget():
    result = run_question(
        "How does the system prevent an infinite loop?"
    )

    revisions = result.get("revisions", 0)

    assert revisions <= 2