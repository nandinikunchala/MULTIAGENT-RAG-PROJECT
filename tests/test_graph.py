from app.graph import graph


def test_rag_question():
    result = graph.invoke({
        "question": "What is RAG and why is it useful?",
        "revisions": 0
    })

    assert result["answer"]
    assert result["decision"] == "PASS"


def test_retriever_agent():
    result = graph.invoke({
        "question": "What is the role of the retriever agent?",
        "revisions": 0
    })

    assert result["answer"]
    assert "retriever" in result["answer"].lower()
    assert result["decision"] == "PASS"


def test_summarizer_agent():
    result = graph.invoke({
        "question": "What is the role of the summarizer agent?",
        "revisions": 0
    })

    assert result["answer"]
    assert "summarizer" in result["answer"].lower()
    assert result["decision"] == "PASS"


def test_critic_agent():
    result = graph.invoke({
        "question": "What is the role of the critic agent?",
        "revisions": 0
    })

    assert result["answer"]
    assert "critic" in result["answer"].lower()
    assert result["decision"] == "PASS"


def test_langgraph():
    result = graph.invoke({
        "question": "Why is LangGraph used in this project?",
        "revisions": 0
    })

    assert result["answer"]
    assert "langgraph" in result["answer"].lower()
    assert result["decision"] == "PASS"


def test_infinite_loop_prevention():
    result = graph.invoke({
        "question": "How does the system prevent an infinite loop?",
        "revisions": 0
    })

    assert result["answer"]
    assert result["revisions"] <= 2