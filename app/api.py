from fastapi import FastAPI
from pydantic import BaseModel

from app.graph import graph


app = FastAPI(
    title="Multi-Agent RAG Research Assistant",
    description="A local Multi-Agent RAG system using LangGraph.",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    question: str
    answer: str
    decision: str
    revisions: int
    feedback: str


@app.get("/")
def root():
    return {
        "message": "Multi-Agent RAG Research Assistant API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):

    result = graph.invoke(
        {
            "question": request.question,
            "revisions": 0
        }
    )

    return {
        "question": request.question,
        "answer": result.get("answer", ""),
        "decision": result.get("decision", ""),
        "revisions": result.get("revisions", 0),
        "feedback": result.get("feedback", "")
    }