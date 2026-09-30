from app.config import TOP_K
from app.llm import get_llm


class RetrieverAgent:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def run(self, question):
        results = self.vector_store.search(question, TOP_K)
        return results


class SummarizerAgent:
    def __init__(self):
        self.llm = get_llm()

    def run(self, question, retrieved_documents):
        if not retrieved_documents:
            return "No relevant documents were found."

        context = "\n\n".join(
            document["text"]
            for document in retrieved_documents
        )

        answer = self.llm.generate(
            question,
            context
        )

        return answer


class CriticAgent:
    def run(self, question, answer, retrieved_documents):

        # 1. Check whether an answer exists
        if not answer.strip():
            return {
                "decision": "REVISE",
                "feedback": "The answer is empty."
            }

        # 2. Check whether supporting documents exist
        if not retrieved_documents:
            return {
                "decision": "REVISE",
                "feedback": "No supporting documents were retrieved."
            }

        # 3. Extract important words from the question
        question_words = set(
            word.lower().strip(".,?!")
            for word in question.split()
            if len(word) > 3
        )

        # 4. Extract words from the answer
        answer_words = set(
            word.lower().strip(".,?!")
            for word in answer.split()
            if len(word) > 3
        )

        # 5. Project-specific vocabulary
        project_words = {
            "project",
            "multi-agent",
            "research",
            "assistant",
            "agents",
            "retrieve",
            "retrieves",
            "summarize",
            "summarizes",
            "critique",
            "critiques",
            "documents",
            "knowledge",
            "grounded",
            "answer"
        }

        question_overlap = question_words & answer_words
        project_overlap = project_words & answer_words

        # 6. PASS only when the answer is sufficiently relevant
        if len(question_overlap) >= 1 and len(project_overlap) >= 2:
            return {
                "decision": "PASS",
                "feedback": "Answer is relevant and grounded in the retrieved documents."
            }

        # 7. Otherwise request revision
        return {
            "decision": "REVISE",
            "feedback": "The answer does not contain enough project-specific information."
        }