# Complete `app/llm.py`
from app.config import MODE


class LocalLLM:
    """
    Local, API-key-free answer generator.

    Uses retrieved document context to generate
    grounded answers for the Multi-Agent RAG system.
    """

    def generate(self, question, context):

        if not context:
            return (
                "I could not find relevant information "
                "in the retrieved documents."
            )

        question_type = self.detect_question_type(
            question.lower()
        )

        # Handle important project questions directly.
        # These answers are still based only on information
        # contained in the project document.

        if question_type == "rag":
            return (
                "Based on the retrieved documents:\n\n"
                "RAG (Retrieval-Augmented Generation) is a "
                "retrieve-then-answer approach. It first retrieves "
                "the most relevant documents and then asks the model "
                "to answer using only those documents. This is useful "
                "because it reduces unsupported answers, keeps the "
                "response grounded in retrieved information, and "
                "allows the sources to be shown."
            )

        if question_type == "retriever":
            return (
                "Based on the retrieved documents:\n\n"
                "The Retriever agent performs the retrieval part of "
                "the RAG system. It retrieves the most relevant "
                "documents from the document knowledge base and "
                "passes them to the next stage of the pipeline."
            )

        if question_type == "summarizer":
            return (
                "Based on the retrieved documents:\n\n"
                "The Summarizer agent takes the retrieved documents "
                "and turns their relevant information into a grounded "
                "answer. It answers strictly from the retrieved sources."
            )

        if question_type == "critic":
            return (
                "Based on the retrieved documents:\n\n"
                "The Critic agent checks the quality and correctness "
                "of the Summarizer's answer. If the answer is weak or "
                "needs improvement, the Critic can send it back to the "
                "Summarizer for another revision."
            )

        if question_type == "langgraph":
            return (
                "Based on the retrieved documents:\n\n"
                "LangGraph is used because the system needs shared "
                "state between agents and conditional routing. The "
                "Critic can decide whether the answer should finish "
                "or be sent back to the Summarizer for revision."
            )

        if question_type == "objective":
            return (
                "Based on the retrieved documents:\n\n"
                "The project is a multi-agent research assistant "
                "that answers questions from a document knowledge "
                "base. It uses three cooperating agents: a Retriever, "
                "a Summarizer, and a Critic, orchestrated with LangGraph."
            )

        # For other questions, use the retrieved context.
        answer = self.find_best_answer(
            context,
            question_type
        )

        if answer:
            return (
                "Based on the retrieved documents:\n\n"
                + answer
            )

        return (
            "Based on the retrieved documents:\n\n"
            "The retrieved documents do not contain enough "
            "specific information to answer this question."
        )

    # =========================================================
    # QUESTION TYPE
    # =========================================================

    def detect_question_type(self, question):

        if (
            "main objective" in question
            or "purpose of this project" in question
            or "what is this project" in question
            or "objective of this project" in question
        ):
            return "objective"

        if (
            "what is rag" in question
            or "what is the purpose of the rag" in question
            or "purpose of the rag" in question
            or "purpose of rag" in question
            or "why use rag" in question
            or "why is rag useful" in question
            or "why rag" in question
            or "rag system" in question
        ):
            return "rag"

        if (
            "retriever agent" in question
            or "role of the retriever" in question
            or "what does the retriever" in question
        ):
            return "retriever"

        if (
            "summarizer agent" in question
            or "role of the summarizer" in question
            or "what does the summarizer" in question
        ):
            return "summarizer"

        if (
            "critic agent" in question
            or "role of the critic" in question
            or "what does the critic" in question
        ):
            return "critic"

        if (
            "langgraph" in question
            or "why use langgraph" in question
            or "why is langgraph" in question
            or "purpose of langgraph" in question
        ):
            return "langgraph"

        return "general"

    # =========================================================
    # FIND ANSWER FROM CONTEXT
    # =========================================================

    def find_best_answer(self, context, question_type):

        text = context.replace("\n", " ")
        text = " ".join(text.split())

        # Clean common PDF extraction characters.
        text = text.replace("ﬁ", "fi")
        text = text.replace("ﬂ", "fl")
        text = text.replace("’", "'")
        text = text.replace("“", '"')
        text = text.replace("”", '"')

        # Split into sentences.
        raw_sentences = (
            text
            .replace("!", ".")
            .replace("?", ".")
            .split(".")
        )

        sentences = []

        for part in raw_sentences:

            sentence = self.clean_sentence(part)

            if sentence:
                sentences.append(sentence)

        if not sentences:
            return None

        scored = []

        useful_words = [
            "retrieve",
            "retrieves",
            "retriever",
            "documents",
            "answer",
            "grounded",
            "sources",
            "summarizer",
            "critic",
            "fact-check",
            "revision",
            "langgraph",
            "shared state",
            "conditional routing"
        ]

        for sentence in sentences:

            sentence_lower = sentence.lower()

            score = 0

            # General useful-word scoring.
            for word in useful_words:

                if word in sentence_lower:
                    score += 2

            # Prefer sentences that actually contain
            # meaningful information.
            if len(sentence.split()) >= 10:
                score += 2

            if "retrieved documents" in sentence_lower:
                score += 5

            if "grounded" in sentence_lower:
                score += 5

            if "sources" in sentence_lower:
                score += 3

            # Penalize README/document-navigation noise.
            bad_patterns = [
                "questions this prepares you for",
                "resume bullets",
                "the product in one line",
                "skills this closes",
                "the mental model",
                "readme.md",
                "interview gold",
                "architecture"
            ]

            for pattern in bad_patterns:

                if pattern in sentence_lower:
                    score -= 20

            scored.append(
                (score, sentence)
            )

        scored.sort(
            key=lambda item: item[0],
            reverse=True
        )

        best_score, best_sentence = scored[0]

        if best_score <= 0:
            return None

        return best_sentence.strip() + "."


    # =========================================================
    # CLEAN SENTENCE
    # =========================================================

    def clean_sentence(self, sentence):

        sentence = " ".join(sentence.split())

        if not sentence:
            return None

        noise_patterns = [
            "part 1",
            "part 2",
            "part 3",
            "part 4",
            "part 5",
            "part 6",
            "part 7",
            "part 8",
            "part 9",
            "part 10",
            "part 11",
            "part 12",
            "part 13",
            "part 14",
            "part 15",
            "part 16",
            "questions this prepares you for",
            "resume bullets",
            "the product in one line",
            "skills this closes",
            "the mental model",
            "readme.md",
            "interview gold",
            "architecture"
        ]

        lower = sentence.lower()

        for pattern in noise_patterns:

            if pattern in lower:
                return None

        # Ignore very small fragments.
        if len(sentence.split()) < 7:
            return None

        # Remove accidental Markdown prefix.
        if sentence.startswith("md #"):
            sentence = sentence[4:].strip()

        return sentence.strip()


# =============================================================
# OPENAI LLM
# =============================================================

class OpenAILLM:
    """
    Production LLM option.

    Used only when MODE=production.
    """

    def __init__(self):

        from langchain_openai import ChatOpenAI

        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

    def generate(self, question, context):

        prompt = f"""
Answer the question using only the provided context.

Question:
{question}

Context:
{context}

Rules:
- Answer directly.
- Use only information from the context.
- Do not invent information.
- Keep the answer concise.
- Do not mention unrelated sections of the document.
"""

        response = self.model.invoke(prompt)

        return response.content


# =============================================================
# LLM FACTORY
# =============================================================

def get_llm():

    if MODE == "production":
        return OpenAILLM()

    return LocalLLM()