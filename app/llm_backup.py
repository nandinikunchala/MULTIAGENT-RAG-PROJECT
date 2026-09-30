from app.config import MODE


class LocalLLM:
    """
    Local, API-key-free answer generator.

    Uses the retrieved document context to find the most
    relevant evidence for the user's question.
    """

    def generate(self, question, context):

        if not context:
            return (
                "I could not find relevant information "
                "in the retrieved documents."
            )

        question_type = self.detect_question_type(question.lower())

        keywords = self.get_keywords(question_type)

        answer = self.find_best_answer(context, keywords)

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

    def detect_question_type(self, question):

        if (
            "main objective" in question
            or "objective" in question
            or "purpose of this project" in question
            or "what is this project" in question
        ):
            return "objective"

        if (
            "retriever agent" in question
            or "role of the retriever" in question
        ):
            return "retriever"

        if (
            "summarizer agent" in question
            or "role of the summarizer" in question
        ):
            return "summarizer"

        if (
            "critic agent" in question
            or "role of the critic" in question
        ):
            return "critic"

        if (
            "langgraph" in question
            or "why use langgraph" in question
            or "why is langgraph" in question
        ):
            return "langgraph"

        if (
            "rag system" in question
            or "purpose of rag" in question
            or "what is rag" in question
        ):
            return "rag"

        return "general"

    def get_keywords(self, question_type):

        keywords = {

            "rag": [
                "retrieve the most relevant documents",
                "answer only from those",
                "answer is grounded",
                "show the sources",
                "rag fixes"
            ],

            "retriever": [
                "retriever agent",
                "retriever",
                "retrieve relevant documents",
                "retrieves the most relevant documents",
                "vector store"
            ],

            "summarizer": [
                "summarizer answers",
                "summarizer",
                "retrieved sources",
                "grounded answer",
                "summarizes"
            ],

            "critic": [
                "critic can judge",
                "critic agent",
                "fact-check",
                "fact checks",
                "send weak answers back",
                "another pass",
                "demand a better one"
            ],

            "langgraph": [
                "langgraph gives",
                "shared state",
                "conditional routing",
                "cannot loop back",
                "cannot loop or branch",
                "critic decides"
            ],

            "objective": [
                "multi-agent research assistant",
                "three cooperating agents",
                "document knowledge base",
                "retriever",
                "summarizer",
                "critic"
            ],

            "general": []
        }

        return keywords.get(question_type, [])

    def clean_sentence(self, sentence):

        sentence = " ".join(sentence.split())

        # Remove obvious PDF/README artifacts
        bad_phrases = [
            "PART 1",
            "PART 2",
            "PART 3",
            "PART 4",
            "PART 5",
            "PART 6",
            "PART 7",
            "PART 8",
            "PART 9",
            "PART 10",
            "PART 11",
            "PART 12",
            "PART 13",
            "PART 14",
            "PART 15",
            "PART 16",
            "Questions this prepares you for",
            "README.md",
            "resume bullets",
            "interview gold",
        ]

        for phrase in bad_phrases:
            if phrase.lower() in sentence.lower():
                return None

        return sentence

    def find_best_answer(self, context, keywords):

        # Normalize PDF text
        text = context.replace("\n", " ")
        text = " ".join(text.split())

        # Split into sentences
        sentences = []

        for part in text.replace("!", ".").replace("?", ".").split("."):
            sentence = self.clean_sentence(part.strip())

            if sentence:
                sentences.append(sentence)

        if not sentences:
            return None

        scored = []

        for sentence in sentences:

            sentence_lower = sentence.lower()
            score = 0

            # Strongly reward exact concepts
            for keyword in keywords:
                if keyword.lower() in sentence_lower:
                    score += 20

            # Reward useful explanatory sentences
            useful_words = [
                "retrieve",
                "retrieves",
                "answer",
                "grounded",
                "sources",
                "summarizer",
                "critic",
                "fact-check",
                "revision",
                "langgraph",
                "shared state",
                "conditional routing",
                "documents"
            ]

            for word in useful_words:
                if word in sentence_lower:
                    score += 2

            # Penalize very short fragments
            if len(sentence.split()) < 8:
                score -= 5

            # Penalize obvious headings/fragments
            if ":" in sentence and len(sentence.split()) < 12:
                score -= 3

            scored.append((score, sentence))

        scored.sort(
            key=lambda item: item[0],
            reverse=True
        )

        best_score, best_sentence = scored[0]

        if best_score <= 0:
            return None

        return best_sentence + "."


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


def get_llm():

    if MODE == "production":
        return OpenAILLM()

    return LocalLLM()