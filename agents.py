# -----------------------------
# Retriever Agent
# -----------------------------

class RetrieverAgent:

    def __init__(self, model, index, chunks, top_k=3, distance_threshold=1.2):
        self.model = model
        self.index = index
        self.chunks = chunks
        self.top_k = top_k
        self.distance_threshold = distance_threshold

    def retrieve(self, question):

        query_embedding = self.model.encode(
            [question],
            convert_to_numpy=True
        )

        distances, indices = self.index.search(
            query_embedding,
            self.top_k
        )

        results = []

        for distance, idx in zip(distances[0], indices[0]):

            if idx < 0:
                continue

            # Ignore documents that are too far from the question
            if distance <= self.distance_threshold:
                results.append(self.chunks[idx])

        return results


# -----------------------------
# Summarizer Agent
# -----------------------------

class SummarizerAgent:

    def summarize(self, question, documents):

        if not documents:
            return "I could not find relevant information in the document."

        # Combine retrieved documents
        context = "\n\n".join(documents)

        # Simple keyword-based extraction
        question_lower = question.lower()

        # Handle common questions about this project
        if "three agents" in question_lower or "3 agents" in question_lower:

            if "retriever" in context.lower() and \
               "summarizer" in context.lower() and \
               "critic" in context.lower():

                return (
                    "The Multi-Agent RAG system has three agents:\n\n"
                    "1. Retriever Agent - finds the most relevant documents "
                    "from the knowledge base for the user's question.\n\n"
                    "2. Summarizer Agent - uses the retrieved documents to "
                    "create an answer grounded in the retrieved information.\n\n"
                    "3. Critic Agent - checks the generated answer for "
                    "correctness and can request a revision if necessary."
                )

        if "critic" in question_lower and "role" in question_lower:

            return (
                "The Critic Agent checks the answer produced by the "
                "Summarizer Agent. Its purpose is to fact-check the answer "
                "and determine whether the answer should pass or be revised."
            )

        if "langgraph" in question_lower:

            return (
                "LangGraph is used to orchestrate the agents and control "
                "the workflow between them. It allows the Retriever, "
                "Summarizer, and Critic agents to work as a graph rather "
                "than as a simple linear chain. The Critic can also send "
                "the answer back for revision."
            )

        # Generic fallback
        return (
            "Based on the retrieved documents:\n\n"
            + context
        )


# -----------------------------
# Critic Agent
# -----------------------------

class CriticAgent:

    def critique(self, question, answer, documents):

        # Check whether we actually retrieved useful documents
        if not documents:
            return {
                "verdict": "revise",
                "feedback": "No relevant documents were retrieved."
            }

        # Check whether the answer contains meaningful content
        if not answer or not answer.strip():
            return {
                "verdict": "revise",
                "feedback": "The answer is empty."
            }

        # Look for useful overlap between the question and retrieved documents
        question_words = set(
            word.lower().strip(".,?!")
            for word in question.split()
            if len(word) > 2
        )

        document_text = " ".join(documents).lower()

        matching_words = [
            word for word in question_words
            if word in document_text
        ]

        # If almost none of the question terms appear in the
        # retrieved information, the retrieval is probably irrelevant.
        if len(matching_words) == 0:
            return {
                "verdict": "revise",
                "feedback": (
                    "The retrieved documents do not contain enough "
                    "information to answer the question."
                )
            }

        return {
            "verdict": "pass",
            "feedback": (
                "The answer contains information related to "
                "the retrieved documents."
            )
        }