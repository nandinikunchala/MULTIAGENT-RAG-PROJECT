import numpy as np

from app.embeddings import get_embedder
from app.config import TOP_K


class LocalVectorStore:
    """
    Local vector store using SentenceTransformer embeddings.

    Documents are embedded and stored in memory.
    Queries are compared using cosine similarity.
    """

    def __init__(self, similarity_threshold=0.25):
        self.embedder = get_embedder()

        self.documents = []
        self.vectors = []

        # Minimum similarity required for a document
        # to be considered relevant.
        self.similarity_threshold = similarity_threshold

    def add_documents(self, documents):
        """
        Add documents and generate embeddings for them.
        """

        if not documents:
            return

        self.documents.extend(documents)

        texts = [
            document.get("text", "")
            for document in documents
        ]

        embeddings = self.embedder.embed_batch(texts)

        # Normalize embeddings so dot product becomes
        # cosine similarity.
        embeddings = self._normalize(embeddings)

        self.vectors.extend(embeddings)

    def search(self, query, k=TOP_K):
        """
        Search for the most relevant documents.

        Documents below the similarity threshold are ignored.
        """

        if not self.documents:
            return []

        if not query or not query.strip():
            return []

        # Create query embedding
        query_vector = self.embedder.embed(query)

        # Normalize query vector
        query_vector = self._normalize(query_vector)

        scores = []

        for index, vector in enumerate(self.vectors):

            # Since both vectors are normalized,
            # dot product = cosine similarity.
            score = float(
                np.dot(query_vector, vector)
            )

            scores.append((score, index))

        # Highest similarity first
        scores.sort(
            key=lambda item: item[0],
            reverse=True
        )

        results = []

        for score, index in scores:

            # Ignore weak matches
            if score < self.similarity_threshold:
                continue

            result = dict(
                self.documents[index]
            )

            result["score"] = score

            results.append(result)

            # Stop after TOP_K relevant documents
            if len(results) >= k:
                break

        return results

    @staticmethod
    def _normalize(vectors):
        """
        Normalize vectors to unit length.

        This makes dot product equivalent to
        cosine similarity.
        """

        vectors = np.asarray(
            vectors,
            dtype=np.float32
        )

        # Handle a single vector
        if vectors.ndim == 1:

            norm = np.linalg.norm(vectors)

            if norm == 0:
                return vectors

            return vectors / norm

        # Handle multiple vectors
        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True
        )

        # Prevent division by zero
        norms = np.maximum(norms, 1e-12)

        return vectors / norms


def get_vector_store():
    """
    Factory function for the local vector store.
    """

    return LocalVectorStore()