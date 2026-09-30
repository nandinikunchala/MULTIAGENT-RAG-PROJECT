import numpy as np

from sentence_transformers import SentenceTransformer

from app.config import MODE


class LocalEmbedder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim = 384

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(
            text,
            convert_to_numpy=True
        ).astype(np.float32)

    def embed_batch(self, texts):
        return self.model.encode(
            texts,
            convert_to_numpy=True
        ).astype(np.float32)


class OpenAIEmbedder:
    dim = 1536

    def __init__(self):
        from langchain_openai import OpenAIEmbeddings

        self._model = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

    def embed(self, text: str):
        return np.array(
            self._model.embed_query(text),
            dtype=np.float32
        )

    def embed_batch(self, texts):
        return np.array(
            self._model.embed_documents(texts),
            dtype=np.float32
        )


def get_embedder():
    if MODE == "production":
        return OpenAIEmbedder()

    return LocalEmbedder()