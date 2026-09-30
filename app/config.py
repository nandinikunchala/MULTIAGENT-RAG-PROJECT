import os

MODE = os.getenv("MODE", "local")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "research-assistant")

TOP_K = int(os.getenv("TOP_K", "3"))
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))

# Minimum similarity required for a document to be considered relevant
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.25"))