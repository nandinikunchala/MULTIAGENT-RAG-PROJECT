from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

from graph import build_graph


# -----------------------------
# 1. Read PDF
# -----------------------------

print("\nReading PDF...")

reader = PdfReader("paper.pdf")

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

print("PDF loaded successfully.")


# -----------------------------
# 2. Split text into chunks
# -----------------------------

chunk_size = 500
chunks = []

for i in range(0, len(text), chunk_size):

    chunk = text[i:i + chunk_size]

    if chunk.strip():
        chunks.append(chunk)

print("Number of chunks:", len(chunks))


# -----------------------------
# 3. Create embeddings
# -----------------------------

print("\nCreating embeddings...")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    chunks,
    convert_to_numpy=True
)

print("Embedding shape:", embeddings.shape)


# -----------------------------
# 4. Create FAISS database
# -----------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Vectors stored in FAISS:", index.ntotal)


# -----------------------------
# 5. Build LangGraph
# -----------------------------

print("\nBuilding LangGraph...")

graph = build_graph(
    model=model,
    index=index,
    chunks=chunks
)

print("LangGraph built successfully.")


# -----------------------------
# 6. Ask question
# -----------------------------

question = input("\nEnter your question: ")


# -----------------------------
# 7. Initial graph state
# -----------------------------

initial_state = {
    "question": question,
    "documents": [],
    "answer": "",
    "verdict": "",
    "feedback": ""
}


# -----------------------------
# 8. Run LangGraph
# -----------------------------

print("\n" + "=" * 60)
print("STARTING MULTI-AGENT RAG PIPELINE")
print("=" * 60)

result = graph.invoke(initial_state)


# -----------------------------
# 9. Display final answer
# -----------------------------

print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(result["answer"])

print("\n" + "=" * 60)
print("CRITIC RESULT")
print("=" * 60)

print("Verdict:", result["verdict"])
print("Feedback:", result["feedback"])

print("\nPipeline completed successfully.")