from app.graph import retriever
from app.llm import get_llm

question = "What is the purpose of the RAG system?"

print("\n========== QUESTION ==========\n")
print(question)

documents = retriever.run(question)

print("\n========== RETRIEVED DOCUMENTS ==========\n")

for i, document in enumerate(documents, 1):
    print(f"\n--- DOCUMENT {i} ---")
    print(document.get("text", ""))
    print(f"\nSCORE: {document.get('score')}")
    print(f"SOURCE: {document.get('source')}")

context = "\n\n".join(
    document.get("text", "")
    for document in documents
)

print("\n========== CONTEXT SENT TO LLM ==========\n")
print(context)

llm = get_llm()

answer = llm.generate(
    question,
    context
)

print("\n========== LLM ANSWER ==========\n")
print(answer)