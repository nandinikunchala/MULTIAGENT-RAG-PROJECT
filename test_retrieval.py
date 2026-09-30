from app.graph import retriever

question = "What is the purpose of the RAG system?"

results = retriever.run(question)

print("\n========== RETRIEVED DOCUMENTS ==========\n")

for i, result in enumerate(results, start=1):
    print(f"--- RESULT {i} ---")
    print("TEXT:")
    print(result.get("text", ""))
    print()
    print("SCORE:", result.get("score"))
    print("SOURCE:", result.get("source"))
    print("\n" + "=" * 60 + "\n")