from app.ingest import ingest_pdf
from eval.testset import TEST_QUERIES


def evaluate_retrieval():
    print("Loading document store...")

    store = ingest_pdf("paper.pdf")

    correct = 0
    total = len(TEST_QUERIES)

    print("\n========== RETRIEVAL EVALUATION ==========\n")

    for i, (question, expected_source) in enumerate(TEST_QUERIES, 1):

        results = store.search(question, 3)

        retrieved_sources = [
            result.get("source", "")
            for result in results
        ]

        passed = expected_source in retrieved_sources

        if passed:
            correct += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"Test {i}: {status}")
        print(f"Question: {question}")
        print(f"Expected source: {expected_source}")
        print(f"Retrieved sources: {retrieved_sources}")
        print()

    accuracy = (correct / total) * 100

    print("========== RESULTS ==========")
    print(f"Correct: {correct}/{total}")
    print(f"Retrieval accuracy: {accuracy:.2f}%")

    return accuracy


if __name__ == "__main__":
    evaluate_retrieval()