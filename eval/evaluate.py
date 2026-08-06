import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rag import answer_question  # noqa: E402

EVAL_SET_PATH = os.path.join(os.path.dirname(__file__), "eval_set.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")
DELAY_BETWEEN_QUERIES_SEC = 2


def main():
    with open(EVAL_SET_PATH) as f:
        eval_set = json.load(f)

    if any("REPLACE ME" in item["question"] for item in eval_set):
        print("⚠️  eval_set.json still has placeholder questions.")
        print("   Replace them with real questions about YOUR indexed documents first.")
        return

    results = []
    correct = 0
    latencies = []

    for item in eval_set:
        r = answer_question(item["question"], log=True)
        is_correct = item["expected_answer_contains"].lower() in r["answer"].lower()
        correct += int(is_correct)
        latencies.append(r["latency_ms"])

        results.append({
            "question": item["question"],
            "answer": r["answer"],
            "expected_contains": item["expected_answer_contains"],
            "correct": is_correct,
            "latency_ms": r["latency_ms"],
            "sources": r["sources"],
        })
        print(f"{'✅' if is_correct else '❌'} {item['question']}")
        time.sleep(DELAY_BETWEEN_QUERIES_SEC)

    accuracy = round(100 * correct / len(eval_set), 1)
    avg_latency = round(sum(latencies) / len(latencies), 1)

    summary = {
        "total_questions": len(eval_set),
        "correct": correct,
        "accuracy_pct": accuracy,
        "avg_latency_ms": avg_latency,
        "results": results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n--- Summary ---")
    print(f"Accuracy: {accuracy}% ({correct}/{len(eval_set)})")
    print(f"Avg latency: {avg_latency} ms")
    print(f"Full results written to {RESULTS_PATH}")
    print("\nUse these real numbers in your resume bullet — no brackets needed.")


if __name__ == "__main__":
    main()
