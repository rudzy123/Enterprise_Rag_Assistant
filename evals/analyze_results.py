#!/usr/bin/env python3
"""
Analyze evaluation JSON results and surface the worst-performing cases.
"""

import argparse
import json
from pathlib import Path


def load_results(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_results(results):
    worst_groundedness = sorted(results, key=lambda x: x["metrics"].get("groundedness_score", 100.0))[:10]
    worst_recall = sorted(results, key=lambda x: x["metrics"].get("retrieval_recall", 1.0))[:10]
    hallucinations = [r for r in results if r["metrics"].get("failure_type") == "hallucination"]

    summary = {
        "total_cases": len(results),
        "worst_groundedness": [
            {
                "question": r["question"],
                "groundedness_score": r["metrics"].get("groundedness_score"),
                "failure_type": r["metrics"].get("failure_type"),
                "retrieved_sources": r["response"].get("sources"),
            }
            for r in worst_groundedness
        ],
        "worst_recall": [
            {
                "question": r["question"],
                "retrieval_recall": r["metrics"].get("retrieval_recall"),
                "failure_type": r["metrics"].get("failure_type"),
                "retrieved_sources": r["response"].get("sources"),
            }
            for r in worst_recall
        ],
        "hallucination_count": len(hallucinations),
        "hallucination_examples": [
            {
                "question": r["question"],
                "groundedness_score": r["metrics"].get("groundedness_score"),
                "retrieval_recall": r["metrics"].get("retrieval_recall"),
                "retrieved_sources": r["response"].get("sources"),
            }
            for r in hallucinations[:10]
        ],
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze eval JSON results")
    parser.add_argument("result_file", type=Path, help="Path to evals/results_*.json")
    args = parser.parse_args()

    results = load_results(args.result_file)
    summary = summarize_results(results)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
