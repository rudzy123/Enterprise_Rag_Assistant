#!/usr/bin/env python3

import sys
from pathlib import Path

# ✅ FIX IMPORT PATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
from datetime import datetime
import time

from retrieval.retrieve_chunks import retrieve_similar_chunks

RESULTS_DIR = Path("evals")
RESULTS_DIR.mkdir(exist_ok=True)


def load_questions(jsonl_path: str) -> list:
    questions = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def call_retrieval_directly(question: str) -> dict:
    """
    Direct retrieval WITH visibility (no silent failures)
    """
    start_time = time.time()

    retrieved_chunks = retrieve_similar_chunks(question)

    latency_ms = (time.time() - start_time) * 1000

    print(f"  DEBUG: Retrieved {len(retrieved_chunks)} chunks")

    sources = list(set(
        c.get("source_file")
        for c in retrieved_chunks
        if c.get("source_file")
    ))

    return {
        "retrieved_chunks": retrieved_chunks,
        "sources": sources,
        "latency_ms": latency_ms
    }


def compute_metrics(question_obj: dict, response: dict) -> dict:
    # ✅ FIX: handle None properly
    expected_sources = question_obj.get("source_doc_id") or []

    if isinstance(expected_sources, str):
        expected_sources = [expected_sources]

    returned_sources = set(response.get("sources", []))
    retrieved_chunks = response.get("retrieved_chunks", [])

    expected_set = set(expected_sources)

    relevant_retrieved = expected_set & returned_sources

    retrieval_hit = bool(relevant_retrieved) if expected_set else True

    retrieval_precision = (
        len(relevant_retrieved) / len(returned_sources)
        if returned_sources else 1.0
    )

    retrieval_recall = (
        len(relevant_retrieved) / len(expected_set)
        if expected_set else 1.0
    )

    return {
        "retrieval_hit": retrieval_hit,
        "retrieval_precision": retrieval_precision,
        "retrieval_recall": retrieval_recall,
        "missing_critical": list(expected_set - returned_sources),
        "irrelevant_chunks": list(returned_sources - expected_set),
        "returned_sources": list(returned_sources),
        "num_retrieved_chunks": len(retrieved_chunks),
        "latency_ms": response.get("latency_ms", 0.0),
    }


def run_evals(jsonl_path: str):
    try:
        questions = load_questions(jsonl_path)
    except FileNotFoundError:
        print(f"Error: Could not find {jsonl_path}")
        sys.exit(1)

    print(f"Loaded {len(questions)} questions\n")
    print("=" * 80)

    results = []

    for q in questions:
        question_text = q.get("question", "")

        print(f"\nQuestion: {question_text}")
        print(f"Expected sources: {q.get('source_doc_id')}")

        response = call_retrieval_directly(question_text)
        metrics = compute_metrics(q, response)

        print(f"Sources: {metrics['returned_sources']}")
        print(f"Chunks: {metrics['num_retrieved_chunks']}")
        print(f"Hit: {'✓' if metrics['retrieval_hit'] else '✗'}")
        print(f"Precision: {metrics['retrieval_precision']:.2f}")
        print(f"Recall: {metrics['retrieval_recall']:.2f}")

        if metrics["missing_critical"]:
            print(f"Missing: {metrics['missing_critical']}")

        if metrics["irrelevant_chunks"]:
            print(f"Irrelevant: {metrics['irrelevant_chunks']}")

        results.append({
            "question": question_text,
            "metrics": metrics
        })

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # ✅ safety guard
    valid_results = [r for r in results if "metrics" in r]

    if not valid_results:
        print("No valid results generated.")
        return

    total = len(valid_results)

    hit_rate = sum(r["metrics"]["retrieval_hit"] for r in valid_results) / total
    avg_precision = sum(r["metrics"]["retrieval_precision"] for r in valid_results) / total
    avg_recall = sum(r["metrics"]["retrieval_recall"] for r in valid_results) / total

    print(f"Total: {total}")
    print(f"Hit Rate: {hit_rate:.2f}")
    print(f"Avg Precision: {avg_precision:.2f}")
    print(f"Avg Recall: {avg_recall:.2f}")

    timestamp = datetime.now().strftime("%Y%m%d")

    with open(RESULTS_DIR / f"results_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(valid_results, f, indent=2, ensure_ascii=False)

    with open(RESULTS_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(valid_results, f, indent=2, ensure_ascii=False)

    print("\n✅ Results saved to evals/results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "jsonl_path",
        nargs="?",
        default="evals/questions.jsonl"
    )
    args = parser.parse_args()

    run_evals(args.jsonl_path)
