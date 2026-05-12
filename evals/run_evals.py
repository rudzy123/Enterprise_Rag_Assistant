#!/usr/bin/env python3

import sys
from pathlib import Path

# ✅ FIX: make project root importable
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json
from datetime import datetime

# ✅ import after path fix
from core.retrieval.retrieve_chunks import retrieve_chunks

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


def compute_metrics(question_obj: dict, retrieved_chunks: list) -> dict:
    expected_sources = question_obj.get("source_doc_id", [])
    if isinstance(expected_sources, str):
        expected_sources = [expected_sources]

    # ✅ safer extraction (avoid None values)
    returned_sources = set(
        [c.get("source_file") for c in retrieved_chunks if c.get("source_file")]
    )

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

    missing_critical = list(expected_set - returned_sources)
    irrelevant_chunks = list(returned_sources - expected_set)

    return {
        "retrieval_hit": retrieval_hit,
        "retrieval_precision": retrieval_precision,
        "retrieval_recall": retrieval_recall,
        "missing_critical": missing_critical,
        "irrelevant_chunks": irrelevant_chunks,
        "num_chunks": len(retrieved_chunks),
    }


def run_evals(jsonl_path: str):
    questions = load_questions(jsonl_path)

    if not questions:
        print("No questions loaded.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(questions)} questions")
    print("=" * 80)

    results = []

    for q in questions:
        question_text = q["question"]

        print(f"\nQuestion: {question_text}")
        print(f"Expected sources: {q.get('source_doc_id')}")

        # ✅ retrieval only
        chunks = retrieve_chunks(question_text)

        metrics = compute_metrics(q, chunks)

        print(f"Retrieved chunks: {metrics['num_chunks']}")
        print(f"Sources: {[c.get('source_file') for c in chunks]}")

        print("Metrics:")
        print(f"  retrieval_hit: {'✓' if metrics['retrieval_hit'] else '✗'}")
        print(f"  precision: {metrics['retrieval_precision']:.2f}")
        print(f"  recall: {metrics['retrieval_recall']:.2f}")

        if metrics["missing_critical"]:
            print(f"  missing_critical: {metrics['missing_critical']}")

        if metrics["irrelevant_chunks"]:
            print(f"  irrelevant: {metrics['irrelevant_chunks']}")

        results.append({
            "question": question_text,
            "metrics": metrics,
        })

    # ✅ SUMMARY
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total = len(results)

    hit_rate = sum(r["metrics"]["retrieval_hit"] for r in results) / total
    avg_precision = sum(r["metrics"]["retrieval_precision"] for r in results) / total
    avg_recall = sum(r["metrics"]["retrieval_recall"] for r in results) / total

    print(f"Total questions: {total}")
    print(f"Hit Rate: {hit_rate:.2f}")
    print(f"Avg Precision: {avg_precision:.2f}")
    print(f"Avg Recall: {avg_recall:.2f}")

    # ✅ SAVE RESULTS
    timestamp = datetime.now().strftime("%Y%m%d")
    timestamp_path = RESULTS_DIR / f"results_{timestamp}.json"
    latest_path = RESULTS_DIR / "results.json"

    with open(timestamp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to:")
    print(f"- {timestamp_path}")
    print(f"- {latest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "jsonl_path",
        nargs="?",
        default=Path("evals/questions.jsonl")
    )
    args = parser.parse_args()

    run_evals(str(args.jsonl_path))