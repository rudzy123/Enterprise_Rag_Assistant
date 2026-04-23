#!/usr/bin/env python3
"""
Evaluation script for Enterprise RAG Assistant.

Loads test questions from evals/questions.jsonl
Calls the /ask endpoint and computes evaluation metrics:
  - retrieval_hit: at least one expected source appears in returned sources
  - answerability_correct: confidence >= 0.3 matches the answerable field
  - citation_present: returned sources list is non-empty when answering
"""

import json
import sys
import requests
from pathlib import Path


def load_questions(jsonl_path: str) -> list:
    """Load questions from JSONL file."""
    questions = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def call_ask_endpoint(question: str, endpoint_url: str = "http://localhost:8000/ask") -> dict:
    """
    Call the /ask endpoint and return the response.
    
    Returns:
        dict with keys: answer, sources, confidence
        Returns None if request fails
    """
    try:
        payload = {"question": question}
        response = requests.post(endpoint_url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling endpoint: {e}", file=sys.stderr)
        return None


def get_confidence_range(confidence: float) -> str:
    """
    Categorize confidence score into range.
    
    Ranges:
    - 0.0–0.3:   Weak (low)    → System refuses to answer
    - 0.3–0.6:   Partial (medium) → Answer given with caution
    - 0.6–1.0:   Strong (high) → High-confidence answer
    
    Args:
        confidence: float between 0.0 and 1.0
    
    Returns:
        str: "low", "medium", or "high"
    """
    if confidence < 0.3:
        return "low"
    elif confidence < 0.6:
        return "medium"
    else:
        return "high"


def compute_metrics(question_obj: dict, response: dict) -> dict:
    """
    Compute evaluation metrics for a single question.
    
    Args:
        question_obj: dict with keys: id, question, expected_sources, answerable
        response: dict with keys: answer, sources, confidence
    
    Returns:
        dict with computed metrics
    """
    expected_sources = set(question_obj.get("expected_sources", []))
    answerable = question_obj.get("answerable", True)
    returned_sources = set(response.get("sources", []))
    confidence = response.get("confidence", 0.0)
    answer = response.get("answer", "")
    
    # Metric 1: retrieval_hit
    # True if at least one expected source appears in returned sources
    if expected_sources:
        retrieval_hit = bool(expected_sources.intersection(returned_sources))
    else:
        # If no expected sources (unanswerable question), pass if no sources returned
        retrieval_hit = len(returned_sources) == 0
    
    # Metric 2: answerability_correct
    # True if confidence >= 0.3 matches the answerable field
    high_confidence = confidence >= 0.3
    answerability_correct = (high_confidence == answerable)
    
    # Metric 3: citation_present
    # True if returned sources list is non-empty when answering (confidence >= 0.3)
    if confidence >= 0.3:
        citation_present = len(returned_sources) > 0
    else:
        # Low confidence answer should not have sources (or empty list is acceptable)
        citation_present = len(returned_sources) == 0
    
    # Confidence range classification
    confidence_range = get_confidence_range(confidence)
    
    return {
        "retrieval_hit": retrieval_hit,
        "answerability_correct": answerability_correct,
        "citation_present": citation_present,
        "confidence": confidence,
        "confidence_range": confidence_range,
        "returned_sources": list(returned_sources),
    }


def run_evals(jsonl_path: str, endpoint_url: str = "http://localhost:8000/ask"):
    """
    Run evaluations on all questions.
    
    Args:
        jsonl_path: path to questions.jsonl
        endpoint_url: full URL to the /ask endpoint
    """
    # Load questions
    try:
        questions = load_questions(jsonl_path)
    except FileNotFoundError:
        print(f"Error: Could not find {jsonl_path}", file=sys.stderr)
        sys.exit(1)
    
    if not questions:
        print("No questions loaded.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(questions)} questions from {jsonl_path}")
    print(f"Calling endpoint: {endpoint_url}\n")
    print("=" * 80)
    
    results = []
    
    # Run each question through the endpoint
    for question_obj in questions:
        q_id = question_obj.get("id", "unknown")
        question_text = question_obj.get("question", "")
        expected_sources = question_obj.get("expected_sources", [])
        answerable = question_obj.get("answerable", True)
        
        print(f"\n[{q_id}] {question_text}")
        print(f"  Expected answerable: {answerable}")
        print(f"  Expected sources: {expected_sources if expected_sources else 'none (unanswerable)'}")
        
        # Call endpoint
        response = call_ask_endpoint(question_text, endpoint_url)
        
        if response is None:
            print(f"  ❌ Failed to get response from endpoint")
            continue
        
        # Compute metrics
        metrics = compute_metrics(question_obj, response)
        
        # Display results
        print(f"  Confidence: {metrics['confidence']:.2f} ({metrics['confidence_range'].upper()})")
        print(f"  Returned sources: {metrics['returned_sources'] if metrics['returned_sources'] else 'none'}")
        print(f"  Metrics:")
        print(f"    - retrieval_hit: {'✓' if metrics['retrieval_hit'] else '✗'}")
        print(f"    - answerability_correct: {'✓' if metrics['answerability_correct'] else '✗'}")
        print(f"    - citation_present: {'✓' if metrics['citation_present'] else '✗'}")
        
        results.append({
            "id": q_id,
            "question": question_text,
            "answerable": answerable,
            "expected_sources": expected_sources,
            "metrics": metrics,
        })
    
    # Print summary metrics
    print("\n" + "=" * 80)
    print("SUMMARY METRICS")
    print("=" * 80)
    
    if not results:
        print("No eval results to summarize.")
        return
    
    retrieval_hits = sum(1 for r in results if r["metrics"]["retrieval_hit"])
    answerability_correct = sum(1 for r in results if r["metrics"]["answerability_correct"])
    citation_present = sum(1 for r in results if r["metrics"]["citation_present"])
    
    # Count confidence ranges
    low_conf = sum(1 for r in results if r["metrics"]["confidence_range"] == "low")
    medium_conf = sum(1 for r in results if r["metrics"]["confidence_range"] == "medium")
    high_conf = sum(1 for r in results if r["metrics"]["confidence_range"] == "high")
    
    total = len(results)
    
    print(f"\nTotal questions: {total}")
    retrieval_hit_rate = (100 * retrieval_hits / total) if total > 0 else 0.0
    answerability_accuracy = (100 * answerability_correct / total) if total > 0 else 0.0
    citation_rate = (100 * citation_present / total) if total > 0 else 0.0
    
    print(f"\nRetrieval Hit Rate: {retrieval_hit_rate:.2f}%")
    print(f"Answerability Accuracy: {answerability_accuracy:.2f}%")
    print(f"Citation Rate: {citation_rate:.2f}%")
    
    avg_confidence = sum(r["metrics"]["confidence"] for r in results) / total if total > 0 else 0.0
    print(f"\nAverage Confidence: {avg_confidence:.2f}")
    
    # Confidence range distribution
    print(f"\nConfidence Range Distribution:")
    print(f"  Low  (0.0–0.3):  {low_conf}/{total}  ({100*low_conf/total:.2f}%)  [Refusal cases]")
    print(f"  Medium (0.3–0.6): {medium_conf}/{total}  ({100*medium_conf/total:.2f}%)  [Cautious answers]")
    print(f"  High (0.6–1.0):  {high_conf}/{total}  ({100*high_conf/total:.2f}%)  [Strong answers]")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Default to local evals file
    evals_path = Path(__file__).parent / "questions.jsonl"
    endpoint = "http://localhost:8000/ask"
    
    # Allow command-line overrides
    if len(sys.argv) > 1:
        evals_path = sys.argv[1]
    if len(sys.argv) > 2:
        endpoint = sys.argv[2]
    
    run_evals(str(evals_path), endpoint)
