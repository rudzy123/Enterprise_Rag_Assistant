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
from sentence_transformers import SentenceTransformer
import re
import os
import openai
from datetime import datetime
import time

def categorize_failure(question_obj: dict, response: dict, groundedness_result: dict) -> str:
    """
    Categorize the type of failure for groundedness < 70%.
    
    Categories:
    - hallucination: unsupported key factual claims
    - weak_retrieval: no relevant chunks retrieved
    - partial_context: missing critical expected sources
    - vague_question: question too broad or ambiguous
    """
    retrieved_chunks = response.get("retrieved_chunks", [])
    importance_breakdown = groundedness_result.get("importance_breakdown", {})
    retrieval_recall = response.get("retrieval_recall", 1.0)
    
    # Check for weak retrieval
    if not retrieved_chunks:
        return "weak_retrieval"
    
    # Check for hallucination (unsupported key claims)
    if importance_breakdown.get("key_unsupported", 0) > 0:
        return "hallucination"
    
    # Check for partial context (low recall)
    if retrieval_recall < 0.5:
        return "partial_context"
    
    # Default to vague question
    return "vague_question"

def classify_sentence_importance(sentences: list) -> list:
    """
    Use LLM to classify each sentence as 'key' (factual claim) or 'supporting' (detail).
    
    Returns list of 'key' or 'supporting' for each sentence.
    """
    if not sentences:
        return []
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # If no API key, treat all as key to avoid penalty
        return ['key'] * len(sentences)
    
    client = openai.OpenAI(api_key=api_key)
    
    # Format sentences for classification
    sentence_list = "\n".join(f"{i+1}. {sent}" for i, sent in enumerate(sentences))
    
    prompt = f"""Classify each sentence as either 'key' or 'supporting':

- 'key': Contains a key factual claim that must be accurate
- 'supporting': Provides context, explanation, or minor detail

Sentences:
{sentence_list}

Respond with only the classifications, one per line:
1. key/supporting
2. key/supporting
..."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1
        )
        classifications_text = response.choices[0].message.content.strip()
        
        # Parse the response
        classifications = []
        for line in classifications_text.split('\n'):
            line = line.strip()
            if 'key' in line.lower():
                classifications.append('key')
            elif 'supporting' in line.lower():
                classifications.append('supporting')
            else:
                classifications.append('supporting')  # default
        
        # Ensure we have the right number
        if len(classifications) != len(sentences):
            return ['supporting'] * len(sentences)  # fallback
        
        return classifications
    
    except Exception as e:
        print(f"Error classifying sentences: {e}", file=sys.stderr)
        return ['supporting'] * len(sentences)  # fallback

def extract_sentences(text: str) -> list:
    """Split text into sentences."""
    # Simple sentence splitter
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def compute_groundedness(answer: str, retrieved_chunks: list, model) -> dict:
    """
    Compute groundedness: how much of the answer is supported by retrieved chunks.
    
    Uses graded scoring with importance weighting:
    - Key claims: strong=1.0, weak=0.5, unsupported=-0.5 (penalty)
    - Supporting details: strong=1.0, weak=0.5, unsupported=0.0
    
    Returns:
        dict with groundedness_score (weighted), support_breakdown, unsupported_claims, importance_breakdown
    """
    if not retrieved_chunks:
        sentences = extract_sentences(answer)
        classifications = classify_sentence_importance(sentences)
        return {
            "groundedness_score": 0.0,
            "support_breakdown": {
                "strong_claims": 0,
                "weak_claims": 0,
                "unsupported_claims": len(sentences)
            },
            "unsupported_claims": sentences,
            "importance_breakdown": {
                "key_unsupported": sum(1 for c in classifications if c == 'key'),
                "supporting_unsupported": sum(1 for c in classifications if c == 'supporting')
            }
        }
    
    sentences = extract_sentences(answer)
    if not sentences:
        return {
            "groundedness_score": 100.0,
            "support_breakdown": {
                "strong_claims": 0,
                "weak_claims": 0,
                "unsupported_claims": 0
            },
            "unsupported_claims": [],
            "importance_breakdown": {
                "key_unsupported": 0,
                "supporting_unsupported": 0
            }
        }
    
    # Classify sentence importance
    classifications = classify_sentence_importance(sentences)
    
    # Embed chunks (cache embeddings)
    chunk_texts = [chunk["text"] for chunk in retrieved_chunks]
    chunk_embeddings = model.encode(chunk_texts)
    
    strong_count = 0
    weak_count = 0
    unsupported_claims = []
    key_unsupported = 0
    supporting_unsupported = 0
    
    total_weight = 0.0
    
    for sentence, importance in zip(sentences, classifications):
        if not sentence:
            continue
        sentence_emb = model.encode(sentence)
        similarities = model.similarity(sentence_emb, chunk_embeddings)
        max_sim = similarities.max().item()
        max_sim_idx = similarities.argmax().item()
        closest_chunk = chunk_texts[max_sim_idx] if chunk_texts else ""
        
        if max_sim >= 0.75:
            weight = 1.0
            strong_count += 1
        elif max_sim >= 0.6:
            weight = 0.5
            weak_count += 1
        else:
            if importance == 'key':
                weight = -0.5  # penalty for unsupported key claims
                key_unsupported += 1
            else:
                weight = 0.0
                supporting_unsupported += 1
            unsupported_claims.append({
                "claim": sentence,
                "importance": importance,
                "closest_chunk": closest_chunk,
                "similarity": max_sim
            })
        
        total_weight += weight
    
    # Normalize to 0-100 scale
    # Weights range from -0.5 to 1.0 per sentence, so total from -0.5*n to 1.0*n
    # Map to 0-100: score = (total_weight + 0.5 * len(sentences)) / (1.5 * len(sentences)) * 100
    max_possible = 1.0 * len(sentences)
    min_possible = -0.5 * len(sentences)
    range_size = max_possible - min_possible
    groundedness_score = ((total_weight - min_possible) / range_size) * 100.0 if range_size > 0 else 100.0
    
    return {
        "groundedness_score": groundedness_score,
        "support_breakdown": {
            "strong_claims": strong_count,
            "weak_claims": weak_count,
            "unsupported_claims": len(unsupported_claims)
        },
        "unsupported_claims": unsupported_claims,
        "importance_breakdown": {
            "key_unsupported": key_unsupported,
            "supporting_unsupported": supporting_unsupported
        }
    }
import re


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
    Call the /ask endpoint and return the response with latency and cost estimates.
    
    Returns:
        dict with keys: answer, sources, confidence, latency_ms, estimated_tokens
        Returns None if request fails
    """
    try:
        payload = {"question": question}
        start_time = time.time()
        response = requests.post(endpoint_url, json=payload, timeout=10)
        end_time = time.time()
        response.raise_for_status()
        
        result = response.json()
        
        # Calculate latency in milliseconds
        latency_ms = (end_time - start_time) * 1000
        
        # Estimate tokens (rough heuristic: ~4 chars per token for English)
        question_tokens = len(question) // 4
        answer_tokens = len(result.get("answer", "")) // 4
        estimated_tokens = question_tokens + answer_tokens
        
        # Add performance metrics to result
        result["latency_ms"] = latency_ms
        result["estimated_tokens"] = estimated_tokens
        
        return result
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
        question_obj: dict with keys: question, expected_answer (optional), source_doc_id
        response: dict with keys: answer, sources, confidence, retrieved_chunks
    
    Returns:
        dict with computed metrics
    """
    expected_answer = question_obj.get("expected_answer")
    expected_sources = question_obj.get("source_doc_id", [])
    if isinstance(expected_sources, str):
        expected_sources = [expected_sources]
    returned_sources = set(response.get("sources", []))
    confidence = response.get("confidence", 0.0)
    latency_ms = response.get("latency_ms", 0.0)
    estimated_tokens = response.get("estimated_tokens", 0)
    answer = response.get("answer", "")
    question = question_obj.get("question", "")
    retrieved_chunks = response.get("retrieved_chunks", [])
    
    # Initialize embedding model for similarities
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Metric 1: retrieval_hit
    # True if at least one expected source appears in returned sources
    if expected_sources:
        retrieval_hit = bool(set(expected_sources).intersection(returned_sources))
    else:
        retrieval_hit = True  # If no expected sources, consider it hit
    
    # Additional retrieval metrics
    expected_set = set(expected_sources) if expected_sources else set()
    retrieved_set = set(returned_sources)
    
    relevant_retrieved = expected_set & retrieved_set
    retrieval_precision = len(relevant_retrieved) / len(retrieved_set) if retrieved_set else 1.0
    retrieval_recall = len(relevant_retrieved) / len(expected_set) if expected_set else 1.0
    
    missing_critical = list(expected_set - retrieved_set)
    irrelevant_chunks = list(retrieved_set - expected_set)
    
    # Metric 2: groundedness (new: faithfulness to retrieved context)
    groundedness_result = compute_groundedness(answer, retrieved_chunks, model)
    groundedness_score = groundedness_result["groundedness_score"]
    support_breakdown = groundedness_result["support_breakdown"]
    unsupported_claims = groundedness_result["unsupported_claims"]
    importance_breakdown = groundedness_result["importance_breakdown"]
    groundedness = groundedness_score >= 70.0  # Pass if >=70% supported
    
    # Failure analysis for groundedness failures
    failure_type = None
    if not groundedness:
        failure_type = categorize_failure(question_obj, response, groundedness_result)
    
    # Metric 3: citation_presence
    # True if the answer mentions any of the returned sources
    citation_present = any(source in answer for source in returned_sources)
    
    # Metric 4: answer_relevance
    # Semantic similarity between question and answer
    question_emb = model.encode(question)
    answer_emb = model.encode(answer)
    relevance_score = model.similarity(question_emb, answer_emb).item()
    answer_relevance = relevance_score >= 0.5  # Threshold for relevant
    
    # Confidence range classification
    confidence_range = get_confidence_range(confidence)
    
    return {
        "retrieval_hit": retrieval_hit,
        "retrieval_precision": retrieval_precision,
        "retrieval_recall": retrieval_recall,
        "missing_critical": missing_critical,
        "irrelevant_chunks": irrelevant_chunks,
        "groundedness": groundedness,
        "groundedness_score": groundedness_score,
        "support_breakdown": support_breakdown,
        "unsupported_claims": unsupported_claims,
        "importance_breakdown": importance_breakdown,
        "failure_type": failure_type,
        "citation_present": citation_present,
        "answer_relevance": answer_relevance,
        "relevance_score": relevance_score,
        "confidence": confidence,
        "confidence_range": confidence_range,
        "returned_sources": list(returned_sources),
        "latency_ms": latency_ms,
        "estimated_tokens": estimated_tokens,
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
        question_text = question_obj.get("question", "")
        expected_answer = question_obj.get("expected_answer")
        expected_sources = question_obj.get("source_doc_id", [])
        if isinstance(expected_sources, str):
            expected_sources = [expected_sources]
        
        print(f"\nQuestion: {question_text}")
        if expected_answer:
            print(f"  Expected answer: {expected_answer[:100]}...")
        print(f"  Expected sources: {expected_sources if expected_sources else 'none'}")
        
        # Call endpoint
        response = call_ask_endpoint(question_text, endpoint_url)
        
        if response is None:
            print("  ❌ Failed to get response from endpoint")
            continue
        
        # Compute metrics
        metrics = compute_metrics(question_obj, response)
        
        # Display results
        print(f"  Confidence: {metrics['confidence']:.2f} ({metrics['confidence_range'].upper()})")
        print(f"  Returned sources: {metrics['returned_sources'] if metrics['returned_sources'] else 'none'}")
        print(f"  Answer: {response['answer'][:200]}...")
        print("  Metrics:")
        print(f"    - retrieval_hit: {'✓' if metrics['retrieval_hit'] else '✗'}")
        print(f"      Precision: {metrics['retrieval_precision']:.2f}, Recall: {metrics['retrieval_recall']:.2f}")
        if metrics['missing_critical']:
            print(f"      Missing critical: {metrics['missing_critical']}")
        if metrics['irrelevant_chunks']:
            print(f"      Irrelevant: {metrics['irrelevant_chunks']}")
        print(f"    - groundedness: {'✓' if metrics['groundedness'] else '✗'} ({metrics['groundedness_score']:.1f}%)")
        breakdown = metrics['support_breakdown']
        print(f"      Strong: {breakdown['strong_claims']}, Weak: {breakdown['weak_claims']}, Unsupported: {breakdown['unsupported_claims']}")
        imp_breakdown = metrics['importance_breakdown']
        print(f"      Key unsupported: {imp_breakdown['key_unsupported']}, Supporting unsupported: {imp_breakdown['supporting_unsupported']}")
        if metrics['unsupported_claims']:
            print("      Unsupported claims:")
            for claim_info in metrics['unsupported_claims']:
                print(f"        - \"{claim_info['claim'][:60]}...\" ({claim_info['importance']})")
                print(f"          Closest chunk: \"{claim_info['closest_chunk'][:60]}...\"")
                print(f"          Similarity: {claim_info['similarity']:.3f}")
        if metrics['failure_type']:
            print(f"      Failure type: {metrics['failure_type']}")
        print(f"    - citation_present: {'✓' if metrics['citation_present'] else '✗'}")
        print(f"    - answer_relevance: {'✓' if metrics['answer_relevance'] else '✗'} ({metrics['relevance_score']:.2f})")
        print(f"    - performance: {metrics['latency_ms']:.0f}ms, ~{metrics['estimated_tokens']} tokens")
        
        results.append({
            "question": question_text,
            "expected_answer": expected_answer,
            "expected_sources": expected_sources,
            "response": response,
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
    retrieval_precisions = [r["metrics"]["retrieval_precision"] for r in results]
    retrieval_recalls = [r["metrics"]["retrieval_recall"] for r in results]
    groundedness_count = sum(1 for r in results if r["metrics"]["groundedness"] is True)
    citation_present = sum(1 for r in results if r["metrics"]["citation_present"])
    answer_relevance_count = sum(1 for r in results if r["metrics"]["answer_relevance"])
    
    total = len(results)
    groundedness_total = sum(1 for r in results if r["metrics"]["groundedness"] is not None)
    
    print(f"\nTotal questions: {total}")
    retrieval_hit_rate = (100 * retrieval_hits / total) if total > 0 else 0.0
    avg_precision = sum(retrieval_precisions) / total if total > 0 else 0.0
    avg_recall = sum(retrieval_recalls) / total if total > 0 else 0.0
    groundedness_rate = (100 * groundedness_count / groundedness_total) if groundedness_total > 0 else 0.0
    citation_rate = (100 * citation_present / total) if total > 0 else 0.0
    relevance_rate = (100 * answer_relevance_count / total) if total > 0 else 0.0
    
    print(f"\nRetrieval Hit Rate: {retrieval_hit_rate:.2f}%")
    print(f"Average Retrieval Precision: {avg_precision:.2f}")
    print(f"Average Retrieval Recall: {avg_recall:.2f}")
    print(f"Groundedness Rate: {groundedness_rate:.2f}% ({groundedness_count}/{groundedness_total})")
    print(f"Citation Rate: {citation_rate:.2f}%")
    print(f"Answer Relevance Rate: {relevance_rate:.2f}%")
    
    avg_confidence = sum(r["metrics"]["confidence"] for r in results) / total if total > 0 else 0.0
    print(f"\nAverage Confidence: {avg_confidence:.2f}")
    
    # Simple summary printout
    hallucination_rate = sum(1 for r in results if r["metrics"]["failure_type"] == "hallucination") / groundedness_total if groundedness_total > 0 else 0.0
    
    print(f"\n" + "=" * 50)
    print("EXECUTIVE SUMMARY")
    print("=" * 50)
    print(f"Groundedness Pass Rate: {groundedness_rate:.1f}% ({groundedness_count}/{groundedness_total})")
    print(f"Average Retrieval Precision: {avg_precision:.2f}")
    print(f"Average Retrieval Recall: {avg_recall:.2f}")
    print(f"Hallucination Rate: {hallucination_rate:.1f}% ({int(hallucination_rate * groundedness_total)}/{groundedness_total})")
    print("=" * 50)
    
    # Confidence range distribution
    low_conf = sum(1 for r in results if r["metrics"]["confidence_range"] == "low")
    medium_conf = sum(1 for r in results if r["metrics"]["confidence_range"] == "medium")
    high_conf = sum(1 for r in results if r["metrics"]["confidence_range"] == "high")
    
    print(f"\nConfidence Range Distribution:")
    print(f"  Low  (0.0–0.3):  {low_conf}/{total}  ({100*low_conf/total:.2f}%)  [Refusal cases]")
    print(f"  Medium (0.3–0.6): {medium_conf}/{total}  ({100*medium_conf/total:.2f}%)  [Cautious answers]")
    print(f"  High (0.6–1.0):  {high_conf}/{total}  ({100*high_conf/total:.2f}%)  [Strong answers]")
    
    # Failure type analysis
    failure_types = {}
    for r in results:
        failure_type = r["metrics"]["failure_type"]
        if failure_type:
            failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
    
    if failure_types:
        print(f"\nFailure Type Analysis (for groundedness < 70%):")
        for failure_type, count in sorted(failure_types.items()):
            print(f"  {failure_type}: {count} cases")
    
    # Save results to JSON with timestamp and latest
    timestamp = datetime.now().strftime("%Y%m%d")
    timestamped_path = Path(__file__).parent / f"results_{timestamp}.json"
    latest_path = Path(__file__).parent / "results.json"
    
    with open(timestamped_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Also save as latest
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {timestamped_path}")
    print(f"Latest results also saved to {latest_path}")
    
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
