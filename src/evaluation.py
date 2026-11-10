"""Simple evaluation for first metrics."""
import json
from pathlib import Path
from .ingestion import load_vectorstore
from .graph import run_query

def compute_recall_at_k(retrieved_sources, relevant_docs, k=5):
    retrieved = set(Path(s).name for s in retrieved_sources[:k])
    relevant = set(relevant_docs)
    if not relevant:
        return 0.0
    return len(retrieved & relevant) / len(relevant)

def simple_answer_score(answer: str, expected_keywords: list) -> float:
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords) if expected_keywords else 0.0

def run_evaluation(vectorstore_path="vectorstore", questions_path="eval/test_questions.json"):
    vs = load_vectorstore(vectorstore_path)
    with open(questions_path) as f:
        questions = json.load(f)

    results = []
    total_recall = 0
    total_correctness = 0
    n = len(questions)

    for q in questions:
        try:
            out = run_query(q["question"], vectorstore_path)
            recall = compute_recall_at_k(out.get("sources", []), q["relevant_docs"])
            correctness = simple_answer_score(out.get("final_answer", ""), q["expected_keywords"])
            results.append({
                "question": q["question"],
                "recall@5": round(recall, 2),
                "answer_score": round(correctness, 2),
                "sources": out.get("sources", []),
                "answer_preview": out.get("final_answer", "")[:150]
            })
            total_recall += recall
            total_correctness += correctness
        except Exception as e:
            results.append({"question": q["question"], "error": str(e)})

    summary = {
        "n_questions": n,
        "avg_recall@5": round(total_recall / n, 2) if n else 0,
        "avg_answer_correctness": round(total_correctness / n, 2) if n else 0,
        "details": results
    }
    return summary

if __name__ == "__main__":
    summary = run_evaluation()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
