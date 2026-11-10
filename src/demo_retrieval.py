"""Demo retrieval without LLM (for quick check)."""
from ingestion import load_vectorstore

def demo():
    vs = load_vectorstore("../vectorstore")
    q = "Чем BFS отличается от DFS?"
    docs = vs.similarity_search(q, k=3)
    print(f"Query: {q}\n")
    for i, d in enumerate(docs, 1):
        print(f"{i}. {d.metadata.get('source', '?')}")
        print(d.page_content[:200])
        print("---")

if __name__ == "__main__":
    demo()
