"""One-command build of the vector store."""
from src.ingestion import load_documents, chunk_documents, create_vectorstore

if __name__ == "__main__":
    print("Loading documents from data/ ...")
    docs = load_documents("data")
    print(f"Loaded {len(docs)} documents")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")
    create_vectorstore(chunks, "vectorstore")
    print("Vector store saved to vectorstore/")
