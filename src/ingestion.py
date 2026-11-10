"""Document ingestion: load, chunk, embed, store in FAISS."""
import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_documents(data_dir: str) -> List:
    docs = []
    data_path = Path(data_dir)
    for file in data_path.glob("*.md"):
        loader = TextLoader(str(file), encoding="utf-8")
        docs.extend(loader.load())
    for file in data_path.glob("*.txt"):
        loader = TextLoader(str(file), encoding="utf-8")
        docs.extend(loader.load())
    return docs

def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(documents)

def create_vectorstore(chunks, persist_path: str = "vectorstore"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(persist_path)
    return vectorstore

def load_vectorstore(persist_path: str = "vectorstore"):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)

if __name__ == "__main__":
    docs = load_documents("data")
    print(f"Loaded {len(docs)} documents")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")
    vs = create_vectorstore(chunks, "vectorstore")
    print("Vector store saved")
