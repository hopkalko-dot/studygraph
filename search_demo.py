from src.ingestion import load_vectorstore

# Загружаем готовую векторную базу
vs = load_vectorstore("vectorstore")

# Вопрос
question = "Чем BFS отличается от DFS?"

print(f"Вопрос: {question}\n")
print("Найденные фрагменты:\n")

docs = vs.similarity_search(question, k=3)
for i, doc in enumerate(docs, 1):
    source = doc.metadata.get("source", "неизвестно")
    print(f"{i}. Источник: {source}")
    print(doc.page_content[:300])
    print("-" * 50)