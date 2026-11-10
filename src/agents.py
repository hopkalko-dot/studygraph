"""Multi-agent components for StudyGraph."""
from typing import TypedDict, List, Annotated
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

# State
class AgentState(TypedDict):
    question: str
    plan: str
    retrieved_chunks: List[str]
    draft_answer: str
    critique: str
    final_answer: str
    sources: List[str]
    memory_context: str
    iteration: int

def get_llm(temperature=0.0):
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY or LLM_API_KEY in environment")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url
    )

def planner_agent(state: AgentState) -> AgentState:
    llm = get_llm()
    prompt = f"""Ты Planner Agent. Проанализируй вопрос студента и составь краткий план поиска.
Вопрос: {state['question']}
Контекст памяти: {state.get('memory_context', 'нет')}
Верни только план (1-3 пункта)."""
    resp = llm.invoke([HumanMessage(content=prompt)])
    state["plan"] = resp.content
    return state

def retriever_agent(state: AgentState, vectorstore) -> AgentState:
    query = state["question"]
    if state.get("plan"):
        query = f"{state['question']} {state['plan']}"
    docs = vectorstore.similarity_search(query, k=5)
    chunks = [d.page_content for d in docs]
    sources = [d.metadata.get("source", "unknown") for d in docs]
    state["retrieved_chunks"] = chunks
    state["sources"] = list(set(sources))
    return state

def synthesizer_agent(state: AgentState) -> AgentState:
    llm = get_llm()
    context = "\n\n".join(state.get("retrieved_chunks", []))
    prompt = f"""Ты Synthesizer Agent. На основе найденных фрагментов составь точный ответ.
Вопрос: {state['question']}
Фрагменты:
{context}
Отвечай только на основе фрагментов. Указывай, откуда информация."""
    resp = llm.invoke([HumanMessage(content=prompt)])
    state["draft_answer"] = resp.content
    return state

def critic_agent(state: AgentState) -> AgentState:
    llm = get_llm()
    context = "\n\n".join(state.get("retrieved_chunks", []))
    prompt = f"""Ты Critic / Verifier Agent. Проверь черновик ответа.
Вопрос: {state['question']}
Черновик: {state.get('draft_answer', '')}
Контекст:
{context}
Оцени: есть ли unsupported claims? Подтверждены ли утверждения источниками?
Если всё хорошо — напиши "OK". Если нет — укажи проблемы кратко."""
    resp = llm.invoke([HumanMessage(content=prompt)])
    state["critique"] = resp.content
    if "OK" in resp.content.upper() or "хорошо" in resp.content.lower():
        state["final_answer"] = state["draft_answer"]
    else:
        state["final_answer"] = state["draft_answer"] + "\n\n[Требует доработки: " + resp.content[:200] + "]"
    return state

def memory_agent(state: AgentState, memory_store: dict) -> AgentState:
    # Simple memory: accumulate questions
    q = state["question"]
    if "history" not in memory_store:
        memory_store["history"] = []
    memory_store["history"].append(q)
    # Simple weak topics detection (keyword based for demo)
    weak = []
    if any(w in q.lower() for w in ["bfs", "dfs", "граф"]):
        weak.append("графы и поиск")
    state["memory_context"] = f"История: {memory_store['history'][-5:]}; Слабые темы: {weak}"
    return state
