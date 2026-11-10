"""LangGraph workflow for StudyGraph multi-agent system."""
from langgraph.graph import StateGraph, END
from agents import AgentState, planner_agent, retriever_agent, synthesizer_agent, critic_agent, memory_agent
from ingestion import load_vectorstore

memory_store = {}

def build_graph(vectorstore):
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", lambda s: planner_agent(s))
    workflow.add_node("retriever", lambda s: retriever_agent(s, vectorstore))
    workflow.add_node("synthesizer", synthesizer_agent)
    workflow.add_node("critic", critic_agent)
    workflow.add_node("memory", lambda s: memory_agent(s, memory_store))

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "synthesizer")
    workflow.add_edge("synthesizer", "critic")
    workflow.add_edge("critic", "memory")
    workflow.add_edge("memory", END)

    return workflow.compile()

def run_query(question: str, vectorstore_path: str = "vectorstore"):
    vs = load_vectorstore(vectorstore_path)
    graph = build_graph(vs)
    initial = {
        "question": question,
        "plan": "",
        "retrieved_chunks": [],
        "draft_answer": "",
        "critique": "",
        "final_answer": "",
        "sources": [],
        "memory_context": "",
        "iteration": 0
    }
    result = graph.invoke(initial)
    return result
