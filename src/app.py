"""Streamlit interface for StudyGraph."""
import streamlit as st
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from ingestion import load_vectorstore
from graph import run_query

st.set_page_config(page_title="StudyGraph", layout="wide")
st.title("StudyGraph — Multi-Agent + Memory")
st.caption("Доказательный поиск по учебным материалам")

if "vectorstore" not in st.session_state:
    try:
        st.session_state.vectorstore = load_vectorstore("vectorstore")
        st.success("Векторная база загружена")
    except Exception as e:
        st.error(f"Сначала постройте vectorstore: python -m src.ingestion\n{e}")
        st.stop()

question = st.text_input("Задайте вопрос по материалам курса:")

if st.button("Ответить") and question:
    with st.spinner("Агенты работают..."):
        try:
            result = run_query(question)
            st.subheader("Ответ")
            st.write(result.get("final_answer", "Нет ответа"))
            st.subheader("Источники")
            for s in result.get("sources", []):
                st.write(f"- {s}")
            with st.expander("Детали агентов"):
                st.write("**План:**", result.get("plan"))
                st.write("**Критика:**", result.get("critique"))
                st.write("**Память:**", result.get("memory_context"))
        except Exception as e:
            st.error(str(e))
