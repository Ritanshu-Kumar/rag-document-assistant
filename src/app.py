"""
Streamlit UI for the RAG chatbot.

Run with:
    streamlit run src/app.py
"""
import streamlit as st
from rag import answer_question
from db import get_stats

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄")

st.title("📄 Document Q&A — RAG Chatbot")
st.caption("Ask questions about the documents you indexed in data/")

with st.sidebar:
    st.subheader("System stats")
    stats = get_stats()
    st.metric("Total queries logged", stats["total_queries"])
    if stats["avg_latency_ms"]:
        st.metric("Avg latency", f"{stats['avg_latency_ms']} ms")

question = st.text_input("Your question:")

if st.button("Ask") and question:
    with st.spinner("Retrieving context and generating answer..."):
        try:
            result = answer_question(question)
            st.markdown("### Answer")
            st.write(result["answer"])
            st.markdown("### Sources")
            st.write(", ".join(result["sources"]) or "None")
            st.caption(f"Retrieved {result['num_chunks_retrieved']} chunks in {result['latency_ms']} ms")
        except Exception as e:
            st.error(f"Error: {e}")
