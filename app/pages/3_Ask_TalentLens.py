"""TalentLens — RAG Chatbot (requires Ollama running locally)."""

from __future__ import annotations

import numpy as np
import streamlit as st

from talentlens.config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH, RETRIEVAL_META_PARQUET

st.set_page_config(page_title="Ask TalentLens | TalentLens", layout="wide")
st.title("🤖 Ask TalentLens")
st.markdown(
    "Ask questions about job postings. TalentLens retrieves relevant jobs using FAISS, "
    "then answers using a local LLM via Ollama."
)

# ── Ollama status check ────────────────────────────────────────────────────────
with st.expander("⚙️ Setup Requirements", expanded=False):
    st.markdown("""
    This page requires **Ollama** running locally with a model pulled:
    ```bash
    # Install: https://ollama.com
    ollama pull llama3.1
    ollama serve
    ```
    The FAISS index must also be built (run notebook 07).
    """)

# ── Model selector ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    question = st.text_area(
        "Ask a question about the job market:",
        placeholder=(
            "e.g. What are the most common requirements for entry-level data scientist roles?\n"
            "e.g. What skills are needed for remote software engineering jobs?"
        ),
        height=100,
    )
with col2:
    model_name = st.selectbox("Ollama model", ["llama3.1", "mistral", "llama3", "phi3"])
    k_results = st.slider("Retrieved jobs (k)", 3, 15, 8)

if st.button("🤖 Ask", type="primary", disabled=not question.strip()):
    if not FAISS_INDEX_PATH.exists():
        st.error("FAISS index not found. Run notebook 07 first.")
        st.stop()

    with st.spinner("Embedding question and retrieving jobs..."):
        try:
            from sentence_transformers import SentenceTransformer
            embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            q_vec = embed_model.encode([question], convert_to_numpy=True)
            q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-9)

            from talentlens.rag import FaissRetriever, answer_question
            retriever = FaissRetriever()
            result = answer_question(
                question=question,
                query_embedding=q_vec[0],
                retriever=retriever,
                k=k_results,
                model=model_name,
            )

            st.subheader("Answer")
            st.markdown(result["answer"])

            with st.expander(f"📄 Retrieved Jobs ({len(result['docs'])} results)"):
                for doc in result["docs"]:
                    st.markdown(
                        f"**{doc['title']}** — {doc['location']} "
                        f"*(score: {doc['score']:.3f}, job_id: {doc['job_id']})*"
                    )
                    st.markdown(f">{doc['content'][:300]}...")
                    st.divider()

        except ImportError as e:
            st.error(f"Missing dependency: {e}. Install langchain-ollama or langchain-community.")
        except Exception as e:
            if "connection refused" in str(e).lower() or "ollama" in str(e).lower():
                st.error(
                    "Cannot connect to Ollama. Make sure it's running: `ollama serve`\n\n"
                    f"Error: {e}"
                )
            else:
                st.error(f"Error: {e}")
