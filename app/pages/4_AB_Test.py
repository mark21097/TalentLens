"""TalentLens — A/B Testing: Compare Prompt A vs Prompt B on the RAG chatbot."""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from talentlens.config import EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH

st.set_page_config(page_title="A/B Test | TalentLens", layout="wide")
st.title("🧪 A/B Test: Prompt Engineering")
st.markdown(
    "Compare two different system prompts on the same query. "
    "Which prompt produces better, more useful answers?"
)

# ── Prompt definitions ─────────────────────────────────────────────────────────
PROMPT_A_TEMPLATE = (
    "You are a helpful assistant answering questions about job postings.\n"
    "Use ONLY the provided job postings to answer. Cite job_ids.\n\n"
    "QUESTION:\n{question}\n\n"
    "CONTEXT:\n{context}\n\n"
    "ANSWER:\n"
)

PROMPT_B_TEMPLATE = (
    "You are TalentLens, an expert job market analyst with deep knowledge of hiring trends.\n"
    "Analyze the retrieved job postings and provide structured insights.\n"
    "Format your answer with:\n"
    "  1. Direct answer to the question\n"
    "  2. Key patterns observed across the postings\n"
    "  3. Notable outliers or exceptions\n"
    "Use specific data from the postings and cite job_ids.\n\n"
    "QUESTION:\n{question}\n\n"
    "RETRIEVED JOB POSTINGS:\n{context}\n\n"
    "STRUCTURED ANALYSIS:\n"
)

# ── Session state for experiment log ──────────────────────────────────────────
if "ab_results" not in st.session_state:
    st.session_state.ab_results = []

# ── UI ─────────────────────────────────────────────────────────────────────────
col_config, col_prompts = st.columns([1, 2])

with col_config:
    st.subheader("⚙️ Configuration")
    model_name = st.selectbox("Ollama model", ["llama3.1", "mistral", "llama3", "phi3"])
    k_results = st.slider("Retrieved jobs (k)", 3, 12, 6)
    question = st.text_area(
        "Test question:",
        value="What are the most common requirements for entry-level data scientist positions?",
        height=100,
    )

with col_prompts:
    st.subheader("📝 Prompts")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Prompt A** — Simple / Generic")
        st.code(PROMPT_A_TEMPLATE[:200] + "...", language="text")
    with col_b:
        st.markdown("**Prompt B** — Structured / Domain-Specialized")
        st.code(PROMPT_B_TEMPLATE[:200] + "...", language="text")

run_test = st.button("▶️ Run A/B Test", type="primary", disabled=not question.strip())

if run_test:
    if not FAISS_INDEX_PATH.exists():
        st.error("FAISS index not found. Run notebook 07 first.")
        st.stop()

    with st.spinner("Running both prompts in parallel..."):
        try:
            from sentence_transformers import SentenceTransformer
            embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            q_vec = embed_model.encode([question], convert_to_numpy=True)
            q_vec = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-9)

            from talentlens.rag import FaissRetriever, build_context
            retriever = FaissRetriever()
            docs = retriever.retrieve(q_vec[0], k=k_results)
            context = build_context(docs)

            from talentlens.rag import _load_langchain_ollama
            llm = _load_langchain_ollama(model=model_name)

            results = {}
            latencies = {}

            for label, template in [("A", PROMPT_A_TEMPLATE), ("B", PROMPT_B_TEMPLATE)]:
                prompt = template.format(question=question, context=context)
                t0 = time.time()
                try:
                    answer = llm.invoke(prompt)
                except Exception:
                    answer = llm(prompt)
                latencies[label] = time.time() - t0
                results[label] = str(answer)

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    # ── Display results side by side ───────────────────────────────────────────
    st.divider()
    col_res_a, col_res_b = st.columns(2)

    with col_res_a:
        st.subheader("🅰️ Prompt A")
        st.caption(f"Latency: {latencies['A']:.1f}s | Words: {len(results['A'].split())}")
        st.markdown(results["A"])

    with col_res_b:
        st.subheader("🅱️ Prompt B")
        st.caption(f"Latency: {latencies['B']:.1f}s | Words: {len(results['B'].split())}")
        st.markdown(results["B"])

    # ── Human rating ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("⭐ Rate the Answers")
    col_rate_a, col_rate_b = st.columns(2)

    with col_rate_a:
        rating_a = st.slider("Rate Prompt A (1=poor, 5=excellent)", 1, 5, 3, key="rating_a")
    with col_rate_b:
        rating_b = st.slider("Rate Prompt B (1=poor, 5=excellent)", 1, 5, 3, key="rating_b")

    preferred = st.radio("Which answer did you prefer overall?", ["A", "B", "Tie"], horizontal=True)

    if st.button("💾 Save Rating"):
        st.session_state.ab_results.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "model": model_name,
            "k": k_results,
            "rating_a": rating_a,
            "rating_b": rating_b,
            "preferred": preferred,
            "latency_a": round(latencies["A"], 2),
            "latency_b": round(latencies["B"], 2),
            "words_a": len(results["A"].split()),
            "words_b": len(results["B"].split()),
        })
        st.success("Rating saved!")

# ── Results log ───────────────────────────────────────────────────────────────
if st.session_state.ab_results:
    st.divider()
    st.subheader("📊 Experiment Results")
    results_df = pd.DataFrame(st.session_state.ab_results)
    st.dataframe(results_df, use_container_width=True)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Avg Rating A", f"{results_df['rating_a'].mean():.2f}/5")
    col_m2.metric("Avg Rating B", f"{results_df['rating_b'].mean():.2f}/5")
    winner = results_df["preferred"].value_counts().idxmax()
    col_m3.metric("Most Preferred", f"Prompt {winner}")

    st.download_button(
        "⬇️ Download Results CSV",
        data=results_df.to_csv(index=False),
        file_name="ab_test_results.csv",
        mime="text/csv",
    )
