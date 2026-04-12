"""TalentLens — Semantic Job Search powered by FAISS."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from talentlens.config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    POSTINGS_FEATURES_PARQUET,
    RETRIEVAL_META_PARQUET,
)

st.set_page_config(page_title="Job Search | TalentLens", layout="wide")
st.title("🔎 Semantic Job Search")
st.markdown(
    "Search 123K job postings by **meaning**, not just keywords. "
    "Powered by FAISS + sentence-transformers embeddings."
)


@st.cache_resource(show_spinner="Loading FAISS index...")
def load_retriever():
    from talentlens.vector_index import load_faiss_index
    if not FAISS_INDEX_PATH.exists():
        return None, None
    index = load_faiss_index(FAISS_INDEX_PATH)
    meta = pd.read_parquet(RETRIEVAL_META_PARQUET)
    return index, meta


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


index, meta = load_retriever()

if index is None:
    st.error(
        "FAISS index not found. Run **notebook 07** (`07-mp-faiss-vector-index.ipynb`) "
        "to build the index first."
    )
    st.stop()

# ── Search UI ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "Describe the job you're looking for:",
        placeholder="e.g. entry level data scientist Python machine learning",
        value="",
    )
with col2:
    k = st.slider("Results to show", min_value=5, max_value=30, value=10)

if st.button("🔍 Search", type="primary", disabled=not query.strip()):
    with st.spinner("Embedding query and searching..."):
        model = load_embed_model()
        query_vec = model.encode([query], convert_to_numpy=True)
        # Normalize for cosine similarity
        query_vec = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-9)

        from talentlens.vector_index import search_faiss
        results = search_faiss(index, query_vec[0], k=k, normalize_query=False)

    hits = meta.iloc[results.indices].copy().reset_index(drop=True)
    hits["score"] = results.scores

    st.subheader(f"Top {len(hits)} results for: *{query}*")

    for i, row in hits.iterrows():
        with st.expander(
            f"**{i+1}. {row.get('title', 'Unknown Title')}** — "
            f"{row.get('location', 'Unknown')} | Score: {row.get('score', 0):.3f}"
        ):
            desc = str(row.get("desc", ""))[:600]
            st.markdown(f"**Description snippet:**\n\n{desc}...")
            col_a, col_b = st.columns(2)
            col_a.write(f"**Job ID**: {row.get('job_id', 'N/A')}")
            col_b.write(f"**Similarity Score**: {row.get('score', 0):.4f}")

# ── How it works ──────────────────────────────────────────────────────────────
with st.expander("ℹ️ How does semantic search work?"):
    st.markdown("""
    1. Your query is embedded into a 384-dimensional vector using `all-MiniLM-L6-v2`
    2. FAISS performs approximate nearest-neighbor search across 123K job embeddings
    3. Results are ranked by **cosine similarity** — how close the meaning is

    Unlike keyword search, this finds jobs that *mean* the same thing even with different words.
    Searching "software engineer" also finds "developer", "programmer", "SWE" roles.
    """)
