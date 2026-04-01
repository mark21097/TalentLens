"""RAG utilities (retrieval + Ollama/LangChain generation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np
import pandas as pd

from talentlens.config import (
    DATABASE_URL,
    EMBEDDINGS_NPY,
    FAISS_INDEX_PATH,
    RETRIEVAL_META_PARQUET,
)
from talentlens.db import get_engine, query_similar_postings
from talentlens.vector_index import load_faiss_index, search_faiss


@dataclass(frozen=True)
class RetrievedDoc:
    job_id: int
    score: float
    title: str
    location: str
    content: str


class Retriever(Protocol):
    def retrieve(self, query_embedding: np.ndarray, k: int = 8) -> list[RetrievedDoc]: ...


class FaissRetriever:
    def __init__(
        self,
        *,
        faiss_index_path=FAISS_INDEX_PATH,
        retrieval_meta_path=RETRIEVAL_META_PARQUET,
    ) -> None:
        self.index = load_faiss_index(faiss_index_path)
        if not retrieval_meta_path.exists():
            raise FileNotFoundError(
                f"Retrieval meta not found: {retrieval_meta_path}. Run notebook 07 first."
            )
        self.meta = pd.read_parquet(retrieval_meta_path)

    def retrieve(self, query_embedding: np.ndarray, k: int = 8) -> list[RetrievedDoc]:
        res = search_faiss(self.index, query_embedding, k=k, normalize_query=True)
        hits = self.meta.iloc[res.indices].reset_index(drop=True)
        out: list[RetrievedDoc] = []
        for rank, row in hits.iterrows():
            out.append(
                RetrievedDoc(
                    job_id=int(row["job_id"]),
                    score=float(res.scores[int(rank)]),
                    title=str(row.get("title", "")),
                    location=str(row.get("location", "")),
                    content=str(row.get("desc", "")),
                )
            )
        return out


class PgvectorRetriever:
    def __init__(self, *, database_url: str | None = None) -> None:
        # Table name is configured globally (PGVECTOR_TABLE), but engine is per-url.
        self.engine = get_engine(database_url or DATABASE_URL)

    def retrieve(self, query_embedding: np.ndarray, k: int = 8) -> list[RetrievedDoc]:
        hits = query_similar_postings(self.engine, query_embedding=query_embedding, k=k)
        return [
            RetrievedDoc(
                job_id=h.job_id,
                score=h.score,
                title=h.title,
                location=h.location,
                content=(h.desc_clean or "")[:600],
            )
            for h in hits
        ]


def _load_langchain_ollama(model: str):
    # LangChain APIs have moved around; support the common options.
    try:
        from langchain_ollama import OllamaLLM  # type: ignore

        return OllamaLLM(model=model)
    except Exception:
        pass

    try:
        from langchain_community.llms import Ollama  # type: ignore

        return Ollama(model=model)
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import an Ollama LLM from LangChain. Install `langchain`, "
            "`langchain-community`, and/or `langchain-ollama`."
        ) from e


def build_context(docs: Iterable[RetrievedDoc]) -> str:
    chunks = []
    for d in docs:
        chunks.append(
            "\n".join(
                [
                    f"job_id: {d.job_id}",
                    f"title: {d.title}",
                    f"location: {d.location}",
                    f"score: {d.score:.3f}",
                    f"text: {d.content}",
                ]
            )
        )
    return "\n\n---\n\n".join(chunks)


def answer_question(
    *,
    question: str,
    query_embedding: np.ndarray,
    retriever: Retriever,
    k: int = 8,
    model: str = "llama3.1",
) -> dict:
    """Retrieve top-k docs and answer with an Ollama-backed LLM."""
    docs = retriever.retrieve(query_embedding, k=k)
    context = build_context(docs)

    llm = _load_langchain_ollama(model=model)

    prompt = (
        "You are TalentLens, an assistant answering questions about job postings.\n"
        "Use ONLY the provided context (retrieved job postings) to answer.\n"
        "If the context is insufficient, say what is missing.\n"
        "Cite job_ids you used.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "ANSWER:\n"
    )

    # LangChain LLMs generally support invoke(str) in newer versions; fall back to __call__.
    try:
        text = llm.invoke(prompt)  # type: ignore[attr-defined]
    except Exception:
        text = llm(prompt)  # type: ignore[operator]

    return {
        "question": question,
        "answer": str(text),
        "docs": [d.__dict__ for d in docs],
    }


def load_query_embedding_from_job(
    *,
    job_idx: int,
) -> np.ndarray:
    """Convenience helper for notebooks: use a job's embedding as the query vector."""
    embeddings = np.load(EMBEDDINGS_NPY)
    return embeddings[int(job_idx)]
