"""Postgres + pgvector helpers for TalentLens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from loguru import logger
import numpy as np
from sqlalchemy import Engine, create_engine, text

from talentlens.config import DATABASE_URL, EMBEDDING_DIM, PGVECTOR_TABLE


def get_engine(database_url: str | None = None) -> Engine:
    url = (database_url or DATABASE_URL).strip()
    if not url:
        raise ValueError(
            "DATABASE_URL is not set. Set it in your environment or .env, e.g. "
            "postgresql+psycopg://user:password@localhost:5432/talentlens"
        )
    return create_engine(url, pool_pre_ping=True)


def init_pgvector(engine: Engine) -> None:
    """Enable pgvector and create the postings table (id + embedding + minimal metadata)."""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {PGVECTOR_TABLE} (
                    job_id BIGINT PRIMARY KEY,
                    title TEXT,
                    location TEXT,
                    desc_clean TEXT,
                    embedding vector({EMBEDDING_DIM})
                );
                """
            )
        )

        # IVFFlat index is common; requires setting lists and ANALYZE, and works best at scale.
        # We keep it optional; create if it doesn't exist.
        conn.execute(
            text(
                f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM pg_indexes
                        WHERE tablename = '{PGVECTOR_TABLE}'
                          AND indexname = '{PGVECTOR_TABLE}_embedding_ivfflat_idx'
                    ) THEN
                        EXECUTE 'CREATE INDEX {PGVECTOR_TABLE}_embedding_ivfflat_idx '
                             'ON {PGVECTOR_TABLE} USING ivfflat (embedding vector_cosine_ops) '
                             'WITH (lists = 100);';
                    END IF;
                END
                $$;
                """
            )
        )

    logger.info("pgvector initialized.")


def _to_pgvector_literal(vec: np.ndarray) -> str:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    return "[" + ",".join(f"{x:.7g}" for x in v.tolist()) + "]"


def upsert_postings_embeddings(
    engine: Engine,
    *,
    job_ids: Iterable[int],
    embeddings: np.ndarray,
    titles: Optional[Iterable[str]] = None,
    locations: Optional[Iterable[str]] = None,
    desc_clean: Optional[Iterable[str]] = None,
    batch_size: int = 1000,
) -> int:
    """Upsert embeddings and minimal fields into pgvector."""
    job_ids_arr = np.asarray(list(job_ids), dtype=np.int64)
    x = np.asarray(embeddings, dtype=np.float32)

    if x.ndim != 2 or x.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Expected embeddings (n, {EMBEDDING_DIM}). Got {x.shape}.")
    if len(job_ids_arr) != len(x):
        raise ValueError("job_ids and embeddings must have the same length.")

    def _arr_or_default(it: Optional[Iterable[str]]) -> np.ndarray:
        if it is None:
            return np.asarray([""] * len(job_ids_arr), dtype=object)
        return np.asarray(list(it), dtype=object)

    titles_arr = _arr_or_default(titles)
    locations_arr = _arr_or_default(locations)
    desc_arr = _arr_or_default(desc_clean)

    total = 0
    with engine.begin() as conn:
        for start in range(0, len(job_ids_arr), batch_size):
            end = min(start + batch_size, len(job_ids_arr))
            rows = []
            for i in range(start, end):
                rows.append(
                    {
                        "job_id": int(job_ids_arr[i]),
                        "title": str(titles_arr[i]) if titles_arr[i] is not None else "",
                        "location": str(locations_arr[i]) if locations_arr[i] is not None else "",
                        "desc_clean": str(desc_arr[i]) if desc_arr[i] is not None else "",
                        "embedding": _to_pgvector_literal(x[i]),
                    }
                )

            conn.execute(
                text(
                    f"""
                    INSERT INTO {PGVECTOR_TABLE} (job_id, title, location, desc_clean, embedding)
                    VALUES (:job_id, :title, :location, :desc_clean, :embedding)
                    ON CONFLICT (job_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        location = EXCLUDED.location,
                        desc_clean = EXCLUDED.desc_clean,
                        embedding = EXCLUDED.embedding;
                    """
                ),
                rows,
            )
            total += len(rows)

    logger.info(f"Upserted {total:,} rows into {PGVECTOR_TABLE}.")
    return total


@dataclass(frozen=True)
class PgvectorHit:
    job_id: int
    score: float
    title: str
    location: str
    desc_clean: str


def query_similar_postings(
    engine: Engine,
    *,
    query_embedding: np.ndarray,
    k: int = 10,
) -> list[PgvectorHit]:
    """Return top-k by cosine similarity using pgvector cosine distance operator."""
    qlit = _to_pgvector_literal(query_embedding)
    with engine.begin() as conn:
        # cosine distance: embedding <=> query ; similarity = 1 - distance
        rows = (
            conn.execute(
                text(
                    f"""
                SELECT
                    job_id,
                    title,
                    location,
                    desc_clean,
                    1 - (embedding <=> :q) AS score
                FROM {PGVECTOR_TABLE}
                ORDER BY embedding <=> :q
                LIMIT :k;
                """
                ),
                {"q": qlit, "k": int(k)},
            )
            .mappings()
            .all()
        )

    hits = [
        PgvectorHit(
            job_id=int(r["job_id"]),
            score=float(r["score"]),
            title=r.get("title") or "",
            location=r.get("location") or "",
            desc_clean=r.get("desc_clean") or "",
        )
        for r in rows
    ]
    return hits


def count_rows(engine: Engine) -> int:
    with engine.begin() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) AS n FROM {PGVECTOR_TABLE};")).scalar_one()
    return int(n)
