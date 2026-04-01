"""FAISS-based vector index utilities.

This module keeps FAISS usage optional at import-time (so the package can still
import even if FAISS isn't installed yet). Functions will raise a clear error
when FAISS is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import faiss  # type: ignore


def _require_faiss():
    try:
        import faiss  # type: ignore

        return faiss
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "FAISS is required for vector indexing. Install `faiss-cpu` "
            "(or `faiss-gpu`) and retry."
        ) from e


def _as_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.float32:
        return x.astype(np.float32, copy=False)
    return x


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization (for cosine similarity via inner product)."""
    x = _as_float32(np.asarray(x))
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


@dataclass(frozen=True)
class FaissSearchResult:
    indices: np.ndarray  # (k,) int64 indices into the original embedding matrix
    scores: np.ndarray  # (k,) float32 similarity scores (inner product)


def build_faiss_index(
    embeddings: np.ndarray,
    *,
    normalize: bool = True,
) -> "faiss.Index":
    """Build a FAISS IndexFlatIP for cosine similarity (via normalized vectors)."""
    faiss = _require_faiss()

    x = _as_float32(np.asarray(embeddings))
    if x.ndim != 2:
        raise ValueError(f"Expected embeddings with shape (n, d). Got {x.shape}.")

    if normalize:
        x = l2_normalize_rows(x)

    dim = x.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(x)
    return index


def save_faiss_index(index: "faiss.Index", path: Path) -> Path:
    faiss = _require_faiss()
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    return path


def load_faiss_index(path: Path) -> "faiss.Index":
    faiss = _require_faiss()
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found: {path}")
    return faiss.read_index(str(path))


def search_faiss(
    index: "faiss.Index",
    query: np.ndarray,
    *,
    k: int = 10,
    normalize_query: bool = True,
) -> FaissSearchResult:
    """Search the index with a single query vector."""
    q = _as_float32(np.asarray(query)).reshape(1, -1)
    if normalize_query:
        q = l2_normalize_rows(q)

    scores, indices = index.search(q, k)
    return FaissSearchResult(indices=indices[0].astype(np.int64), scores=scores[0])


def build_retrieval_meta(
    *,
    job_ids: Iterable[int],
    titles: Iterable[str],
    locations: Iterable[str],
    desc_clean: Optional[Iterable[str]] = None,
    max_desc_chars: int = 280,
) -> "np.ndarray":
    """Create a compact structured array for display in notebooks.

    This returns a NumPy structured array that is easy to convert to a DataFrame
    without pulling in the entire original dataset.
    """
    job_ids = np.asarray(list(job_ids))
    titles = np.asarray(list(titles), dtype=object)
    locations = np.asarray(list(locations), dtype=object)

    if desc_clean is None:
        desc = np.asarray([""] * len(job_ids), dtype=object)
    else:
        desc = np.asarray(list(desc_clean), dtype=object)
        desc = np.asarray(
            [d[:max_desc_chars] if isinstance(d, str) else "" for d in desc], dtype=object
        )

    if not (len(job_ids) == len(titles) == len(locations) == len(desc)):
        raise ValueError("All meta fields must be the same length.")

    meta = np.zeros(
        len(job_ids), dtype=[("job_id", "i8"), ("title", "O"), ("location", "O"), ("desc", "O")]
    )
    meta["job_id"] = job_ids
    meta["title"] = titles
    meta["location"] = locations
    meta["desc"] = desc
    return meta
