"""NLP feature engineering utilities for TalentLens.

This module provides reusable functions for text preprocessing,
feature extraction, sentiment analysis, and embedding generation.

Usage:
    from talentlens.features import clean_text, lemmatize_texts, generate_embeddings
"""

from __future__ import annotations

from html import unescape
import re

from loguru import logger
import numpy as np
import pandas as pd
from textblob import TextBlob
from tqdm import tqdm

from talentlens.config import EMBEDDING_MODEL_NAME

# ── Text Cleaning ────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """Strip HTML tags, normalize whitespace, and lowercase.

    Args:
        text: Raw text string (may contain HTML).

    Returns:
        Cleaned, lowercased text with normalized whitespace.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Decode HTML entities (&amp; → &, etc.)
    text = unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Normalize whitespace (tabs, newlines, multiple spaces → single space)
    text = re.sub(r"\s+", " ", text).strip()

    # Lowercase
    text = text.lower()

    return text


def add_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add word count and character count columns for description and title.

    Args:
        df: DataFrame with 'description' and 'title' columns.

    Returns:
        DataFrame with added columns: desc_word_count, desc_char_count, title_word_count.
    """
    df = df.copy()

    # Description stats
    df["desc_word_count"] = df["description"].fillna("").str.split().str.len()
    df["desc_char_count"] = df["description"].fillna("").str.len()

    # Title stats
    df["title_word_count"] = df["title"].fillna("").str.split().str.len()

    logger.info(
        f"Text stats: median desc words = {df['desc_word_count'].median():.0f}, "
        f"median title words = {df['title_word_count'].median():.0f}"
    )

    return df


# ── spaCy Processing ─────────────────────────────────────────────


def lemmatize_texts(
    texts: pd.Series,
    batch_size: int = 500,
    n_process: int = 1,
    max_chars: int = 100_000,
) -> pd.Series:
    """Batch lemmatize texts using spaCy, removing stopwords and punctuation.

    Uses spaCy's efficient nlp.pipe() for batch processing instead of
    processing one text at a time (which would be extremely slow on 123K+ texts).

    Args:
        texts: Series of text strings to lemmatize.
        batch_size: Number of texts to process per batch (keep <=500 to avoid MemoryError).
        n_process: Number of parallel processes (1 = single-threaded, safest).
        max_chars: Truncate texts longer than this to prevent memory issues.

    Returns:
        Series of lemmatized text strings.
    """
    import spacy  # lazy import — avoid triggering torch DLL chain at module level

    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    # We only need the tokenizer and lemmatizer — disabling NER and parser
    # makes it ~3x faster since we don't need entity recognition or
    # dependency parsing for lemmatization.
    nlp.max_length = max_chars + 1000  # Allow slightly over to avoid hard failures

    logger.info(f"Lemmatizing {len(texts):,} texts with spaCy (batch_size={batch_size})...")

    results = []
    # Truncate very long texts to prevent memory blowup
    text_list = []
    for t in texts.fillna("").tolist():
        if len(t) > max_chars:
            text_list.append(t[:max_chars])
        else:
            text_list.append(t)

    for doc in tqdm(
        nlp.pipe(text_list, batch_size=batch_size, n_process=n_process),
        total=len(text_list),
        desc="Lemmatizing",
    ):
        # Keep only alphabetic tokens that aren't stopwords
        tokens = []
        for token in doc:
            if token.is_alpha and not token.is_stop:
                tokens.append(token.lemma_.lower())
        results.append(" ".join(tokens))

    logger.info("Lemmatization complete.")
    return pd.Series(results, index=texts.index)


# ── Sentiment Analysis ───────────────────────────────────────────


def add_sentiment(
    df: pd.DataFrame,
    text_col: str = "description",
) -> pd.DataFrame:
    """Add sentiment polarity and subjectivity columns using TextBlob.

    Polarity: -1 (negative) to +1 (positive)
    Subjectivity: 0 (objective/factual) to 1 (subjective/opinion)

    Args:
        df: DataFrame with a text column.
        text_col: Name of the column to analyze.

    Returns:
        DataFrame with added polarity and subjectivity columns.
    """
    df = df.copy()

    logger.info(f"Computing sentiment for {len(df):,} texts...")

    polarities = []
    subjectivities = []

    for text in tqdm(df[text_col].fillna(""), desc="Sentiment analysis"):
        blob = TextBlob(text)
        polarities.append(blob.sentiment.polarity)
        subjectivities.append(blob.sentiment.subjectivity)

    df["sentiment_polarity"] = polarities
    df["sentiment_subjectivity"] = subjectivities

    logger.info(
        f"Sentiment: mean polarity = {df['sentiment_polarity'].mean():.3f}, "
        f"mean subjectivity = {df['sentiment_subjectivity'].mean():.3f}"
    )

    return df


# ── Entry-Level Paradox (Theme 3) ────────────────────────────────


# Keywords that signal senior-level expectations
SENIOR_SIGNAL_PATTERNS = [
    r"\b\d{1,2}\+?\s*years?\b(?!\s*(?:old|salary|yearly|annual|wage))",  # "5+ years" but NOT "75000 yearly"
    r"\bsenior\b",  # "senior"
    r"\blead\b",  # "lead"
    r"\bmanage[rd]?\b",  # "manager", "managed", "manage"
    r"\bexpert\b",  # "expert"
    r"\badvanced\b",  # "advanced"
    r"\bextensive experience\b",  # "extensive experience"
    r"\bproven track record\b",  # "proven track record"
    r"\bstrategic\b",  # "strategic"
]

# Max realistic years of experience for any job (safety cap)
MAX_YEARS_CAP = 30


def detect_senior_signals(text: str) -> dict:
    """Count senior-level keywords in a job description.

    This supports the Entry-Level Paradox research theme: do jobs labeled
    "entry-level" actually demand senior skills?

    Args:
        text: Job description text (should be lowercased).

    Returns:
        Dict with 'total_signals' count and 'years_mentioned' (max years found).
    """
    if not isinstance(text, str) or not text.strip():
        return {"total_signals": 0, "years_mentioned": 0}

    text_lower = text.lower()
    total = 0

    for pattern in SENIOR_SIGNAL_PATTERNS:
        matches = re.findall(pattern, text_lower)
        total += len(matches)

    # Extract the maximum years mentioned (e.g., "5+ years" → 5)
    # Only match 1-2 digit numbers, exclude "yearly/salary/annual" (salary false positives)
    year_matches = re.findall(
        r"\b(\d{1,2})\+?\s*years?\b(?!\s*(?:old|salary|yearly|annual|wage))",
        text_lower,
    )
    max_years = 0
    if year_matches:
        for y in year_matches:
            val = int(y)
            if val > MAX_YEARS_CAP:
                continue  # Skip unrealistic values
            if val > max_years:
                max_years = val

    return {"total_signals": total, "years_mentioned": max_years}


def add_senior_signals(df: pd.DataFrame, text_col: str = "description") -> pd.DataFrame:
    """Add senior signal columns to the DataFrame.

    Args:
        df: DataFrame with a text column.
        text_col: Name of the column to analyze.

    Returns:
        DataFrame with added senior_signal_count and max_years_required columns.
    """
    df = df.copy()

    logger.info(f"Detecting senior signals in {len(df):,} descriptions...")

    signal_counts = []
    years_required = []

    for text in tqdm(df[text_col].fillna(""), desc="Senior signals"):
        result = detect_senior_signals(text)
        signal_counts.append(result["total_signals"])
        years_required.append(result["years_mentioned"])

    df["senior_signal_count"] = signal_counts
    df["max_years_required"] = years_required

    logger.info(
        f"Senior signals: mean count = {df['senior_signal_count'].mean():.1f}, "
        f"postings mentioning years = {(df['max_years_required'] > 0).sum():,}"
    )

    return df


# ── Embeddings ───────────────────────────────────────────────────


def generate_embeddings(
    texts: list[str],
    model_name: str = EMBEDDING_MODEL_NAME,
    batch_size: int = 256,
    show_progress: bool = True,
) -> np.ndarray:
    """Generate sentence-transformer embeddings in batches.

    Uses all-MiniLM-L6-v2 by default (384 dimensions, fast, free).
    Processes in batches with a progress bar since 123K+ texts takes time.

    Args:
        texts: List of text strings to embed.
        model_name: HuggingFace model name for sentence-transformers.
        batch_size: Number of texts per batch.
        show_progress: Whether to show a progress bar.

    Returns:
        NumPy array of shape (n_texts, embedding_dim).
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"Generating embeddings for {len(texts):,} texts (batch_size={batch_size})...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )

    logger.info(f"Embeddings shape: {embeddings.shape} (dtype: {embeddings.dtype})")
    return embeddings
