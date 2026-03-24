"""Tests for NLP feature engineering utilities."""

import numpy as np
import pandas as pd
import pytest

from talentlens.features import (
    add_senior_signals,
    add_sentiment,
    add_text_stats,
    clean_text,
    detect_senior_signals,
    generate_embeddings,
    lemmatize_texts,
)


# ── TestCleanText ────────────────────────────────────────────────


class TestCleanText:
    """Test HTML stripping, whitespace normalization, lowercasing."""

    def test_strips_html_tags(self):
        result = clean_text("<p>Hello <b>world</b></p>")
        assert result == "hello world"

    def test_decodes_html_entities(self):
        result = clean_text("R&amp;D department")
        assert result == "r&d department"

    def test_removes_urls(self):
        result = clean_text("Apply at https://example.com/jobs today")
        assert result == "apply at today"

    def test_normalizes_whitespace(self):
        result = clean_text("hello   world\n\tnow")
        assert result == "hello world now"

    def test_lowercases(self):
        result = clean_text("SENIOR Software Engineer")
        assert result == "senior software engineer"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_none_input(self):
        assert clean_text(None) == ""

    def test_whitespace_only(self):
        assert clean_text("   ") == ""


# ── TestTextStats ────────────────────────────────────────────────


class TestTextStats:
    """Test word count and character count columns."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "description": ["hello world foo", "one two", None],
            "title": ["Software Engineer", "Data Scientist", "PM"],
        })

    def test_adds_desc_word_count(self, sample_df):
        result = add_text_stats(sample_df)
        assert "desc_word_count" in result.columns
        assert result["desc_word_count"].iloc[0] == 3
        assert result["desc_word_count"].iloc[1] == 2

    def test_adds_desc_char_count(self, sample_df):
        result = add_text_stats(sample_df)
        assert "desc_char_count" in result.columns
        assert result["desc_char_count"].iloc[0] == 15

    def test_adds_title_word_count(self, sample_df):
        result = add_text_stats(sample_df)
        assert "title_word_count" in result.columns
        assert result["title_word_count"].iloc[0] == 2

    def test_handles_null_descriptions(self, sample_df):
        result = add_text_stats(sample_df)
        # Null descriptions should get word count of 0
        assert result["desc_word_count"].iloc[2] == 0


# ── TestSentiment ────────────────────────────────────────────────


class TestSentiment:
    """Test sentiment analysis returns valid ranges."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "description": [
                "This is an amazing opportunity with great benefits!",
                "Terrible working conditions and low pay.",
                "The role involves data analysis and reporting.",
            ],
        })

    def test_polarity_range(self, sample_df):
        result = add_sentiment(sample_df)
        assert result["sentiment_polarity"].between(-1, 1).all()

    def test_subjectivity_range(self, sample_df):
        result = add_sentiment(sample_df)
        assert result["sentiment_subjectivity"].between(0, 1).all()

    def test_positive_text_has_positive_polarity(self, sample_df):
        result = add_sentiment(sample_df)
        assert result["sentiment_polarity"].iloc[0] > 0

    def test_adds_both_columns(self, sample_df):
        result = add_sentiment(sample_df)
        assert "sentiment_polarity" in result.columns
        assert "sentiment_subjectivity" in result.columns


# ── TestSeniorSignals ────────────────────────────────────────────


class TestSeniorSignals:
    """Test entry-level paradox keyword detection."""

    def test_detects_years(self):
        result = detect_senior_signals("Requires 5+ years of experience")
        assert result["total_signals"] >= 1
        assert result["years_mentioned"] == 5

    def test_detects_senior_keyword(self):
        result = detect_senior_signals("Looking for a senior developer with lead experience")
        assert result["total_signals"] >= 2  # "senior" + "lead"

    def test_empty_text(self):
        result = detect_senior_signals("")
        assert result["total_signals"] == 0
        assert result["years_mentioned"] == 0

    def test_no_signals(self):
        result = detect_senior_signals("Entry level position, no experience required")
        assert result["years_mentioned"] == 0

    def test_max_years_picks_highest(self):
        result = detect_senior_signals("3 years Python, 5 years Java, 2 years SQL")
        assert result["years_mentioned"] == 5

    def test_add_senior_signals_columns(self):
        df = pd.DataFrame({
            "description": ["Need 5+ years senior experience", "Junior role"],
        })
        result = add_senior_signals(df)
        assert "senior_signal_count" in result.columns
        assert "max_years_required" in result.columns
        assert result["senior_signal_count"].iloc[0] > result["senior_signal_count"].iloc[1]

    def test_ignores_salary_yearly(self):
        """Regression: '$75,000 yearly salary' should NOT extract 75000 as years."""
        result = detect_senior_signals("compensation details: 60000-75000 yearly salary")
        assert result["years_mentioned"] == 0

    def test_ignores_large_numbers(self):
        """Regression: numbers like 225000 should not be treated as years."""
        result = detect_senior_signals("annual salary range 225000 yearly")
        assert result["years_mentioned"] == 0

    def test_caps_at_30_years(self):
        """Years above 30 should be ignored as unrealistic."""
        result = detect_senior_signals("requires 50 years of deep experience")
        assert result["years_mentioned"] == 0

    def test_still_detects_valid_years(self):
        """Normal year patterns should still work after the fix."""
        result = detect_senior_signals("3+ years Python, 5 years Java required")
        assert result["years_mentioned"] == 5
        assert result["total_signals"] >= 2


# ── TestEmbeddings ───────────────────────────────────────────────


class TestEmbeddings:
    """Test sentence-transformer embedding generation."""

    def test_output_shape(self):
        texts = ["hello world", "data science is fun"]
        embeddings = generate_embeddings(texts, show_progress=False)
        assert embeddings.shape == (2, 384)

    def test_output_dtype(self):
        texts = ["test embedding"]
        embeddings = generate_embeddings(texts, show_progress=False)
        assert embeddings.dtype == np.float32

    def test_different_texts_different_embeddings(self):
        texts = ["software engineer", "pastry chef"]
        embeddings = generate_embeddings(texts, show_progress=False)
        # Cosine similarity should be < 1 (they're different)
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        assert similarity < 0.95


# ── TestLemmatize ────────────────────────────────────────────────


class TestLemmatize:
    """Test spaCy lemmatization."""

    def test_removes_stopwords(self):
        texts = pd.Series(["the cat is running quickly"])
        result = lemmatize_texts(texts)
        # "the", "is" should be removed as stopwords
        assert "the" not in result.iloc[0]
        assert "is" not in result.iloc[0]

    def test_lemmatizes_words(self):
        texts = pd.Series(["the dogs were running"])
        result = lemmatize_texts(texts)
        assert "dog" in result.iloc[0]
        assert "run" in result.iloc[0]

    def test_preserves_index(self):
        texts = pd.Series(["hello world", "test text"], index=[10, 20])
        result = lemmatize_texts(texts)
        assert list(result.index) == [10, 20]
