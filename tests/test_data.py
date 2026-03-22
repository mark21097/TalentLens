"""Tests for data loading, validation, and cleaning utilities."""

import pandas as pd
import pytest

from talentlens.config import (
    BENEFITS_CSV,
    COMPANIES_CSV,
    HOURS_PER_YEAR,
    INDUSTRIES_MAP_CSV,
    JOB_INDUSTRIES_CSV,
    JOB_SKILLS_CSV,
    POSTINGS_CSV,
    SALARIES_CSV,
    SKILLS_MAP_CSV,
)
from talentlens.dataset import clean_postings, load_raw_postings, normalize_salary


# ── Raw file existence ────────────────────────────────────────────


class TestRawFilesExist:
    """Verify that all expected raw data files are present."""

    @pytest.mark.parametrize(
        "path",
        [
            POSTINGS_CSV,
            COMPANIES_CSV,
            BENEFITS_CSV,
            JOB_SKILLS_CSV,
            JOB_INDUSTRIES_CSV,
            SALARIES_CSV,
            INDUSTRIES_MAP_CSV,
            SKILLS_MAP_CSV,
        ],
    )
    def test_raw_file_exists(self, path):
        assert path.exists(), f"Missing raw data file: {path}"


# ── Salary normalization ─────────────────────────────────────────


class TestNormalizeSalary:
    """Test hourly-to-yearly salary conversion."""

    def test_yearly_passthrough(self):
        row = pd.Series({"med_salary": 100_000, "pay_period": "YEARLY"})
        assert normalize_salary(row) == 100_000

    def test_hourly_to_yearly(self):
        row = pd.Series({"med_salary": 50.0, "pay_period": "HOURLY"})
        assert normalize_salary(row) == 50.0 * HOURS_PER_YEAR

    def test_missing_salary_returns_none(self):
        row = pd.Series({"med_salary": None, "pay_period": "YEARLY"})
        assert normalize_salary(row) is None

    def test_missing_pay_period_defaults_to_yearly(self):
        row = pd.Series({"med_salary": 80_000, "pay_period": None})
        assert normalize_salary(row) == 80_000


# ── Cleaning logic ────────────────────────────────────────────────


class TestCleanPostings:
    """Test the cleaning pipeline on a small synthetic DataFrame."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "job_id": [1, 2, 3, 4, 2],  # job_id 2 is duplicated
                "title": ["SWE", "DS", None, "MLE", "DS dupe"],
                "description": ["Build APIs", "Train models", "No title", None, "Dupe"],
                "remote_allowed": [1.0, 0.0, 1.0, 0.0, 1.0],
                "formatted_experience_level": ["Entry level", "Mid-Senior level", None, "Associate", "Entry level"],
                "listed_time": [1_700_000_000_000, 1_700_100_000_000, None, None, 1_700_000_000_000],
                "closed_time": [1_705_000_000_000, None, None, None, 1_705_000_000_000],
                "min_salary": [50_000, 80_000, None, None, 50_000],
                "med_salary": [60_000, 90_000, None, None, 60_000],
                "max_salary": [70_000, 100_000, None, None, 70_000],
                "pay_period": ["YEARLY", "YEARLY", None, None, "YEARLY"],
            }
        )

    def test_drops_null_descriptions(self, sample_df):
        result = clean_postings(sample_df)
        assert result["description"].isna().sum() == 0

    def test_drops_null_titles(self, sample_df):
        result = clean_postings(sample_df)
        assert result["title"].isna().sum() == 0

    def test_deduplicates_job_ids(self, sample_df):
        result = clean_postings(sample_df)
        assert result["job_id"].is_unique

    def test_creates_is_remote_column(self, sample_df):
        result = clean_postings(sample_df)
        assert "is_remote" in result.columns
        assert result["is_remote"].dtype == bool

    def test_creates_experience_level_column(self, sample_df):
        result = clean_postings(sample_df)
        assert "experience_level" in result.columns
        assert (result["experience_level"] != "").all()

    def test_creates_yearly_salary_columns(self, sample_df):
        result = clean_postings(sample_df)
        assert "med_salary_yearly" in result.columns
