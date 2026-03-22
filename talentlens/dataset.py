"""Dataset loading, cleaning, and processing utilities.

This module provides functions to load raw CSV data, apply cleaning
transformations, and produce processed parquet files ready for analysis.

Usage:
    python -m talentlens.dataset          # Run the full pipeline
    make data                             # Via Makefile
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger
import pandas as pd

from talentlens.config import (
    BENEFITS_CSV,
    COMPANIES_CSV,
    COMPANY_INDUSTRIES_CSV,
    COMPANY_SPECIALITIES_CSV,
    EMPLOYEE_COUNTS_CSV,
    HOURS_PER_YEAR,
    INDUSTRIES_MAP_CSV,
    JOB_INDUSTRIES_CSV,
    JOB_SKILLS_CSV,
    POSTINGS_CLEAN_PARQUET,
    POSTINGS_CSV,
    PROCESSED_DIR,
    SALARIES_CSV,
    SKILLS_MAP_CSV,
)


def load_raw_postings(
    sample_n: Optional[int] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load the raw postings CSV with appropriate dtypes.

    Args:
        sample_n: If set, return a random sample of this many rows.
        nrows: If set, only read this many rows from the CSV (faster for dev).

    Returns:
        Raw postings DataFrame.
    """
    logger.info(f"Loading raw postings from {POSTINGS_CSV}")

    dtype_map = {
        "job_id": "int64",
        "company_id": "float64",
        "title": "str",
        "company_name": "str",
        "location": "str",
        "formatted_work_type": "str",
        "formatted_experience_level": "str",
        "skills_desc": "str",
        "remote_allowed": "float64",
        "views": "float64",
        "applies": "float64",
        "sponsored": "float64",
        "currency": "str",
        "compensation_type": "str",
        "pay_period": "str",
        "work_type": "str",
        "posting_domain": "str",
        "application_type": "str",
        "zip_code": "str",
        "fips": "float64",
    }

    df = pd.read_csv(
        POSTINGS_CSV,
        dtype=dtype_map,
        nrows=nrows,
        low_memory=False,
    )

    logger.info(f"Loaded {len(df):,} raw postings ({df.shape[1]} columns)")

    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=42)
        logger.info(f"Sampled down to {len(df):,} rows")

    return df


def load_secondary_tables() -> dict[str, pd.DataFrame]:
    """Load all secondary CSV tables into a dictionary.

    Returns:
        Dict mapping table name to DataFrame.
    """
    tables = {
        "companies": COMPANIES_CSV,
        "company_industries": COMPANY_INDUSTRIES_CSV,
        "company_specialities": COMPANY_SPECIALITIES_CSV,
        "employee_counts": EMPLOYEE_COUNTS_CSV,
        "benefits": BENEFITS_CSV,
        "job_industries": JOB_INDUSTRIES_CSV,
        "job_skills": JOB_SKILLS_CSV,
        "salaries": SALARIES_CSV,
        "industries_map": INDUSTRIES_MAP_CSV,
        "skills_map": SKILLS_MAP_CSV,
    }

    result = {}
    for name, path in tables.items():
        if path.exists():
            result[name] = pd.read_csv(path, low_memory=False)
            logger.info(f"Loaded {name}: {len(result[name]):,} rows")
        else:
            logger.warning(f"File not found: {path}")

    return result


def normalize_salary(
    row: pd.Series,
    salary_col: str = "med_salary",
    pay_period_col: str = "pay_period",
) -> float | None:
    """Convert a salary value to yearly, handling hourly pay periods.

    Args:
        row: A DataFrame row.
        salary_col: Column name for the salary value.
        pay_period_col: Column name for the pay period.

    Returns:
        Yearly salary as float, or None if data is missing.
    """
    salary = row.get(salary_col)
    pay_period = row.get(pay_period_col)

    if pd.isna(salary):
        return None

    if isinstance(pay_period, str) and pay_period.upper() == "HOURLY":
        return float(salary) * HOURS_PER_YEAR

    return float(salary)


def clean_postings(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning transformations to raw postings.

    Steps:
        1. Drop rows missing description or title
        2. Deduplicate on job_id
        3. Parse timestamps to datetime
        4. Normalize salaries (hourly → yearly)
        5. Create derived columns (is_remote, days_open, experience_level)

    Args:
        df: Raw postings DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    initial_count = len(df)
    logger.info(f"Cleaning {initial_count:,} postings...")

    # 1. Drop rows missing critical text fields
    df = df.dropna(subset=["description"])
    logger.info(f"After dropping null descriptions: {len(df):,}")

    df = df.dropna(subset=["title"])
    logger.info(f"After dropping null titles: {len(df):,}")

    # 2. Deduplicate on job_id
    df = df.drop_duplicates(subset=["job_id"], keep="first")
    logger.info(f"After deduplication: {len(df):,}")

    # 3. Parse timestamps
    for col in ["original_listed_time", "listed_time", "closed_time", "expiry"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")

    # 4. Normalize salaries to yearly
    for salary_col in ["min_salary", "med_salary", "max_salary"]:
        if salary_col in df.columns:
            yearly_col = f"{salary_col}_yearly"
            df[yearly_col] = df.apply(
                lambda row: normalize_salary(row, salary_col, "pay_period"),
                axis=1,
            )

    # 5. Derived columns
    df["is_remote"] = df["remote_allowed"].fillna(0).astype(bool)

    if "listed_time" in df.columns and "closed_time" in df.columns:
        df["days_open"] = (df["closed_time"] - df["listed_time"]).dt.days

    df["experience_level"] = (
        df["formatted_experience_level"].fillna("Unknown").str.strip()
    )

    logger.info(
        f"Cleaning complete: {initial_count:,} → {len(df):,} postings "
        f"({initial_count - len(df):,} removed)"
    )

    return df


def make_dataset(nrows: Optional[int] = None) -> Path:
    """Run the full data processing pipeline.

    Args:
        nrows: If set, only process this many rows (for development).

    Returns:
        Path to the output parquet file.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_postings(nrows=nrows)
    df = clean_postings(df)

    df.to_parquet(POSTINGS_CLEAN_PARQUET, index=False)
    logger.info(f"Saved cleaned postings to {POSTINGS_CLEAN_PARQUET}")

    return POSTINGS_CLEAN_PARQUET


if __name__ == "__main__":
    make_dataset()
