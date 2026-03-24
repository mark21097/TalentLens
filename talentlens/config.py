"""Centralized configuration and file paths for TalentLens"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project structure ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ── Raw data file paths ───────────────────────────────────────────
# Main postings table
POSTINGS_CSV = RAW_DIR / "postings.csv"

# Companies
COMPANIES_CSV = RAW_DIR / "companies" / "companies.csv"
COMPANY_INDUSTRIES_CSV = RAW_DIR / "companies" / "company_industries.csv"
COMPANY_SPECIALITIES_CSV = RAW_DIR / "companies" / "company_specialities.csv"
EMPLOYEE_COUNTS_CSV = RAW_DIR / "companies" / "employee_counts.csv"

# Jobs
BENEFITS_CSV = RAW_DIR / "jobs" / "benefits.csv"
JOB_INDUSTRIES_CSV = RAW_DIR / "jobs" / "job_industries.csv"
JOB_SKILLS_CSV = RAW_DIR / "jobs" / "job_skills.csv"
SALARIES_CSV = RAW_DIR / "jobs" / "salaries.csv"

# Mappings (lookup tables)
INDUSTRIES_MAP_CSV = RAW_DIR / "mappings" / "industries.csv"
SKILLS_MAP_CSV = RAW_DIR / "mappings" / "skills.csv"

# ── Processed data paths ──────────────────────────────────────────
POSTINGS_CLEAN_PARQUET = PROCESSED_DIR / "postings_clean.parquet"
POSTINGS_NLP_PARQUET = PROCESSED_DIR / "postings_nlp.parquet"
POSTINGS_FEATURES_PARQUET = PROCESSED_DIR / "postings_features.parquet"
EMBEDDINGS_NPY = PROCESSED_DIR / "description_embeddings.npy"
EMBEDDING_JOB_IDS_NPY = PROCESSED_DIR / "embedding_job_ids.npy"

# ── Model paths ───────────────────────────────────────────────────
FAISS_INDEX_DIR = MODELS_DIR / "faiss_index"
SALARY_MODEL_PATH = MODELS_DIR / "salary_model.joblib"
GHOST_JOB_MODEL_PATH = MODELS_DIR / "ghost_job_model.joblib"
BERTOPIC_MODEL_DIR = MODELS_DIR / "bertopic_model"

# ── Constants ─────────────────────────────────────────────────────
RANDOM_SEED = 42
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
HOURS_PER_YEAR = 2080  # For converting hourly salary to yearly
