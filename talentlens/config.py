"""Centralized configuration and file paths for TalentLens"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Prevent crash when Anaconda's libiomp5md.dll and PyTorch's libomp.dll are
# both present in the same process (common on Windows + Anaconda environments).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
TOPIC_ASSIGNMENTS_PARQUET = PROCESSED_DIR / "topic_assignments.parquet"

# Minimal metadata used for retrieval demos (kept small for fast loads)
RETRIEVAL_META_PARQUET = PROCESSED_DIR / "retrieval_meta.parquet"

# ── Model paths ───────────────────────────────────────────────────
FAISS_INDEX_DIR = MODELS_DIR / "faiss_index"
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "postings.index"
SALARY_MODEL_PATH = MODELS_DIR / "salary_model.joblib"
GHOST_JOB_MODEL_PATH = MODELS_DIR / "ghost_job_model.joblib"
ENTRY_LEVEL_MODEL_PATH = MODELS_DIR / "entry_level_paradox_model.joblib"
EMPLOYER_BRANDING_MODEL_PATH = MODELS_DIR / "employer_branding_model.joblib"
BERTOPIC_MODEL_DIR = MODELS_DIR / "bertopic_model"

# ── Database (Postgres + pgvector) ────────────────────────────────
# Example: postgresql+psycopg://user:password@localhost:5432/talentlens
DATABASE_URL = os.getenv("DATABASE_URL", "")
PGVECTOR_TABLE = os.getenv("PGVECTOR_TABLE", "job_postings")

# ── Constants ─────────────────────────────────────────────────────
RANDOM_SEED = 42
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
HOURS_PER_YEAR = 2080  # For converting hourly salary to yearly
