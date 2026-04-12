# Phase 0: Project Scaffolding & Environment Setup

## What We Built
A properly structured Python package (`talentlens/`) with centralized configuration,
data loading utilities, and automated tests.

## Key Concepts Learned

### 1. Python Packaging with Flit
**What**: `flit` is a lightweight build tool that turns your folder into an installable Python package.

**Why it matters**: The `-e .` (editable install) in `requirements.txt` means you run
`pip install -e .` once, and then you can `from talentlens.config import POSTINGS_CSV`
from any notebook or script — no `sys.path` hacks needed.

**How it works**: `pyproject.toml` declares the package name, version, and dependencies.
Flit reads this and registers `talentlens/` as an importable package in your environment.

```toml
[build-system]
requires = ["flit_core >=3.2,<4"]
build-backend = "flit_core.buildapi"
```

### 2. Configuration Management
**What**: All file paths and constants live in one file: `talentlens/config.py`.

**Why it matters**: If your data folder moves, you change ONE file instead of hunting
through 14 notebooks. Every module imports paths from the same source of truth.

**Pattern**:
```python
from talentlens.config import POSTINGS_CSV, RANDOM_SEED
```

### 3. The Cookiecutter Data Science Structure
**What**: A standardized project layout used across the data science industry.

**Key directories**:
- `data/raw/` → Original, immutable data (never modify these files)
- `data/interim/` → Intermediate transformed data
- `data/processed/` → Final clean datasets (parquet files)
- `notebooks/` → Exploration and storytelling (numbered for ordering)
- `talentlens/` → Reusable, tested source code
- `models/` → Trained models and indexes
- `tests/` → Automated tests

**Why it matters**: Anyone opening your GitHub repo instantly understands where to find
the data, the code, and the results. This is the standard hiring managers expect.

### 4. Environment Variables with python-dotenv
**What**: Secrets (API keys) go in a `.env` file that is gitignored.

**Why it matters**: Never commit API keys to GitHub. `load_dotenv()` in `config.py`
reads the `.env` file automatically, so `os.getenv("OPENAI_API_KEY")` works everywhere.

### 5. Defensive Testing with Pytest
**What**: Tests that verify your data pipeline works correctly before you run the
full analysis.

**Examples from our tests**:
- Raw CSV files exist where expected
- Salary normalization correctly converts hourly → yearly
- Cleaning drops null descriptions and deduplicates
- Output has all expected columns

**Run with**: `make test` or `pytest tests/`

## Files Created
| File | Purpose |
|------|---------|
| `talentlens/__init__.py` | Makes the folder an importable Python package |
| `talentlens/config.py` | Centralized paths, constants, and configuration |
| `talentlens/dataset.py` | Data loading and cleaning functions |
| `tests/test_data.py` | Automated data validation tests |
| `.env` | Environment variables (gitignored) |
| `CLAUDE.md` | Project context for AI-assisted development |

## Next Steps
→ Phase 1: Load the raw data, clean it, and run exploratory data analysis (EDA).
