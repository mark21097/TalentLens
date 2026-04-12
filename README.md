# TalentLens

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

An end-to-end RAG-powered data science pipeline analyzing 123,842 LinkedIn job postings across five research themes.

## Research Questions
- **Q1 — Ghost Job Detection**: Are companies posting "ghost jobs" with no real intent to hire? We classify probable ghost postings using engagement signals (views, applications, duration open).
- **Q2 — Entry-Level Paradox**: Do jobs labeled "entry-level" actually demand senior qualifications? ~40% of entry-level postings mention 4+ years of experience.
- **Q3 — Salary Prediction**: Can NLP features from job descriptions predict median salary? We train regression models on text stats, embeddings, and structured features.
- **Q4 — Employer Branding**: What drives application-to-view ratio? We model which description characteristics (sentiment, length, remote status) attract more applicants.
- **Q5 — A/B Testing RAG**: Does prompt engineering improve the quality of LLM answers over retrieved job postings? Prompt A vs Prompt B evaluated in the Streamlit app.

## Pipeline Architecture

```
Raw CSVs (3.38M rows)
  → Phase 1: Clean + EDA            → postings_clean.parquet (123K rows)
  → Phase 2: NLP Features           → postings_features.parquet (47 cols)
                                    → description_embeddings.npy (123K × 384)
  → Phase 3: ML Models              → salary_model.joblib, ghost_job_model.joblib
  → Phase 4: RAG Pipeline (FAISS)   → postings.index (FAISS)
  → Phase 5: Streamlit App          → Multi-page app + A/B test chatbot
```

## Tech Stack
- **Vector DB:** FAISS (local, no server required)
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (384-dim, free)
- **LLM:** Ollama (mistral / llama3.1, local, no API costs)
- **Data:** Pandas, PyArrow (Parquet)
- **ML:** scikit-learn, XGBoost, SHAP
- **NLP:** spaCy, TextBlob, BERTopic, sentence-transformers
- **RAG:** LangChain + FAISS
- **App:** Streamlit, Plotly

## Quick Start

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         talentlens and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── talentlens   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes talentlens a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

