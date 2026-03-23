# TalentLens

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A learning end-to-end pipeline project that will analyze a series of problems in today's job market.

## Research Questions
- Q1: Are jobs really hiring? A problem in in today's job market are companies throwing "Ghost jobs" to make it look like the company is growing without the actual intent to hire.
- Q2: How has the definition of "entry-level" positions changed over the last three years. We will compare a datasets from the years (2019, 2023, and 2026)
- Q3: Are there salary penalty for fully remote jobs in SWE/AI/ML compared to roles that are hybrid or fully in-office positions (e.g tech hubs).
- Q4:

## Pipeline Architecture
[Check image]

## Tech Stack
- **Databases:** PostgreSQL + pgvector
- **Data Preprocessing:** Pandas & SQLAlchemy
- **RAG Pipelines:** LangChain
- **Embedding/LLM:** Ollama
- **API / App:** FastAPI & Streamlit

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

