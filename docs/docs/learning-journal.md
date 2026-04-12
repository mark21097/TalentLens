# TalentLens Learning Journal

This document tracks what was learned at each phase of the project.
Updated incrementally as the pipeline grows.

---

## Phase 0: Project Scaffolding (Complete)
**Date**: 2026-03-21

### Concepts covered
- Python packaging with flit and editable installs
- Cookiecutter Data Science project structure
- Configuration management (centralized paths in config.py)
- Environment variables with python-dotenv
- Pytest for data validation
- Loguru for structured logging

### Key takeaways
- Always separate reusable code (package) from exploration (notebooks)
- Centralize file paths so you change one place, not fourteen
- Write tests early — they catch broken assumptions before they waste hours
- Use `.env` for secrets, never commit API keys

### What's next
- Phase 1: Load 3.38M rows, clean and filter, save to parquet, run EDA
- Key question to answer: How many usable postings remain after cleaning?

---

## Phase 1: Data Cleaning & EDA (Complete)
**Date**: 2026-03-22

### Concepts covered
- Loading large CSVs with dtype specification for memory efficiency
- Parquet format (columnar compression, type preservation, 5x smaller than CSV)
- Data cleaning pipeline: null handling, deduplication, salary normalization
- Structured EDA: univariate, bivariate, and temporal analysis
- Reusable visualization module (`talentlens/plots.py`)
- Relational data schemas and join operations

### Key takeaways
- Always profile your data before cleaning — understand what's missing and why
- Document every cleaning decision (what was dropped, why, how many rows affected)
- Parquet is always better than CSV for analytics workloads
- EDA should be structured around research questions, not random exploration

### Key findings
- **Cleaned row count: 123,842** (from 3.38M raw — only LinkedIn US postings subset used)
- 5% of postings have salary data → sparse target for regression
- Ghost job signals confirmed: open >60 days with <5 applies
- Entry-level paradox preliminary evidence: senior language appears in entry-level postings
- Remote jobs get ~27% higher application rates than on-site

---

## Phase 2: NLP Feature Engineering (Complete)
**Date**: 2026-04-04

### Concepts covered
- Text cleaning pipeline: HTML removal, URL stripping, whitespace normalization, lowercasing
- Lemmatization with NLTK WordNet (50% word reduction, removes stopwords)
- TF-IDF: sparse matrix representation of term importance across documents
- Sentiment analysis with TextBlob (polarity + subjectivity)
- Regex-based senior signal detection (years mentioned, seniority keywords)
- Sentence embeddings with sentence-transformers (all-MiniLM-L6-v2, 384-dim)
- Topic modeling with BERTopic (UMAP dimensionality reduction + HDBSCAN clustering)

### Key takeaways
- Always cache expensive operations (embeddings take ~20 min on 123K texts)
- Sparse matrices (TF-IDF) vs dense arrays (embeddings) serve different purposes
- Embeddings capture *semantic meaning*; TF-IDF captures *term frequency* — both are useful
- BERTopic is unsupervised; expect ~40-50% outliers on heterogeneous job data

### Key findings
- **Entry-Level Paradox confirmed**: 40% of entry-level jobs mention years of experience; median = 4 years
- 235 natural job clusters discovered by BERTopic (top topics: Sales, Legal, Patient Care, Accounting)
- Sentiment: mean polarity 0.164 (slightly positive) — job descriptions are mildly optimistic
- 98.6% of postings have structured skill tags; top skills: IT, Sales, Management

### Outputs
- `postings_nlp.parquet` — cleaned + lemmatized text
- `postings_features.parquet` — all 47 features, ML-ready
- `description_embeddings.npy` — 123,842 × 384 float32
- `models/bertopic_model/` — saved topic model

---

## Phase 3: ML Models (In Progress)
**Date started**: 2026-04-11

### What we're building
- Salary prediction regression (XGBoost + SHAP)
- Ghost job binary classifier (heuristic labels → XGBoost)
- Entry-level paradox statistical analysis (Mann-Whitney U + flagging)
- Employer branding engagement model (apply-rate regression)

### Key challenge
- Salary target is 95% null — train on the labeled 6K rows only
- Ghost job labels must be constructed from behavior signals (views, applies ratio)
- Statistical tests require careful non-parametric methods (data is not normally distributed)

---

## Phase 4: RAG Pipeline (Skeleton Complete)
**Date**: 2026-04-04

### What's built
- FAISS index over 123K job description embeddings (notebook 07)
- `FaissRetriever` class in `talentlens/rag.py`
- `answer_question()` RAG pipeline using LangChain + Ollama
- pgvector ingestion code (notebook 08 — optional backend)

### What's needed
- Install and run Ollama locally (`ollama pull llama3.1`)
- Test full end-to-end: query → embed → retrieve → generate answer

---

## Phase 5: Streamlit App
*Coming soon — depends on Phase 3 models and Ollama running*
