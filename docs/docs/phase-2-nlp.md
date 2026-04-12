# Phase 2: NLP Feature Engineering & Embeddings

## What We Built

Three notebooks that transform raw job description text into numerical features
and dense vector embeddings ready for ML models and RAG search.

| Notebook | What it does |
|----------|-------------|
| 04 - NLP Text Preprocessing | HTML cleaning, text normalization, lemmatization with spaCy |
| 05 - Feature Engineering | TF-IDF analysis, sentiment, entry-level paradox detection, skill joins |
| 06 - Embeddings & Topics | Sentence-transformer embeddings, BERTopic topic modeling |

## Key Concepts Learned

### 1. Text Preprocessing (Why It Matters)

**Problem**: Raw job descriptions contain HTML tags, URLs, inconsistent formatting.
ML models would learn patterns from `<br>` and `&amp;` instead of actual content.

**Solution**: A cleaning pipeline: decode HTML entities → strip tags → remove URLs → normalize whitespace → lowercase.

**Why it matters**: In any NLP project, 80% of the work is preprocessing. Clean text is the foundation everything else builds on. Garbage in → garbage out.

### 2. Lemmatization vs Stemming

| Aspect | Stemming (NLTK) | Lemmatization (spaCy) |
|--------|-----------------|----------------------|
| Approach | Crude suffix removal | Dictionary + grammar rules |
| "running" | "run" ✓ | "run" ✓ |
| "university" | "univers" ✗ | "university" ✓ |
| "better" | "better" ✗ | "good" ✓ |
| Speed | Very fast | Fast (with `nlp.pipe()`) |
| Quality | Can produce non-words | Always real words |

**Decision**: spaCy lemmatization — we need real words for topic modeling and human readability.

**Performance trick**: Use `nlp.pipe()` for batch processing, disable unused pipeline components (NER, parser) → 3x faster.

### 3. TF-IDF: Turning Text into Numbers

**What it is**: Term Frequency–Inverse Document Frequency. Measures how important a word is to a document within a collection.

**Intuition**: A word that appears often in one document (high TF) but rarely across all documents (high IDF) is distinctive for that document.

- "python" in a data science job → high TF-IDF (distinctive)
- "experience" in every job → low TF-IDF (too common)

**How we used it**: Compared top TF-IDF terms across experience levels to see what language distinguishes entry-level from executive roles.

**Limitation**: TF-IDF treats every word independently. "machine" and "learning" are two separate features — it doesn't understand the phrase "machine learning." That's why we also use embeddings.

### 4. Sentiment Analysis

**What it measures**: How positive/negative (polarity) and factual/opinionated (subjectivity) text is.

**Tool**: TextBlob — a simple rule-based sentiment analyzer.

**Key insight**: Job descriptions are almost always slightly positive (companies are "selling" the role). The interesting question is whether *more* positive descriptions get more applications.

**Limitation**: TextBlob is not sophisticated enough for nuanced text. For production, use a fine-tuned transformer model. We use it because it's fast, free, and good enough for feature engineering.

### 5. Sentence Embeddings: Why They're Better Than TF-IDF

| Feature | TF-IDF | Embeddings |
|---------|--------|-----------|
| Dimension | Sparse, ~5000 | Dense, 384 |
| Similarity | Exact word match only | Captures meaning |
| "dog" vs "canine" | Completely different | Very similar |
| Memory | Sparse matrix (efficient) | Dense matrix (larger) |
| Use case | Analysis, traditional ML | Similarity search, RAG, modern ML |

**Model**: `all-MiniLM-L6-v2` — a sentence-transformer model that converts text to 384-dimensional vectors. Similar texts get similar vectors.

**Why this model**: Free, fast, runs on CPU, good quality. It's the default choice for most embedding tasks.

### 6. BERTopic: Unsupervised Topic Discovery

**What it does**: Finds natural groupings in text without manual labels.

**How it works** (3 steps):
1. **UMAP**: Dimensionality reduction (384-dim → 5-dim) to make clustering feasible
2. **HDBSCAN**: Density-based clustering that automatically finds the number of clusters
3. **c-TF-IDF**: Labels each cluster with its most representative terms

**Why not K-Means?** HDBSCAN doesn't require specifying the number of clusters, handles clusters of different sizes, and identifies outliers.

**Why not LDA?** LDA (Latent Dirichlet Allocation) works on word counts — it's a bag-of-words model. BERTopic uses embeddings, so it captures semantic meaning.

### 7. The NLP Feature Engineering Mindset

The mental model for any NLP feature engineering project:

```
Raw text → Clean → Extract features → Ready for ML

Features can be:
├── Simple statistics (word count, char count)
├── Regex patterns (years mentioned, keywords)
├── Sentiment scores (polarity, subjectivity)
├── TF-IDF (word importance)
├── Embeddings (dense semantic vectors)
└── Topics (cluster assignments)
```

Each method captures different information. Use multiple approaches and let the ML model figure out which features matter most.

## Files Created

| File | Purpose |
|------|---------|
| `talentlens/features.py` | Reusable NLP functions (clean_text, lemmatize, sentiment, embeddings) |
| `tests/test_features.py` | 28 unit tests for all feature functions |
| `notebooks/04-mp-nlp-text-preprocessing.ipynb` | Text cleaning + spaCy lemmatization |
| `notebooks/05-mp-feature-engineering.ipynb` | TF-IDF, sentiment, entry-level paradox, skills |
| `notebooks/06-mp-embeddings-and-topics.ipynb` | Sentence embeddings + BERTopic |
| `data/processed/postings_nlp.parquet` | Cleaned + lemmatized text |
| `data/processed/postings_features.parquet` | All NLP features added |
| `data/processed/description_embeddings.npy` | 123K × 384 embedding array |
| `data/processed/embedding_job_ids.npy` | Job ID ↔ embedding index mapping |
| `models/bertopic_model/` | Saved BERTopic model |

## Research Theme Progress

| Theme | Phase 2 Contribution |
|-------|---------------------|
| Salary Prediction | Text stats, sentiment, skill count as features |
| Ghost Job Detection | Text stats (short descriptions might indicate ghost jobs) |
| Entry-Level Paradox | Senior signal detection — do "entry-level" jobs demand "5+ years"? |
| Employer Branding | Sentiment features — do positive descriptions get more applications? |

## Next Steps

→ **Phase 3**: ML models — use these features to predict salary, classify ghost jobs, and run statistical tests on the entry-level paradox.
