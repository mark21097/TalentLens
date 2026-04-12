# Phase 1: Data Ingestion, Cleaning, and EDA

## What We Built
Three notebooks that load 3.38M raw job postings, clean them into a Parquet file,
and visualize distributions, trends, and correlations across all research themes.

## Key Concepts Learned

### 1. Loading Large CSVs with Pandas
**Problem**: A 493MB CSV with 3.38M rows can consume 2-4GB of RAM if pandas guesses types wrong.

**Solution**: Specify `dtype` for each column and use `nrows` for development:
```python
df = pd.read_csv("postings.csv", dtype={"zip_code": "str", "remote_allowed": "float64"}, nrows=10000)
```

**Why it matters**: In production, you'd use chunked reading (`chunksize=50000`) or tools like
Dask/Polars for out-of-memory data. For this dataset, specifying dtypes is enough.

### 2. Parquet vs CSV
| Feature | CSV | Parquet |
|---------|-----|---------|
| Size | 493 MB | ~100 MB |
| Load speed | 30-60s | 2-5s |
| Types preserved | No (everything is string) | Yes |
| Columnar access | No (reads all columns) | Yes (reads only what you need) |

Parquet uses **columnar compression** — it stores all values for one column together,
which means similar values compress extremely well (e.g., a column of "YEARLY"/"HOURLY" strings).

### 3. Data Cleaning Pipeline
A reproducible cleaning pipeline means anyone can re-run your code and get the same result.
Key decisions:

1. **Drop null descriptions**: Can't do NLP on empty text
2. **Deduplicate on job_id**: Same posting may appear multiple times
3. **Normalize salaries**: Convert hourly pay × 2,080 hours = yearly equivalent
4. **Parse timestamps**: Unix milliseconds → Python datetime for time-based analysis

### 4. Exploratory Data Analysis (EDA)
EDA is not random exploration — it's structured investigation:

1. **Univariate**: What does each variable look like individually? (histograms, value counts)
2. **Bivariate**: How do two variables relate? (box plots, scatter, correlations)
3. **Temporal**: How do things change over time? (time series of posting volume)

Each visualization should answer a specific question tied to your research themes.

### 5. Reusable Plot Functions
Instead of copying matplotlib code in every notebook cell, we created `talentlens/plots.py`
with functions like `plot_distribution()` and `plot_top_categories()`. Benefits:
- Consistent style across all notebooks
- Auto-saves figures to `reports/figures/`
- Less code in notebooks = easier to read

## Files Created
| File | Purpose |
|------|---------|
| `notebooks/01-mp-data-loading-and-schema.ipynb` | Load all tables, profile columns, document schema |
| `notebooks/02-mp-data-cleaning.ipynb` | Full cleaning pipeline, save to Parquet |
| `notebooks/03-mp-exploratory-data-analysis.ipynb` | Visualizations across all research themes |
| `talentlens/plots.py` | Reusable visualization functions |
| `references/data_dictionary.md` | Complete schema documentation |

## Next Steps
→ Phase 2: NLP feature engineering — extract skills, run topic modeling, generate embeddings.
