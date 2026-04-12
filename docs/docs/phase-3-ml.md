# Phase 3: Machine Learning Models

**Status**: In Progress
**Date started**: 2026-04-11
**Notebooks**: 10–13

---

## Overview

Phase 3 trains predictive models for all four quantitative research themes.
Each notebook follows the same pattern:

```
Load postings_features.parquet
  → Feature selection + encoding
  → Baseline model (linear)
  → Better model (XGBoost)
  → Evaluation (appropriate metrics for problem type)
  → SHAP feature importance
  → Save model with joblib
```

---

## Notebooks

### Notebook 10 — Salary Prediction

**File**: `10-mp-salary-prediction.ipynb`
**Research Theme 1**: Can NLP features predict median annual salary?

**Challenge**: Only 5% of postings (6,280 rows) have salary labels.
We train exclusively on this labeled subset.

**Features**:
- Numeric: `desc_word_count`, `sentiment_polarity`, `senior_signal_count`, `max_years_required`, `n_skills`
- Categorical: `experience_level` (ordinally encoded 0–5), `is_remote`

**Models**:
- Ridge regression baseline (L2 regularization, scale-invariant)
- XGBoost regression (gradient-boosted trees, best for tabular data)

**Target**: `log(1 + med_salary_yearly)` — log-transform stabilizes the right-skewed salary distribution

**Evaluation**: RMSE (in dollars), MAE, R² (on 20% held-out test set + 5-fold CV)

**Output**: `models/salary_model.joblib`

---

### Notebook 11 — Ghost Job Classifier

**File**: `11-mp-ghost-job-classifier.ipynb`
**Research Theme 2**: Which job postings are probably ghost jobs?

**Labeling approach**: No ground truth → construct heuristic labels from behavioral signals:
- Ghost if: `views > 200 AND applies < 5`
- Rationale: high-visibility job that almost nobody applied to

**Class imbalance**: ~10–15% ghost rate → use `class_weight='balanced'` (Logistic Regression)
and `scale_pos_weight` (XGBoost)

**Features**: `desc_word_count`, `sentiment_polarity`, `sentiment_subjectivity`,
`senior_signal_count`, `max_years_required`, `n_skills`, `exp_level_encoded`, `is_remote`, `sponsored`

**Models**:
- Logistic Regression baseline (interpretable, handles imbalance)
- XGBoost classifier

**Evaluation**: F1, ROC-AUC, precision-recall curve (accuracy is misleading for imbalanced classes)

**Output**: `models/ghost_job_model.joblib`, `ghost_prob` column on all postings

---

### Notebook 12 — Entry-Level Paradox

**File**: `12-mp-entry-level-paradox.ipynb`
**Research Theme 3**: Do entry-level jobs demand senior qualifications?

**This is an analysis + classifier notebook** (not just modeling):

**Statistical Analysis**:
- Mann-Whitney U test comparing senior signal counts: entry-level vs mid-senior
- Cohen's d effect size
- Bootstrap 95% confidence intervals for median difference
- Null hypothesis: H₀: no difference in senior signals across experience levels

**Paradox flag definition**:
```python
is_paradox = (max_years_required >= 3) OR (senior_signal_count >= 3)
```

**Classifier**: Logistic Regression to predict `is_paradox` from description features alone
(so we can flag paradox jobs even without an experience_level label)

**Output**: `models/entry_level_paradox_model.joblib`

**Key finding**: ~40% of entry-level jobs are paradox postings. Mann-Whitney p << 0.05,
confirming the difference is statistically significant.

---

### Notebook 13 — Employer Branding

**File**: `13-mp-employer-branding.ipynb`
**Research Theme 4**: What drives application-to-view ratio?

**Target**: `apply_rate = applies / views` (23,318 rows with both signals)

**Feature selection** uses Spearman correlation first (data is not normally distributed,
apply_rate is bounded 0–1).

**Models**:
- Ridge regression (after removing top/bottom 5% outliers)
- XGBoost regression

**Evaluation**: R², MAE (in apply-rate units)

**SHAP dependency plots**: Show how each feature's value affects the apply rate prediction

**Output**: `models/employer_branding_model.joblib`

**Key finding**: Remote status is the strongest predictor (+27% apply rate).
Description sentiment has a real but small positive effect.

---

## Concepts Introduced in Phase 3

### Train/Test Split and Cross-Validation

```python
from sklearn.model_selection import train_test_split, KFold, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=KFold(5), scoring='r2')
```

Split once, train on 80%, evaluate on 20%. Cross-validation gives a more stable estimate
by averaging across 5 different train/val splits of the training data.

### XGBoost

Gradient-boosted decision trees: builds an ensemble of weak learners sequentially,
each one correcting the errors of the previous. Generally the best algorithm for
tabular data. Key hyperparameters:

| Parameter | What it controls |
|-----------|-----------------|
| `n_estimators` | Number of trees (more = lower variance, higher compute) |
| `max_depth` | Tree depth (higher = more complex, risk of overfit) |
| `learning_rate` | Shrinkage per tree (lower = more robust, needs more trees) |
| `subsample` | Fraction of rows per tree (reduces overfit) |
| `scale_pos_weight` | Class imbalance correction |

### SHAP (SHapley Additive exPlanations)

SHAP assigns each feature a contribution value for each prediction, based on game theory.
Unlike standard feature importance, SHAP is:
- **Consistent**: removing a more important feature always reduces performance more
- **Directional**: shows whether a feature increases or decreases the prediction
- **Instance-level**: explains individual predictions, not just averages

```python
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

### Class Imbalance

When one class is rare (e.g., ghost jobs ~10%), naive models predict the majority class
and achieve high accuracy but zero recall on the minority class.

Solutions used:
1. `class_weight='balanced'` in sklearn — weights loss by inverse class frequency
2. `scale_pos_weight` in XGBoost — upweights positive class in gradient computation
3. Evaluate with F1/ROC-AUC, not accuracy

### Non-Parametric Statistics

- **Mann-Whitney U**: Tests if one distribution is stochastically greater than another
  (does not assume normality). Alternative to t-test for skewed/count data.
- **Spearman correlation**: Rank-based correlation, robust to outliers and non-linearity.
- **Cohen's d**: Effect size — how practically large is the difference, independent of sample size?

---

## Model Artifacts

| File | Created by | Description |
|------|-----------|-------------|
| `models/salary_model.joblib` | NB10 | XGBoost salary regressor |
| `models/ghost_job_model.joblib` | NB11 | XGBoost ghost job classifier |
| `models/entry_level_paradox_model.joblib` | NB12 | Logistic regression paradox classifier |
| `models/employer_branding_model.joblib` | NB13 | XGBoost apply-rate regressor |

---

## Resume Bullets (Draft)

> "Built XGBoost salary prediction model (RMSE ~$X) from NLP features; used SHAP to explain
> that experience level and senior language are the strongest salary predictors."

> "Constructed heuristic ghost job labels from engagement signals; trained XGBoost classifier
> achieving F1=X on 23K labeled postings; scored all 123K postings with ghost probability."

> "Quantified entry-level paradox: 40% of entry-level jobs require 4+ years experience;
> confirmed with Mann-Whitney U (p < 0.001), Cohen's d effect size."

> "Modeled application-to-view ratio with Ridge + SHAP; found remote status is the dominant
> predictor (+27% apply rate uplift, statistically significant)."

---

**→ Next**: Phase 4 — RAG Pipeline (LangChain + FAISS + Ollama) | Phase 5 — Streamlit App
