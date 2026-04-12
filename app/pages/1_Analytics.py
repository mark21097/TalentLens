"""TalentLens — Analytics page: Phase 3 model results and EDA charts."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from talentlens.config import (
    GHOST_JOB_MODEL_PATH,
    POSTINGS_FEATURES_PARQUET,
    SALARY_MODEL_PATH,
)

st.set_page_config(page_title="Analytics | TalentLens", layout="wide")
st.title("📊 Analytics")

# ── Layout constants ───────────────────────────────────────────────────────────
_H_TALL = 400       # standard chart height (px)
_H_MID = 380        # slightly shorter for side-by-side pairs
_H_SHORT = 350      # compact charts (histograms)
_H_MINI = 300       # small supplementary charts
_SCATTER_SAMPLE = 3_000   # max rows in scatter plots (keeps Plotly fast)
_GHOST_VIEWS_MIN = 200    # min views to be considered a ghost job signal
_GHOST_APPLIES_MAX = 5    # max applies to be considered a ghost job signal


@st.cache_data(show_spinner="Loading data...")
def load_data() -> pd.DataFrame:
    return pd.read_parquet(POSTINGS_FEATURES_PARQUET)


df = load_data()

tab1, tab2, tab3, tab4 = st.tabs([
    "💰 Salary Prediction",
    "👻 Ghost Jobs",
    "🎓 Entry-Level Paradox",
    "📈 Employer Branding",
])

# ── Tab 1: Salary Prediction ──────────────────────────────────────────────────
with tab1:
    st.header("💰 Salary Prediction")
    st.markdown("**Theme 1**: Can NLP features from job descriptions predict salary?")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Salary Distribution by Experience Level")
        df_sal = df[df["med_salary_yearly"].notna() & df["med_salary_yearly"].between(10_000, 600_000)]
        exp_order = ["Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]
        df_sal_filtered = df_sal[df_sal["experience_level"].isin(exp_order)]
        fig = px.box(
            df_sal_filtered,
            x="experience_level",
            y="med_salary_yearly",
            category_orders={"experience_level": exp_order},
            labels={"med_salary_yearly": "Median Annual Salary ($)", "experience_level": ""},
            color="experience_level",
        )
        fig.update_layout(showlegend=False, height=_H_TALL)
        fig.update_yaxes(tickprefix="$", tickformat=",")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Remote vs On-Site Salary")
        df_remote = df_sal.copy()
        df_remote["work_type"] = df_remote["is_remote"].map({True: "Remote", False: "On-site"})
        fig2 = px.violin(
            df_remote,
            x="work_type",
            y="med_salary_yearly",
            box=True,
            labels={"med_salary_yearly": "Median Annual Salary ($)", "work_type": ""},
            color="work_type",
        )
        fig2.update_layout(showlegend=False, height=_H_TALL)
        fig2.update_yaxes(tickprefix="$", tickformat=",")
        st.plotly_chart(fig2, use_container_width=True)

    if SALARY_MODEL_PATH.exists():
        st.success(f"✅ Salary model trained and saved at `{SALARY_MODEL_PATH.name}`")
        st.markdown("Run **notebook 10** to see RMSE, R², and SHAP feature importance plots.")
    else:
        st.warning("⚠️ Salary model not found — run notebook 10 first.")

# ── Tab 2: Ghost Jobs ──────────────────────────────────────────────────────────
with tab2:
    st.header("👻 Ghost Job Detection")
    st.markdown("**Theme 2**: Are companies posting jobs with no intent to hire?")

    df_eng = df[df["views"].notna() & df["applies"].notna()].copy()
    df_eng["apply_rate"] = (df_eng["applies"] / df_eng["views"].clip(lower=1)).clip(upper=1.0)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Apply Rate Distribution")
        fig = px.histogram(
            df_eng[df_eng["apply_rate"] < 0.5],
            x="apply_rate",
            nbins=60,
            labels={"apply_rate": "Apply Rate (applies / views)"},
        )
        fig.add_vline(
            x=df_eng["apply_rate"].median(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Median: {df_eng['apply_rate'].median():.3f}",
        )
        fig.update_layout(height=_H_SHORT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("High Views, Low Applies (Ghost Signal)")
        ghost_heuristic = df_eng[(df_eng["views"] > _GHOST_VIEWS_MIN) & (df_eng["applies"] < _GHOST_APPLIES_MAX)]
        st.metric("Probable Ghost Jobs", f"{len(ghost_heuristic):,}")
        st.metric("Ghost Rate", f"{len(ghost_heuristic)/len(df_eng)*100:.1f}%")
        st.dataframe(
            ghost_heuristic.nlargest(10, "views")[["title", "company_name", "views", "applies"]].reset_index(drop=True),
            use_container_width=True,
        )

    if GHOST_JOB_MODEL_PATH.exists():
        model = joblib.load(GHOST_JOB_MODEL_PATH)

        exp_order_map = {
            "Internship": 0, "Entry level": 1, "Associate": 2,
            "Mid-Senior level": 3, "Director": 4, "Executive": 5, "Unknown": 2
        }
        # Work on a local copy — do NOT mutate the @st.cache_data result
        df_scored = df.copy()
        df_scored["exp_level_encoded"] = df_scored["experience_level"].map(exp_order_map).fillna(2)
        df_scored["is_remote_int"] = df_scored["is_remote"].astype(int)
        df_scored["sponsored_int"] = df_scored["sponsored"].fillna(0).astype(int)

        GHOST_FEATURE_COLS = [
            "desc_word_count", "sentiment_polarity", "sentiment_subjectivity",
            "senior_signal_count", "max_years_required", "n_skills",
            "exp_level_encoded", "is_remote_int", "sponsored_int",
        ]
        X_all = df_scored[GHOST_FEATURE_COLS].fillna(0).values
        df_scored["ghost_prob"] = model.predict_proba(X_all)[:, 1]

        st.subheader("Ghost Probability Distribution (All Postings)")
        fig3 = px.histogram(df_scored, x="ghost_prob", nbins=50, labels={"ghost_prob": "Ghost Probability"})
        fig3.add_vline(x=0.5, line_dash="dash", line_color="red", annotation_text="Threshold 0.5")
        fig3.update_layout(height=_H_MINI)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("ℹ️ Run notebook 11 to train the ghost job classifier.")

# ── Tab 3: Entry-Level Paradox ─────────────────────────────────────────────────
with tab3:
    st.header("🎓 Entry-Level Paradox")
    st.markdown("**Theme 3**: Do entry-level jobs demand senior qualifications?")

    exp_order = ["Internship", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive"]
    df_known = df[df["experience_level"].isin(exp_order)].copy()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Senior Signals by Experience Level")
        df_box = df_known[df_known["senior_signal_count"] <= 15]
        fig = px.box(
            df_box,
            x="experience_level",
            y="senior_signal_count",
            category_orders={"experience_level": exp_order},
            labels={"senior_signal_count": "Senior Signal Count", "experience_level": ""},
            color="experience_level",
        )
        fig.update_layout(showlegend=False, height=_H_TALL)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("% Mentioning Years of Experience")
        pct_years = df_known.groupby("experience_level").apply(
            lambda x: (x["max_years_required"] > 0).mean() * 100
        ).reindex(exp_order).dropna()
        fig2 = px.bar(
            x=pct_years.index,
            y=pct_years.values,
            labels={"x": "", "y": "% Postings Mentioning Years"},
            color=pct_years.values,
            color_continuous_scale="Blues",
        )
        fig2.update_layout(height=_H_TALL, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    entry_df = df_known[df_known["experience_level"] == "Entry level"]
    entry_with_years = entry_df[entry_df["max_years_required"] > 0]

    col3, col4, col5 = st.columns(3)
    col3.metric("Entry-Level Jobs", f"{len(entry_df):,}")
    col4.metric("% Mentioning Years", f"{len(entry_with_years)/len(entry_df)*100:.1f}%")
    col5.metric("Median Years Required", f"{entry_with_years['max_years_required'].median():.0f} years")

# ── Tab 4: Employer Branding ───────────────────────────────────────────────────
with tab4:
    st.header("📈 Employer Branding")
    st.markdown("**Theme 4**: What drives higher application-to-view ratios?")

    df_eng2 = df[df["views"].notna() & df["applies"].notna()].copy()
    df_eng2["apply_rate"] = (df_eng2["applies"] / df_eng2["views"].clip(lower=1)).clip(upper=1.0)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Apply Rate: Remote vs On-Site")
        df_eng2["work_type"] = df_eng2["is_remote"].map({True: "Remote", False: "On-site", None: "Unknown"})
        fig = px.box(
            df_eng2[df_eng2["apply_rate"] < 0.5],
            x="work_type",
            y="apply_rate",
            labels={"apply_rate": "Apply Rate", "work_type": ""},
            color="work_type",
        )
        fig.update_layout(showlegend=False, height=_H_MID)
        st.plotly_chart(fig, use_container_width=True)

        remote_med = df_eng2[df_eng2["is_remote"] == True]["apply_rate"].median()
        onsite_med = df_eng2[df_eng2["is_remote"] == False]["apply_rate"].median()
        if onsite_med > 0:
            uplift = (remote_med - onsite_med) / onsite_med * 100
            st.metric("Remote Apply Rate Uplift", f"+{uplift:.1f}%")

    with col2:
        st.subheader("Apply Rate vs Sentiment Polarity")
        sample = df_eng2[df_eng2["apply_rate"] < 0.5].sample(min(_SCATTER_SAMPLE, len(df_eng2)), random_state=42)
        fig2 = px.scatter(
            sample,
            x="sentiment_polarity",
            y="apply_rate",
            opacity=0.3,
            trendline="lowess",
            labels={"sentiment_polarity": "Description Sentiment (polarity)", "apply_rate": "Apply Rate"},
        )
        fig2.update_layout(height=_H_MID)
        st.plotly_chart(fig2, use_container_width=True)
