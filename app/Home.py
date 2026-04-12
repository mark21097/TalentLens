"""TalentLens Streamlit App — Home page."""

import streamlit as st

st.set_page_config(
    page_title="TalentLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔍 TalentLens")
st.subheader("Job Market Intelligence Dashboard")

st.markdown("""
TalentLens analyzes **123,842 LinkedIn job postings** to surface actionable insights
about the modern job market.

---

### Research Questions Answered

| # | Theme | Status |
|---|-------|--------|
| 1 | **Salary Prediction** — What features predict job salary? | ✅ Phase 3 |
| 2 | **Ghost Job Detection** — Are companies posting fake jobs? | ✅ Phase 3 |
| 3 | **Entry-Level Paradox** — Do entry-level jobs demand senior experience? | ✅ Phase 3 |
| 4 | **Employer Branding** — What drives more applications? | ✅ Phase 3 |
| 5 | **A/B Testing RAG** — Which chatbot prompt answers better? | 🔬 Phase 5 |

---

### Navigate

Use the **sidebar** to explore:

- 📊 **Analytics** — Charts and model results for all 4 research themes
- 🔎 **Job Search** — Semantic job search powered by FAISS embeddings
- 🤖 **Ask TalentLens** — RAG chatbot over job postings (requires Ollama)
- 🧪 **A/B Test** — Compare Prompt A vs Prompt B on the same retrieved docs

---
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Postings", "123,842")
    st.metric("With Salary Data", "6,280 (5%)")
with col2:
    st.metric("Entry-Level Jobs", "36,708")
    st.metric("Paradox Rate", "~40%")
with col3:
    st.metric("With Engagement Data", "23,318")
    st.metric("Median Apply Rate", "14.9%")
