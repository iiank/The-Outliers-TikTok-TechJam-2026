"""
Main application landing page.
Run with: streamlit run submission/src/frontend/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="CRIS",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛍️ TechJam Conversational Recommender & Intelligent Search (CRIS)")
st.markdown("### The Outliers")
st.markdown("#### Problem Statement 4: AI Conversational Search and Recommendations")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        #### 📊 Page 1: Evaluation Dashboard
        Inspect CRIS performance over the simulated test sessions:
        * **KPIs**: Hit Rate@10, MRR, MTTC, Efficiency, and Composite Technical Score.
        * **Scenarios**: Performance segmented by `buying`, `browsing`, `intent_override`, and `boundary` scenarios.
        * **Token & Cost Telemetry**: Usage tracking for LLM state tracking calls.
        """
    )
    if st.button("Go to Evaluation Dashboard ➡️"):
        st.switch_page("pages/1_Evaluation.py")

with col2:
    st.markdown(
        """
        #### 💬 Page 2: Interactive Recommender Demo
        Test CRIS live in a conversational session:
        * **Interactive 10-turn dialogue interface**.
        * **Dynamic State Tracking (DST)** inspector displaying parsed slots.
        * **Hard-filter drops & candidate pool counters** (50k -> Filter -> RRF -> Reranker).
        * **Cross-Encoder scores** and **Attribute Entropy distributions**.
        """
    )
    if st.button("Go to Interactive Recommender Demo ➡️"):
        st.switch_page("pages/2_Recommender_Demo.py")

st.markdown("---")
st.caption("TikTok TechJam 2026 • Conversational Recommender & Intelligent Search (CRIS) • The Outliers")