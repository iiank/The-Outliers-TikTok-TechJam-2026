"""
Page 1: Evaluation Dashboard
Displays aggregate benchmarks, scenario breakdowns, and session-level metrics.
"""

from pathlib import Path
import streamlit as st
import pandas as pd

from utils.ui_helpers import load_evaluation_data, parse_scenario_metrics, parse_sessions

st.set_page_config(page_title="Evaluation Dashboard", page_icon="📊", layout="wide")

st.title("📊 CRIS Evaluation Dashboard")
st.caption("Dashboard of `results.json` metrics created by `local_evaluator.py` across sessions.")

# -----------------------------------------------------------------------------
# Sidebar Config & results.json Loading
# -----------------------------------------------------------------------------
st.sidebar.header("📁 Data Source")
default_path = "results.json"
results_file_path = st.sidebar.text_input("Path to `results.json` (default `root/`):", value=default_path)

data, error = load_evaluation_data(results_file_path)

if error or not data:
    st.error(f"⚠️ {error}")
    st.info("Ensure you have executed `python -m evaluator.local_evaluator` from the project root to generate `results.json`.")
    st.stop()

# -----------------------------------------------------------------------------
# Aggregate KPIs
# -----------------------------------------------------------------------------
st.subheader("Global Performance Summary")

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
kpi1.metric("Total Samples", f"{data.get('sample_count', 0):,}")
kpi2.metric("Hit Rate @ 10", f"{data.get('hit_rate_at_10', 0.0):.2%}")
kpi3.metric("MRR", f"{data.get('mrr', 0.0):.4f}")
kpi4.metric("MTTC (Turns)", f"{data.get('mttc', 0.0):.2f}")
kpi5.metric("Efficiency", f"{data.get('efficiency', 0.0):.3f}")
kpi6.metric("Technical Score", f"{data.get('recommended_technical_score', 0.0):.5f}")

st.markdown("---")

# -----------------------------------------------------------------------------
# Scenario Breakdown & Token Usage
# -----------------------------------------------------------------------------
col_scenario, col_tokens = st.columns([2.2, 0.8])

with col_scenario:
    st.subheader("Scenario-Level Performance")
    scenario_metrics = data.get("scenario_metrics", {})
    if scenario_metrics:
        df_scenarios = parse_scenario_metrics(scenario_metrics)
        
        # Display Table
        st.dataframe(
            df_scenarios,
            width='stretch',
            hide_index=True,
            column_config={
                "Scenario Type": st.column_config.TextColumn("Scenario Type"),
                "Sample Count": st.column_config.NumberColumn("Samples"),
                "Hit Rate @ 10": st.column_config.ProgressColumn("Hit Rate @ 10", min_value=0.0, max_value=1.0, format="%.2f"),
                "MRR": st.column_config.NumberColumn("MRR", format="%.4f"),
                "MTTC (Turns)": st.column_config.NumberColumn("MTTC", format="%.2f"),
            }
        )
    else:
        st.info("No scenario breakdown found in results payload.")

with col_tokens:
    st.subheader("Token Usage")
    token_usage = data.get("reported_token_usage", {})
    prompt_toks = token_usage.get("prompt_tokens", 0)
    comp_toks = token_usage.get("completion_tokens", 0)
    total_toks = token_usage.get("total_tokens", 0)

    st.metric("Total Tokens", f"{total_toks:,}")
    st.metric("Prompt Tokens", f"{prompt_toks:,}")
    st.metric("Completion Tokens", f"{comp_toks:,}")

st.markdown("---")

# -----------------------------------------------------------------------------
# Per-Session Breakdown Table
# -----------------------------------------------------------------------------
st.subheader("Session Level Records")

raw_sessions = data.get("sessions", [])
if raw_sessions:
    df_sessions = parse_sessions(raw_sessions)

    # Filter Controls
    f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
    with f_col1:
        scenarios_available = ["All"] + sorted(df_sessions["scenario_type"].unique().tolist())
        selected_scenario = st.selectbox("Filter Scenario", scenarios_available)
    with f_col2:
        hit_filter = st.selectbox("Filter Hit Status", ["All", "Hits Only (True)", "Misses Only (False)"])
    with f_col3:
        search_sample = st.text_input("Search Sample ID", "")

    # Apply Filters
    filtered_df = df_sessions.copy()
    if selected_scenario != "All":
        filtered_df = filtered_df[filtered_df["scenario_type"] == selected_scenario]
    if hit_filter == "Hits Only (True)":
        filtered_df = filtered_df[filtered_df["hit"] == True]
    elif hit_filter == "Misses Only (False)":
        filtered_df = filtered_df[filtered_df["hit"] == False]
    if search_sample.strip():
        filtered_df = filtered_df[filtered_df["sample_id"].str.contains(search_sample.strip(), case=False)]

    st.caption(f"Displaying **{len(filtered_df)}** of **{len(df_sessions)}** total sessions")

    st.dataframe(
        filtered_df,
        width='stretch',
        hide_index=True,
        column_config={
            "sample_id": st.column_config.TextColumn("Sample ID"),
            "scenario_type": st.column_config.TextColumn("Scenario"),
            "hit": st.column_config.CheckboxColumn("Hit @ 10"),
            "first_hit_turn": st.column_config.TextColumn("First Hit Turn"),
            "best_rank": st.column_config.TextColumn("Best Rank (1-10)"),
            "reciprocal_rank": st.column_config.NumberColumn("Reciprocal Rank", format="%.4f"),
        }
    )
else:
    st.info("No individual session logs present in results.json.")

# -----------------------------------------------------------------------------
# Raw JSON Inspector
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("🔍 View Raw `results.json`"):
    st.json(data)