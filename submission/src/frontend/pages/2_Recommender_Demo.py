"""
Page 2: Interactive Recommender Demo
Left Panel: Scrollable 10-turn Chat
Right Panel: Pipeline State Inspector
"""

import sys
import uuid
from pathlib import Path
from typing import List
import pandas as pd
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "submission") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "submission"))

from submission.agent import Agent

MAX_TURNS = 10
MAX_USER_CHARS = 250
AVAILABLE_PREFERENCE_TAGS = [
    'fit', 'comfort', 'durability', 'style', 'material',
    'weather', 'warmth', 'performance', 'general shopping'
]

st.set_page_config(
    page_title="Interactive Recommender Demo",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Agent Loader & Session Initialization
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initialising Agent...")
def load_live_agent() -> Agent:
    """Instantiates Shopping Agent."""
    return Agent()


def init_session_state():
    """Initialises session state trackers."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"demo-session-{uuid.uuid4().hex[:8]}"
    if "current_turn" not in st.session_state:
        st.session_state.current_turn = 1
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "latest_diagnostics" not in st.session_state:
        st.session_state.latest_diagnostics = None
    if "user_preference_tags" not in st.session_state:
        st.session_state.user_preference_tags = []
    if "profile_confirmed" not in st.session_state:
        st.session_state.profile_confirmed = False


def reset_conversation():
    """Clears history and resets agent state."""
    st.session_state.session_id = f"demo-session-{uuid.uuid4().hex[:8]}"
    st.session_state.current_turn = 1
    st.session_state.messages = []
    st.session_state.latest_diagnostics = None
    st.session_state.profile_confirmed = False
    st.session_state.user_preference_tags = []


init_session_state()

try:
    active_agent = load_live_agent()
    agent_load_error = None
except Exception as e:
    active_agent = None
    agent_load_error = str(e)

# -----------------------------------------------------------------------------
# Sidebar: Session Controls & Active Tags Indicator
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Session Controls")
st.sidebar.caption(f"Session ID: `{st.session_state.session_id}`")

progress_val = min((st.session_state.current_turn - 1) / MAX_TURNS, 1.0)
st.sidebar.progress(progress_val, text=f"Turn {st.session_state.current_turn} of {MAX_TURNS}")

if agent_load_error:
    st.sidebar.error(f"⚠️ Could not load Agent: {agent_load_error}")
else:
    st.sidebar.success("✅ Live Agent Loaded")

if st.sidebar.button("Reset Session", use_container_width=True):
    reset_conversation()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Active User Profile")
if st.session_state.profile_confirmed:
    tags_html = " ".join([f"`{tag}`" for tag in st.session_state.user_preference_tags])
    st.sidebar.markdown(tags_html)
else:
    st.sidebar.caption("Profile not yet configured.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Session Constraints**:
    * Max Turns: **10**
    * Input Limit: **250 Characters**
    * Profile Tags: **1-5**
    * Recommendations: **Top 10 ASINs**
    """
)

# -----------------------------------------------------------------------------
# Main Layout: Split Screen
# -----------------------------------------------------------------------------
col_chat, col_inspect = st.columns([1.1, 0.9], gap="large")

# =============================================================================
# LEFT PANEL: Interactive Chat
# =============================================================================
with col_chat:
    st.subheader("💬 Shop with CRIS")

    # Step 1: Pre-session Profile Setup
    if not st.session_state.profile_confirmed:
        st.info("👋 Welcome! Please select **1-5 preference tags** to initialise your profile before starting.")
        selected_tags = st.multiselect(
            "Select User Profile Preferences:",
            options=AVAILABLE_PREFERENCE_TAGS,
            default=["comfort", "durability"],
            max_selections=5,
        )

        if st.button("Start Conversation 🚀", type="primary", disabled=len(selected_tags) == 0):
            st.session_state.user_preference_tags = selected_tags
            st.session_state.profile_confirmed = True

            # Reset agent state with initialized profile
            if active_agent is not None:
                try:
                    active_agent.reset(
                        session_id=st.session_state.session_id,
                        user_profile={
                            "preference_tags": selected_tags,
                            "summary": f"Prior purchases emphasize {', '.join(selected_tags)}.",
                        },
                    )
                except Exception as e:
                    st.error(f"Failed to reset agent with profile: {e}")

            st.rerun()

    # Step 2: Active Chat Interface
    else:
        tags_badges = " ".join([f"`{tag}`" for tag in st.session_state.user_preference_tags])
        st.markdown(f"**Customer Preferences:** {tags_badges}")

        chat_scroll_box = st.container(height=500)

        with chat_scroll_box:
            if not st.session_state.messages:
                st.caption("Start by describing what item you are looking for (e.g., *'I need lightweight running shoes'*)...")

            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.write(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(msg["content"])

                        if msg.get("ask_attribute"):
                            st.caption(f"🎯 **Target Clarification Attribute:** `{msg['ask_attribute']}`")

                        recommendations = msg.get("recommendations", [])
                        if recommendations:
                            with st.expander(f"📦 Top {len(recommendations)} Recommendations (Turn {msg.get('turn', '-')})", expanded=True):
                                top_picks = recommendations[:2]
                                c1, c2 = st.columns(2)
                                for idx, item in enumerate(top_picks):
                                    col_target = c1 if idx == 0 else c2
                                    with col_target:
                                        asin = item.get("parent_asin", "N/A")
                                        title = item.get("title", f"Product {asin}")
                                        score = item.get("score")
                                        score_txt = f" • Score: `{score:.4f}`" if score is not None else ""
                                        st.markdown(
                                            f"""
                                            <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; margin-bottom: 5px; background-color: rgba(128,128,128,0.05);">
                                                <strong>#{idx+1} {asin}</strong>{score_txt}<br/>
                                                <span style="font-size: 0.85em;">{title[:80]}...</span>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )

                                if len(recommendations) > 2:
                                    st.dataframe(
                                        [
                                            {
                                                "Rank": i + 3,
                                                "ASIN": r.get("parent_asin", "N/A"),
                                                "Title": r.get("title", f"Product {r.get('parent_asin')}"),
                                            }
                                            for i, r in enumerate(recommendations[2:])
                                        ],
                                        use_container_width=True,
                                        hide_index=True,
                                    )

        # Chat Input Bar
        if st.session_state.current_turn <= MAX_TURNS:
            user_input = st.chat_input(
                f"Turn {st.session_state.current_turn}/{MAX_TURNS}: Enter query (max {MAX_USER_CHARS} chars, no typos)...",
                max_chars=MAX_USER_CHARS,
            )

            if user_input:
                clean_input = user_input.strip()
                if clean_input:
                    if active_agent is None:
                        st.error("Agent is not initialized. Please ensure the agent backend is available.")
                    else:
                        # 1. Record User Turn
                        st.session_state.messages.append({"role": "user", "content": clean_input})

                        # 2. Process Turn through Agent
                        with st.spinner("Processing turn through CRIS pipeline..."):
                            try:
                                turn_output = active_agent.respond_chat(
                                    session_id=st.session_state.session_id,
                                    user_message=clean_input,
                                    turn=st.session_state.current_turn,
                                    top_k=10,
                                )
                                # 3. Append Assistant Message
                                assistant_msg = {
                                    "role": "assistant",
                                    "content": turn_output.get("message", "Here are my current recommendations:"),
                                    "ask_attribute": turn_output.get("ask_attribute", ""),
                                    "recommendations": turn_output.get("recommendations", []),
                                    "turn": st.session_state.current_turn,
                                }
                                st.session_state.messages.append(assistant_msg)
                                st.session_state.latest_diagnostics = turn_output.get("diagnostics", {})

                                # 4. Increment Turn Counter
                                st.session_state.current_turn += 1
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error executing agent turn: {e}")
        else:
            st.warning("🛑 Maximum turns (10/10) reached for this session. Please reset the session to start again.")

# =============================================================================
# RIGHT PANEL: Pipeline Inspector
# =============================================================================
with col_inspect:
    st.subheader("⚙️ Pipeline State Inspector")

    diagnostics = st.session_state.latest_diagnostics

    if not diagnostics:
        st.info("Awaiting initial user input. State info from the most recent turn will appear here.")
    else:
        current_diag_turn = diagnostics.get("turn", st.session_state.current_turn - 1)
        st.caption(f"Displaying pipeline for **Turn {current_diag_turn}**")

        # ---------------------------------------------------------------------
        # 1. Candidate Pruning Funnel
        # ---------------------------------------------------------------------
        st.markdown("**1. Candidate Pruning Funnel**")
        counts = diagnostics.get("retrieval_counts", {})
        dropped_hard = diagnostics.get("hard_filters_dropped", 0)
        pre_filtered = counts.get("pre_filtered_pool", 50000 - dropped_hard)
        rrf_count = counts.get("rrf_pool", 100)

        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        f_col1.metric("Catalog", "50,000")
        f_col2.metric("Masked Pool", f"{pre_filtered:,}", delta=f"-{dropped_hard:,}", delta_color="blue")
        f_col3.metric("RRF Pool", f"{rrf_count}")
        f_col4.metric("Reranker Final", "10")

        st.markdown("---")

        # ---------------------------------------------------------------------
        # 2. Dynamic State Tracking (DST)
        # ---------------------------------------------------------------------
        st.markdown("**2. Dynamic State Tracking (DST)**")
        dyn_state = diagnostics.get("dynamic_state", {})
        session_profile = dyn_state.get("session_profile", {})
        detected_intent = dyn_state.get("intent", "buying")

        st.caption(f"Inferred Intent: `{detected_intent.upper()}`")

        active_slots = {
            k: v for k, v in session_profile.items()
            if v and k not in ("rejected",)
        }
        rejected_slots = session_profile.get("rejected", [])

        dst_col1, dst_col2 = st.columns(2)
        with dst_col1:
            st.markdown("##### Positive Constraints")
            if active_slots:
                for slot, vals in active_slots.items():
                    val_str = ", ".join(vals) if isinstance(vals, list) else str(vals)
                    st.markdown(f"* **{slot.title()}**: `{val_str}`")
            else:
                st.caption("No positive constraints extracted yet.")

        with dst_col2:
            st.markdown("##### Exclusions & Filters")
            if rejected_slots:
                st.markdown(f"* **Rejected**: `{', '.join(rejected_slots)}`")
            if "category" in active_slots:
                st.markdown(f"* **Hard Category Filter**: `{', '.join(active_slots['category'])}`")
            if not rejected_slots and "category" not in active_slots:
                st.caption("No active exclusions or budget caps.")

        st.markdown("---")

        # ---------------------------------------------------------------------
        # 3. Attribute Information Gain / Entropy
        # ---------------------------------------------------------------------
        st.markdown("**3. Attribute Information Gain (Entropy)**")
        entropy_data = diagnostics.get("entropy_scores", {})
        if entropy_data:
            df_entropy = (
                pd.DataFrame(list(entropy_data.items()), columns=["Attribute", "Entropy"])
                .sort_values(by="Entropy", ascending=True)
                .set_index("Attribute")
            )
            st.bar_chart(df_entropy, horizontal=True)
        else:
            st.caption("No entropy distribution available for this turn.")

        st.markdown("---")

        # ---------------------------------------------------------------------
        # 4. Top 10 Cross-Encoder Candidate Rankings
        # ---------------------------------------------------------------------
        st.markdown("**4. Reranker (Cross-Encoder) Top 10 Scoring**")
        ce_candidates = diagnostics.get("top_candidates_ce", [])

        if ce_candidates:
            df_ce = pd.DataFrame(
                [
                    {
                        "Rank": idx + 1,
                        "parent_asin": item.get("parent_asin", "N/A"),
                        "Score": item.get("score", 0.0),
                        "Title": item.get("title", f"Product {item.get('parent_asin')}"),
                    }
                    for idx, item in enumerate(ce_candidates[:10])
                ]
            )

            st.dataframe(
                df_ce,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank"),
                    "parent_asin": st.column_config.TextColumn("Parent ASIN"),
                    "Score": st.column_config.NumberColumn("Reranker Score", format="%.4f"),
                    "Title": st.column_config.TextColumn("Product Title"),
                },
            )
        else:
            st.caption("No reranker (cross-encoder) candidate scores reported.")