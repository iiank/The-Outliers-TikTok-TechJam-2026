"""
Page 2: Interactive Recommender Demo
Left Panel: 10-turn Conversational Chat, Attribute Inquiries, and Recommendations.
Right Panel: Pipeline Inspector (Placeholder).
"""

import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[3] 
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "submission") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "submission"))

MAX_TURNS = 10
MAX_USER_CHARS = 250

st.set_page_config(
    page_title="Interactive Recommender Demo",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Mock Response Generator (For standalone testing)
# -----------------------------------------------------------------------------
def generate_mock_turn_output(user_message: str, turn: int) -> Dict[str, Any]:
    mock_attributes = ["material", "color", "size", "brand", "style", "feature", "budget"]
    next_attr = mock_attributes[(turn - 1) % len(mock_attributes)]

    mock_items = [
        {
            "parent_asin": f"B08SAMPLE{i:02d}",
            "title": f"Sample Recommended Product {i+1} - Lightweight Comfort Edition",
            "score": round(0.95 - (i * 0.07), 4),
            "price": 29.99 + (i * 5),
            "category": "Clothing, Shoes & Jewelry > Men > Footwear",
        }
        for i in range(10)
    ]

    return {
        "message": f"I found several matching items based on '{user_message}'. To help narrow this down further, what preference do you have for **{next_attr}**?",
        "ask_attribute": next_attr,
        "recommendations": mock_items,
        "diagnostics": {
            "dynamic_state": {
                "session_profile": {
                    "category": ["footwear"],
                    "material": ["leather"] if turn > 1 else [],
                    "color": ["black"] if turn > 2 else [],
                    "budget": ["<=100"],
                    "feature": [],
                    "use_case": [],
                    "other": [],
                    "rejected": [],
                },
                "intent": "buying",
            },
            "hard_filters_dropped": 4210,
            "retrieval_counts": {
                "pre_filtered_pool": 45790,
                "bm25_top": 50,
                "dense_top": 50,
                "rrf_pool": 100,
            },
            "top_candidates_ce": [
                {"asin": item["parent_asin"], "score": item["score"], "title": item["title"]}
                for item in mock_items[:5]
            ],
            "entropy_scores": {
                "material": 1.42,
                "color": 1.35,
                "brand": 1.10,
                "size": 0.88,
                "style": 0.65,
                "budget": 0.40,
            },
        },
    }


# -----------------------------------------------------------------------------
# Agent Loader & Session Initialization
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initialising Agent...")
def load_live_agent():
    """Attempts to instantiate Shopping Agent."""
    try:
        from src.agent.shopping_agent import Agent
        agent = Agent()
        return agent, None
    except Exception as e:
        return None, str(e)


def init_session_state():
    """Initializes chat history, turn trackers, and session ID."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"demo-session-{uuid.uuid4().hex[:8]}"
    if "current_turn" not in st.session_state:
        st.session_state.current_turn = 1
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "latest_diagnostics" not in st.session_state:
        st.session_state.latest_diagnostics = None


def reset_conversation():
    """Clears history and resets agent state."""
    st.session_state.session_id = f"demo-session-{uuid.uuid4().hex[:8]}"
    st.session_state.current_turn = 1
    st.session_state.messages = []
    st.session_state.latest_diagnostics = None
    if "live_agent" in st.session_state and st.session_state.live_agent is not None:
        try:
            st.session_state.live_agent.reset(
                session_id=st.session_state.session_id,
                user_profile={"preference_tags": ["quality", "comfort"]},
            )
        except Exception:
            pass


init_session_state()

# -----------------------------------------------------------------------------
# Sidebar: Session Controls & Model Status
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ Session Controls")
st.sidebar.caption(f"Session ID: `{st.session_state.session_id}`")

# Turn progress tracking
progress_val = min((st.session_state.current_turn - 1) / MAX_TURNS, 1.0)
st.sidebar.progress(progress_val, text=f"Turn {st.session_state.current_turn} of {MAX_TURNS}")

agent_mode = st.sidebar.radio(
    "Agent Backend",
    ["Mock Agent (Standalone UI Test)", "Live Agent Pipeline"],
    index=0,
)

if agent_mode == "Live Agent Pipeline":
    live_agent, err = load_live_agent()
    if err:
        st.sidebar.warning(f"⚠️ Could not load Agent: {err}")
        st.sidebar.info("Falling back to Mock Agent.")
        active_agent = None
    else:
        st.sidebar.success("✅ Live Agent Loaded")
        active_agent = live_agent
else:
    active_agent = None

if st.sidebar.button("🔄 Reset Conversation", use_container_width=True):
    reset_conversation()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Session Constraints**:
    * Max Turns: **10**
    * Input Limit: **250 Characters**
    * Recommendations: **Top 10 ASINs**
    """
)

# -----------------------------------------------------------------------------
# Main Layout: Split Screen
# -----------------------------------------------------------------------------
col_chat, col_inspect = st.columns([1.15, 0.85], gap="medium")

# =============================================================================
# LEFT PANEL: Interactive Chat & Recommendations
# =============================================================================
with col_chat:
    st.subheader("💬 Customer Dialogue & Recommendations")
    
    # Render Chat History
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.info("👋 Welcome! Start by entering what product you are looking for (e.g., *'I am looking for lightweight hiking boots'*).")
        
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["content"])
                    
                    # Clarifying attribute badge
                    if msg.get("ask_attribute"):
                        st.caption(f"🎯 **Target Clarification Attribute:** `{msg['ask_attribute']}`")

                    # Product Recommendations Shelf
                    recommendations = msg.get("recommendations", [])
                    if recommendations:
                        with st.expander(f"📦 Top {len(recommendations)} Recommendations (Turn {msg.get('turn', '-')})", expanded=True):
                            # Render top 2 in detail, remaining 8 in a compact table
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
                            
                            # Compact list for Ranks 3-10
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

    # Chat Input Handling
    if st.session_state.current_turn <= MAX_TURNS:
        user_input = st.chat_input(
            f"Turn {st.session_state.current_turn}/{MAX_TURNS}: Describe what you're looking for (max {MAX_USER_CHARS} chars)...",
            max_chars=MAX_USER_CHARS,
        )

        if user_input:
            clean_input = user_input.strip()
            if clean_input:
                # 1. Record User Message
                st.session_state.messages.append({"role": "user", "content": clean_input})

                # 2. Query Agent or Mock Pipeline
                with st.spinner("Processing turn through CRS pipeline..."):
                    if active_agent is not None:
                        try:
                            turn_output = active_agent.respond_chat(
                                session_id=st.session_state.session_id,
                                user_message=clean_input,
                                turn=st.session_state.current_turn,
                                top_k=10,
                            )
                        except Exception as e:
                            st.error(f"Error calling Agent: {e}")
                            turn_output = generate_mock_turn_output(clean_input, st.session_state.current_turn)
                    else:
                        turn_output = generate_mock_turn_output(clean_input, st.session_state.current_turn)

                # 3. Record Assistant Response
                assistant_msg = {
                    "role": "assistant",
                    "content": turn_output.get("message", "Here are my current recommendations:"),
                    "ask_attribute": turn_output.get("ask_attribute", ""),
                    "recommendations": turn_output.get("recommendations", []),
                    "turn": st.session_state.current_turn,
                }
                st.session_state.messages.append(assistant_msg)
                st.session_state.latest_diagnostics = turn_output.get("diagnostics", {})

                # 4. Advance Turn Counter
                st.session_state.current_turn += 1
                st.rerun()
    else:
        st.warning("🛑 Maximum turns (10/10) reached for this dialogue session. Please reset the conversation to start a new session.")

# =============================================================================
# RIGHT PANEL: Pipeline Inspector (Placeholder)
# =============================================================================
with col_inspect:
    st.subheader("⚙️ Pipeline State Inspector")
    
    st.info(
        """
        **Pipeline Diagnostics View**
        
        This panel is ready to visualize intermediate telemetry on every turn:
        * 🏷️ **Dynamic State Tracking (DST)**: Parsed hard & soft slots
        * 🚫 **Hard Filter Masking**: Out-of-budget & category drops
        * 📊 **Candidate Pool Counts**: 50k -> Masked -> RRF(100)
        * 🎯 **Cross-Encoder Scoring**: Top 5 neural reranking values
        * 🧮 **Attribute Entropy**: Information gain per attribute
        """
    )

    if st.session_state.latest_diagnostics:
        with st.expander("🔍 Raw Turn Diagnostics Data", expanded=True):
            st.json(st.session_state.latest_diagnostics)
    else:
        st.caption("Awaiting first dialogue turn to capture pipeline state telemetry...")