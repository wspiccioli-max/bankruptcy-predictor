"""
Main Streamlit entry point.

Run with:
  streamlit run dashboard/app.py

Uses Streamlit's multi-page pattern via st.sidebar radio buttons.
Each page is a separate module that gets called here.
"""

import sys
from pathlib import Path

# Allow imports from the project root (utils/, models/, etc.)
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Bankruptcy & Fraud Risk Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("📉 Risk Predictor")
st.sidebar.caption("BA870-AC820 · SEC EDGAR + yfinance")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏆  Bankruptcy Leaderboard",
        "🔍  Company Lookup",
        "📊  Model Comparison",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data sources**\n"
    "- SEC EDGAR 10-K filings\n"
    "- yfinance market data\n\n"
    "**Models**\n"
    "- Logistic Regression\n"
    "- Altman Z-Score\n"
    "- Fraud Risk Score"
)

# ── Route to the selected page ────────────────────────────────────────────────
if page == "🏆  Bankruptcy Leaderboard":
    from dashboard.page1_leaderboard import render
    render()

elif page == "🔍  Company Lookup":
    from dashboard.page2_lookup import render
    render()

elif page == "📊  Model Comparison":
    from dashboard.page3_comparison import render
    render()
