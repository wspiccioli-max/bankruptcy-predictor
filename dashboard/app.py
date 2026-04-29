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
    page_title="Bankruptcy Filing & Fraud Risk Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("📉 Filing Risk Predictor")
st.sidebar.caption("BA870-AC820 · SEC EDGAR + yfinance")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏆  S&P 500 Leaderboard",
        "📌  Watchlist",
        "🔍  Company Lookup",
        "📊  Model Validation",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data sources**\n"
    "- SEC EDGAR 10-K filings\n"
    "- yfinance market data\n\n"
    "**Models**\n"
    "- Logistic Regression\n"
    "- Random Forest\n"
    "- Altman Z-Score (rule-based benchmark)\n"
    "- Fraud Risk Score (rule-based indicator)\n\n"
    "**Focus**\n"
    "- Estimated bankruptcy filing risk"
)

# ── Route to the selected page ────────────────────────────────────────────────
if page == "🏆  S&P 500 Leaderboard":
    from dashboard.page1_leaderboard import render
    render()

elif page == "📌  Watchlist":
    from dashboard.page4_watchlist import render
    render()

elif page == "🔍  Company Lookup":
    from dashboard.page2_lookup import render
    render()

elif page == "📊  Model Validation":
    from dashboard.page3_comparison import render
    render()
