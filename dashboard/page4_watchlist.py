"""
Page 4 - Watchlist

Screening view of current S&P 500 companies ranked by estimated bankruptcy
filing risk. This is not investment advice.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.formatting import (
    PROBABILITY_DISCLAIMER,
    format_probability,
)
from dashboard.metadata import methodology_caption

ROOT = Path(__file__).parent.parent
SP500_PATH = ROOT / "data/processed/sp500_predictions.csv"


@st.cache_data(ttl=3600)
def load_sp500() -> pd.DataFrame:
    if not SP500_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SP500_PATH)


def _z_score(row) -> float | None:
    parts = [row.get(f"current_z_x{i}") for i in range(1, 6)]
    if any(pd.isna(v) for v in parts):
        return None
    return 1.2 * parts[0] + 1.4 * parts[1] + 3.3 * parts[2] + 0.6 * parts[3] + parts[4]


def _drivers(row) -> str:
    drivers = []
    cr = row.get("current_current_ratio")
    dte = row.get("current_debt_to_equity")
    roa = row.get("current_return_on_assets")
    ic = row.get("current_interest_coverage")
    margin = row.get("current_net_margin")
    if pd.notna(cr) and cr < 1:
        drivers.append("low liquidity")
    if pd.notna(dte) and (dte > 3 or dte < 0):
        drivers.append("high leverage")
    if pd.notna(roa) and roa < 0:
        drivers.append("negative ROA")
    if pd.notna(margin) and margin < 0:
        drivers.append("negative margin")
    if pd.notna(ic) and ic < 1.5:
        drivers.append("weak interest coverage")
    return ", ".join(drivers[:3]) or "model-weighted ratio profile"


def render():
    st.title("📌 Watchlist")
    st.caption("Screening list only. Not investment advice.")
    st.caption(methodology_caption())
    st.caption(PROBABILITY_DISCLAIMER)

    df = load_sp500()
    if df.empty:
        st.error("No S&P 500 predictions file found. Run `python refresh_sp500.py`.")
        return

    df = df[df["data_available"] == True].copy()
    st.sidebar.markdown("### Watchlist Filters")
    sectors = ["All"] + sorted(df["sector"].dropna().unique().tolist())
    sel_sector = st.sidebar.selectbox("Watchlist sector", sectors)
    risk_levels = st.sidebar.multiselect(
        "Watchlist risk level",
        ["🔴 High", "🟡 Medium", "🟢 Low"],
        default=["🔴 High", "🟡 Medium"],
    )
    top_n = st.sidebar.slider("Companies shown", 10, 100, 25, step=5)

    if sel_sector != "All":
        df = df[df["sector"] == sel_sector]
    if risk_levels:
        df = df[df["risk_bucket"].isin(risk_levels)]

    df["altman_z_score"] = df.apply(_z_score, axis=1)
    df["risk_drivers"] = df.apply(_drivers, axis=1)
    df = df.sort_values("current_prob", ascending=False).head(top_n)
    lr_score = df["lr_current_prob"] if "lr_current_prob" in df.columns else df["current_prob"]
    rf_score = (
        df["rf_current_prob"] if "rf_current_prob" in df.columns
        else pd.Series([None] * len(df), index=df.index)
    )

    display = pd.DataFrame({
        "Ticker": df["ticker"],
        "Company": df["name"],
        "Sector": df.get("sector"),
        "Filing Risk": df["current_prob"].apply(format_probability),
        "Risk Category": df["risk_bucket"],
        "Altman Z": df["altman_z_score"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—"),
        "Logistic": lr_score.apply(format_probability),
        "Random Forest": rf_score.apply(format_probability),
        "Key Drivers": df["risk_drivers"],
    })
    st.dataframe(display, use_container_width=True, hide_index=True, height=560)
