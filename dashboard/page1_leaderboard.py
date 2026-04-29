"""
Page 1 — Live S&P 500 Bankruptcy Filing Risk Leaderboard

Shows every S&P 500 company ranked by the logistic regression bankruptcy filing
probability, using the most recent SEC filing (10-K or 10-Q) that EDGAR has.

Key features:
  - "Data As Of" column: the fiscal-period-end date of the filing we used
  - "Δ Since Prior Filing" column: how the risk changed quarter-over-quarter
  - Sector and risk-level filters
  - Sortable by biggest risk increase (most useful view for forward-looking use)
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent.parent
SP500_PATH       = ROOT / "data/processed/sp500_predictions.csv"
HISTORICAL_PATH  = ROOT / "data/processed/predictions.csv"


@st.cache_data(ttl=3600)  # refresh every hour within a Streamlit Cloud session
def load_sp500() -> pd.DataFrame:
    if not SP500_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SP500_PATH)


def risk_color(p: float) -> str:
    if pd.isna(p):    return "#95a5a6"
    if p >= 0.65:     return "#e74c3c"
    if p >= 0.40:     return "#f39c12"
    return "#27ae60"


def render():
    st.title("🏆 S&P 500 Bankruptcy Filing Risk Leaderboard")

    df = load_sp500()
    if df.empty:
        st.error(
            "No S&P 500 predictions file found.\n\n"
            "Run `python refresh_sp500.py` from the project root to generate it."
        )
        return

    df = df[df["data_available"] == True].copy()

    refresh_ts = df["refresh_utc"].iloc[0] if "refresh_utc" in df.columns else "unknown"
    st.caption(
        f"Live scoring of {len(df)} S&P 500 companies. "
        f"**Data refreshed:** `{refresh_ts}` · "
        "Model: Logistic Regression trained on historical filing examples "
        "and public controls. Δ shows period-over-period change in filing risk."
    )

    # ── Sidebar filters ───────────────────────────────────────────────────────
    st.sidebar.markdown("### Filters")

    sectors = ["All"] + sorted(df["sector"].dropna().unique().tolist())
    sel_sector = st.sidebar.selectbox("Sector", sectors)

    risk_levels = st.sidebar.multiselect(
        "Risk level",
        ["🔴 High", "🟡 Medium", "🟢 Low"],
        default=["🔴 High", "🟡 Medium", "🟢 Low"],
    )

    sort_options = {
        "Current filing risk probability (highest first)": ("current_prob", False),
        "Biggest risk INCREASE this quarter":              ("delta_prob",  False),
        "Biggest risk DECREASE this quarter":              ("delta_prob",  True),
        "Ticker (A–Z)":                                    ("ticker",      True),
    }
    sort_label = st.sidebar.radio("Sort by", list(sort_options.keys()))
    sort_col, sort_asc = sort_options[sort_label]

    # ── Apply filters ─────────────────────────────────────────────────────────
    filt = df.copy()
    if sel_sector != "All":
        filt = filt[filt["sector"] == sel_sector]
    if risk_levels:
        filt = filt[filt["risk_bucket"].isin(risk_levels)]
    filt = filt.sort_values(sort_col, ascending=sort_asc, na_position="last")

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Companies", len(filt))
    c2.metric("Avg filing risk", f"{filt['current_prob'].mean():.1%}")
    c3.metric("High risk (≥65%)",   int((filt["current_prob"] >= 0.65).sum()))
    rising = int(filt["delta_prob"].gt(0.02).sum())
    falling = int(filt["delta_prob"].lt(-0.02).sum())
    c4.metric("Rising / Falling", f"{rising}  /  {falling}")

    st.markdown("---")

    # ── Main leaderboard ──────────────────────────────────────────────────────
    st.subheader("Ranked by " + sort_label.lower())

    display = filt.copy()
    display["Current Prob"]  = display["current_prob"].apply(
        lambda x: f"{x:.1%}" if pd.notna(x) else "—"
    )
    display["Prior Prob"]    = display["prior_prob"].apply(
        lambda x: f"{x:.1%}" if pd.notna(x) else "—"
    )
    display["Δ"]             = display.apply(
        lambda r: f"{r['delta_arrow']} {r['delta_prob']:+.1%}"
                   if pd.notna(r["delta_prob"]) else "—", axis=1,
    )
    display["Data As Of"]    = display.apply(
        lambda r: f"{r['current_period_end']} ({r['current_form']})"
                   if pd.notna(r["current_period_end"]) else "—", axis=1,
    )

    st.dataframe(
        display[[
            "ticker", "name", "sector",
            "risk_bucket", "Current Prob", "Prior Prob", "Δ",
            "Data As Of",
        ]].rename(columns={
            "ticker": "Ticker", "name": "Company",
            "sector": "Sector", "risk_bucket": "Risk",
        }),
        use_container_width=True,
        height=600,
    )

    st.markdown("---")

    # ── Visuals row ───────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Risk Distribution")
        fig = px.histogram(
            filt, x="current_prob", nbins=25,
            color_discrete_sequence=["#2980b9"],
        )
        fig.add_vline(x=0.40, line_dash="dash", line_color="#f39c12",
                      annotation_text="Medium", annotation_position="top")
        fig.add_vline(x=0.65, line_dash="dash", line_color="#e74c3c",
                      annotation_text="High", annotation_position="top")
        fig.update_xaxes(tickformat=".0%", title="Current Filing Risk Probability")
        fig.update_yaxes(title="# Companies")
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Biggest Quarter-Over-Quarter Changes")
        moves = filt.dropna(subset=["delta_prob"]).copy()
        moves["abs_delta"] = moves["delta_prob"].abs()
        top_moves = moves.nlargest(15, "abs_delta").sort_values("delta_prob")
        fig2 = go.Figure(go.Bar(
            x=top_moves["delta_prob"],
            y=top_moves["ticker"],
            orientation="h",
            marker_color=[("#e74c3c" if d > 0 else "#27ae60")
                          for d in top_moves["delta_prob"]],
            text=[f"{d:+.1%}" for d in top_moves["delta_prob"]],
            textposition="outside",
        ))
        fig2.update_layout(
            height=350, xaxis_tickformat=".0%",
            xaxis_title="Δ Filing Risk Probability",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Sector breakdown ──────────────────────────────────────────────────────
    st.subheader("Average Risk by Sector")
    by_sector = (
        filt.groupby("sector")["current_prob"]
        .mean().sort_values(ascending=False).reset_index()
    )
    fig3 = px.bar(
        by_sector, x="current_prob", y="sector", orientation="h",
        color="current_prob", color_continuous_scale="RdYlGn_r",
        text=by_sector["current_prob"].apply(lambda p: f"{p:.1%}"),
    )
    fig3.update_layout(
        height=400, xaxis_tickformat=".0%",
        xaxis_title="Average Filing Risk Probability",
        yaxis_title="Sector", coloraxis_showscale=False,
    )
    st.plotly_chart(fig3, use_container_width=True)
