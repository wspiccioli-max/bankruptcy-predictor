"""
Page 2 — Company Lookup

Search any S&P 500 company by ticker.
Shows KPI cards, a health gauge, the Altman Z-Score, and quarter-over-quarter
change in bankruptcy filing risk probability.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT            = Path(__file__).parent.parent
SP500_PATH      = ROOT / "data/processed/sp500_predictions.csv"
HISTORICAL_PATH = ROOT / "data/processed/predictions.csv"


@st.cache_data(ttl=3600)
def load_sp500() -> pd.DataFrame:
    if not SP500_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(SP500_PATH)


@st.cache_data
def load_historical() -> pd.DataFrame:
    return pd.read_csv(HISTORICAL_PATH)


def health_gauge(p: float) -> go.Figure:
    score = (1 - p) * 100
    color = "#27ae60" if score >= 67 else ("#f39c12" if score >= 34 else "#e74c3c")
    label = "Healthy" if score >= 67 else ("At Risk" if score >= 34 else "Distressed")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 100"},
        title={"text": f"Financial Health Score<br><sub>{label}</sub>"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0, 33],   "color": "#fadbd8"},
                {"range": [33, 67],  "color": "#fdebd0"},
                {"range": [67, 100], "color": "#d5f5e3"},
            ],
        },
    ))
    fig.update_layout(height=300, margin=dict(t=70, b=10, l=30, r=30))
    return fig


def render():
    st.title("🔍 Company Lookup")

    sp500 = load_sp500()
    sp500 = sp500[sp500.get("data_available", False) == True].copy() if not sp500.empty else sp500

    if sp500.empty:
        st.warning(
            "S&P 500 data not yet generated. Run `python refresh_sp500.py` first."
        )
        return

    tickers = sorted(sp500["ticker"].tolist())
    default = tickers.index("AAPL") if "AAPL" in tickers else 0
    selected = st.selectbox("Choose a company (S&P 500)", tickers, index=default)

    row = sp500[sp500["ticker"] == selected].iloc[0]

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        f"## {row['name']} &nbsp; `{row['ticker']}`\n"
        f"**Sector:** {row.get('sector', '—')} &nbsp;|&nbsp; "
        f"**Industry:** {row.get('industry', '—')}"
    )

    if pd.notna(row.get("current_period_end")):
        st.info(
            f"📅 **Data as of:** {row['current_period_end']} "
            f"({row['current_form']}) &nbsp; · &nbsp; "
            f"**Prior filing:** {row.get('prior_period_end', '—')} "
            f"({row.get('prior_form', '—')})"
        )

    st.markdown("---")

    # ── KPI row — current + delta ─────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    current_prob = row["current_prob"]
    prior_prob   = row.get("prior_prob")
    delta        = row.get("delta_prob")

    c1.metric(
        "Current Filing Risk",
        f"{current_prob:.1%}" if pd.notna(current_prob) else "—",
        delta=(f"{delta:+.1%}" if pd.notna(delta) else None),
        delta_color="inverse",   # rising risk shown in red
        help="Logistic regression probability of a bankruptcy filing risk signal based on the most recent SEC filing.",
    )
    c2.metric(
        "Prior Filing Prob",
        f"{prior_prob:.1%}" if pd.notna(prior_prob) else "—",
        help="Same model applied to the previous period's filing.",
    )
    cr = row.get("current_current_ratio")
    c3.metric(
        "Current Ratio",
        f"{cr:.2f}" if pd.notna(cr) else "—",
        help="Current Assets / Current Liabilities. Below 1.0 = can't cover short-term bills.",
    )
    dte = row.get("current_debt_to_equity")
    c4.metric(
        "Debt / Equity",
        f"{dte:.2f}" if pd.notna(dte) else "—",
        help="Higher = more leveraged.",
    )
    roa = row.get("current_return_on_assets")
    c5.metric(
        "Return on Assets",
        f"{roa:.1%}" if pd.notna(roa) else "—",
        help="Net Income / Total Assets.",
    )

    st.markdown("---")

    # ── Gauges ────────────────────────────────────────────────────────────────
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.plotly_chart(health_gauge(float(current_prob)), use_container_width=True)

    with col_g2:
        # Build a simple "before vs. after" bar
        fig = go.Figure(go.Bar(
            x=["Prior filing", "Current filing"],
            y=[prior_prob if pd.notna(prior_prob) else 0,
               current_prob if pd.notna(current_prob) else 0],
            marker_color=["#95a5a6",
                          "#e74c3c" if current_prob >= 0.65 else
                          ("#f39c12" if current_prob >= 0.40 else "#27ae60")],
            text=[
                f"{prior_prob:.1%}" if pd.notna(prior_prob) else "n/a",
                f"{current_prob:.1%}" if pd.notna(current_prob) else "n/a",
            ],
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Period-Over-Period Filing Risk Probability",
            yaxis_tickformat=".0%",
            yaxis_range=[0, max(1.0, (current_prob or 0) + 0.1)],
            height=300,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Feature breakdown ─────────────────────────────────────────────────────
    st.subheader("Financial Ratio Breakdown (current filing)")

    feat_map = {
        "current_current_ratio":     ("Current Ratio",            "> 1.0 is healthy"),
        "current_debt_to_equity":    ("Debt / Equity",             "< 2.0 is conservative"),
        "current_return_on_assets":  ("Return on Assets",          "> 5% is strong"),
        "current_net_margin":        ("Net Margin",                "> 10% is strong"),
        "current_interest_coverage": ("Interest Coverage",         "> 3x is safe"),
    }
    breakdown = []
    for col, (label, benchmark) in feat_map.items():
        v = row.get(col)
        breakdown.append({
            "Ratio":     label,
            "Value":     (f"{v:.3f}" if pd.notna(v) else "—"),
            "Benchmark": benchmark,
        })
    st.table(pd.DataFrame(breakdown))
