"""
Page 2 — Company Lookup

Search any company in our dataset by ticker.
Shows KPI cards, a health gauge, Z-score breakdown, and fraud flags.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data/processed/predictions.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def health_gauge(health_score: float, label: str) -> go.Figure:
    """
    Build a speedometer-style gauge chart.
    health_score: 0 (worst) → 100 (best)
    """
    if health_score >= 67:
        color = "#27ae60"   # green
    elif health_score >= 34:
        color = "#f39c12"   # amber
    else:
        color = "#e74c3c"   # red

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_score,
        title={"text": f"Financial Health Score<br><sub>{label}</sub>"},
        delta={"reference": 50, "valueformat": ".0f"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 33],  "color": "#fadbd8"},
                {"range": [33, 67], "color": "#fdebd0"},
                {"range": [67, 100], "color": "#d5f5e3"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
        number={"suffix": " / 100"},
    ))
    fig.update_layout(height=300, margin=dict(t=80, b=0, l=30, r=30))
    return fig


def zscore_gauge(z: float) -> go.Figure:
    """Visualise the Altman Z-Score on a colour-coded axis."""
    color = "#27ae60" if z > 2.99 else ("#f39c12" if z >= 1.81 else "#e74c3c")
    zone  = "Safe Zone" if z > 2.99 else ("Grey Zone" if z >= 1.81 else "Distress Zone")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=z,
        title={"text": f"Altman Z-Score<br><sub>{zone}</sub>"},
        gauge={
            "axis": {"range": [0, 6], "tickvals": [0, 1.81, 2.99, 6]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 1.81],    "color": "#fadbd8"},
                {"range": [1.81, 2.99], "color": "#fdebd0"},
                {"range": [2.99, 6],    "color": "#d5f5e3"},
            ],
        },
    ))
    fig.update_layout(height=300, margin=dict(t=80, b=0, l=30, r=30))
    return fig


def kpi_card(col, label: str, value: str, delta: str = None, help_text: str = None):
    col.metric(label=label, value=value, delta=delta, help=help_text)


def render():
    st.title("🔍 Company Lookup")
    st.caption("Select a company to see its full financial health profile.")

    df = load_data()

    # ── Company selector ──────────────────────────────────────────────────────
    tickers = sorted(df["ticker"].tolist())
    selected = st.selectbox(
        "Choose a company ticker",
        tickers,
        index=tickers.index("AAPL") if "AAPL" in tickers else 0,
    )

    row = df[df["ticker"] == selected].iloc[0]

    # ── Header ────────────────────────────────────────────────────────────────
    outcome_badge = "💀 Filed for Bankruptcy" if row["bankrupt"] == 1 else "✅ Still Operating"
    outcome_color = "#e74c3c" if row["bankrupt"] == 1 else "#27ae60"

    st.markdown(
        f"## {row['name']} &nbsp; `{row['ticker']}`\n"
        f"**Sector:** {row['sector']} &nbsp;|&nbsp; "
        f"**Size:** {row['company_size']} &nbsp;|&nbsp; "
        f"**Actual outcome:** "
        f"<span style='color:{outcome_color}; font-weight:bold'>{outcome_badge}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── KPI cards row ─────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)

    prob = row["lr_prob_bankrupt"]
    kpi_card(
        c1, "Bankruptcy Probability",
        f"{prob:.1%}",
        help_text="Logistic regression prediction. Higher = more risk.",
    )

    cr = row["current_ratio"]
    kpi_card(
        c2, "Current Ratio",
        f"{cr:.2f}" if pd.notna(cr) else "—",
        delta="< 1.0 = danger" if pd.notna(cr) and cr < 1.0 else None,
        help_text="Current Assets / Current Liabilities. Below 1.0 means the company can't cover short-term debts.",
    )

    dte = row["debt_to_equity"]
    kpi_card(
        c3, "Debt / Equity",
        f"{dte:.2f}" if pd.notna(dte) else "—",
        help_text="Total Liabilities / Stockholders Equity. Higher = more leveraged.",
    )

    roa = row["return_on_assets"]
    kpi_card(
        c4, "Return on Assets",
        f"{roa:.1%}" if pd.notna(roa) else "—",
        help_text="Net Income / Total Assets. How efficiently the company converts assets to profit.",
    )

    ic = row["interest_coverage"]
    kpi_card(
        c5, "Interest Coverage",
        f"{ic:.1f}x" if pd.notna(ic) else "—",
        delta="< 1.5x = strain" if pd.notna(ic) and ic < 1.5 else None,
        help_text="EBIT / Interest Expense. How many times over the company can pay its interest. Below 1.5x is a red flag.",
    )

    st.markdown("---")

    # ── Gauges row ────────────────────────────────────────────────────────────
    col_gauge, col_z = st.columns(2)

    with col_gauge:
        # Convert bankruptcy probability → health score (inverse)
        health_score = (1 - prob) * 100
        if health_score >= 67:
            hlabel = "Healthy"
        elif health_score >= 34:
            hlabel = "At Risk"
        else:
            hlabel = "Distressed"
        st.plotly_chart(health_gauge(health_score, hlabel), use_container_width=True)

    with col_z:
        z = row.get("z_score")
        if pd.notna(z):
            st.plotly_chart(zscore_gauge(float(z)), use_container_width=True)
        else:
            st.info(
                "**Altman Z-Score unavailable**\n\n"
                "This company's Z-score could not be computed because "
                "market cap data was not available from yfinance (the stock "
                "may be delisted or Yahoo Finance was rate-limited). "
                "The logistic regression still works using EDGAR financial ratios."
            )

    st.markdown("---")

    # ── Fraud flag breakdown ──────────────────────────────────────────────────
    st.subheader("Fraud & Distress Red Flags")

    flags = {
        "Extreme Leverage (D/E > 3)":         row.get("flag_leverage"),
        "Negative Profit Margin":              row.get("flag_neg_margin"),
        "Low Liquidity (Current Ratio < 1)":   row.get("flag_low_liquidity"),
        "Asset Bloat (Assets > 3× Revenue)":   row.get("flag_asset_bloat"),
        "Interest Strain (Coverage < 1.5×)":   row.get("flag_interest_strain"),
    }

    # Load the features CSV which has the flag columns
    features_path = Path(__file__).parent.parent / "data/processed/features.csv"
    feat_df = pd.read_csv(features_path)
    feat_row = feat_df[feat_df["ticker"] == selected]

    if not feat_row.empty:
        feat_row = feat_row.iloc[0]
        flags = {
            "Extreme Leverage (D/E > 3)":         feat_row.get("flag_leverage", 0),
            "Negative Profit Margin":              feat_row.get("flag_neg_margin", 0),
            "Low Liquidity (Current Ratio < 1)":   feat_row.get("flag_low_liquidity", 0),
            "Asset Bloat (Assets > 3× Revenue)":   feat_row.get("flag_asset_bloat", 0),
            "Interest Strain (Coverage < 1.5×)":   feat_row.get("flag_interest_strain", 0),
        }

    total_flags = sum(v for v in flags.values() if pd.notna(v))
    risk_label = row["fraud_risk_label"]
    risk_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(risk_label, "⚪")

    st.markdown(
        f"**Fraud Risk Level:** {risk_emoji} {risk_label} &nbsp;|&nbsp; "
        f"**Total flags:** {total_flags} / 5"
    )

    flag_cols = st.columns(5)
    for i, (flag_name, flag_val) in enumerate(flags.items()):
        is_set = bool(flag_val) if pd.notna(flag_val) else False
        flag_cols[i].markdown(
            f"{'🚩' if is_set else '✅'}  \n**{flag_name}**",
        )

    st.markdown("---")

    # ── Model predictions summary ─────────────────────────────────────────────
    st.subheader("What Each Model Says")

    pred_data = {
        "Model": ["Logistic Regression", "Altman Z-Score", "Fraud Risk Score"],
        "Prediction": [
            "💀 Bankrupt" if row["lr_pred_bankrupt"] == 1 else "✅ Healthy",
            "💀 Bankrupt" if row["z_pred_bankrupt"] == 1 else "✅ Healthy",
            "💀 Bankrupt" if row["fraud_pred_bankrupt"] == 1 else "✅ Healthy",
        ],
        "Confidence / Score": [
            f"{row['lr_prob_bankrupt']:.1%} probability",
            f"Z = {row['z_score']:.2f}" if pd.notna(row['z_score']) else "Z = N/A",
            f"{int(row['fraud_risk_score'])} / 5 flags",
        ],
    }
    st.table(pd.DataFrame(pred_data))

    actual = "💀 Bankrupt" if row["bankrupt"] == 1 else "✅ Healthy"
    st.success(f"**Actual outcome:** {actual}")
