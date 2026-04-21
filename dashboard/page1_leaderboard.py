"""
Page 1 — Bankruptcy Risk Leaderboard

Shows all companies ranked by their logistic regression bankruptcy probability.
Users can filter by sector and company size to narrow the view.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data/processed/predictions.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def risk_color(prob: float) -> str:
    """Map a bankruptcy probability to a traffic-light color."""
    if prob >= 0.65:
        return "#e74c3c"   # red
    elif prob >= 0.40:
        return "#f39c12"   # amber
    else:
        return "#27ae60"   # green


def risk_label(prob: float) -> str:
    if prob >= 0.65:
        return "🔴 High"
    elif prob >= 0.40:
        return "🟡 Medium"
    else:
        return "🟢 Low"


def render():
    st.title("🏆 Bankruptcy Risk Leaderboard")
    st.caption(
        "Companies ranked by logistic regression bankruptcy probability. "
        "Use the filters on the left to narrow the view."
    )

    df = load_data()

    # ── Sidebar filters ───────────────────────────────────────────────────────
    st.sidebar.markdown("### Filters")

    sectors = ["All"] + sorted(df["sector"].dropna().unique().tolist())
    selected_sector = st.sidebar.selectbox("Sector", sectors)

    sizes = ["All"] + sorted(df["company_size"].dropna().unique().tolist())
    selected_size = st.sidebar.selectbox("Company Size", sizes)

    show_only = st.sidebar.radio(
        "Show",
        ["All companies", "Bankrupt only", "Healthy only"],
    )

    # ── Apply filters ─────────────────────────────────────────────────────────
    filtered = df.copy()
    if selected_sector != "All":
        filtered = filtered[filtered["sector"] == selected_sector]
    if selected_size != "All":
        filtered = filtered[filtered["company_size"] == selected_size]
    if show_only == "Bankrupt only":
        filtered = filtered[filtered["bankrupt"] == 1]
    elif show_only == "Healthy only":
        filtered = filtered[filtered["bankrupt"] == 0]

    filtered = filtered.sort_values("lr_prob_bankrupt", ascending=False)

    # ── KPI row ───────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Companies shown", len(filtered))
    col2.metric("Bankrupt (actual)", int(filtered["bankrupt"].sum()))
    col3.metric(
        "Avg bankruptcy probability",
        f"{filtered['lr_prob_bankrupt'].mean():.1%}",
    )
    col4.metric(
        "High-risk companies",
        int((filtered["lr_prob_bankrupt"] >= 0.65).sum()),
    )

    st.markdown("---")

    # ── Main leaderboard table ────────────────────────────────────────────────
    st.subheader("Ranked by Bankruptcy Probability")

    display_df = filtered[[
        "ticker", "name", "sector", "company_size", "bankrupt",
        "lr_prob_bankrupt", "z_score", "z_zone",
        "fraud_risk_score", "current_ratio", "debt_to_equity",
    ]].copy()

    display_df["Risk Level"] = display_df["lr_prob_bankrupt"].apply(risk_label)
    display_df["Actual Outcome"] = display_df["bankrupt"].map(
        {1: "💀 Bankrupt", 0: "✅ Healthy"}
    )
    display_df["Bankruptcy Prob"] = display_df["lr_prob_bankrupt"].apply(
        lambda x: f"{x:.1%}"
    )
    display_df["Z-Score"] = display_df["z_score"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "—"
    )
    display_df["Current Ratio"] = display_df["current_ratio"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "—"
    )
    display_df["D/E Ratio"] = display_df["debt_to_equity"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "—"
    )

    st.dataframe(
        display_df[[
            "ticker", "name", "sector", "company_size",
            "Actual Outcome", "Risk Level", "Bankruptcy Prob",
            "Z-Score", "z_zone", "fraud_risk_score",
            "Current Ratio", "D/E Ratio",
        ]].rename(columns={
            "ticker": "Ticker",
            "name": "Company",
            "sector": "Sector",
            "company_size": "Size",
            "z_zone": "Z-Zone",
            "fraud_risk_score": "Fraud Flags",
        }),
        use_container_width=True,
        height=500,
    )

    st.markdown("---")

    # ── Bar chart: bankruptcy probability by company ───────────────────────────
    st.subheader("Bankruptcy Probability by Company")
    colors = [risk_color(p) for p in filtered["lr_prob_bankrupt"]]

    fig = go.Figure(go.Bar(
        x=filtered["ticker"],
        y=filtered["lr_prob_bankrupt"],
        marker_color=colors,
        text=[f"{p:.0%}" for p in filtered["lr_prob_bankrupt"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Bankruptcy prob: %{y:.1%}<br>"
            "<extra></extra>"
        ),
    ))
    fig.add_hline(y=0.65, line_dash="dash", line_color="#e74c3c",
                  annotation_text="High risk threshold (65%)")
    fig.add_hline(y=0.40, line_dash="dash", line_color="#f39c12",
                  annotation_text="Medium risk threshold (40%)")
    fig.update_layout(
        yaxis_tickformat=".0%",
        yaxis_title="Predicted Bankruptcy Probability",
        xaxis_title="Company Ticker",
        showlegend=False,
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Sector breakdown pie ──────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Companies by Sector")
        sector_counts = filtered["sector"].value_counts().reset_index()
        sector_counts.columns = ["Sector", "Count"]
        fig_pie = px.pie(
            sector_counts, names="Sector", values="Count",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.subheader("Fraud Risk Distribution")
        fraud_counts = filtered["fraud_risk_score"].value_counts().sort_index().reset_index()
        fraud_counts.columns = ["Fraud Flags (0–5)", "Companies"]
        fig_bar = px.bar(
            fraud_counts,
            x="Fraud Flags (0–5)", y="Companies",
            color="Fraud Flags (0–5)",
            color_continuous_scale=["#27ae60", "#f39c12", "#e74c3c"],
        )
        fig_bar.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
