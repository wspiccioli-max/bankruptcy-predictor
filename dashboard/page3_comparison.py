"""
Page 3 — Model Validation (Historical Backtest)

Evaluates the logistic regression model on the 30 hand-picked historical
companies (15 bankrupt, 15 healthy) used to train it. This establishes
whether the model is good enough to trust on the live S&P 500 data
shown in Page 1.

Shows:
  - ROC curves for logistic regression + two rule-based benchmarks
  - Feature importance bar chart (what does the model rely on?)
  - Confusion matrices
  - Comparison table
"""

import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix

ROOT = Path(__file__).parent.parent
PRED_PATH    = ROOT / "data/processed/predictions.csv"
FEAT_PATH    = ROOT / "data/processed/features.csv"
MODEL_PATH   = ROOT / "models/logistic_model.pkl"
COMPARE_PATH = ROOT / "data/processed/model_comparison.csv"
FI_PATH      = ROOT / "data/processed/feature_importance.csv"


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PRED_PATH)


@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_csv(FEAT_PATH)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def make_roc_figure(df: pd.DataFrame) -> go.Figure:
    """
    Build a Plotly ROC curve for all three models on one chart.

    ROC = Receiver Operating Characteristic.
    The curve shows the trade-off between:
      - True Positive Rate (TPR): did we catch the bankruptcies?
      - False Positive Rate (FPR): did we falsely flag healthy companies?
    A perfect model hugs the top-left corner. A random guess follows
    the diagonal dashed line (AUC = 0.5).
    """
    y_true = df["bankrupt"].values

    fig = go.Figure()

    # ── Logistic Regression ───────────────────────────────────────────────────
    lr_probs = df["lr_prob_bankrupt"].values
    fpr, tpr, _ = roc_curve(y_true, lr_probs)
    lr_auc = auc(fpr, tpr)
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"Logistic Regression (AUC = {lr_auc:.3f})",
        line=dict(color="#2980b9", width=2.5),
    ))

    # ── Altman Z-Score (only companies with a Z-score) ────────────────────────
    z_sub = df.dropna(subset=["z_score"])
    if len(z_sub) > 1 and z_sub["bankrupt"].nunique() > 1:
        z_probs = 1 - (z_sub["z_score"] / z_sub["z_score"].max()).clip(0, 1)
        fpr_z, tpr_z, _ = roc_curve(z_sub["bankrupt"].values, z_probs.values)
        z_auc = auc(fpr_z, tpr_z)
        fig.add_trace(go.Scatter(
            x=fpr_z, y=tpr_z,
            mode="lines",
            name=f"Altman Z-Score (AUC = {z_auc:.3f}, n={len(z_sub)})",
            line=dict(color="#e74c3c", width=2.5, dash="dot"),
        ))

    # ── Fraud Risk Score ──────────────────────────────────────────────────────
    f_probs = (df["fraud_risk_score"] / 5).values
    fpr_f, tpr_f, _ = roc_curve(y_true, f_probs)
    f_auc = auc(fpr_f, tpr_f)
    fig.add_trace(go.Scatter(
        x=fpr_f, y=tpr_f,
        mode="lines",
        name=f"Fraud Risk Score (AUC = {f_auc:.3f})",
        line=dict(color="#27ae60", width=2.5, dash="dash"),
    ))

    # ── Random baseline ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random Guess (AUC = 0.500)",
        line=dict(color="grey", width=1, dash="longdash"),
    ))

    fig.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate (healthy companies wrongly flagged)",
        yaxis_title="True Positive Rate (bankruptcies correctly caught)",
        legend=dict(x=0.55, y=0.08),
        height=480,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
    )
    return fig


def make_feature_importance_chart() -> go.Figure:
    """
    Horizontal bar chart of logistic regression feature importances.
    Each bar = absolute value of the model's coefficient for that feature.
    Bigger bar = that ratio matters more for the prediction.
    """
    fi = pd.read_csv(FI_PATH)
    fi = fi.sort_values("importance", ascending=True)

    # Human-readable feature names
    name_map = {
        "return_on_assets":  "Return on Assets",
        "debt_to_equity":    "Debt / Equity",
        "z_x1":              "Z-X1: Working Capital / Assets",
        "z_x3":              "Z-X3: EBIT / Assets",
        "z_x5":              "Z-X5: Revenue / Assets",
        "current_ratio":     "Current Ratio",
        "interest_coverage": "Interest Coverage",
        "z_x2":              "Z-X2: Retained Earnings / Assets",
        "net_margin":        "Net Profit Margin",
        "z_x4":              "Z-X4: Equity / Liabilities",
    }
    fi["label"] = fi["feature"].map(name_map).fillna(fi["feature"])

    fig = go.Figure(go.Bar(
        x=fi["importance"],
        y=fi["label"],
        orientation="h",
        marker_color=px.colors.sequential.Blues_r[:len(fi)],
        text=[f"{v:.3f}" for v in fi["importance"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Feature Importance (Logistic Regression Coefficients)",
        xaxis_title="Absolute Coefficient Value",
        height=420,
        margin=dict(l=200),
    )
    return fig


def confusion_matrix_fig(cm: np.ndarray, title: str) -> go.Figure:
    """Heatmap-style confusion matrix."""
    labels = ["Healthy", "Bankrupt"]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=[f"Pred {l}" for l in labels],
        y=[f"True {l}" for l in labels],
        colorscale=[[0, "#d5f5e3"], [1, "#fadbd8"]],
        text=cm,
        texttemplate="%{text}",
        showscale=False,
    ))
    fig.update_layout(title=title, height=280, margin=dict(t=40))
    return fig


def render():
    st.title("📊 Model Validation — Historical Backtest")
    st.caption(
        "Evaluates the logistic regression on 30 hand-picked historical companies "
        "(15 bankrupt, 15 healthy). This is the backtest that justifies using the "
        "model to score the live S&P 500 data on Page 1."
    )

    df_pred = load_predictions()
    y_true  = df_pred["bankrupt"].values

    # ── Comparison table at the top ───────────────────────────────────────────
    st.subheader("Summary Statistics")

    compare = pd.read_csv(COMPARE_PATH)
    st.dataframe(compare, use_container_width=True, hide_index=True)

    st.info(
        "**Why is accuracy so low?** We only have 24 companies — far too few "
        "for a reliable ML model. The logistic regression has 10 features but "
        "only 24 samples (rule of thumb: need 10–20 samples *per feature*). "
        "In a real project you'd pull thousands of 10-K filings. "
        "The ROC-AUC and confusion matrices still reveal meaningful patterns."
    )

    st.markdown("---")

    # ── ROC curve ─────────────────────────────────────────────────────────────
    st.subheader("ROC Curves")
    st.caption(
        "The closer the curve hugs the top-left corner, the better. "
        "AUC = 1.0 is perfect. AUC = 0.5 is a random guess."
    )
    st.plotly_chart(make_roc_figure(df_pred), use_container_width=True)

    st.markdown("---")

    # ── Feature importance ────────────────────────────────────────────────────
    st.subheader("What Does the Logistic Regression Rely On?")
    st.caption(
        "Larger bar = that ratio had more influence on the model's prediction. "
        "Return on Assets and Debt/Equity dominate — assets turning profit "
        "and how much a company owes vs. owns are the clearest warning signs."
    )
    st.plotly_chart(make_feature_importance_chart(), use_container_width=True)

    st.markdown("---")

    # ── Confusion matrices ────────────────────────────────────────────────────
    st.subheader("Confusion Matrices")
    st.caption(
        "Each cell shows how many companies fell into each prediction bucket. "
        "Bottom-right = correctly caught bankruptcies (true positives). "
        "Bottom-left = missed bankruptcies (false negatives — the dangerous kind)."
    )

    col1, col2, col3 = st.columns(3)

    # Logistic regression
    lr_preds = df_pred["lr_pred_bankrupt"].values
    cm_lr = confusion_matrix(y_true, lr_preds)
    col1.plotly_chart(
        confusion_matrix_fig(cm_lr, "Logistic Regression"),
        use_container_width=True,
    )

    # Z-score (on subset)
    z_sub = df_pred.dropna(subset=["z_score"])
    if len(z_sub) >= 2:
        z_preds = z_sub["z_pred_bankrupt"].values
        cm_z = confusion_matrix(z_sub["bankrupt"].values, z_preds)
        col2.plotly_chart(
            confusion_matrix_fig(cm_z, f"Altman Z-Score (n={len(z_sub)})"),
            use_container_width=True,
        )
    else:
        col2.info("Not enough Z-scores to display.")

    # Fraud risk score
    f_preds = df_pred["fraud_pred_bankrupt"].values
    cm_f = confusion_matrix(y_true, f_preds)
    col3.plotly_chart(
        confusion_matrix_fig(cm_f, "Fraud Risk Score"),
        use_container_width=True,
    )

    st.markdown("---")

    # ── Per-company prediction table ──────────────────────────────────────────
    st.subheader("Per-Company Predictions vs. Reality")

    display = df_pred[[
        "ticker", "name", "bankrupt",
        "lr_pred_bankrupt", "z_pred_bankrupt", "fraud_pred_bankrupt",
        "lr_prob_bankrupt", "z_score", "fraud_risk_score",
    ]].copy()

    display["Actual"]  = display["bankrupt"].map({1: "💀 Bankrupt", 0: "✅ Healthy"})
    display["LR Pred"] = display["lr_pred_bankrupt"].map({1: "💀", 0: "✅"})
    display["Z Pred"]  = display["z_pred_bankrupt"].map({1: "💀", 0: "✅"})
    display["FR Pred"] = display["fraud_pred_bankrupt"].map({1: "💀", 0: "✅"})
    display["LR Prob"] = display["lr_prob_bankrupt"].apply(lambda x: f"{x:.1%}")
    display["Z-Score"] = display["z_score"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "—"
    )
    display["Fraud Flags"] = display["fraud_risk_score"].apply(lambda x: f"{int(x)}/5")

    # Highlight rows where at least 2 models disagree
    st.dataframe(
        display[[
            "ticker", "name", "Actual",
            "LR Pred", "Z Pred", "FR Pred",
            "LR Prob", "Z-Score", "Fraud Flags",
        ]].rename(columns={
            "ticker": "Ticker", "name": "Company",
        }),
        use_container_width=True,
        height=450,
    )
