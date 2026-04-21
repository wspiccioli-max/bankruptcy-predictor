"""
Phase 3 — Model Training & Evaluation

Run this script after feature_engineering.py:
  python train_models.py

What it does:
  1. Loads the feature matrix from Phase 2
  2. Trains a logistic regression model using cross-validation
     (cross-val is essential here because we only have 24 companies —
      a simple 80/20 split would leave us with ~5 test samples, which is
      too few to trust any metric)
  3. Evaluates the Altman Z-Score as a rule-based benchmark
  4. Evaluates the Fraud Risk Score as a second benchmark
  5. Prints accuracy, ROC-AUC, and confusion matrix for all three
  6. Saves the trained model + scaler for use in the dashboard
  7. Saves per-company predictions to data/processed/predictions.csv
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, cross_val_score
)
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, RocCurveDisplay
)

FEATURES_PATH = Path("data/processed/features.csv")
MODELS_DIR    = Path("models")
RESULTS_DIR   = Path("data/processed")
MODELS_DIR.mkdir(exist_ok=True)

# The 10 financial features our logistic regression will train on
MODEL_FEATURES = [
    "current_ratio",
    "debt_to_equity",
    "return_on_assets",
    "net_margin",
    "interest_coverage",
    "z_x1", "z_x2", "z_x3", "z_x4", "z_x5",
]


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading feature matrix...")
df = pd.read_csv(FEATURES_PATH)
print(f"  {len(df)} companies | {df['bankrupt'].sum()} bankrupt | "
      f"{(df['bankrupt']==0).sum()} healthy\n")

X = df[MODEL_FEATURES].values
y = df["bankrupt"].values


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: Logistic Regression (trained on our financial features)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("MODEL 1: Logistic Regression")
print("=" * 60)

# StandardScaler transforms all features to have mean=0 and SD=1.
# This is required for logistic regression because features like
# total_assets (billions) would otherwise swamp current_ratio (1–3).
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Logistic regression with class_weight='balanced' gives extra weight
# to the minority class (bankrupt=1) so the model doesn't just predict
# "healthy" for everything.
lr = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
)

# StratifiedKFold ensures each fold has roughly the same bankrupt/healthy ratio.
# With n_splits=5 and 24 samples we get ~5 test companies per fold.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# cross_val_predict trains on 4 folds, predicts the held-out 5th fold,
# repeating for all 5 folds → every company gets exactly one prediction
# from a model that never saw it. Much more honest than a single split.
lr_probs = cross_val_predict(lr, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
lr_preds = (lr_probs >= 0.5).astype(int)

lr_acc   = accuracy_score(y, lr_preds)
lr_auc   = roc_auc_score(y, lr_probs)
lr_cm    = confusion_matrix(y, lr_preds)

print(f"  Accuracy : {lr_acc:.1%}")
print(f"  ROC-AUC  : {lr_auc:.3f}  (1.0 = perfect, 0.5 = coin flip)")
print(f"\n  Confusion Matrix:")
print(f"              Predicted Healthy  Predicted Bankrupt")
print(f"  True Healthy        {lr_cm[0,0]:>2}                  {lr_cm[0,1]:>2}")
print(f"  True Bankrupt       {lr_cm[1,0]:>2}                  {lr_cm[1,1]:>2}")
print(f"\n  Classification Report:")
print(classification_report(y, lr_preds, target_names=["Healthy", "Bankrupt"]))

# Now fit on ALL data so we have a model to save for the dashboard
lr.fit(X_scaled, y)
lr_feature_importance = pd.Series(
    np.abs(lr.coef_[0]),
    index=MODEL_FEATURES
).sort_values(ascending=False)

print("  Feature Importance (by absolute coefficient):")
for feat, val in lr_feature_importance.items():
    bar = "█" * int(val * 10)
    print(f"    {feat:<22} {val:.3f}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: Altman Z-Score (rule-based benchmark, no training)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 2: Altman Z-Score Benchmark")
print("=" * 60)
print("  Rule: Z < 1.81 → Bankrupt,  Z >= 1.81 → Healthy")
print("  (Z-score is unknown for companies missing market cap data —")
print("   those are excluded from this benchmark's metrics)\n")

z_mask = df["z_score"].notna()
z_sub  = df[z_mask]
y_z    = z_sub["bankrupt"].values

# Z < 1.81 → predict bankrupt
z_preds = (z_sub["z_score"] < 1.81).astype(int).values
z_probs = 1 - (z_sub["z_score"] / z_sub["z_score"].max()).clip(0, 1).values

z_acc = accuracy_score(y_z, z_preds)
z_auc = roc_auc_score(y_z, z_probs) if len(np.unique(y_z)) > 1 else float("nan")
z_cm  = confusion_matrix(y_z, z_preds)

print(f"  Evaluated on {z_mask.sum()} companies with available Z-scores")
print(f"  Accuracy : {z_acc:.1%}")
print(f"  ROC-AUC  : {z_auc:.3f}")
print(f"\n  Confusion Matrix:")
print(f"              Predicted Healthy  Predicted Bankrupt")
print(f"  True Healthy        {z_cm[0,0]:>2}                  {z_cm[0,1]:>2}")
print(f"  True Bankrupt       {z_cm[1,0]:>2}                  {z_cm[1,1]:>2}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: Fraud Risk Score (heuristic benchmark)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL 3: Fraud Risk Score Benchmark")
print("=" * 60)
print("  Rule: fraud_risk_score >= 2 → Bankrupt, < 2 → Healthy\n")

f_preds = (df["fraud_risk_score"] >= 2).astype(int).values
f_probs = (df["fraud_risk_score"] / 5).values  # normalise to [0,1]

f_acc = accuracy_score(y, f_preds)
f_auc = roc_auc_score(y, f_probs)
f_cm  = confusion_matrix(y, f_preds)

print(f"  Accuracy : {f_acc:.1%}")
print(f"  ROC-AUC  : {f_auc:.3f}")
print(f"\n  Confusion Matrix:")
print(f"              Predicted Healthy  Predicted Bankrupt")
print(f"  True Healthy        {f_cm[0,0]:>2}                  {f_cm[0,1]:>2}")
print(f"  True Bankrupt       {f_cm[1,0]:>2}                  {f_cm[1,1]:>2}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary comparison table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)
comparison = pd.DataFrame({
    "Model":    ["Logistic Regression", "Altman Z-Score", "Fraud Risk Score"],
    "Accuracy": [f"{lr_acc:.1%}", f"{z_acc:.1%}", f"{f_acc:.1%}"],
    "ROC-AUC":  [f"{lr_auc:.3f}", f"{z_auc:.3f}", f"{f_auc:.3f}"],
    "Sample N": [len(y), int(z_mask.sum()), len(y)],
})
print(comparison.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────────────────────────────────────
print("\nSaving model artifacts...")

# 1. Trained logistic regression + scaler (needed by the dashboard)
with open(MODELS_DIR / "logistic_model.pkl", "wb") as f:
    pickle.dump({"model": lr, "scaler": scaler, "features": MODEL_FEATURES}, f)
print("  Saved → models/logistic_model.pkl")

# 2. Per-company predictions (used in dashboard leaderboard)
df["lr_prob_bankrupt"]  = lr_probs
df["lr_pred_bankrupt"]  = lr_preds
df["z_pred_bankrupt"]   = (df["z_score"] < 1.81).astype(int)
df["fraud_pred_bankrupt"] = (df["fraud_risk_score"] >= 2).astype(int)

pred_cols = [
    "ticker", "name", "sector", "company_size", "bankrupt",
    "z_score", "z_zone", "fraud_risk_score", "fraud_risk_label",
    "lr_prob_bankrupt", "lr_pred_bankrupt", "z_pred_bankrupt", "fraud_pred_bankrupt",
    "current_ratio", "debt_to_equity", "return_on_assets",
    "net_margin", "interest_coverage",
]
df[pred_cols].to_csv(RESULTS_DIR / "predictions.csv", index=False)
print("  Saved → data/processed/predictions.csv")

# 3. Model comparison table (used in dashboard Page 3)
comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
print("  Saved → data/processed/model_comparison.csv")

# 4. Feature importance (used in dashboard bar chart)
lr_feature_importance.reset_index().rename(
    columns={"index": "feature", 0: "importance"}
).to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
print("  Saved → data/processed/feature_importance.csv")

print("\nPhase 3 complete. Ready for Phase 4 (Streamlit Dashboard).")
