"""
Phase 3 - Model Training & Evaluation

Run this script after feature_engineering.py:
  python train_models.py

The supervised models are evaluated with sklearn Pipelines so imputation,
scaling, and model fitting happen inside each cross-validation fold. This avoids
preprocessing leakage from held-out companies into the training folds.
"""

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = Path("data/processed/features.csv")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("data/processed")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_FEATURES = [
    "current_ratio",
    "debt_to_equity",
    "return_on_assets",
    "net_margin",
    "interest_coverage",
    "z_x1", "z_x2", "z_x3", "z_x4", "z_x5",
]


def _metric_row(name: str, y_true, preds, scores, sample_n: int) -> dict:
    auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else np.nan
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, preds),
        "ROC-AUC": auc,
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "Sample N": sample_n,
    }


def _print_metrics(row: dict, cm: np.ndarray) -> None:
    print(f"  Accuracy : {row['Accuracy']:.1%}")
    print(f"  ROC-AUC  : {row['ROC-AUC']:.3f}")
    print(f"  Precision: {row['Precision']:.1%}")
    print(f"  Recall   : {row['Recall']:.1%}")
    print("\n  Confusion Matrix:")
    print("              Predicted Control  Predicted Filing")
    print(f"  True Control        {cm[0,0]:>3}               {cm[0,1]:>3}")
    print(f"  True Filing         {cm[1,0]:>3}               {cm[1,1]:>3}")


print("Loading feature matrix...")
df = pd.read_csv(FEATURES_PATH)
label_col = "bankruptcy_filing" if "bankruptcy_filing" in df.columns else "bankrupt"
df = df.dropna(subset=[label_col]).copy()
df[label_col] = df[label_col].astype(int)
if label_col != "bankruptcy_filing":
    df["bankruptcy_filing"] = df[label_col]
    label_col = "bankruptcy_filing"
print(
    f"  {len(df)} companies | {df[label_col].sum()} filing positives | "
    f"{(df[label_col] == 0).sum()} controls\n"
)

X = df[MODEL_FEATURES]
y = df[label_col].values
min_class = int(pd.Series(y).value_counts().min())
if min_class < 2:
    raise ValueError("Need at least two samples in each class for stratified CV.")
n_splits = min(5, min_class)
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ]),
}

comparison_rows = []
prediction_cols = {}
fitted_models = {}

for model_name, pipeline in models.items():
    print("=" * 60)
    print(model_name.upper())
    print("=" * 60)
    probs = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    preds = (probs >= 0.5).astype(int)
    row = _metric_row(model_name, y, preds, probs, len(y))
    cm = confusion_matrix(y, preds, labels=[0, 1])
    _print_metrics(row, cm)
    comparison_rows.append(row)

    prefix = "lr" if model_name == "Logistic Regression" else "rf"
    prediction_cols[f"{prefix}_prob_filing"] = probs
    prediction_cols[f"{prefix}_pred_filing"] = preds
    # Backward-compatible names used by the current dashboard.
    if prefix == "lr":
        prediction_cols["lr_prob_bankrupt"] = probs
        prediction_cols["lr_pred_bankrupt"] = preds

    pipeline.fit(X, y)
    fitted_models[model_name] = pipeline
    print()


print("=" * 60)
print("RULE-BASED BENCHMARKS")
print("=" * 60)

z_mask = df["z_score"].notna()
z_sub = df[z_mask]
if len(z_sub) > 0:
    y_z = z_sub[label_col].astype(int).values
    z_preds = (z_sub["z_score"] < 1.81).astype(int).values
    z_scores = -z_sub["z_score"].values
    z_row = _metric_row("Altman Z-Score", y_z, z_preds, z_scores, len(y_z))
    z_cm = confusion_matrix(y_z, z_preds, labels=[0, 1])
    print("\nAltman Z-Score")
    _print_metrics(z_row, z_cm)
    comparison_rows.append(z_row)

f_preds = (df["fraud_risk_score"] >= 2).astype(int).values
f_scores = df["fraud_risk_score"].values
f_row = _metric_row("Fraud Risk Score", y, f_preds, f_scores, len(y))
f_cm = confusion_matrix(y, f_preds, labels=[0, 1])
print("\nFraud Risk Score")
_print_metrics(f_row, f_cm)
comparison_rows.append(f_row)

comparison = pd.DataFrame(comparison_rows)
comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
print("\nSaved -> data/processed/model_comparison.csv")

training_metadata = {
    "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="minutes"),
    "scoring_model": "Logistic Regression",
    "training_sample_size": int(len(df)),
    "filing_positive_count": int(df[label_col].sum()),
    "control_count": int((df[label_col] == 0).sum()),
    "cv_splits": int(n_splits),
}
(RESULTS_DIR / "training_metadata.json").write_text(
    json.dumps(training_metadata, indent=2) + "\n"
)
print("Saved -> data/processed/training_metadata.json")

print("\nSaving model artifacts...")
with open(MODELS_DIR / "logistic_model.pkl", "wb") as f:
    pickle.dump({
        "model": fitted_models["Logistic Regression"],
        "models": fitted_models,
        "features": MODEL_FEATURES,
        "label": "bankruptcy_filing",
        "cv_splits": n_splits,
        "metadata": training_metadata,
    }, f)
print("  Saved -> models/logistic_model.pkl")

for col, values in prediction_cols.items():
    df[col] = values
df["rf_prob_bankrupt"] = df["rf_prob_filing"]
df["rf_pred_bankrupt"] = df["rf_pred_filing"]
df["z_pred_filing"] = (df["z_score"] < 1.81).astype(int)
df["fraud_pred_filing"] = (df["fraud_risk_score"] >= 2).astype(int)
df["z_pred_bankrupt"] = df["z_pred_filing"]
df["fraud_pred_bankrupt"] = df["fraud_pred_filing"]

pred_cols = [
    "ticker", "name", "sector", "company_size", label_col, "bankrupt",
    "filing_date", "source_filing_end", "source_filing_filed",
    "z_score", "z_zone", "fraud_risk_score", "fraud_risk_label",
    "lr_prob_filing", "lr_pred_filing", "rf_prob_filing", "rf_pred_filing",
    "lr_prob_bankrupt", "lr_pred_bankrupt", "rf_prob_bankrupt", "rf_pred_bankrupt",
    "z_pred_filing", "fraud_pred_filing", "z_pred_bankrupt", "fraud_pred_bankrupt",
    "current_ratio", "debt_to_equity", "return_on_assets",
    "net_margin", "interest_coverage",
]
available_pred_cols = [c for c in pred_cols if c in df.columns]
df[available_pred_cols].to_csv(RESULTS_DIR / "predictions.csv", index=False)
print("  Saved -> data/processed/predictions.csv")

lr_model = fitted_models["Logistic Regression"].named_steps["model"]
lr_feature_importance = pd.Series(
    np.abs(lr_model.coef_[0]),
    index=MODEL_FEATURES,
).sort_values(ascending=False)
lr_feature_importance.reset_index().rename(
    columns={"index": "feature", 0: "importance"}
).to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
print("  Saved -> data/processed/feature_importance.csv")

print("\nPhase 3 complete. Ready for S&P 500 refresh.")
