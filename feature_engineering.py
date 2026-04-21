"""
Phase 2 — Feature Engineering

Run this script after collect_data.py:
  python feature_engineering.py

What it does:
  1. Loads the raw financials CSV from Phase 1
  2. Fixes known data quality issues (duplicate CIKs, extreme outliers)
  3. Drops companies with too many missing values to be usable
  4. Computes Altman Z-Score (bankruptcy benchmark) for each company
  5. Computes a Fraud Risk Score (Beneish-inspired heuristic)
  6. Imputes remaining missing values with the column median
     (so the model can train on every row)
  7. Saves the final feature matrix to data/processed/features.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from utils.altman_zscore import add_zscores
from utils.fraud_score import add_fraud_scores

INPUT_PATH  = Path("data/processed/financials.csv")
OUTPUT_PATH = Path("data/processed/features.csv")


# ── Step 1: Load ──────────────────────────────────────────────────────────────
print("Loading raw financials...")
df = pd.read_csv(INPUT_PATH)
print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")


# ── Step 2: Fix known data quality issues ─────────────────────────────────────
print("\nFixing data quality issues...")

# Circuit City (CCTYQ) was accidentally given J&J's CIK (0000200406).
# Circuit City was a mid-size electronics retailer — total assets ~$3B.
# J&J is a $200B healthcare giant. We can detect the collision by size.
# Fix: remove Circuit City since its financial data is actually J&J's.
before = len(df)
df = df[~((df["ticker"] == "CCTYQ") & (df["total_assets"] > 100e9))]
print(f"  Removed {before - len(df)} row(s) with CIK collision (CCTYQ/JNJ)")

# Lehman Brothers shows total_assets of $8.3M (clearly wrong — it was $691B).
# This is because EDGAR only has their post-bankruptcy wind-down entity.
# Fix: drop Lehman (can't train on bad data — better to have no row than a wrong one).
before = len(df)
df = df[~((df["ticker"] == "LEHMQ") & (df["total_assets"] < 1e9))]
print(f"  Removed {before - len(df)} row(s) with implausible Lehman data")

# PCRXQ was incorrectly mapped to a small company (likely wrong CIK).
# Assets of $83M are implausible for a Nortel-scale entity — drop it.
before = len(df)
df = df[~(df["ticker"] == "PCRXQ")]
print(f"  Removed {before - len(df)} row(s) with bad PCRXQ CIK mapping")


# ── Step 3: Drop rows with no usable financial data ───────────────────────────
print("\nDropping companies with no EDGAR financial data...")

# These companies pre-date mandatory XBRL filing (2009), so EDGAR has no data
REQUIRED_COLS = ["total_assets", "revenue", "net_income"]
before = len(df)
df = df.dropna(subset=REQUIRED_COLS, how="all")
print(f"  Dropped {before - len(df)} rows — kept {len(df)} companies")


# ── Step 4: Compute Altman Z-Score ────────────────────────────────────────────
print("\nComputing Altman Z-Scores...")
df = add_zscores(df)
z_computed = df["z_score"].notna().sum()
print(f"  Z-Score computed for {z_computed}/{len(df)} companies")
print(f"  Zone distribution:\n{df['z_zone'].value_counts().to_string()}")


# ── Step 5: Compute Fraud Risk Score ─────────────────────────────────────────
print("\nComputing Fraud Risk Scores...")
df = add_fraud_scores(df)
print(f"  Risk distribution:\n{df['fraud_risk_label'].value_counts().to_string()}")


# ── Step 6: Impute missing values ─────────────────────────────────────────────
print("\nImputing missing values with column medians...")

# These are the features our ML model will actually train on
MODEL_FEATURES = [
    "current_ratio",
    "debt_to_equity",
    "return_on_assets",
    "net_margin",
    "interest_coverage",
    "z_x1", "z_x2", "z_x3", "z_x4", "z_x5",
]

for col in MODEL_FEATURES:
    n_missing = df[col].isna().sum()
    if n_missing > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"  {col}: filled {n_missing} missing → median {median_val:.4f}")


# ── Step 7: Clip extreme outliers ─────────────────────────────────────────────
# Cap at ±10 standard deviations to prevent one outlier from dominating the model
print("\nClipping extreme outliers (±10 SD)...")
for col in MODEL_FEATURES:
    mean, sd = df[col].mean(), df[col].std()
    if sd > 0:
        df[col] = df[col].clip(mean - 10*sd, mean + 10*sd)


# ── Step 8: Final summary and save ────────────────────────────────────────────
print("\n── Final Feature Matrix ──────────────────────────────────────")
print(f"  Companies : {len(df)}")
print(f"  Bankrupt  : {df['bankrupt'].sum()}")
print(f"  Healthy   : {(df['bankrupt']==0).sum()}")
print(f"  Features  : {len(MODEL_FEATURES)} ML features + Z-score + fraud score")
print()

# Display a clean summary table
summary = df[[
    "ticker", "name", "bankrupt", "z_score", "z_zone",
    "fraud_risk_score", "fraud_risk_label", "current_ratio", "debt_to_equity"
]].sort_values("z_score")

pd.set_option("display.max_rows", 50)
pd.set_option("display.width", 120)
print(summary.to_string(index=False))

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved feature matrix → {OUTPUT_PATH}")
print("Phase 2 complete. Ready for Phase 3 (Model Training).")
