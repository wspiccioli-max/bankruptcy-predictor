"""
Phase 5 — S&P 500 Live Refresh

This is the single script you run (or that GitHub Actions runs weekly) to:
  1. Fetch the current S&P 500 constituent list from Wikipedia
  2. Pull their two most-recent SEC filings (10-K or 10-Q) from EDGAR
  3. Compute financial ratios for both periods
  4. Score both periods with the already-trained logistic regression model
  5. Compute period-over-period change in bankruptcy filing risk probability
  6. Save results to data/processed/sp500_predictions.csv

Usage:
  python refresh_sp500.py              # full refresh (~5–8 minutes)
  python refresh_sp500.py --limit 50   # quick test on 50 companies
"""

import argparse
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.sp500_universe  import fetch_sp500
from utils.edgar_quarterly import fetch_current_and_prior, compute_ratios

MODEL_PATH  = Path("models/logistic_model.pkl")
OUTPUT_PATH = Path("data/processed/sp500_predictions.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL_FEATURES = [
    "current_ratio", "debt_to_equity", "return_on_assets",
    "net_margin", "interest_coverage",
    "z_x1", "z_x2", "z_x3", "z_x4", "z_x5",
]


def score_snapshot(ratios: dict, model, scaler=None, medians: pd.Series | None = None) -> float:
    """Run one snapshot of ratios through the trained model to get p(filing)."""
    # Build feature vector in the order the model expects
    row = []
    for f in MODEL_FEATURES:
        v = ratios.get(f)
        if (v is None or pd.isna(v)) and medians is not None:
            v = medians[f]
        row.append(v)

    X = pd.DataFrame([row], columns=MODEL_FEATURES)
    if scaler is not None:
        X_scaled = scaler.transform(np.array(row, dtype=float).reshape(1, -1))
        return float(model.predict_proba(X_scaled)[0, 1])
    return float(model.predict_proba(X)[0, 1])


def risk_bucket(prob: float) -> str:
    if prob >= 0.65: return "🔴 High"
    if prob >= 0.40: return "🟡 Medium"
    return "🟢 Low"


def delta_arrow(delta: float) -> str:
    if pd.isna(delta): return "—"
    if delta >  0.02:  return "🔺"
    if delta < -0.02:  return "🔻"
    return "➡️"


def main(limit: int = None):
    print("=" * 65)
    print(f"S&P 500 Refresh — {datetime.utcnow().isoformat(timespec='minutes')}Z")
    print("=" * 65)

    # ── 1. Load trained model ────────────────────────────────────────────────
    if not MODEL_PATH.exists():
        print(f"ERROR: {MODEL_PATH} not found. Run train_models.py first.")
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    rf_model = bundle.get("models", {}).get("Random Forest")
    scaler = bundle.get("scaler")
    print(f"  Loaded model ({len(bundle['features'])} features)\n")

    # Medians to impute missing values (loaded from training features.csv)
    medians = None
    if scaler is not None:
        train_features = pd.read_csv("data/processed/features.csv")
        medians = train_features[MODEL_FEATURES].median()

    # ── 2. Fetch S&P 500 universe ────────────────────────────────────────────
    print("Fetching S&P 500 constituent list...")
    universe = fetch_sp500()
    if limit:
        universe = universe.head(limit)
        print(f"  [--limit] Restricted to first {limit} companies\n")
    else:
        print()

    # ── 3. For each company, fetch 2 snapshots and score them ────────────────
    rows = []
    print(f"Fetching EDGAR data for {len(universe)} companies...")
    for _, company in tqdm(universe.iterrows(), total=len(universe), unit="co"):
        cur, prior, meta = fetch_current_and_prior(company["cik"], company["ticker"])

        if not meta["data_available"]:
            # No XBRL data — skip but keep the company in the table
            rows.append({
                "ticker": company["ticker"],
                "name":   company["name"],
                "sector": company.get("sector"),
                "industry": company.get("industry"),
                "cik":    company["cik"],
                "data_available": False,
            })
            continue

        cur_ratios   = compute_ratios(cur)
        prior_ratios = compute_ratios(prior)

        cur_prob   = score_snapshot(cur_ratios,   model, scaler, medians)
        prior_prob = score_snapshot(prior_ratios, model, scaler, medians) \
                     if meta["prior_period_end"] else np.nan
        rf_cur_prob = score_snapshot(cur_ratios, rf_model) if rf_model is not None else np.nan
        rf_prior_prob = (
            score_snapshot(prior_ratios, rf_model)
            if rf_model is not None and meta["prior_period_end"] else np.nan
        )

        rows.append({
            "ticker": company["ticker"],
            "name":   company["name"],
            "sector": company.get("sector"),
            "industry": company.get("industry"),
            "cik":    company["cik"],
            "data_available":     True,
            "current_period_end": meta["current_period_end"],
            "current_form":       meta["current_form"],
            "prior_period_end":   meta["prior_period_end"],
            "prior_form":         meta["prior_form"],
            "current_prob":       cur_prob,
            "prior_prob":         prior_prob,
            "delta_prob":         (cur_prob - prior_prob) if pd.notna(prior_prob) else np.nan,
            "lr_current_prob":    cur_prob,
            "lr_prior_prob":      prior_prob,
            "rf_current_prob":    rf_cur_prob,
            "rf_prior_prob":      rf_prior_prob,
            # Keep the current ratios for the dashboard's company lookup page
            **{f"current_{k}": v for k, v in cur_ratios.items()},
        })

    df = pd.DataFrame(rows)

    # ── 4. Annotate risk buckets and delta arrows ────────────────────────────
    df["risk_bucket"] = df["current_prob"].apply(
        lambda p: risk_bucket(p) if pd.notna(p) else "—"
    )
    df["delta_arrow"] = df["delta_prob"].apply(delta_arrow)
    df["refresh_utc"] = datetime.utcnow().isoformat(timespec="minutes") + "Z"

    # ── 5. Save ──────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows → {OUTPUT_PATH}")

    # Summary
    with_data = df[df["data_available"] == True]
    print("\n── Summary ───────────────────────────────────────────────────")
    print(f"  Companies with data : {len(with_data)} / {len(df)}")
    if len(with_data) > 0:
        print(f"  Mean filing risk prob: {with_data['current_prob'].mean():.1%}")
        print(f"  High-risk (≥65%)    : {(with_data['current_prob'] >= 0.65).sum()}")
        print(f"  Medium (40–65%)     : {((with_data['current_prob'] >= 0.40) & (with_data['current_prob'] < 0.65)).sum()}")
        print(f"  Low (<40%)          : {(with_data['current_prob'] < 0.40).sum()}")
        rising = with_data.dropna(subset=["delta_prob"]).nlargest(5, "delta_prob")
        print("\n  Top 5 biggest risk INCREASES this quarter:")
        for _, r in rising.iterrows():
            print(f"    {r['ticker']:<6} {r['name'][:40]:<40}  "
                  f"Δ {r['delta_prob']:+.1%}  ({r['prior_prob']:.1%} → {r['current_prob']:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N companies (for quick testing)")
    args = parser.parse_args()
    main(limit=args.limit)
