"""
Phase 1 — Data Collection Orchestrator

Run this script once to build the master dataset:
  python collect_data.py

What it does (step by step):
  1. Loops through our company list (utils/companies.py)
  2. For each company, fetches financial data from SEC EDGAR
  3. Then fetches market data (price, market cap) from yfinance
  4. Merges everything into one row per company
  5. Saves two CSV files:
       data/processed/financials.csv  — all raw numbers + ratios
       data/labels/labels.csv         — just ticker + bankrupt label

These CSVs are what the model training scripts (Phase 2) will read.
"""

import pandas as pd
from pathlib import Path
from utils.companies import COMPANIES
from utils.edgar_fetcher import fetch_financials
from utils.yfinance_fetcher import fetch_market_data

# ── Output paths ──────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
LABELS_DIR    = Path("data/labels")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)


def build_dataset() -> pd.DataFrame:
    rows = []

    print(f"\nCollecting data for {len(COMPANIES)} companies...\n")

    for company in COMPANIES:
        print(f"[{company['ticker']}] {company['name']}")

        # 1. Pull balance sheet / income statement numbers from EDGAR
        edgar_data = fetch_financials(company["cik"], company["ticker"])

        # 2. Pull current market data from Yahoo Finance
        market_data = fetch_market_data(company["ticker"])

        # 3. Merge everything into one flat record
        row = {
            # Identifiers
            "ticker":      company["ticker"],
            "name":        company["name"],
            "sector":      company["sector"],
            "year_filed":  company["year_filed"],

            # Label (target variable for our model)
            "bankrupt":    company["bankrupt"],

            # EDGAR financial data (raw numbers, in USD)
            **edgar_data,

            # yfinance market data
            **market_data,
        }
        rows.append(row)
        print()

    return pd.DataFrame(rows)


def add_company_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket companies into Small / Mid / Large by total assets.
    This is used for the dashboard's size filter.
    """
    def size_bucket(assets):
        if assets is None:
            return "Unknown"
        if assets < 1e9:           # < $1 billion
            return "Small"
        elif assets < 10e9:        # $1B – $10B
            return "Mid"
        else:                      # > $10B
            return "Large"

    df["company_size"] = df["total_assets"].apply(size_bucket)
    return df


if __name__ == "__main__":
    # Build the full dataset
    df = build_dataset()

    # Add size buckets
    df = add_company_size(df)

    # Save master financials CSV
    out_path = PROCESSED_DIR / "financials.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows → {out_path}")

    # Save a lightweight labels-only CSV (used during model training)
    labels_df = df[["ticker", "name", "sector", "bankrupt", "company_size"]]
    labels_path = LABELS_DIR / "labels.csv"
    labels_df.to_csv(labels_path, index=False)
    print(f"Saved labels     → {labels_path}")

    # Quick summary
    print("\n── Dataset Summary ──────────────────────────────────")
    print(f"  Total companies : {len(df)}")
    print(f"  Bankrupt        : {df['bankrupt'].sum()}")
    print(f"  Healthy         : {(df['bankrupt'] == 0).sum()}")
    print(f"  Sectors         : {df['sector'].nunique()}")
    print("\nSample rows:")
    print(df[["ticker", "name", "bankrupt", "total_assets", "current_ratio"]].to_string())
    print("\nPhase 1 complete. Ready for Phase 2 (Feature Engineering).")
