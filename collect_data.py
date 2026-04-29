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
       data/labels/labels.csv         — just ticker + filing-risk label

These CSVs are what the model training scripts (Phase 2) will read.
"""

import argparse
import pandas as pd
from pathlib import Path
from utils.companies import COMPANIES
from utils.edgar_fetcher import fetch_financials
from utils.sp500_universe import fetch_sp500
from utils.yfinance_fetcher import fetch_market_data

# ── Output paths ──────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/processed")
LABELS_DIR    = Path("data/labels")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)


def _auto_sp500_controls(existing: list[dict], limit: int) -> list[dict]:
    """
    Add a deterministic, sector-spread set of current public controls.

    This reduces manual public-control selection bias while keeping the data
    source public/free and compatible with CI. Current S&P 500 constituents are
    not treated as perfect "never bankrupt" companies; they are controls with no
    known bankruptcy filing as of collection time.
    """
    if limit <= 0:
        return []

    existing_ciks = {str(c["cik"]).lstrip("0") for c in existing}
    existing_tickers = {c["ticker"] for c in existing}
    sp500 = fetch_sp500()
    sp500 = sp500[
        ~sp500["cik"].astype(str).str.lstrip("0").isin(existing_ciks)
        & ~sp500["ticker"].isin(existing_tickers)
    ].copy()
    sp500 = sp500.sort_values(["sector", "ticker"])

    controls = []
    # Round-robin by sector instead of taking the first N alphabetically.
    for _, sector_df in sp500.groupby("sector", sort=True):
        controls.extend(sector_df.head(max(1, limit // max(1, sp500["sector"].nunique()))).to_dict("records"))
    if len(controls) < limit:
        used = {r["ticker"] for r in controls}
        controls.extend(sp500[~sp500["ticker"].isin(used)].head(limit - len(controls)).to_dict("records"))

    return [
        {
            "ticker": row["ticker"],
            "cik": str(row["cik"]).zfill(10),
            "name": row["name"],
            "bankruptcy_filing": 0,
            "bankrupt": 0,
            "sector": row.get("sector"),
            "filing_date": None,
            "year_filed": None,
            "sample_source": "s&p500_auto_control",
        }
        for row in controls[:limit]
    ]


def build_dataset(control_limit: int = 120) -> pd.DataFrame:
    rows = []
    companies = [dict(c) for c in COMPANIES]
    companies.extend(_auto_sp500_controls(companies, control_limit))

    print(f"\nCollecting data for {len(companies)} companies...\n")

    for company in companies:
        print(f"[{company['ticker']}] {company['name']}")

        filing_label = company.get("bankruptcy_filing", company.get("bankrupt", 0))
        cutoff_date = company.get("filing_date") if filing_label else None

        # 1. Pull balance sheet / income statement numbers from EDGAR.
        # Bankruptcy-filing examples use only the final 10-K filed before the
        # filing date so post-filing and successor financials are excluded.
        edgar_data = fetch_financials(company["cik"], company["ticker"], cutoff_date=cutoff_date)

        # 2. Pull current market data from Yahoo Finance
        # Do not use current market data for historical bankruptcy-filing rows;
        # that would mix post-filing prices/market caps into pre-filing features.
        market_data = (
            {k: None for k in [
                "market_cap", "price", "beta", "pe_ratio", "week52_high",
                "week52_low", "analyst_rating", "shares_outstanding",
            ]}
            if filing_label
            else fetch_market_data(company["ticker"])
        )

        # 3. Merge everything into one flat record
        row = {
            # Identifiers
            "ticker":      company["ticker"],
            "name":        company["name"],
            "sector":      company["sector"],
            "year_filed":  company["year_filed"],
            "filing_date": company.get("filing_date"),
            "sample_source": company.get("sample_source", "manual"),

            # Label (target variable for our model)
            "bankruptcy_filing": filing_label,
            "bankrupt":    filing_label,

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
        if assets is None or pd.isna(assets):
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--control-limit",
        type=int,
        default=120,
        help="Number of deterministic S&P 500 controls to add to manual controls.",
    )
    args = parser.parse_args()

    # Build the full dataset
    df = build_dataset(control_limit=args.control_limit)

    # Add size buckets
    df = add_company_size(df)

    # Save master financials CSV
    out_path = PROCESSED_DIR / "financials.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows → {out_path}")

    # Save a lightweight labels-only CSV (used during model training)
    labels_df = df[[
        "ticker", "name", "sector", "bankruptcy_filing", "bankrupt",
        "filing_date", "source_filing_end", "source_filing_filed",
        "company_size", "sample_source",
    ]]
    labels_path = LABELS_DIR / "labels.csv"
    labels_df.to_csv(labels_path, index=False)
    print(f"Saved labels     → {labels_path}")

    # Quick summary
    print("\n── Dataset Summary ──────────────────────────────────")
    print(f"  Total companies : {len(df)}")
    print(f"  Filing examples: {df['bankruptcy_filing'].sum()}")
    print(f"  Controls       : {(df['bankruptcy_filing'] == 0).sum()}")
    print(f"  Sectors         : {df['sector'].nunique()}")
    print("\nSample rows:")
    print(df[["ticker", "name", "bankruptcy_filing", "total_assets", "current_ratio"]].to_string())
    print("\nPhase 1 complete. Ready for Phase 2 (Feature Engineering).")
