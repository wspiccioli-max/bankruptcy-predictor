"""
S&P 500 Universe Fetcher

Pulls the current list of S&P 500 constituents with their SEC EDGAR CIK numbers.

Source: Wikipedia maintains a live table of S&P 500 companies at
  https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
This table includes Symbol, Security name, GICS Sector, and the all-important CIK.

We use pandas.read_html() which parses HTML tables directly into DataFrames.
If Wikipedia is unreachable (e.g. inside a CI runner with no internet),
we fall back to a cached local copy if one exists.
"""

from io import StringIO
from pathlib import Path
import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# Wikipedia blocks default urllib User-Agent; send a browser-like UA instead.
WIKI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (BA870-AC820 research; contact: research@university.edu)",
}
CACHE_PATH = Path(__file__).parent.parent / "data/raw/sp500_list.csv"


def fetch_sp500() -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      ticker, name, cik, sector, industry, date_added
    """
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Fetch with a real User-Agent (Wikipedia 403s the default urllib one)
        resp = requests.get(WIKI_URL, headers=WIKI_HEADERS, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text), header=0)
        sp500 = tables[0].copy()

        # Normalise column names (Wikipedia may change these over time)
        col_map = {
            "Symbol":            "ticker",
            "Security":          "name",
            "CIK":               "cik",
            "GICS Sector":       "sector",
            "GICS Sub-Industry": "industry",
            "Date added":        "date_added",
            "Date first added":  "date_added",
        }
        sp500 = sp500.rename(columns={k: v for k, v in col_map.items() if k in sp500.columns})

        # Keep only the columns we actually use
        keep = [c for c in ["ticker", "name", "cik", "sector", "industry", "date_added"]
                if c in sp500.columns]
        sp500 = sp500[keep]

        # CIK comes in as an int without leading zeros — pad to 10 digits
        sp500["cik"] = sp500["cik"].astype(str).str.zfill(10)

        # Normalise ticker symbols (some have "." like BRK.B which EDGAR/yfinance need as "BRK-B")
        sp500["ticker"] = sp500["ticker"].str.replace(".", "-", regex=False)

        # Cache a copy so future offline runs still work
        sp500.to_csv(CACHE_PATH, index=False)
        print(f"  Fetched {len(sp500)} S&P 500 companies from Wikipedia")
        return sp500

    except Exception as e:
        print(f"  [warn] Wikipedia fetch failed ({e}) — falling back to cache")
        if CACHE_PATH.exists():
            return pd.read_csv(CACHE_PATH, dtype={"cik": str})
        raise RuntimeError(
            "Could not fetch S&P 500 list and no cached copy exists. "
            "Run this script once while online to create the cache."
        )


if __name__ == "__main__":
    # Quick diagnostic: `python -m utils.sp500_universe`
    df = fetch_sp500()
    print(df.head(10).to_string(index=False))
    print(f"\nTotal: {len(df)} companies | Sectors: {df['sector'].nunique()}")
