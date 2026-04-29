"""
Fetches financial statement data from SEC EDGAR's free JSON API.

How it works:
  The SEC publishes every company's XBRL financial data at:
    https://data.sec.gov/api/xbrl/companyfacts/{CIK}.json

  That JSON contains every line-item a company ever reported in a 10-K:
  total assets, liabilities, revenue, etc. — going back to the early 2000s.

  We extract the most recent annual (10-K) value for each metric we need,
  then compute financial ratios from those raw numbers.

Rate limit: SEC allows 10 requests/second. We add a small sleep between calls.
"""

import time
import requests
import pandas as pd
from typing import Optional

# Tell the SEC who we are (required by their API terms of service)
HEADERS = {
    "User-Agent": "BA870-AC820 Research Project research@university.edu",
    "Accept-Encoding": "gzip, deflate",
}

BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts"

# Map our friendly names → XBRL concept names used in EDGAR filings
XBRL_CONCEPTS = {
    "total_assets":          "Assets",
    "total_liabilities":     "Liabilities",
    "current_assets":        "AssetsCurrent",
    "current_liabilities":   "LiabilitiesCurrent",
    "retained_earnings":     "RetainedEarningsAccumulatedDeficit",
    "revenue":               "Revenues",
    "net_income":            "NetIncomeLoss",
    "ebit":                  "OperatingIncomeLoss",   # closest proxy for EBIT
    "interest_expense":      "InterestExpense",
    "stockholders_equity":   "StockholdersEquity",
    "long_term_debt":        "LongTermDebt",
}


def _get_company_facts(cik: str) -> Optional[dict]:
    """Download the full XBRL facts blob for one company."""
    # CIK must be zero-padded to 10 digits
    cik_padded = cik.lstrip("0").zfill(10)
    url = f"{BASE_URL}/CIK{cik_padded}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"    [EDGAR] Failed for CIK {cik}: {e}")
        return None


def _latest_annual_entry(
    facts: dict,
    concept: str,
    cutoff_date: Optional[str] = None,
    anchor_end: Optional[str] = None,
) -> Optional[dict]:
    """
    Pull the latest annual filing entry for a concept.

    EDGAR organises data as:
      facts → us-gaap → {concept} → units → USD → [ list of filings ]

    Each filing entry looks like:
      { "form": "10-K", "end": "2023-12-31", "val": 123456789, ... }

    For bankruptcy-filing labels, cutoff_date prevents post-filing or successor
    financials from entering the training set. When anchor_end is supplied, use
    values from the same fiscal period or earlier.
    """
    try:
        entries = (
            facts["facts"]["us-gaap"][concept]["units"]["USD"]
        )
    except KeyError:
        return None

    # Keep only annual 10-K filings (not 10-Q quarterly reports).
    annual = [e for e in entries if e.get("form") in ("10-K", "10-K/A")]
    if cutoff_date:
        annual = [e for e in annual if e.get("filed", e.get("end", "")) < cutoff_date]
    if anchor_end:
        annual = [e for e in annual if e.get("end", "") <= anchor_end]
    if not annual:
        return None

    # Sort by fiscal period and filing date so amendments after the same period
    # do not accidentally push us to a later fiscal period.
    annual.sort(key=lambda e: (e.get("end", ""), e.get("filed", "")), reverse=True)
    return annual[0]


def _extract_annual_value(
    facts: dict,
    concept: str,
    cutoff_date: Optional[str] = None,
    anchor_end: Optional[str] = None,
) -> Optional[float]:
    entry = _latest_annual_entry(facts, concept, cutoff_date, anchor_end)
    return None if entry is None else entry.get("val")


def _extract_time_series(facts: dict, concept: str) -> pd.Series:
    """
    Return a time series of annual values for a concept (used for Z-score charts).
    Returns a pandas Series indexed by fiscal year-end date.
    """
    try:
        entries = facts["facts"]["us-gaap"][concept]["units"]["USD"]
    except KeyError:
        return pd.Series(dtype=float)

    annual = [e for e in entries if e.get("form") in ("10-K", "10-K/A")]
    if not annual:
        return pd.Series(dtype=float)

    series = (
        pd.DataFrame(annual)[["end", "val"]]
        .drop_duplicates("end")
        .set_index("end")["val"]
        .sort_index()
    )
    return series


def fetch_financials(
    cik: str,
    ticker: str,
    cutoff_date: Optional[str] = None,
) -> dict:
    """
    Main entry point: fetch all key financial metrics for one company.

    Returns a flat dict of raw numbers + computed ratios.
    Returns None values for any metrics EDGAR doesn't have.
    """
    cutoff_msg = f", cutoff {cutoff_date}" if cutoff_date else ""
    print(f"  Fetching EDGAR data for {ticker} (CIK {cik}{cutoff_msg})...")
    time.sleep(0.12)  # stay under SEC's 10 req/s rate limit

    facts = _get_company_facts(cik)
    if facts is None:
        return {k: None for k in XBRL_CONCEPTS}

    anchor = _latest_annual_entry(facts, "Assets", cutoff_date=cutoff_date)
    anchor_end = anchor.get("end") if anchor else None

    row = {}
    for friendly_name, concept in XBRL_CONCEPTS.items():
        row[friendly_name] = _extract_annual_value(
            facts,
            concept,
            cutoff_date=cutoff_date,
            anchor_end=anchor_end,
        )

    row["source_filing_end"] = anchor_end
    row["source_filing_filed"] = anchor.get("filed") if anchor else None
    row["source_filing_form"] = anchor.get("form") if anchor else None

    # ── Derived ratios ────────────────────────────────────────────────────
    ta  = row["total_assets"]
    tl  = row["total_liabilities"]
    ca  = row["current_assets"]
    cl  = row["current_liabilities"]
    re  = row["retained_earnings"]
    ebit = row["ebit"]
    rev = row["revenue"]
    ni  = row["net_income"]
    ie  = row["interest_expense"]
    eq  = row["stockholders_equity"]
    ltd = row["long_term_debt"]

    def safe_div(num, den):
        """Return num/den, or None if either value is missing or denominator is 0."""
        if num is None or den is None or den == 0:
            return None
        return num / den

    # Leverage: how much of the company is funded by debt
    row["debt_to_equity"] = safe_div(tl, eq)

    # Liquidity: can the company pay short-term bills?
    row["current_ratio"] = safe_div(ca, cl)

    # Profitability: how much profit per dollar of assets
    row["return_on_assets"] = safe_div(ni, ta)

    # Interest coverage: can the company afford its interest payments?
    row["interest_coverage"] = safe_div(ebit, ie)

    # Profit margin: profit as % of revenue
    row["net_margin"] = safe_div(ni, rev)

    return row


def fetch_zscore_series(cik: str, ticker: str) -> pd.DataFrame:
    """
    Fetch multi-year time series of Altman Z-score inputs for one company.
    Used to draw the Z-score trend chart on the dashboard.
    """
    time.sleep(0.12)
    facts = _get_company_facts(cik)
    if facts is None:
        return pd.DataFrame()

    series = {}
    for friendly_name, concept in XBRL_CONCEPTS.items():
        series[friendly_name] = _extract_time_series(facts, concept)

    df = pd.DataFrame(series)
    df.index.name = "date"
    df["ticker"] = ticker
    return df
