"""
Quarterly EDGAR Fetcher — pulls the two most recent filings per company.

Difference from edgar_fetcher.py:
  - That module only pulls the most recent 10-K (annual).
  - This module pulls the 2 most recent filings of ANY type (10-K OR 10-Q)
    so we can compute period-over-period changes in filing risk.

Design decisions for quarterly data:
  - Balance sheet items (Assets, Liabilities, Equity) are point-in-time
    values so they're directly comparable between 10-K and 10-Q filings.
  - Income statement items (Revenue, Net Income, EBIT) are flow measures
    that differ by reporting period. To keep ratios comparable across
    filings we always pull the most recent TRAILING-12-MONTH income
    statement values (from the latest annual 10-K).

Result: each company gets a (current, prior) pair of snapshots where:
  - Balance sheet numbers reflect the most recent filing date
  - Income numbers are annual (most recent 10-K)
  - We record the filing type and fiscal-period-end for display
"""

import time
import requests
import pandas as pd
from typing import Optional, Tuple

HEADERS = {
    "User-Agent": "BA870-AC820 Research Project research@university.edu",
    "Accept-Encoding": "gzip, deflate",
}
BASE_URL = "https://data.sec.gov/api/xbrl/companyfacts"

# Concepts to pull. Tag each as 'bs' (balance sheet, point-in-time)
# or 'is' (income statement, period-based).
CONCEPTS = {
    "total_assets":        ("Assets",                             "bs"),
    "total_liabilities":   ("Liabilities",                        "bs"),
    "current_assets":      ("AssetsCurrent",                      "bs"),
    "current_liabilities": ("LiabilitiesCurrent",                 "bs"),
    "retained_earnings":   ("RetainedEarningsAccumulatedDeficit", "bs"),
    "stockholders_equity": ("StockholdersEquity",                 "bs"),
    "long_term_debt":      ("LongTermDebt",                       "bs"),
    "revenue":             ("Revenues",                           "is"),
    "net_income":          ("NetIncomeLoss",                      "is"),
    "ebit":                ("OperatingIncomeLoss",                "is"),
    "interest_expense":    ("InterestExpense",                    "is"),
}


def _get_facts(cik: str) -> Optional[dict]:
    """Fetch the full XBRL facts blob for one company."""
    url = f"{BASE_URL}/CIK{cik.zfill(10)}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _latest_bs_entries(facts: dict, concept: str, top_n: int = 2) -> list:
    """
    For a balance-sheet concept, return the top_n most recent entries
    (from any 10-K or 10-Q filing), sorted newest first.
    """
    try:
        entries = facts["facts"]["us-gaap"][concept]["units"]["USD"]
    except KeyError:
        return []

    # Keep periodic filings only
    keep = [e for e in entries if e.get("form") in ("10-K", "10-Q", "10-K/A", "10-Q/A")]
    # Deduplicate by fiscal period end (sometimes the same date appears twice)
    seen = {}
    for e in keep:
        key = e["end"]
        # Prefer amended filings when both exist for the same fiscal period.
        if key not in seen or (e["form"].endswith("/A") and not seen[key]["form"].endswith("/A")):
            seen[key] = e
    sorted_entries = sorted(seen.values(), key=lambda e: e["end"], reverse=True)
    return sorted_entries[:top_n]


def _latest_annual_value(facts: dict, concept: str) -> Optional[float]:
    """For an income-statement concept, return the most recent ANNUAL value."""
    try:
        entries = facts["facts"]["us-gaap"][concept]["units"]["USD"]
    except KeyError:
        return None
    # 10-K filings report the full year. Use that to avoid quarterly/annual mismatch.
    annual = [e for e in entries if e.get("form") in ("10-K", "10-K/A")]
    if not annual:
        return None
    annual.sort(key=lambda e: e["end"], reverse=True)
    return annual[0]["val"]


def fetch_current_and_prior(cik: str, ticker: str) -> Tuple[dict, dict, dict]:
    """
    Pull two snapshots of one company: the most recent filing and the prior one.

    Returns (current_snapshot, prior_snapshot, metadata) where:
      - current_snapshot and prior_snapshot are dicts of raw financial values
      - metadata has filing type (10-K/10-Q), fiscal period end dates, etc.
    """
    time.sleep(0.11)  # stay under SEC's 10 req/s limit
    facts = _get_facts(cik)

    meta = {
        "current_period_end": None,
        "prior_period_end":   None,
        "current_form":       None,
        "prior_form":         None,
        "data_available":     False,
    }
    current = {name: None for name in CONCEPTS}
    prior   = {name: None for name in CONCEPTS}

    if facts is None:
        return current, prior, meta

    # Use 'Assets' to anchor the two periods — every filer reports this.
    anchor = _latest_bs_entries(facts, "Assets", top_n=2)
    if len(anchor) < 1:
        return current, prior, meta

    meta["data_available"]     = True
    meta["current_period_end"] = anchor[0]["end"]
    meta["current_form"]       = anchor[0]["form"]
    if len(anchor) > 1:
        meta["prior_period_end"] = anchor[1]["end"]
        meta["prior_form"]       = anchor[1]["form"]

    # For every balance-sheet concept, pull values at each anchor date.
    for name, (concept, kind) in CONCEPTS.items():
        if kind == "bs":
            entries = _latest_bs_entries(facts, concept, top_n=10)
            by_date = {e["end"]: e["val"] for e in entries}
            current[name] = by_date.get(meta["current_period_end"])
            if meta["prior_period_end"]:
                prior[name] = by_date.get(meta["prior_period_end"])
        else:  # income-statement: use latest annual value for both snapshots
            val = _latest_annual_value(facts, concept)
            current[name] = val
            prior[name]   = val

    return current, prior, meta


def compute_ratios(snap: dict) -> dict:
    """Compute standard financial ratios from a snapshot dict."""
    def sd(num, den):
        if num is None or den is None or den == 0 or pd.isna(num) or pd.isna(den):
            return None
        return num / den

    ta, tl = snap["total_assets"], snap["total_liabilities"]
    ca, cl = snap["current_assets"], snap["current_liabilities"]
    re, ebit = snap["retained_earnings"], snap["ebit"]
    rev, ni  = snap["revenue"], snap["net_income"]
    ie, eq   = snap["interest_expense"], snap["stockholders_equity"]

    out = {
        "debt_to_equity":    sd(tl, eq),
        "current_ratio":     sd(ca, cl),
        "return_on_assets":  sd(ni, ta),
        "interest_coverage": sd(ebit, ie),
        "net_margin":        sd(ni, rev),
    }

    # Altman Z-Score components
    out["z_x1"] = sd((ca - cl) if (ca is not None and cl is not None) else None, ta)
    out["z_x2"] = sd(re, ta)
    out["z_x3"] = sd(ebit, ta)
    out["z_x4"] = sd(eq, tl)        # book equity / liabilities (market cap fallback)
    out["z_x5"] = sd(rev, ta)
    return out
