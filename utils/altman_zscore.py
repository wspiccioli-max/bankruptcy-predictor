"""
Altman Z-Score Calculator

The Altman Z-Score was invented in 1968 by NYU professor Edward Altman.
It uses five financial ratios combined into a single number to predict
whether a company will go bankrupt within the next two years.

Formula (for publicly traded manufacturing companies):
  Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

The five components and what they measure:
  X1 = Working Capital / Total Assets        → short-term liquidity
  X2 = Retained Earnings / Total Assets      → accumulated profitability
  X3 = EBIT / Total Assets                   → operating efficiency
  X4 = Market Cap / Total Liabilities        → solvency (market vs. book debt)
  X5 = Revenue / Total Assets                → asset utilization

Interpretation zones:
  Z > 2.99   → "Safe Zone"   (low bankruptcy risk)
  1.81–2.99  → "Grey Zone"   (uncertain — monitor closely)
  Z < 1.81   → "Distress Zone" (high bankruptcy risk)

Note on X4: ideally we use market cap (shares × price).
When market cap is unavailable (e.g., delisted bankrupt companies),
we substitute book value of equity as a conservative fallback.
"""

import pandas as pd
import numpy as np


def compute_zscore(row: pd.Series) -> dict:
    """
    Compute the Altman Z-Score for one company row.

    Accepts a pandas Series (one row of our financials DataFrame).
    Returns a dict with the five components + final Z-score.
    """
    ta  = row.get("total_assets")
    ca  = row.get("current_assets")
    cl  = row.get("current_liabilities")
    re  = row.get("retained_earnings")
    ebit = row.get("ebit")
    tl  = row.get("total_liabilities")
    rev = row.get("revenue")
    eq  = row.get("stockholders_equity")

    # Use market cap if available; fall back to book equity
    # Must use pd.isna() — NaN is truthy in Python, so "NaN or eq" returns NaN
    raw_mktcap = row.get("market_cap")
    mktcap = raw_mktcap if (raw_mktcap is not None and not pd.isna(raw_mktcap)) else eq

    result = {
        "z_x1": None,  # Working Capital / Total Assets
        "z_x2": None,  # Retained Earnings / Total Assets
        "z_x3": None,  # EBIT / Total Assets
        "z_x4": None,  # Market Cap / Total Liabilities
        "z_x5": None,  # Revenue / Total Assets
        "z_score": None,
        "z_zone": "Unknown",
    }

    def _val(v):
        """Return v if it's a real number, else None."""
        return None if (v is None or pd.isna(v)) else v

    ta   = _val(ta)
    ca   = _val(ca)
    cl   = _val(cl)
    re   = _val(re)
    ebit = _val(ebit)
    tl   = _val(tl)
    rev  = _val(rev)
    mktcap = _val(mktcap)

    if not ta or ta == 0:
        return result

    # X1: Working Capital / Total Assets
    if ca is not None and cl is not None:
        result["z_x1"] = (ca - cl) / ta

    # X2: Retained Earnings / Total Assets
    if re is not None:
        result["z_x2"] = re / ta

    # X3: EBIT / Total Assets
    if ebit is not None:
        result["z_x3"] = ebit / ta

    # X4: Market Cap (or book equity) / Total Liabilities
    if mktcap is not None and tl is not None and tl != 0:
        result["z_x4"] = mktcap / tl

    # X5: Revenue / Total Assets
    if rev is not None:
        result["z_x5"] = rev / ta

    # Only compute final Z-score if we have all five components
    components = [result["z_x1"], result["z_x2"], result["z_x3"],
                  result["z_x4"], result["z_x5"]]

    if all(c is not None for c in components):
        z = (1.2 * result["z_x1"] +
             1.4 * result["z_x2"] +
             3.3 * result["z_x3"] +
             0.6 * result["z_x4"] +
             1.0 * result["z_x5"])
        result["z_score"] = round(z, 4)

        # Classify into Altman zones (guard against NaN)
        if pd.isna(z):
            result["z_zone"] = "Unknown"
        elif z > 2.99:
            result["z_zone"] = "Safe"
        elif z >= 1.81:
            result["z_zone"] = "Grey"
        else:
            result["z_zone"] = "Distress"

    return result


def add_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """Apply compute_zscore to every row and merge results back into df."""
    zscore_rows = df.apply(compute_zscore, axis=1)
    zscore_df = pd.DataFrame(list(zscore_rows))
    return pd.concat([df.reset_index(drop=True), zscore_df], axis=1)
