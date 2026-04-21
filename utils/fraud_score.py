"""
Fraud Risk Scoring — simplified Beneish M-Score indicators

The Beneish M-Score (1999) detects earnings manipulation using 8 ratios
that require TWO consecutive years of financial data (year-over-year changes).

Since our dataset only has the most recent 10-K snapshot per company,
we can't compute the full M-Score. Instead we build a "fraud risk index"
from the indicators we CAN compute with a single year of data.

Our proxy variables:

  1. LEVERAGE_FLAG    → debt-to-equity > 3.0 (dangerously leveraged)
  2. MARGIN_PRESSURE  → net margin < 0 (losing money)
  3. LIQUIDITY_FLAG   → current ratio < 1.0 (can't pay short-term bills)
  4. ASSET_BLOAT      → total assets >> revenue (assets not generating sales)
  5. INTEREST_STRAIN  → interest coverage < 1.5 (barely covering interest)

Each flag is scored 0 or 1. The fraud_risk_score = sum of flags (0–5).
Higher = more red flags.

Note: This is an educational heuristic, NOT the actual Beneish M-Score.
In a production system you would pull 2+ years of 10-K data to compute
the true M-Score with all 8 components.
"""

import pandas as pd


def compute_fraud_score(row: pd.Series) -> dict:
    """
    Compute a fraud/distress risk score for one company.

    Returns a dict with individual flags and a total score (0–5).
    """
    flags = {}

    # Flag 1: Extreme leverage (debt-to-equity > 3)
    dte = row.get("debt_to_equity")
    flags["flag_leverage"] = int(dte is not None and dte > 3.0)

    # Flag 2: Negative profit margin (losing money on sales)
    margin = row.get("net_margin")
    flags["flag_neg_margin"] = int(margin is not None and margin < 0)

    # Flag 3: Current ratio below 1 (can't cover short-term obligations)
    cr = row.get("current_ratio")
    flags["flag_low_liquidity"] = int(cr is not None and cr < 1.0)

    # Flag 4: Asset bloat — total assets more than 3× revenue
    # Healthy companies convert assets into revenue efficiently
    ta = row.get("total_assets")
    rev = row.get("revenue")
    if ta and rev and rev > 0:
        flags["flag_asset_bloat"] = int(ta / rev > 3.0)
    else:
        flags["flag_asset_bloat"] = 0

    # Flag 5: Low interest coverage — struggling to pay interest
    ic = row.get("interest_coverage")
    flags["flag_interest_strain"] = int(ic is not None and ic < 1.5)

    # Total score: count of red flags
    fraud_risk_score = sum(flags.values())

    # Risk label for display
    if fraud_risk_score >= 4:
        risk_label = "High"
    elif fraud_risk_score >= 2:
        risk_label = "Medium"
    else:
        risk_label = "Low"

    return {
        **flags,
        "fraud_risk_score": fraud_risk_score,
        "fraud_risk_label": risk_label,
    }


def add_fraud_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Apply compute_fraud_score to every row and merge results back."""
    fraud_rows = df.apply(compute_fraud_score, axis=1)
    fraud_df = pd.DataFrame(list(fraud_rows))
    return pd.concat([df.reset_index(drop=True), fraud_df], axis=1)
