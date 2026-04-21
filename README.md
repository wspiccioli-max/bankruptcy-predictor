# Bankruptcy & Fraud Risk Predictor

**Course:** BA870-AC820  
**Data:** SEC EDGAR 10-K / 10-Q filings + Yahoo Finance  
**Scope:** Live scoring of the current S&P 500 with quarter-over-quarter risk changes

## What it does

1. **Historical training** — Trains a logistic regression model on 30 hand-picked companies (15 confirmed bankruptcies, 15 healthy controls) using their SEC EDGAR financial ratios.
2. **Live S&P 500 scoring** — Uses that trained model to score all ~500 current S&P 500 companies, pulling their most recent 10-K *or* 10-Q filing from EDGAR.
3. **Quarter-over-quarter deltas** — Also scores each company's prior filing to show how the bankruptcy risk changed period-over-period (a 🔺 rising-risk company is the most useful signal).
4. **Automated weekly refresh** — A GitHub Actions workflow (`.github/workflows/refresh.yml`) reruns the full S&P 500 data pull every Sunday night so the live dashboard never goes stale.

## Project Structure
```
bankruptcy-predictor/
├── analysis.ipynb            # Full academic write-up of the historical backtest
├── collect_data.py           # Phase 1: pull training-set data (30 companies)
├── feature_engineering.py    # Phase 2: compute ratios + Altman Z-score
├── train_models.py           # Phase 3: train logistic regression
├── refresh_sp500.py          # Phase 5: live S&P 500 refresh (weekly cron)
├── dashboard/                # Streamlit multi-page app
│   ├── app.py
│   ├── page1_leaderboard.py  # Live S&P 500 leaderboard with deltas
│   ├── page2_lookup.py       # Per-company KPI dashboard
│   └── page3_comparison.py   # Historical backtest (model validation)
├── utils/                    # Reusable helpers (EDGAR, yfinance, scoring)
├── models/                   # Saved model artifacts (.pkl)
├── data/
│   ├── raw/                  # Cached Wikipedia S&P 500 list
│   └── processed/            # All CSVs used by the dashboard
└── .github/workflows/        # GitHub Actions cron for weekly refresh
```

## Run locally
```bash
pip install -r requirements.txt

# One-time: train the model on historical companies
python collect_data.py
python feature_engineering.py
python train_models.py

# Score the live S&P 500 (takes ~5 minutes)
python refresh_sp500.py

# Launch dashboard
streamlit run dashboard/app.py
```

## Data refresh on GitHub Actions
The workflow at `.github/workflows/refresh.yml` runs every Sunday at 22:00 UTC, re-pulls SEC data for all S&P 500 companies, and commits the updated CSV back to the repo. To trigger a manual refresh, go to the **Actions** tab on GitHub and click "Run workflow".

When deployed on Streamlit Community Cloud, the app auto-redeploys whenever this workflow pushes new data — so the dashboard always shows data that is at most 7 days old.
