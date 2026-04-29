# Bankruptcy Filing Risk Predictor

Educational/prototype screening tool for estimated bankruptcy filing risk. The dashboard uses public SEC EDGAR financial statement data and yfinance market data to compare companies against historical filing examples and public-company controls.

Model probabilities are estimated filing-risk similarity scores, not certainty, predictions of final liquidation, or investment advice.

## Current Methodology

- **Training sample:** 178 companies in the current committed artifacts: 44 bankruptcy-filing positives and 134 controls.
- **Positive label:** bankruptcy filing risk, not confirmed permanent failure.
- **Filing examples:** use the final 10-K available before the bankruptcy filing date where available, avoiding post-filing or successor filings.
- **Controls:** public companies without a known bankruptcy filing at collection time.
- **Supervised models:** Logistic Regression and Random Forest.
- **Validation:** grouped cross-validation keeps the same ticker/company group out of both train and validation folds. The current artifact has one row per ticker, 178 validation groups, and a separate 45-row stratified holdout.
- **Probability quality:** model comparison includes Brier score. Random Forest probabilities are sigmoid-calibrated with `CalibratedClassifierCV`.
- **Benchmarks:** Altman Z-Score and Fraud Risk Score are rule-based indicators, not supervised models.
- **S&P 500 scoring model:** Logistic Regression.
- **Market signal demo:** verified pre-filing yfinance price examples are currently limited to `PCG` and `CZR`; broader historical ticker coverage is a future enhancement because delisted symbols are not reliably available.

## Dashboard Pages

- **S&P 500 Leaderboard:** latest S&P 500 filing-risk scores and period-over-period deltas.
- **Watchlist:** top current S&P 500 companies by estimated filing risk, with risk category and financial drivers.
- **Company Lookup:** per-company filing-risk score, filing metadata, and key ratios.
- **Model Validation:** grouped cross-validation, holdout metrics, majority-class baseline, rule-based benchmarks, Brier score, confusion matrices, feature importance, and selected pre-filing stock-price examples.

## Project Structure

```text
bankruptcy-predictor/
├── collect_data.py           # Pull filing examples + public controls
├── feature_engineering.py    # Compute ratios, Altman Z-score, fraud flags
├── train_models.py           # Train/evaluate Logistic Regression + Random Forest
├── refresh_sp500.py          # Score current S&P 500 filings
├── dashboard/                # Streamlit multi-page app
├── utils/                    # EDGAR, yfinance, scoring helpers
├── data/processed/           # CSV/JSON artifacts used by the dashboard
└── .github/workflows/        # Weekly refresh workflow
```

## Run Locally

```bash
pip install -r requirements.txt

# Rebuild training artifacts
python collect_data.py --control-limit 120
python feature_engineering.py
python train_models.py

# Refresh S&P 500 scores
python refresh_sp500.py

# Launch dashboard
streamlit run dashboard/app.py
```

Quick smoke test:

```bash
python refresh_sp500.py --limit 25
streamlit run dashboard/app.py
```

## Deployment

The GitHub Actions workflow at `.github/workflows/refresh.yml` runs weekly. It rebuilds model artifacts, refreshes S&P 500 scores, and commits updated dashboard data files back to the repo. Streamlit Community Cloud redeploys when those files change.

## Limitations

- This is a prototype screening tool, not a production credit model.
- Filing-risk scores are not calibrated guarantees.
- Reported metrics are prototype screening performance on a curated educational dataset and may not generalize to larger real-world samples.
- Historical bankruptcy labels depend on available public filing metadata.
- yfinance often lacks delisted historical tickers, so the stock-price signal is intentionally limited to verified examples.
