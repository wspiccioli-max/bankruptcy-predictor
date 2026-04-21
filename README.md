# Bankruptcy & Fraud Risk Predictor
**Course:** BA870-AC820  
**Data:** SEC EDGAR 10-K filings (2000–2024) + yfinance real-time data

## Project Structure
```
bankruptcy-predictor/
├── data/
│   ├── raw/          # Raw EDGAR downloads and yfinance pulls
│   ├── processed/    # Cleaned DataFrames with financial ratios
│   └── labels/       # Bankrupt vs. healthy company labels
├── models/           # Saved model files (.pkl)
├── dashboard/        # Streamlit app pages
├── notebooks/        # Exploration notebooks
└── utils/            # Shared helper functions
```

## Phases
1. Data Collection (SEC EDGAR + yfinance)
2. Feature Engineering (financial ratios, Altman Z-score)
3. Model Training (logistic regression, fraud scorer)
4. Streamlit Dashboard (3 pages)

## Run
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```
