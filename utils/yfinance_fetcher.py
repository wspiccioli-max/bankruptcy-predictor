"""
Fetches real-time market data from Yahoo Finance via the yfinance library.

Why do we need this on top of EDGAR?
  EDGAR gives us accounting numbers (balance sheet, income statement).
  yfinance gives us market-based numbers:
    - Current stock price
    - Market capitalization (shares × price) ← needed for Altman Z-score
    - 52-week high/low (useful for the health gauge)
    - Beta (how volatile the stock is vs. the market)
    - Analyst ratings / credit ratings when available

Note: Bankrupt companies are delisted, so yfinance returns limited data for them.
We handle that gracefully by returning None for missing fields.
"""

import yfinance as yf
import pandas as pd


def fetch_market_data(ticker: str) -> dict:
    """
    Pull current market snapshot for one ticker.

    Returns a dict with:
      market_cap, price, beta, pe_ratio, week52_high, week52_low
    """
    print(f"  Fetching yfinance data for {ticker}...")
    try:
        info = yf.Ticker(ticker).info

        return {
            "market_cap":   info.get("marketCap"),
            "price":        info.get("currentPrice") or info.get("previousClose"),
            "beta":         info.get("beta"),
            "pe_ratio":     info.get("trailingPE"),
            "week52_high":  info.get("fiftyTwoWeekHigh"),
            "week52_low":   info.get("fiftyTwoWeekLow"),
            "analyst_rating": info.get("recommendationKey"),  # e.g. "buy", "hold"
            "shares_outstanding": info.get("sharesOutstanding"),
        }

    except Exception as e:
        print(f"    [yfinance] Failed for {ticker}: {e}")
        return {
            "market_cap": None,
            "price": None,
            "beta": None,
            "pe_ratio": None,
            "week52_high": None,
            "week52_low": None,
            "analyst_rating": None,
            "shares_outstanding": None,
        }


def fetch_price_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Download daily closing prices for the past N years.
    Used for the Z-score time series chart on the dashboard.
    """
    try:
        df = yf.download(ticker, period=period, progress=False)
        return df[["Close"]].rename(columns={"Close": "close_price"})
    except Exception as e:
        print(f"    [yfinance] Price history failed for {ticker}: {e}")
        return pd.DataFrame()
