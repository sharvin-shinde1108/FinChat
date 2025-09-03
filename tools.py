from typing import Dict, Any
from datetime import datetime
import pandas as pd
import yfinance as yf

from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volume import VolumeWeightedAveragePrice

from langchain_core.tools import tool


def _series_to_points(s: pd.Series, last_n: int = 12):
    """Convert a pandas Series to a list of {date, value} points (ISO date)."""
    s = s.dropna().tail(last_n)
    return [
        {"date": d.strftime("%Y-%m-%d"), "value": float(v)}
        for d, v in s.items()
    ]


# @tool
# def get_stock_prices(ticker: str) -> Dict[str, Any]:
#     """
#     Fetch historical OHLCV data and compute RSI, Stochastic, MACD, VWAP.

#     Args:
#         ticker: Stock symbol, e.g., "AAPL", "TSLA".

#     Returns:
#         dict with 'ticker', 'period', 'interval', 'latest_close', 'indicators', and 'sample_prices'.
#     """
#     if not ticker or not isinstance(ticker, str):
#         raise ValueError("Please provide a valid ticker symbol, e.g., 'AAPL'.")

#     period = "18mo"
#     interval = "1wk"

#     # Download price data
#     df = yf.download(
#         tickers=ticker,
#         period=period,
#         interval=interval,
#         auto_adjust=False,
#         progress=False,
#         threads=True,
#     )

#     if df.empty:
#         raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol or try a different period.")

#     # Ensure columns
#     for col in ["Open", "High", "Low", "Close", "Volume"]:
#         if col not in df.columns:
#             raise ValueError(f"Expected column '{col}' not found in data.")

#     # Indicators
#     close = df["Close"]
#     high = df["High"]
#     low = df["Low"]
#     volume = df["Volume"]

#     # RSI
#     rsi = RSIIndicator(close=close, window=14).rsi()

#     # Stochastic Oscillator
#     stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
#     stoch_k = stoch.stoch()          # %K
#     stoch_d = stoch.stoch_signal()   # %D

#     # MACD
#     macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
#     macd = macd_ind.macd()
#     macd_signal = macd_ind.macd_signal()
#     macd_hist = macd_ind.macd_diff()

#     # VWAP (uses a rolling window; here we use full-series for context)
#     vwap = VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=volume, window=14).volume_weighted_average_price()

#     # Build response
#     latest_close = float(close.dropna().iloc[-1])
#     result = {
#         "ticker": ticker.upper(),
#         "period": period,
#         "interval": interval,
#         "latest_close": latest_close,
#         "indicators": {
#             "rsi": _series_to_points(rsi),
#             "stochastic_k": _series_to_points(stoch_k),
#             "stochastic_d": _series_to_points(stoch_d),
#             "macd": _series_to_points(macd),
#             "macd_signal": _series_to_points(macd_signal),
#             "macd_hist": _series_to_points(macd_hist),
#             "vwap": _series_to_points(vwap),
#         },
#         # small sample of price history for the LLM context
#         "sample_prices": [
#             {
#                 "date": idx.strftime("%Y-%m-%d"),
#                 "open": float(row["Open"]),
#                 "high": float(row["High"]),
#                 "low": float(row["Low"]),
#                 "close": float(row["Close"]),
#                 "volume": int(row["Volume"]),
#             }
#             for idx, row in df.tail(16).iterrows()
#         ],
#     }
#     return result

@tool
def get_stock_prices(ticker: str) -> Dict[str, Any]:
    """
    Fetch historical OHLCV data and compute RSI, Stochastic, MACD, VWAP.
    Robust: handles yfinance MultiIndex columns and falls back weekly→daily.
    """
    import pandas as pd
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.trend import MACD
    from ta.volume import VolumeWeightedAveragePrice

    if not ticker or not isinstance(ticker, str):
        raise ValueError("Please provide a valid ticker symbol, e.g., 'AAPL'.")

    tried, used = [], None
    df = None
    try:
        # Weekly (as in the tutorial)
        df = fetch_prices(ticker, period="18mo", interval="1wk", auto_adjust=True)
        used = {"period": "18mo", "interval": "1wk"}
    except Exception as e_weekly:
        tried.append({"period": "18mo", "interval": "1wk", "error": str(e_weekly)})
        # Daily fallback (typically more reliable)
        df = fetch_prices(ticker, period="6mo", interval="1d", auto_adjust=True)
        used = {"period": "6mo", "interval": "1d"}

    # Columns are single-level now: Open/High/Low/Close/Volume
    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    # Compute indicators with guards
    indicators: Dict[str, Any] = {}
    n = len(df.dropna())
    if n >= 30:
        rsi = RSIIndicator(close=close, window=14).rsi()
        stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
        macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)

        def _series_to_points(s: pd.Series, last_n: int = 12):
            s = s.dropna().tail(last_n)
            return [{"date": d.strftime("%Y-%m-%d"), "value": float(v)} for d, v in s.items()]

        indicators.update({
            "rsi": _series_to_points(rsi),
            "stochastic_k": _series_to_points(stoch.stoch()),
            "stochastic_d": _series_to_points(stoch.stoch_signal()),
            "macd": _series_to_points(macd_ind.macd()),
            "macd_signal": _series_to_points(macd_ind.macd_signal()),
            "macd_hist": _series_to_points(macd_ind.macd_diff()),
        })
    else:
        indicators["warning"] = f"Not enough data ({n} rows) to compute all indicators."

    try:
        vwap = VolumeWeightedAveragePrice(
            high=high, low=low, close=close, volume=volume, window=14
        ).volume_weighted_average_price()
        indicators["vwap"] = [{"date": d.strftime("%Y-%m-%d"), "value": float(v)}
                              for d, v in vwap.dropna().tail(12).items()]
    except Exception as e:
        indicators["vwap_error"] = f"VWAP computation failed: {e!s}"

    latest_close = float(close.dropna().iloc[-1])
    sample = [{
        "date": idx.strftime("%Y-%m-%d"),
        "open": float(row["Open"]),
        "high": float(row["High"]),
        "low": float(row["Low"]),
        "close": float(row["Close"]),
        "volume": int(row["Volume"]),
    } for idx, row in df.tail(16).iterrows()]

    return {
        "ticker": ticker.upper(),
        "fetch_used": used,
        "fetch_tried": tried,
        "latest_close": latest_close,
        "indicators": indicators,
        "sample_prices": sample,
    }



@tool
def get_financial_metrics(ticker: str) -> Dict[str, Any]:
    """
    Fetch basic financial metrics using yfinance.

    Returns:
        dict with commonly used valuation/quality ratios where available.
    """
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Please provide a valid ticker symbol, e.g., 'AAPL'.")

    tk = yf.Ticker(ticker)
    info = {}
    # yfinance 'info' can be slow or partially populated; guard with try/except
    try:
        info = tk.info or {}
    except Exception:
        info = {}

    # Prefer 'fast_info' for some fields when available
    fast = {}
    try:
        fast = tk.fast_info or {}
    except Exception:
        fast = {}

    def _get(key, default=None):
        return info.get(key, default)

    metrics = {
        "ticker": ticker.upper(),
        "market_cap": fast.get("market_cap") or _get("marketCap"),
        "trailing_pe": _get("trailingPE"),
        "forward_pe": _get("forwardPE"),
        "price_to_book": _get("priceToBook"),
        "debt_to_equity": _get("debtToEquity"),
        "profit_margins": _get("profitMargins"),
        "return_on_equity": _get("returnOnEquity"),
        "currency": fast.get("currency") or _get("currency"),
    }
    # Clean numeric types (best-effort)
    for k, v in list(metrics.items()):
        if k in {"ticker", "currency"}:
            continue
        try:
            if v is not None:
                metrics[k] = float(v)
        except Exception:
            pass

    return metrics

import pandas as pd
import yfinance as yf

def _normalize_yf(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Flatten yfinance MultiIndex columns to single level: ['Open','High','Low','Close','Volume'].
    Works when df has columns like ('Close','TSLA') with names ['Price','Ticker'].
    """
    if isinstance(df.columns, pd.MultiIndex):
        # If ticker is a second-level column, select that cross-section
        lvl_names = list(df.columns.names or [])
        # Prefer selecting by the 'Ticker' level if present
        if "Ticker" in lvl_names:
            df = df.xs(ticker, axis=1, level="Ticker", drop_level=False)
        else:
            # Fallback: assume the right-most level is ticker
            df = df.xs(ticker, axis=1, level=-1, drop_level=False)
        # After xs, drop the remaining level to get single-level columns
        df.columns = df.columns.get_level_values(0)  # e.g., 'Open','High','Low','Close','Volume'
    # Ensure expected columns exist and proper order
    expected = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns after normalization: {missing}; got {df.columns.tolist()}")
    return df[expected]

def fetch_prices(ticker: str, period="18mo", interval="1wk", auto_adjust=True) -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,   # yfinance now defaults to True; we keep it explicit
        progress=False,
        # threads=False  # avoid threads on Windows if you hit hangs
    )
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker} (period={period}, interval={interval}).")
    df = _normalize_yf(df, ticker)
    # Make sure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    return df
