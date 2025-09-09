# from typing import Dict, Any, Annotated
# from datetime import datetime
# import pandas as pd
# import yfinance as yf

# from ta.momentum import RSIIndicator, StochasticOscillator
# from ta.trend import MACD
# from ta.volume import VolumeWeightedAveragePrice

# from langchain_core.tools import tool

# from langsmith import traceable

# def _series_to_points(s: pd.Series, last_n: int = 12):
#     """Convert a pandas Series to a list of {date, value} points (ISO date)."""
#     s = s.dropna().tail(last_n)
#     return [
#         {"date": d.strftime("%Y-%m-%d"), "value": float(v)}
#         for d, v in s.items()
#     ]

# # @traceable(name="get_stock_prices")
# @tool
# def get_stock_prices(ticker: str) -> Dict[str, Any]:
#     """
#     Fetch historical OHLCV data and compute RSI, Stochastic, MACD, VWAP.
#     Robust: handles yfinance MultiIndex columns and falls back weekly→daily.
#     """
#     import pandas as pd
#     from ta.momentum import RSIIndicator, StochasticOscillator
#     from ta.trend import MACD
#     from ta.volume import VolumeWeightedAveragePrice

#     if not ticker or not isinstance(ticker, str):
#         raise ValueError("Please provide a valid ticker symbol, e.g., 'AAPL'.")

#     tried, used = [], None
#     df = None
#     try:
#         # Weekly (as in the tutorial)
#         df = fetch_prices(ticker, period="18mo", interval="1wk", auto_adjust=True)
#         used = {"period": "18mo", "interval": "1wk"}
#     except Exception as e_weekly:
#         tried.append({"period": "18mo", "interval": "1wk", "error": str(e_weekly)})
#         # Daily fallback (typically more reliable)
#         df = fetch_prices(ticker, period="6mo", interval="1d", auto_adjust=True)
#         used = {"period": "6mo", "interval": "1d"}

#     # Columns are single-level now: Open/High/Low/Close/Volume
#     close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

#     # Compute indicators with guards
#     indicators: Dict[str, Any] = {}
#     n = len(df.dropna())
#     if n >= 30:
#         rsi = RSIIndicator(close=close, window=14).rsi()
#         stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
#         macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)

#         def _series_to_points(s: pd.Series, last_n: int = 12):
#             s = s.dropna().tail(last_n)
#             return [{"date": d.strftime("%Y-%m-%d"), "value": float(v)} for d, v in s.items()]

#         indicators.update({
#             "rsi": _series_to_points(rsi),
#             "stochastic_k": _series_to_points(stoch.stoch()),
#             "stochastic_d": _series_to_points(stoch.stoch_signal()),
#             "macd": _series_to_points(macd_ind.macd()),
#             "macd_signal": _series_to_points(macd_ind.macd_signal()),
#             "macd_hist": _series_to_points(macd_ind.macd_diff()),
#         })
#     else:
#         indicators["warning"] = f"Not enough data ({n} rows) to compute all indicators."

#     try:
#         vwap = VolumeWeightedAveragePrice(
#             high=high, low=low, close=close, volume=volume, window=14
#         ).volume_weighted_average_price()
#         indicators["vwap"] = [{"date": d.strftime("%Y-%m-%d"), "value": float(v)}
#                               for d, v in vwap.dropna().tail(12).items()]
#     except Exception as e:
#         indicators["vwap_error"] = f"VWAP computation failed: {e!s}"

#     latest_close = float(close.dropna().iloc[-1])
#     sample = [{
#         "date": idx.strftime("%Y-%m-%d"),
#         "open": float(row["Open"]),
#         "high": float(row["High"]),
#         "low": float(row["Low"]),
#         "close": float(row["Close"]),
#         "volume": int(row["Volume"]),
#     } for idx, row in df.tail(16).iterrows()]

#     return {
#         "ticker": ticker.upper(),
#         "fetch_used": used,
#         "fetch_tried": tried,
#         "latest_close": latest_close,
#         "indicators": indicators,
#         "sample_prices": sample,
#     }


# # @traceable(name="get_financial_metrics")
# @tool
# def get_financial_metrics(ticker: str) -> Dict[str, Any]:
#     """
#     Fetch basic financial metrics using yfinance.

#     Returns:
#         dict with commonly used valuation/quality ratios where available.
#     """
#     if not ticker or not isinstance(ticker, str):
#         raise ValueError("Please provide a valid ticker symbol, e.g., 'AAPL'.")

#     tk = yf.Ticker(ticker)
#     info = {}
#     # yfinance 'info' can be slow or partially populated; guard with try/except
#     try:
#         info = tk.info or {}
#     except Exception:
#         info = {}

#     # Prefer 'fast_info' for some fields when available
#     fast = {}
#     try:
#         fast = tk.fast_info or {}
#     except Exception:
#         fast = {}

#     def _get(key, default=None):
#         return info.get(key, default)

#     metrics = {
#         "ticker": ticker.upper(),
#         "market_cap": fast.get("market_cap") or _get("marketCap"),
#         "trailing_pe": _get("trailingPE"),
#         "forward_pe": _get("forwardPE"),
#         "price_to_book": _get("priceToBook"),
#         "debt_to_equity": _get("debtToEquity"),
#         "profit_margins": _get("profitMargins"),
#         "return_on_equity": _get("returnOnEquity"),
#         "currency": fast.get("currency") or _get("currency"),
#     }
#     # Clean numeric types (best-effort)
#     for k, v in list(metrics.items()):
#         if k in {"ticker", "currency"}:
#             continue
#         try:
#             if v is not None:
#                 metrics[k] = float(v)
#         except Exception:
#             pass

#     return metrics

# import pandas as pd
# import yfinance as yf

# def _normalize_yf(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
#     """
#     Flatten yfinance MultiIndex columns to single level: ['Open','High','Low','Close','Volume'].
#     Works when df has columns like ('Close','TSLA') with names ['Price','Ticker'].
#     """
#     if isinstance(df.columns, pd.MultiIndex):
#         # If ticker is a second-level column, select that cross-section
#         lvl_names = list(df.columns.names or [])
#         # Prefer selecting by the 'Ticker' level if present
#         if "Ticker" in lvl_names:
#             df = df.xs(ticker, axis=1, level="Ticker", drop_level=False)
#         else:
#             # Fallback: assume the right-most level is ticker
#             df = df.xs(ticker, axis=1, level=-1, drop_level=False)
#         # After xs, drop the remaining level to get single-level columns
#         df.columns = df.columns.get_level_values(0)  # e.g., 'Open','High','Low','Close','Volume'
#     # Ensure expected columns exist and proper order
#     expected = ["Open", "High", "Low", "Close", "Volume"]
#     missing = [c for c in expected if c not in df.columns]
#     if missing:
#         raise ValueError(f"Missing expected columns after normalization: {missing}; got {df.columns.tolist()}")
#     return df[expected]

# def fetch_prices(ticker: str, period="18mo", interval="1wk", auto_adjust=True) -> pd.DataFrame:
#     df = yf.download(
#         tickers=ticker,
#         period=period,
#         interval=interval,
#         auto_adjust=auto_adjust,   # yfinance now defaults to True; we keep it explicit
#         progress=False,
#         # threads=False  # avoid threads on Windows if you hit hangs
#     )
#     if df is None or df.empty:
#         raise ValueError(f"No data returned for {ticker} (period={period}, interval={interval}).")
#     df = _normalize_yf(df, ticker)
#     # Make sure index is DatetimeIndex
#     if not isinstance(df.index, pd.DatetimeIndex):
#         df.index = pd.to_datetime(df.index, errors="coerce")
#     return df


# tools.py
from __future__ import annotations
import warnings
from typing import Dict, List, Any

import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volume import VolumeWeightedAveragePrice

from langchain.tools import tool
from pydantic import BaseModel, Field

# ---------- helpers ----------

# def _series_to_list(s: pd.Series, value_name: str = "value", limit: int | None = 180) -> List[Dict[str, Any]]:
#     """Convert a Series with a DatetimeIndex into [{date, value}], drop NaNs, clip to last N."""
#     if s is None or len(s) == 0:
#         return []
#     s = s.dropna()
#     if s.empty:
#         return []
#     if limit:
#         s = s.tail(limit)
#     # ensure tz-naive isoformat
#     idx = pd.to_datetime(s.index).tz_localize(None, nonexistent="shift_forward", ambiguous="NaT", errors="ignore")
#     out = [{"date": d.isoformat(), "value": float(v)} for d, v in zip(idx, s.values)]
#     return out
def _series_to_list(s: pd.Series, value_name: str = "value", limit: int | None = 180):
    if s is None or len(s) == 0:
        return []
    s = s.dropna()
    if s.empty:
        return []
    if limit:
        s = s.tail(limit)

    # Make index tz-naive safely across pandas versions
    idx = pd.to_datetime(s.index)
    try:
        # works when index is tz-aware and this pandas supports tz_localize(None)
        idx = idx.tz_localize(None)
    except Exception:
        try:
            # some pandas combos require tz_convert(None) instead
            idx = idx.tz_convert(None)
        except Exception:
            # leave as-is if already naive
            pass

    out = [{"date": d.isoformat(), "value": float(v)} for d, v in zip(idx, s.values)]
    return out

def _history_with_fallbacks(ticker: str) -> pd.DataFrame:
    """Try a few period/interval combos to avoid empty frames."""
    tried = [
        ("6mo", "1d"),
        ("1y", "1d"),
        ("max", "1d"),
    ]
    for period, interval in tried:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    return pd.DataFrame()

def _compute_indicators(df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    if df.empty:
        # structured empty payload
        return dict(prices=[], vwap=[], rsi=[], macd=[], macd_signal=[], stoch_k=[], stoch_d=[])

    # canonical columns
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float)

    # RSI (default 14)
    rsi = RSIIndicator(close=close, window=14).rsi()

    # Stochastic (14, 3, 3)
    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    stoch_k = stoch.stoch()
    stoch_d = stoch.stoch_signal()

    # MACD (12, 26, 9)
    macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    macd = macd_ind.macd()
    macd_signal = macd_ind.macd_signal()

    # VWAP (requires high/low/close/volume)
    try:
        vwap = VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=vol).volume_weighted_average_price()
    except Exception:
        vwap = pd.Series(index=close.index, dtype=float)

    payload = {
        "prices": _series_to_list(close.rename("close"), value_name="close"),
        "vwap": _series_to_list(vwap),
        "rsi": _series_to_list(rsi),
        "macd": _series_to_list(macd),
        "macd_signal": _series_to_list(macd_signal),
        "stoch_k": _series_to_list(stoch_k),
        "stoch_d": _series_to_list(stoch_d),
    }
    return payload

# ---------- LangChain Tool ----------

class PriceInput(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, e.g., AAPL, MSFT, GOOG")

@tool("get_stock_prices", args_schema=PriceInput)
def get_stock_prices(ticker: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch OHLCV with yfinance and compute RSI, Stochastic, MACD, and VWAP.
    Returns a dict with lists of {date, value} (prices uses Close).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = _history_with_fallbacks(ticker)

    return _compute_indicators(df)
