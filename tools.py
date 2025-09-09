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