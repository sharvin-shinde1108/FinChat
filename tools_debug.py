"""
Patched tools with verbose debugging, retries, and safe fallbacks.
Save as tools_debug.py and temporarily import these in app.py for troubleshooting.
"""

from typing import Dict, Any, Optional
import time
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volume import VolumeWeightedAveragePrice
from langchain_core.tools import tool


def _series_to_points(s: pd.Series, last_n: int = 12):
    s = s.dropna().tail(last_n)
    return [{"date": d.strftime("%Y-%m-%d"), "value": float(v)} for d, v in s.items()]


def _try_download(
    ticker: str,
    period: str,
    interval: str,
    auto_adjust: bool = True,
    attempts: int = 2,
    pause: float = 1.0,
) -> pd.DataFrame:
    """Retry wrapper around yfinance.download with controlled args (no threads)."""
    last_err: Optional[Exception] = None
    for i in range(1, attempts + 1):
        try:
            df = yf.download(
                tickers=ticker,
                period=period,
                interval=interval,
                auto_adjust=auto_adjust,
                progress=False,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception as e:
            last_err = e
        time.sleep(pause)
    if last_err:
        raise last_err
    return pd.DataFrame()


def _fallback_history(ticker: str, period: str, interval: str, auto_adjust: bool = True) -> pd.DataFrame:
    """Fallback path using Ticker.history() API."""
    t = yf.Ticker(ticker)
    try:
        df = t.history(period=period, interval=interval, auto_adjust=auto_adjust)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()


def _compute_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    # Ensure necessary columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in data. Got: {df.columns.tolist()}")

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    out = {"indicators": {}}

    # Compute safely only if enough rows
    n = len(df.dropna())
    if n >= 20:
        rsi = RSIIndicator(close=close, window=14).rsi()
        out["indicators"]["rsi"] = _series_to_points(rsi)
        stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
        out["indicators"]["stochastic_k"] = _series_to_points(stoch.stoch())
        out["indicators"]["stochastic_d"] = _series_to_points(stoch.stoch_signal())
        macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        out["indicators"]["macd"] = _series_to_points(macd_ind.macd())
        out["indicators"]["macd_signal"] = _series_to_points(macd_ind.macd_signal())
        out["indicators"]["macd_hist"] = _series_to_points(macd_ind.macd_diff())
    else:
        out["indicators"]["warning"] = f"Not enough data ({n} rows) to compute all indicators."

    # VWAP needs highs/lows/volumes; a 14-period rolling window is typical
    try:
        vwap = VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=volume, window=14)\
            .volume_weighted_average_price()
        out["indicators"]["vwap"] = _series_to_points(vwap)
    except Exception as e:
        out["indicators"]["vwap_error"] = f"VWAP computation failed: {e!s}"

    return out


@tool
def get_stock_prices_debug(ticker: str) -> Dict[str, Any]:
    """
    Debug version: tries multiple download strategies and intervals.
    Order:
      1) period=18mo, interval=1wk
      2) period=2y, interval=1wk
      3) period=6mo, interval=1d
      4) Ticker.history(period=18mo, interval=1wk)
    Returns verbose metadata describing what worked.
    """
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Please provide a valid ticker symbol, e.g., 'AAPL'.")

    attempts = [
        ("download", {"period": "18mo", "interval": "1wk"}),
        ("download", {"period": "2y", "interval": "1wk"}),
        ("download", {"period": "6mo", "interval": "1d"}),
        ("history", {"period": "18mo", "interval": "1wk"}),
    ]

    meta = {"attempts": [], "success": False, "used": None}
    df = None
    for (kind, params) in attempts:
        try:
            if kind == "download":
                df_try = _try_download(ticker, **params)
            else:
                df_try = _fallback_history(ticker, **params)

            meta["attempts"].append(
                {"kind": kind, **params, "rows": 0 if df_try is None else int(len(df_try))}
            )
            if df_try is not None and not df_try.empty:
                df = df_try
                meta["success"] = True
                meta["used"] = {"kind": kind, **params}
                break
        except Exception as e:
            meta["attempts"].append({"kind": kind, **params, "error": str(e)})

    if df is None or df.empty:
        return {
            "ticker": ticker.upper(),
            "error": "Failed to download price data via multiple strategies.",
            "debug": meta,
        }

    # Normalize index to datetime (yfinance sometimes returns tz-aware)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    latest_close = float(df["Close"].dropna().iloc[-1])
    indicators = _compute_indicators(df)

    sample = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"]),
        }
        for idx, row in df.tail(16).iterrows()
    ]

    return {
        "ticker": ticker.upper(),
        "latest_close": latest_close,
        "period_used": meta["used"],
        "debug": meta,
        **indicators,
        "sample_prices": sample,
    }
