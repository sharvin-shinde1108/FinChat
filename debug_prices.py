import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volume import VolumeWeightedAveragePrice
import argparse

def normalize_for_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        # Select the column slice for this ticker
        if "Ticker" in (df.columns.names or []):
            df = df.xs(ticker, axis=1, level="Ticker", drop_level=False)
        else:
            df = df.xs(ticker, axis=1, level=-1, drop_level=False)
        df.columns = df.columns.get_level_values(0)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    return df[cols]

def try_combo(ticker, period, interval):
    print(f"\n=== Trying download(period={period}, interval={interval}) ===")
    df = yf.download(tickers=ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    print("Raw shape:", getattr(df, "shape", None))
    print("Raw columns:", getattr(df, "columns", None))
    if df is not None and not df.empty:
        df = normalize_for_ticker(df, ticker)
        print("Normalized columns:", df.columns.tolist())
        print("Head:\n", df.head())
        print("Tail:\n", df.tail())
    return df

def compute_indicators(df):
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]

    rsi = RSIIndicator(close=close, window=14).rsi()
    print("RSI last 5:\n", rsi.tail())

    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    print("%K last 5:\n", stoch.stoch().tail())
    print("%D last 5:\n", stoch.stoch_signal().tail())

    macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    print("MACD last 5:\n", macd_ind.macd().tail())
    print("Signal last 5:\n", macd_ind.macd_signal().tail())
    print("Hist last 5:\n", macd_ind.macd_diff().tail())

    vwap = VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=vol, window=14)\
        .volume_weighted_average_price()
    print("VWAP last 5:\n", vwap.tail())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="TSLA")
    args = ap.parse_args()

    combos = [("18mo", "1wk"), ("2y", "1wk"), ("6mo", "1d")]
    df = None
    for period, interval in combos:
        df = try_combo(args.ticker, period, interval)
        if df is not None and not df.empty:
            break

    if df is None or df.empty:
        print("\n❌ No data fetched. Possible causes: network/proxy, yfinance version, ticker symbol, rate limiting.")
        print("Tips: upgrade yfinance (uv pip install -U yfinance), try a different network, or switch intervals.")
        return

    print("\n=== Computing indicators ===")
    compute_indicators(df)


if __name__ == "__main__":
    main()
