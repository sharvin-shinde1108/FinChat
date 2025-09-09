import os
import json
import tempfile
from datetime import datetime
import traceback
import pandas as pd

import gradio as gr
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Reuse your existing graph + tool
from app import build_graph            # your LangGraph app builder
from tools import get_stock_prices     # tool returning prices + indicators

from langsmith import traceable

from app import run_cli 
from app import run_for_ui  # uses the routed graph

load_dotenv()
GRAPH = build_graph()  # compile once at startup

# def save_graph_png(GRAPH):
#     # Get the PNG bytes
#     png_bytes = GRAPH.get_graph().draw_mermaid_png()

#     # Save to file
#     with open("graph.png", "wb") as f:
#         f.write(png_bytes)

# save_graph_png(GRAPH)

# ---------- helpers: robust series extraction ----------

def _series_from_list(items, value_key_candidates=("value","close")):
    if not items:
        return pd.Series(dtype=float)
    vk = None
    for cand in value_key_candidates:
        if cand in items[0]:
            vk = cand; break
    if vk is None:
        for k in items[0].keys():
            if k != "date":
                vk = k; break
    dates = pd.to_datetime([it.get("date") for it in items])
    vals = []
    for it in items:
        v = it.get(vk)
        try:
            vals.append(float(v) if v is not None else None)
        except Exception:
            vals.append(None)
    s = pd.Series(vals, index=dates).dropna()
    return s

def _get_prices_df(prices_payload: dict):
    out = {}
    if not isinstance(prices_payload, dict):
        return out
    out["price"]        = _series_from_list(prices_payload.get("prices", []), ("value","close"))
    out["vwap"]         = _series_from_list(prices_payload.get("vwap", []))
    out["rsi"]          = _series_from_list(prices_payload.get("rsi", []))
    out["macd"]         = _series_from_list(prices_payload.get("macd", []))
    out["macd_signal"]  = _series_from_list(prices_payload.get("macd_signal", []))
    out["stoch_k"]      = _series_from_list(prices_payload.get("stoch_k", []))
    out["stoch_d"]      = _series_from_list(prices_payload.get("stoch_d", []))
    return out

# ---------- plotting (1 fig per chart, no styles/colors set) ----------

def plot_price_vwap(series_map, ticker: str):
    price = series_map.get("price", pd.Series(dtype=float))
    vwap  = series_map.get("vwap", pd.Series(dtype=float))
    fig = plt.figure()
    ax = plt.gca()
    drew = False
    if not price.empty:
        price.sort_index().plot(ax=ax, label="Close"); drew = True
    if not vwap.empty:
        vwap.sort_index().plot(ax=ax, label="VWAP"); drew = True
    ax.set_title(f"{ticker} — Price & VWAP"); ax.set_xlabel("Date"); ax.set_ylabel("Price")
    if drew: ax.legend()
    fig.tight_layout(); return fig

def plot_rsi(series_map, ticker: str):
    rsi = series_map.get("rsi", pd.Series(dtype=float))
    fig = plt.figure(); ax = plt.gca(); drew = False
    if not rsi.empty:
        rsi.sort_index().plot(ax=ax, label="RSI"); drew = True
    ax.axhline(70, linestyle="--"); ax.axhline(30, linestyle="--")
    ax.set_title(f"{ticker} — RSI"); ax.set_xlabel("Date"); ax.set_ylabel("RSI")
    if drew: ax.legend()
    fig.tight_layout(); return fig

def plot_macd(series_map, ticker: str):
    macd = series_map.get("macd", pd.Series(dtype=float))
    sig  = series_map.get("macd_signal", pd.Series(dtype=float))
    fig = plt.figure(); ax = plt.gca(); drew = False
    if not macd.empty:
        macd.sort_index().plot(ax=ax, label="MACD"); drew = True
    if not sig.empty:
        sig.sort_index().plot(ax=ax, label="Signal"); drew = True
    if not macd.empty and not sig.empty:
        df = pd.concat([macd.rename("macd"), sig.rename("signal")], axis=1).dropna()
        if not df.empty:
            (df["macd"] - df["signal"]).plot(kind="bar", ax=ax, alpha=0.3, label="Histogram"); drew = True
    ax.set_title(f"{ticker} — MACD"); ax.set_xlabel("Date"); ax.set_ylabel("MACD")
    if drew: ax.legend()
    fig.tight_layout(); return fig

def plot_stoch(series_map, ticker: str):
    k = series_map.get("stoch_k", pd.Series(dtype=float))
    d = series_map.get("stoch_d", pd.Series(dtype=float))
    fig = plt.figure(); ax = plt.gca(); drew = False
    if not k.empty:
        k.sort_index().plot(ax=ax, label="%K"); drew = True
    if not d.empty:
        d.sort_index().plot(ax=ax, label="%D"); drew = True
    ax.axhline(80, linestyle="--"); ax.axhline(20, linestyle="--")
    ax.set_title(f"{ticker} — Stochastic"); ax.set_xlabel("Date"); ax.set_ylabel("Oscillator")
    if drew: ax.legend()
    fig.tight_layout(); return fig

# ---------- Gradio wiring ----------

def on_submit(user_input: str):
    try:
        result = run_for_ui(user_input)
        mode = (result.get("mode") or "").lower()
        answer = result.get("answer") or ""

        # default: no charts (educational)
        figs = [None, None, None, None]

        if mode == "analysis":
            tr = result.get("tool_results") or {}
            ticker = tr.get("ticker", "TICKER")
            prices_payload = (tr.get("prices") or {})
            s = _get_prices_df(prices_payload)

            # build figures (each returns a Matplotlib Figure)
            figs = [
                plot_price_vwap(s, ticker),
                plot_rsi(s, ticker),
                plot_macd(s, ticker),
                plot_stoch(s, ticker),
            ]

        return [answer, *figs]
    except Exception as e:
        return [f"Error: {e}", None, None, None, None]

with gr.Blocks() as demo:
    gr.Markdown("# Financial Analyst Agent — RAG + Analysis\nAsk a finance concept (educational) or a stock analysis question (analysis).")

    inp = gr.Textbox(label="Your question")
    out_md = gr.Markdown(label="Answer")

    with gr.Row():
        plot1 = gr.Plot(label="Price & VWAP")
        plot2 = gr.Plot(label="RSI")
    with gr.Row():
        plot3 = gr.Plot(label="MACD")
        plot4 = gr.Plot(label="Stochastic")

    btn = gr.Button("Run")
    btn.click(on_submit, inputs=inp, outputs=[out_md, plot1, plot2, plot3, plot4])
    inp.submit(on_submit, inputs=inp, outputs=[out_md, plot1, plot2, plot3, plot4])

if __name__ == "__main__":
    demo.launch()