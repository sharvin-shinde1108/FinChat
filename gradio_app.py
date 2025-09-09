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

# ---------- plotting helpers (return FILE PATHS) ----------

# def _plot_line(dates, values, title, ylab):
#     if not dates or not values:
#         return None
#     fig, ax = plt.subplots(figsize=(7, 3.2))
#     ax.plot(dates, values, linewidth=2)   # no explicit colors/styles per guidance
#     ax.set_title(title)
#     ax.set_xlabel("Date")
#     ax.set_ylabel(ylab)
#     ax.tick_params(axis="x", rotation=45)
#     fig.tight_layout()

#     tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
#     fig.savefig(tmp.name, format="png", dpi=160, bbox_inches="tight")
#     plt.close(fig)
#     return tmp.name

# def _plot_price(sample_prices):
#     if not sample_prices:
#         return None
#     dates = [p["date"] for p in sample_prices]
#     closes = [p["close"] for p in sample_prices]
#     return _plot_line(dates, closes, "Recent Close Prices (sample)", "Close")

# def _plot_rsi(indicators):
#     rsi_pts = (indicators or {}).get("rsi") or []
#     if not rsi_pts:
#         return None
#     dates = [p["date"] for p in rsi_pts]
#     vals = [p["value"] for p in rsi_pts]
#     return _plot_line(dates, vals, "RSI (last points)", "RSI")

# def _plot_macd(indicators):
#     macd_pts = (indicators or {}).get("macd") or []
#     sig_pts  = (indicators or {}).get("macd_signal") or []
#     if not macd_pts or not sig_pts:
#         return None
#     # assume same dates length/order
#     dates = [p["date"] for p in macd_pts]
#     macd_vals = [p["value"] for p in macd_pts]
#     sig_vals  = [p["value"] for p in sig_pts]

#     fig, ax = plt.subplots(figsize=(7, 3.2))
#     ax.plot(dates, macd_vals, linewidth=2, label="MACD")
#     ax.plot(dates, sig_vals,  linewidth=2, label="Signal")
#     ax.set_title("MACD vs Signal (last points)")
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Value")
#     ax.tick_params(axis="x", rotation=45)
#     ax.legend()
#     fig.tight_layout()

#     tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
#     fig.savefig(tmp.name, format="png", dpi=160, bbox_inches="tight")
#     plt.close(fig)
#     return tmp.name


# ---------- core handler ----------

# @traceable(name="gradio_analyze")
# def analyze(ticker: str, question: str, model_name: str, show_charts: bool):
#     """
#     1) Set model via OPENAI_MODEL (used in your app).
#     2) Stream the LangGraph and return final LLM message.
#     3) Optionally render charts (Close / RSI / MACD) from the tool output.
#     """
#     ticker = (ticker or "").strip().upper()
#     if not ticker:
#         return "Please enter a ticker (e.g., TSLA).", None, None, None, ""

#     question = (question or "").strip() or "Should I buy this stock?"
#     os.environ["OPENAI_MODEL"] = model_name  # app reads this env var

#     # Run your graph
#     try:
#         inputs = {"messages": [HumanMessage(content=question)], "stock": ticker}
#         final_msg = None
#         for ev in GRAPH.stream(inputs, stream_mode="values"):
#             if "messages" in ev and ev["messages"]:
#                 final_msg = ev["messages"][-1]
#         analysis = final_msg.content if final_msg else "No output produced."
#     except Exception:
#         analysis = f"❌ Error while running analysis:\n```\n{traceback.format_exc()}\n```"

#     # Fetch tool output for charts + meta (don’t fail UI if charting fails)
#     price_path = rsi_path = macd_path = None
#     meta_txt = ""
#     try:
#         tool_out = get_stock_prices.invoke({"ticker": ticker})
#         meta_txt = json.dumps(
#             {
#                 "fetch_used": tool_out.get("fetch_used") or {
#                     "period": tool_out.get("period"),
#                     "interval": tool_out.get("interval"),
#                 },
#                 "fetch_tried": tool_out.get("fetch_tried"),
#                 "latest_close": tool_out.get("latest_close"),
#             },
#             indent=2,
#         )
#         if show_charts:
#             price_path = _plot_price(tool_out.get("sample_prices", []))
#             indicators = tool_out.get("indicators", {})
#             rsi_path   = _plot_rsi(indicators)
#             macd_path  = _plot_macd(indicators)
#     except Exception as e:
#         meta_txt = f"Note: unable to fetch chart/tool data: {e}"

#     ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     analysis = f"_(Generated {ts})_\n\n" + analysis
#     return analysis, price_path, rsi_path, macd_path, meta_txt


# ---------- UI ----------

# with gr.Blocks(title="Agentic Financial Analyst") as demo:
#     gr.Markdown(
#         "# Agentic Financial Analyst\n"
#         "Enter a ticker and a question. This UI uses your LangGraph agent and tools to fetch data, compute indicators, and produce a structured analysis."
#     )

#     with gr.Row():
#         ticker = gr.Textbox(label="Ticker", value="TSLA", scale=1, placeholder="e.g., AAPL, TSLA, MSFT")
#         model = gr.Dropdown(
#             label="OpenAI Model",
#             choices=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
#             value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
#             scale=1,
#         )

#     question = gr.Textbox(label="Question", value="Should I buy this stock?", lines=2)
#     show_charts = gr.Checkbox(label="Show charts (Close / RSI / MACD)", value=True)
#     run_btn = gr.Button("Run Analysis", variant="primary")

#     analysis_md = gr.Markdown(label="Analysis")
#     with gr.Row():
#         price_img = gr.Image(label="Close Price (sample)", type="filepath")
#         rsi_img   = gr.Image(label="RSI", type="filepath")
#         macd_img  = gr.Image(label="MACD vs Signal", type="filepath")
#     meta_out = gr.Textbox(label="Fetch Meta (debug)", lines=8)

#     run_btn.click(
#         analyze,
#         inputs=[ticker, question, model, show_charts],
#         outputs=[analysis_md, price_img, rsi_img, macd_img, meta_out],
#     )

# def on_submit(user_input):
#     try:
#         answer = run_cli(user_input)
#         return answer
#     except Exception as e:
#         return f"Error: {e}"

# with gr.Blocks() as demo:
#     gr.Markdown("# Financial Analyst Agent + FAQ RAG")
#     inp = gr.Textbox(label="Ask a finance question or request a stock analysis")
#     out = gr.Markdown()
#     btn = gr.Button("Run")
#     btn.click(on_submit, inputs=inp, outputs=out)
#     inp.submit(on_submit, inputs=inp, outputs=out)

# if __name__ == "__main__":
#     # Set share=True if you want a public link
#     demo.launch()



# New code for graph/plot
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