import os
import json
import tempfile
from datetime import datetime
import traceback

import gradio as gr
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Reuse your existing graph + tool
from app import build_graph            # your LangGraph app builder
from tools import get_stock_prices     # tool returning prices + indicators

load_dotenv()
GRAPH = build_graph()  # compile once at startup


# ---------- plotting helpers (return FILE PATHS) ----------

def _plot_line(dates, values, title, ylab):
    if not dates or not values:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(dates, values, linewidth=2)   # no explicit colors/styles per guidance
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylab)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return tmp.name

def _plot_price(sample_prices):
    if not sample_prices:
        return None
    dates = [p["date"] for p in sample_prices]
    closes = [p["close"] for p in sample_prices]
    return _plot_line(dates, closes, "Recent Close Prices (sample)", "Close")

def _plot_rsi(indicators):
    rsi_pts = (indicators or {}).get("rsi") or []
    if not rsi_pts:
        return None
    dates = [p["date"] for p in rsi_pts]
    vals = [p["value"] for p in rsi_pts]
    return _plot_line(dates, vals, "RSI (last points)", "RSI")

def _plot_macd(indicators):
    macd_pts = (indicators or {}).get("macd") or []
    sig_pts  = (indicators or {}).get("macd_signal") or []
    if not macd_pts or not sig_pts:
        return None
    # assume same dates length/order
    dates = [p["date"] for p in macd_pts]
    macd_vals = [p["value"] for p in macd_pts]
    sig_vals  = [p["value"] for p in sig_pts]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(dates, macd_vals, linewidth=2, label="MACD")
    ax.plot(dates, sig_vals,  linewidth=2, label="Signal")
    ax.set_title("MACD vs Signal (last points)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return tmp.name


# ---------- core handler ----------

def analyze(ticker: str, question: str, model_name: str, show_charts: bool):
    """
    1) Set model via OPENAI_MODEL (used in your app).
    2) Stream the LangGraph and return final LLM message.
    3) Optionally render charts (Close / RSI / MACD) from the tool output.
    """
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return "Please enter a ticker (e.g., TSLA).", None, None, None, ""

    question = (question or "").strip() or "Should I buy this stock?"
    os.environ["OPENAI_MODEL"] = model_name  # app reads this env var

    # Run your graph
    try:
        inputs = {"messages": [HumanMessage(content=question)], "stock": ticker}
        final_msg = None
        for ev in GRAPH.stream(inputs, stream_mode="values"):
            if "messages" in ev and ev["messages"]:
                final_msg = ev["messages"][-1]
        analysis = final_msg.content if final_msg else "No output produced."
    except Exception:
        analysis = f"❌ Error while running analysis:\n```\n{traceback.format_exc()}\n```"

    # Fetch tool output for charts + meta (don’t fail UI if charting fails)
    price_path = rsi_path = macd_path = None
    meta_txt = ""
    try:
        tool_out = get_stock_prices.invoke({"ticker": ticker})
        meta_txt = json.dumps(
            {
                "fetch_used": tool_out.get("fetch_used") or {
                    "period": tool_out.get("period"),
                    "interval": tool_out.get("interval"),
                },
                "fetch_tried": tool_out.get("fetch_tried"),
                "latest_close": tool_out.get("latest_close"),
            },
            indent=2,
        )
        if show_charts:
            price_path = _plot_price(tool_out.get("sample_prices", []))
            indicators = tool_out.get("indicators", {})
            rsi_path   = _plot_rsi(indicators)
            macd_path  = _plot_macd(indicators)
    except Exception as e:
        meta_txt = f"Note: unable to fetch chart/tool data: {e}"

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    analysis = f"_(Generated {ts})_\n\n" + analysis
    return analysis, price_path, rsi_path, macd_path, meta_txt


# ---------- UI ----------

with gr.Blocks(title="Agentic Financial Analyst") as demo:
    gr.Markdown(
        "# Agentic Financial Analyst\n"
        "Enter a ticker and a question. This UI uses your LangGraph agent and tools to fetch data, compute indicators, and produce a structured analysis."
    )

    with gr.Row():
        ticker = gr.Textbox(label="Ticker", value="TSLA", scale=1, placeholder="e.g., AAPL, TSLA, MSFT")
        model = gr.Dropdown(
            label="OpenAI Model",
            choices=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
            value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            scale=1,
        )

    question = gr.Textbox(label="Question", value="Should I buy this stock?", lines=2)
    show_charts = gr.Checkbox(label="Show charts (Close / RSI / MACD)", value=True)
    run_btn = gr.Button("Run Analysis", variant="primary")

    analysis_md = gr.Markdown(label="Analysis")
    with gr.Row():
        price_img = gr.Image(label="Close Price (sample)", type="filepath")
        rsi_img   = gr.Image(label="RSI", type="filepath")
        macd_img  = gr.Image(label="MACD vs Signal", type="filepath")
    meta_out = gr.Textbox(label="Fetch Meta (debug)", lines=8)

    run_btn.click(
        analyze,
        inputs=[ticker, question, model, show_charts],
        outputs=[analysis_md, price_img, rsi_img, macd_img, meta_out],
    )

if __name__ == "__main__":
    # Set share=True if you want a public link
    demo.launch()
