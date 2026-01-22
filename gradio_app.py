import os
import json
import tempfile
import re
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
 
# Set matplotlib style for professional looking charts
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('ggplot')
 
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
    
    # Ensure sorted by date for correct plotting
    if not macd.empty: macd = macd.sort_index()
    if not sig.empty: sig = sig.sort_index()
    
    if not macd.empty:
        ax.plot(macd.index, macd.values, label="MACD"); drew = True
    if not sig.empty:
        ax.plot(sig.index, sig.values, label="Signal"); drew = True
        
    if not macd.empty and not sig.empty:
        df = pd.concat([macd.rename("macd"), sig.rename("signal")], axis=1).dropna()
        if not df.empty:
            hist = df["macd"] - df["signal"]
            # Use ax.bar with datetime index instead of pandas plot(kind='bar')
            # Pandas bar plot treats index as categorical strings, causing x-axis clutter
            ax.bar(hist.index, hist.values, color='gray', alpha=0.3, label="Histogram", width=1.0); drew = True
            
    ax.set_title(f"{ticker} — MACD"); ax.set_xlabel("Date"); ax.set_ylabel("MACD")
    
    if drew:
        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate() # rotates and aligns labels
        ax.legend()
        
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
 
# ---------- Formatting ----------
 
CUSTOM_CSS = """
.analysis-card {
    background-color: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    transition: all 0.2s ease;
}
.analysis-card:hover {
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
}
.analysis-card h2 {
    margin-top: 0 !important;
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    color: var(--primary-500);
    border-bottom: 1px solid var(--border-color-primary);
    padding-bottom: 10px;
    margin-bottom: 15px !important;
    display: flex;
    align-items: center;
}
"""
 
def format_analysis_output(text: str) -> str:
    """Wraps text sections in nice HTML cards if Markdown headers are found."""
    if not text or "## " not in text:
        return text
    
    try:
        # Split by '## ' at start of lines
        parts = re.split(r'(?m)^## ', text)
        if not parts:
            return text
        
        html_out = ""
        
        # Intro text
        if parts[0].strip():
             html_out += f'<div class="analysis-card">{parts[0].strip()}</div>\n'
             
        for p in parts[1:]:
            lines = p.split('\n', 1)
            header = lines[0].strip()
            body = lines[1].strip() if len(lines) > 1 else ""
            
            # Simple icon mapping based on keywords
            icon = "📊"
            l_head = header.lower()
            if "overview" in l_head: icon = "🧐"
            elif "technical" in l_head: icon = "📈"
            elif "risk" in l_head: icon = "⚠️"
            elif "bottom" in l_head or "summary" in l_head: icon = "💡"
            elif "fundamental" in l_head: icon = "🏢"
            
            card = f"""<div class="analysis-card">
<h2>{icon} {header}</h2>
 
{body}
</div>\n"""
            html_out += card
            
        return html_out
    except Exception:
        return text
 
# ---------- Gradio wiring ----------
 
def on_submit(user_input: str, progress=gr.Progress()):
    try:
        progress(0.1, desc="Consulting Agent...")
        
        # Use the global GRAPH built at startup instead of rebuilding it (was run_for_ui)
        init = {"question": user_input, "mode": "", "tool_results": {}, "final": None}
        state = GRAPH.invoke(init)
        result = {
            "mode": state.get("mode"),
            "answer": state.get("final"),
            "tool_results": state.get("tool_results", {}),
        }
        
        progress(0.5, desc="Processing Result...")
        mode = (result.get("mode") or "").lower()
        answer = result.get("answer") or ""
 
        # Apply formatting (organized boxes)
        answer = format_analysis_output(answer)
 
        if mode == "analysis":
            progress(0.7, desc="Rendering Charts...")
            tr = result.get("tool_results") or {}
            ticker = tr.get("ticker", "TICKER")
            prices_payload = (tr.get("prices") or {})
            s = _get_prices_df(prices_payload)
 
            # build figures (each returns a Matplotlib Figure)
            # Create figures with a specific size for consistency
            figs = [
                plot_price_vwap(s, ticker),
                plot_rsi(s, ticker),
                plot_macd(s, ticker),
                plot_stoch(s, ticker),
            ]
            
            progress(1.0, desc="Done")
            # Show the plots container
            return [answer, gr.update(visible=True), *figs]
        
        progress(1.0, desc="Done")
        # If not analysis mode, hide the charts and clear them
        empty_plot = gr.update(value=None)
        return [answer, gr.update(visible=False), empty_plot, empty_plot, empty_plot, empty_plot]
    except Exception as e:
        # returns error message and Nones for plots
        empty_plot = gr.update(value=None)
        return [f"Error: {e}", gr.update(visible=False), empty_plot, empty_plot, empty_plot, empty_plot]
 
# Define a professional theme
theme = gr.themes.Soft(
    primary_hue="sky",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)
 
with gr.Blocks(theme=theme, title="FinChat Analyst", css=CUSTOM_CSS) as demo:
    with gr.Column(variant="panel"):
        gr.Markdown(
            """
            # 📈 Financial Analyst Agent
            ### Your AI-powered assistant for financial education and stock market analysis.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Textbox(
                    label="Ask a Question",
                    placeholder="e.g., 'What is P/E ratio?' or 'Analyze AAPL stock'",
                    lines=2
                )
                btn = gr.Button("🚀 Run Analysis", variant="primary")
            
            with gr.Column(scale=2):
                with gr.Accordion("📚 Sample Queries", open=True):
                    gr.Examples(
                        examples=[
                            ["What is Current Ratio in finance?"],
                            ["Analyze NVDA stock including technical indicators."],
                            ["Explain the concept of beta in finance."],
                            ["Show me the price and RSI for TSLA."],
                        ],
                        inputs=inp,
                        label=None
                    )
 
    with gr.Row():
        out_md = gr.Markdown(label="Analysis Result", header_links=True)
 
    with gr.Group(visible=False) as plots_holder:
        gr.Markdown("### 📊 Technical Indicators")
        with gr.Row():
            plot1 = gr.Plot(label="Price & VWAP")
            plot2 = gr.Plot(label="RSI")
        with gr.Row():
            plot3 = gr.Plot(label="MACD")
            plot4 = gr.Plot(label="Stochastic")
 
    # Event handlers
    btn.click(fn=on_submit, inputs=inp, outputs=[out_md, plots_holder, plot1, plot2, plot3, plot4])
    inp.submit(fn=on_submit, inputs=inp, outputs=[out_md, plots_holder, plot1, plot2, plot3, plot4])
 
if __name__ == "__main__":
    demo.launch()
 