# Agentic Financial Analyst — Run & Use Guide

This guide explains how to set up, run, and interact with the project using the **Gradio UI** (`gradio_app.py`) and **tools** (`tools.py`). It also includes LangSmith tracing and troubleshooting.

---

## 1) Requirements

- Python **3.11+** (3.12 OK)
- **uv** (recommended): https://docs.astral.sh/uv/
- An **OpenAI API key**
- Internet access (for `yfinance`)

---

## 2) Project layout (final)

```
project-root/
├─ app.py              # LangGraph app (backend)
├─ gradio_app.py       # Gradio UI (RUN THIS)
├─ tools.py            # yfinance tools (prices + indicators; financial ratios)
├─ .env.example        # sample env vars (copy → .env and fill)
├─ README_Run_and_Use.md  # this document
├─ pyproject.toml      # Dependencies
└─ uv.lock              # Tracks dependencies for reproducible installs
```

---

## 3) Install dependencies

Using **uv**:

```powershell
uv venv
uv sync
```

Using **pip**:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -U langgraph langchain langchain_openai pandas ta python-dotenv yfinance gradio matplotlib langsmith
```

---

## 4) Environment variables

Create **`.env`** from `.env.example`:

```env
OPENAI_API_KEY=sk-...your-key...
OPENAI_MODEL=gpt-4o-mini

# LangSmith (observability)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=lsv2_...your-langsmith-key...
LANGCHAIN_PROJECT=financial-analyst-agent
```

Do **not** submit your real `.env`; submit `.env.example`.

---

## 5) Run the UI

```powershell
uv run python .\gradio_app.py
```
- Open the local URL (e.g., `http://127.0.0.1:7860`).
- Enter **Ticker** (e.g., `TSLA`) and **Question** (e.g., “Should I buy this stock?”).
- Pick **OpenAI Model** and click **Run Analysis**.

You’ll see a structured report with sections:
- Price & Trend
- Technical Indicators (RSI/Stochastic/MACD/VWAP)
- Financial Health (P/E, P/B, D/E, Margins, ROE, Market Cap)
- Conclusion
- Answer to User

The UI also shows:
- **Charts**: Close, RSI, MACD (optional)
- **Fetch Meta**: the price data interval actually used (weekly or daily fallback)

For a public demo link, change the last line:
```python
demo.launch(share=True)
```

---

## 🚀 Features
- **Stock Data Retrieval:** Real-time and historical data with [yfinance](https://pypi.org/project/yfinance/).  
- **Technical Indicators:** RSI, MACD, Stochastic Oscillator, VWAP via [ta](https://pypi.org/project/ta/).  
- **Financial Metrics:** P/E ratio, Debt-to-Equity, Profit Margins.  
- **AI-Generated Analysis:** Summarized insights with GPT-4 (or gpt-4o-mini).  
- **Agentic Flow:** Built with LangGraph for orchestrating tools and conversational logic.  

---

## 🛠️ Tech Stack
- [LangGraph](https://github.com/langchain-ai/langgraph)  
- [LangChain](https://github.com/langchain-ai/langchain)  
- [OpenAI API](https://platform.openai.com/)  
- [pandas](https://pandas.pydata.org/)  
- [ta](https://pypi.org/project/ta/)  
- [yfinance](https://pypi.org/project/yfinance/)  
- [python-dotenv](https://pypi.org/project/python-dotenv/)  

## 6) How it works

- `app.py` builds a **LangGraph** with an analyst node and a **ToolNode**.
- The analyst node prompts **OpenAI** and produces the final report.
- If tools are needed, `ToolNode` executes:
  - `get_stock_prices(ticker)` → OHLCV + indicators
  - `get_financial_metrics(ticker)` → ratios
- `gradio_app.py` streams the graph and renders results; it separately calls the prices tool to render small charts.

---

## 7) LangSmith tracing

With the `.env` set above, you’ll see runs in LangSmith:
- `gradio_analyze` (each UI click)
- `fundamental_analyst` (LLM node)
- Tool runs: `get_stock_prices`, `get_financial_metrics`

Open LangSmith → your project to inspect inputs/outputs and timings.

---

## 8) Troubleshooting

**A) OpenAI error about 'tool' messages**  
If you ever see:
```
Invalid parameter: messages with role 'tool' must be a response to a preceding message with 'tool_calls'
```
Ensure the analyst node **does not** bind tools on the model. The included `app.py` uses `llm.invoke(...)` directly (correct for LangGraph’s `ToolNode`).

**B) `NameError: Annotated is not defined`**  
Add `from typing import Annotated` in any module that uses it (e.g., `app.py`). Tools already include it.

**C) Gradio image error (`BytesIO`)**  
The UI saves charts to temp files and returns **file paths** (already fixed).

**D) yfinance MultiIndex**  
Handled by normalization; weekly first, then daily fallback if needed.

**E) Corporate proxy**  
If downloads fail, set `HTTP_PROXY`/`HTTPS_PROXY` before running.

---

## 9) What to submit

- ✅ `app.py`
- ✅ `gradio_app.py`
- ✅ `tools.py`
- ✅ `.env.example`
- ✅ This README

**Omit**: virtual envs, caches, temp images, and debug scripts.

---

## 10) Quick commands

```powershell
# Setup (uv)
uv venv
uv sync

# Run UI
uv run python .\gradio_app.py
```

---

## 11) Project summary

This Agentic Financial Analyst automates stock analysis using:
- **yfinance** for recent and historical prices
- **ta** for RSI, Stochastic, MACD, VWAP
- **OpenAI** (via LangChain) for concise, sectioned analysis
- **LangGraph** to orchestrate tool use
- **LangSmith** for tracing

It produces an objective, actionable summary and can be extended later with RAG/MCP.
