# Agentic Financial Analyst

An AI-powered agent that automates stock analysis by combining technical indicators, financial metrics, and structured insights using **LangGraph** and **OpenAI**.

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

---

## 📦 Installation

Using [uv](https://github.com/astral-sh/uv) (recommended for speed):

```bash
uv venv
uv pip install -U langgraph langchain langchain_openai pandas ta python-dotenv yfinance
```

Or with pip:

```bash
pip install -U langgraph langchain langchain_openai pandas ta python-dotenv yfinance
```

---

## ▶️ Usage

1. Set your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

2. Run the agent:
   ```bash
   uv run python app.py
   ```
   or
   ```bash
   python app.py
   ```

---

## 📖 Example Output
- Fetches stock data (e.g., AAPL, TSLA).  
- Computes RSI, MACD, VWAP.  
- Evaluates financial ratios (P/E, D/E, Profit Margin).  
- Produces a structured AI-generated analysis with conclusions and next steps.  

---

## 🔮 Future Enhancements
- Expand into a **Portfolio Manager Agent** with multiple specialized sub-agents.  
- Add sentiment analysis from financial news.  
- Support risk assessment and portfolio optimization.  

---

## 📄 License
MIT License  
