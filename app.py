from typing import TypedDict, Annotated, List
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langsmith import traceable

from tools import get_stock_prices, get_financial_metrics

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "financial-analyst-agent")

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    stock: str


FUNDAMENTAL_ANALYST_PROMPT = """You are an Agentic Financial Analyst.
You will analyze the stock: {company}.

You have access to two tools:
1) get_stock_prices(ticker): returns recent OHLCV price history and indicators (RSI, Stochastic %K/%D, MACD, VWAP).
2) get_financial_metrics(ticker): returns basic ratios like P/E (trailing/forward), P/B, Debt/Equity, Profit Margins, ROE, Market Cap.

Process:
- First, think about which tool(s) to call. Use them to gather data.
- Then, produce a concise, structured analysis using the following sections:

# Price & Trend
Summarize recent price movement, notable trends (up/down/sideways), and any volatility observations.

# Technical Indicators
Briefly interpret RSI, Stochastic, MACD, and VWAP (e.g., momentum overbought/oversold, crossovers, signal vs. histogram, price vs. VWAP).

# Financial Health
Summarize valuation and quality using available metrics: P/E (trailing/forward), P/B, Debt/Equity, Profit Margins, ROE, Market Cap, with a short take on what they imply.

# Conclusion
Provide an objective conclusion (bullish/bearish/neutral) with 2–3 clear bullet points of evidence.

# Answer to User
Answer the user's specific question directly and succinctly.

Be measured and avoid definitive financial advice. If data is missing, state so briefly.
"""

@traceable(name="fundamental_analyst")   # NEW: trace this node as a chain
def fundamental_analyst(state: State):
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2)
    tools = [get_stock_prices, get_financial_metrics]
    llm_with_tools = llm.bind_tools(tools)

    system_msg = SystemMessage(
        content=FUNDAMENTAL_ANALYST_PROMPT.format(company=state["stock"])
    )
    msgs = [system_msg] + state["messages"]

    ai_msg = llm_with_tools.invoke(msgs)
    return {"messages": [ai_msg]}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("fundamental_analyst", fundamental_analyst)
    graph.add_node("tools", ToolNode([get_stock_prices, get_financial_metrics]))

    graph.add_edge(START, "fundamental_analyst")
    graph.add_conditional_edges("fundamental_analyst", tools_condition)
    graph.add_edge("tools", "fundamental_analyst")
    return graph.compile()


def main():
    load_dotenv()
    app = build_graph()

    stock = os.getenv("STOCK_SYMBOL", "TSLA")
    user_question = os.getenv("USER_QUESTION", "Should I buy this stock?")

    inputs = {
        "messages": [HumanMessage(content=user_question)],
        "stock": stock,
    }

    print(f"\\n=== Running Agentic Financial Analyst for {stock} ===\\n")
    final = None
    for event in app.stream(inputs, stream_mode="values"):
        if "messages" in event and event["messages"]:
            final = event["messages"][-1]

    if final:
        print(final.content)
    else:
        print("No output produced.")

if __name__ == "__main__":
    main()
