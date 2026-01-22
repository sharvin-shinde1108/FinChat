# app.py
import os
import argparse
from typing import Dict, Any
 
from dotenv import load_dotenv
load_dotenv()
 
# LangChain / LangGraph
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langchain_core.messages import SystemMessage, HumanMessage
 
# Our router + RAG chain
from router import route
from edu_chain import answer_educational
 
# Your existing tools; this imports your indicator function
from tools import get_stock_prices
 
 
# New code for graph plot issue
# --- app.py (helpers) ---
import json
from typing import Any, Dict, List
 
def _extract_ticker(text: str) -> str:
    import re
    m = re.findall(r"\b[A-Z]{1,5}\b", text)
    return m[-1] if m else "AAPL"
 
def _coerce_prices_payload(raw: Any) -> Dict[str, List[Dict[str, Any]]]:
    """
    Accepts: dict OR JSON string from the tool.
    Normalizes keys to: prices, vwap, rsi, macd, macd_signal, stoch_k, stoch_d
    Each value is a list of {date, value} (prices may be {date, close}).
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return {}
    if not isinstance(raw, dict):
        return {}
 
    def _norm_list(lst, value_keys=("value", "close", "Close")):
        out = []
        for item in lst or []:
            d = item.get("date") or item.get("Date") or item.get("timestamp")
            v = None
            for k in value_keys:
                if k in item:
                    v = item[k]; break
            # sometimes yfinance dicts are like {"Close": x}
            if v is None:
                # pick first numeric-looking field (fallback)
                for k, val in item.items():
                    if k.lower() != "date" and isinstance(val, (int, float)):
                        v = val; break
            if d is not None and v is not None:
                out.append({"date": d, "value": float(v)})
        return out
 
    norm = {}
    norm["prices"]       = _norm_list(raw.get("prices") or raw.get("price") or raw.get("closes"), value_keys=("close","Close","value"))
    norm["vwap"]         = _norm_list(raw.get("vwap"))
    norm["rsi"]          = _norm_list(raw.get("rsi"))
    norm["macd"]         = _norm_list(raw.get("macd"))
    norm["macd_signal"]  = _norm_list(raw.get("macd_signal") or raw.get("signal"))
    norm["stoch_k"]      = _norm_list(raw.get("stoch_k") or raw.get("%K"))
    norm["stoch_d"]      = _norm_list(raw.get("stoch_d") or raw.get("%D"))
    return norm
 
 
############################
# Shared LLMs
############################
ANALYST_SYSTEM = (
    "You are a disciplined equity analyst. Use the provided tool results to craft a balanced, sourced analysis. "
    "You MUST use valid Markdown headers (lines starting with '## ') for each section. "
    "Required sections:\n"
    "## Quick Overview\n"
    "## Technical Analysis\n"
    "## Key Risks\n"
    "## Bottom Line\n\n"
    "Add other sections like '## Fundamentals' only if relevant."
)
 
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
 
############################
# LangGraph State
############################
from typing import TypedDict, Optional
 
class GraphState(TypedDict):
    question: str
    mode: str              # 'educational' | 'analysis'
    tool_results: Dict[str, Any]
    final: Optional[str]
 
############################
# Nodes
############################
def route_node(state: GraphState) -> Command[GraphState]:
    mode = route(state["question"])
    return Command(update={"mode": mode})
 
def edu_node(state: GraphState) -> Command[GraphState]:
    q = state["question"]
    out = answer_educational(q)
    ans = out["answer"]
    cites = "\n".join(f"- [{s['id']}] {s['source']}" for s in out["sources"])
    final = f"{ans}\n\n**Sources**:\n{cites}"
    return Command(update={"final": final})
 
def analysis_tools_node(state: GraphState) -> Command[GraphState]:
    q = state["question"]
    ticker = _extract_ticker(q)
 
    # Call the tool the modern way
    try:
        if hasattr(get_stock_prices, "invoke"):
            # If a schema is defined, pass a dict; otherwise pass the arg directly
            if getattr(get_stock_prices, "args_schema", None):
                raw = get_stock_prices.invoke({"ticker": ticker})
            else:
                raw = get_stock_prices.invoke(ticker)
        else:
            raw = get_stock_prices(ticker)
    except Exception as e:
        raw = {"error": str(e)}
 
    payload = _coerce_prices_payload(raw)
    print("DEBUG prices keys:", list(payload.keys()))
    print("DEBUG price sample:", payload.get("prices", [])[:3])
    return Command(update={"tool_results": {"ticker": ticker, "prices": payload}})
 
def analysis_llm_node(state: GraphState) -> Command[GraphState]:
    tr = state["tool_results"]
    ticker = tr["ticker"]
    prices = tr["prices"]
    sys = SystemMessage(content=ANALYST_SYSTEM)
    human = HumanMessage(content=(
        f"User asked: {state['question']}\n\n"
        f"Analyze {ticker} using the provided technical indicators JSON below. "
        f"Include clear takeaways and avoid overclaiming.\n\n"
        f"DATA JSON:\n{prices}"
    ))
    ai = llm.invoke([sys, human])
    return Command(update={"final": ai.content})
 
############################
# Graph Wiring
############################
def build_graph():
    g = StateGraph(GraphState)
 
    g.add_node("route", route_node)
    g.add_node("edu", edu_node)
    g.add_node("analysis_tools", analysis_tools_node)
    g.add_node("analysis_llm", analysis_llm_node)
 
    g.set_entry_point("route")
 
    # Conditional edges
    def on_route(state: GraphState):
        if state["mode"] == "educational":
            return "edu"
        return "analysis_tools"
 
    g.add_conditional_edges("route", on_route, ["edu", "analysis_tools"])
    g.add_edge("analysis_tools", "analysis_llm")
    g.add_edge("edu", END)
    g.add_edge("analysis_llm", END)
 
    return g.compile()
 
def run_cli(question: str):
    app = build_graph()
    # Initial graph state
    init = {"question": question, "mode": "", "tool_results": {}, "final": None}
    # Direct (non-streaming) execution returns the final state dict
    result = app.invoke(init)
    return result.get("final") or "(no answer produced)"
 
def run_for_ui(question: str):
    app = build_graph()
    init = {"question": question, "mode": "", "tool_results": {}, "final": None}
    state = app.invoke(init)
    # Return everything the UI might need
    return {
        "mode": state.get("mode"),
        "answer": state.get("final"),
        "tool_results": state.get("tool_results", {}),
    }
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", "--question", dest="question", required=True,
                        help="Ask either an educational finance question or a stock analysis question.")
    args = parser.parse_args()
    print(run_cli(args.question))
 
 