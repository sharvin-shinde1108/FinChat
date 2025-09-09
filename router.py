# router.py
import re
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

Intent = Literal["educational", "analysis"]

# naive ticker-ish detector (you can expand)
TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")

EDU_KEYWORDS = [
    "what is", "what's", "explain", "meaning of", "how does", "how do", "why does",
    "difference between", "define", "definition", "interpret", "when to use",
    "pros and cons", "advantages", "disadvantages"
]

def heuristic_intent(user_input: str) -> Intent:
    text = user_input.lower()
    if any(kw in text for kw in EDU_KEYWORDS):
        return "educational"
    # crude: if looks like a ticker or mentions analyze/price/target
    if TICKER_RE.search(user_input) or any(
        w in text for w in ["analyze", "analysis", "target", "price", "buy", "sell"]
    ):
        return "analysis"
    # default to educational for purely conceptual asks
    return "educational"

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an intent router. Output only one word: 'educational' if the user asks to explain a concept, ratio, indicator, definition, or asks general finance questions. "
     "Output 'analysis' if the user asks to analyze a specific stock/ticker, price, recommendation, catalysts, risks, charts, indicators for a given ticker."),
    ("user", "{q}")
])

def llm_intent(user_input: str) -> Intent:
    resp = _llm.invoke(_prompt.format_messages(q=user_input))
    out = resp.content.strip().lower()
    return "educational" if "educational" in out else "analysis"

def route(user_input: str) -> Intent:
    # fast guess first
    guess = heuristic_intent(user_input)
    # confidence hack: if heuristic said educational and we see a clear ticker cue, override
    if guess == "educational" and TICKER_RE.search(user_input):
        return "analysis"
    # for ambiguous inputs, confirm with LLM (optional, cheap)
    if guess == "educational":
        return "educational"
    # if heuristic says analysis but no concrete ticker words present, double-check
    return llm_intent(user_input)
