# edu_chain.py
from typing import Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from rag_store import get_retriever

EDU_SYSTEM = (
    "You are a concise finance tutor. Answer only from the given context."
    " If the answer is not in context, say you don't know. Be clear, neutral, and include a brief example when helpful."
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", EDU_SYSTEM),
    ("user", "Question: {q}\n\nContext:\n{context}\n\nAnswer succinctly:")
])

def _fmt_context(docs) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "faq")
        parts.append(f"[{i}] ({src}) {d.page_content}")
    return "\n\n".join(parts)

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def answer_educational(q: str) -> Dict:
    retriever = get_retriever(k=4)
    # docs = retriever.get_relevant_documents(q)
    docs = retriever.invoke(q)
    ctx = _fmt_context(docs)
    msg = PROMPT.format_messages(q=q, context=ctx)
    ai = _llm.invoke(msg)
    sources = []
    for i, d in enumerate(docs, 1):
        src = (d.metadata or {}).get("source", "faq_data/faqs.md")
        sources.append({"id": i, "source": src})
    return {"answer": ai.content, "sources": sources}
