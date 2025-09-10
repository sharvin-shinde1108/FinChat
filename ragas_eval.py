from __future__ import annotations
import re
from pathlib import Path
from typing import List, Dict

import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

from rag_store import get_retriever
from edu_chain import answer_educational

from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

DATA_MD = Path("faq_data/faqs.md")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

SECTION_RE = re.compile(r"^##\s+(.*)")


# --- Parser: each FAQ question + full answer in one ground_truth ---
def parse_faqs_markdown(md_path: Path) -> List[Dict[str, str]]:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    items, cur_q, cur_buf = [], None, []

    def flush():
        nonlocal cur_q, cur_buf
        if cur_q and cur_buf:
            gt = " ".join([ln.strip() for ln in cur_buf if ln.strip()])
            if gt:
                items.append({"question": cur_q.strip(), "ground_truth": gt})
        cur_q, cur_buf = None, []

    for ln in lines:
        m = SECTION_RE.match(ln.strip())
        if m:
            flush()
            cur_q, cur_buf = m.group(1), []
        else:
            cur_buf.append(ln)

    flush()
    return items


# --- Pipeline runner ---
def run_pipeline(q: str) -> Dict:
    retriever = get_retriever(k=4)
    docs = retriever.invoke(q)
    contexts = [d.page_content.strip() for d in docs if d.page_content.strip()]

    # keep only top-1 context to avoid ragas IndexError
    if contexts:
        contexts = [contexts[0]]

    out = answer_educational(q)
    return {"answer": out["answer"], "contexts": contexts}


# --- Main evaluation ---
def main(limit: int | None = None):
    items = parse_faqs_markdown(DATA_MD)
    if limit:
        items = items[:limit]

    records = []
    for it in items:
        pred = run_pipeline(it["question"])
        if not pred["contexts"] or not it["ground_truth"].strip():
            continue
        records.append({
            "question": it["question"],
            "contexts": pred["contexts"],
            "answer": pred["answer"],
            "ground_truth": it["ground_truth"],
        })

    if not records:
        print("No valid records for evaluation.")
        return

    # Build dataset in ragas-friendly format
    eval_data = {
        "question": [r["question"] for r in records],
        "contexts": [r["contexts"] for r in records],
        "answer": [r["answer"] for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
    }
    dataset = Dataset.from_dict(eval_data)

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    emb = OpenAIEmbeddings(model="text-embedding-3-small")

    result = evaluate(dataset, metrics=metrics, llm=llm, embeddings=emb)

    # Save outputs
    df_scores = result.to_pandas()
    summary = getattr(result, "overall", getattr(result, "metrics", {}))

    df_scores.to_csv(REPORTS_DIR / "ragas_scores.csv", index=False)

    md_path = REPORTS_DIR / "ragas_report.md"
    md = [
        "# RAGAS Evaluation Report",
        f"- Samples: {len(df_scores)}",
        *(f"- {k}: {v:.3f}" for k, v in summary.items()),
        "",
        "## First rows",
        df_scores.head(10).to_markdown(index=False),
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")

    print("=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:>18}: {v:.3f}")
    print("Saved reports to:", REPORTS_DIR.resolve())


if __name__ == "__main__":
    main(limit=None)
