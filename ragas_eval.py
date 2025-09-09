# ragas_eval.py
from __future__ import annotations
import re
from pathlib import Path
from typing import List, Dict

import pandas as pd
from dotenv import load_dotenv

from rag_store import get_retriever
from edu_chain import answer_educational

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
DATA_MD = Path("faq_data/faqs.md")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

SECTION_RE = re.compile(r"^##\s+(.*)")

def parse_faqs_markdown(md_path: Path) -> List[Dict[str, str]]:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    items, cur_q, cur_buf = [], None, []
    for ln in lines:
        m = SECTION_RE.match(ln.strip())
        if m:
            if cur_q and cur_buf:
                items.append({"question": cur_q, "ground_truth": "\n".join(cur_buf).strip()})
            cur_q, cur_buf = m.group(1), []
        else:
            cur_buf.append(ln)
    if cur_q and cur_buf:
        items.append({"question": cur_q, "ground_truth": "\n".join(cur_buf).strip()})
    return [x for x in items if x.get("ground_truth")]

def run_pipeline(q: str) -> Dict:
    retriever = get_retriever(k=4)
    docs = retriever.invoke(q)
    contexts = [d.page_content for d in docs if d.page_content.strip()]
    out = answer_educational(q)
    return {"answer": out["answer"], "contexts": contexts}

def main(limit: int | None = None):
    items = parse_faqs_markdown(DATA_MD)
    if limit:
        items = items[:limit]

    records = []
    for it in items:
        pred = run_pipeline(it["question"])
        if not pred["contexts"]:  # skip if retrieval returned nothing
            continue
        records.append({
            "question": it["question"],
            "contexts": pred["contexts"],
            "answer": pred["answer"],
            "ground_truth": it["ground_truth"],
        })

    if not records:
        print("No valid evaluation samples (all had empty contexts).")
        return

    ds = Dataset.from_pandas(pd.DataFrame.from_records(records))

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    emb = OpenAIEmbeddings(model="text-embedding-3-small")

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=emb,
    )

    df_scores = result.to_pandas()

    # Try new API first, else fall back
    summary = getattr(result, "overall", None)
    if summary is None:
        summary = getattr(result, "metrics", {})

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
