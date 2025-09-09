# rag_ingest.py
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Choose one vector store backend. Default FAISS.
USE_CHROMA = False

if USE_CHROMA:
    from langchain_community.vectorstores import Chroma
else:
    from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings

load_dotenv()

DATA_DIR = Path("faq_data")
STORE_DIR = Path("storage/faiss_edu")  # if CHROMA, this is the persist directory
STORE_DIR.mkdir(parents=True, exist_ok=True)

def load_docs():
    paths = list(DATA_DIR.glob("*.md")) + list(DATA_DIR.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No FAQ files found in {DATA_DIR.resolve()}")
    docs = []
    for p in paths:
        loader = TextLoader(str(p), encoding="utf-8")
        docs.extend(loader.load())
    return docs

def main():
    docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if USE_CHROMA:
        print("Building Chroma store…")
        vs = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=str(STORE_DIR))
        vs.persist()
    else:
        print("Building FAISS store…")
        vs = FAISS.from_documents(chunks, embedding=embeddings)
        vs.save_local(str(STORE_DIR))
    print("Done. Store at:", STORE_DIR.resolve())

if __name__ == "__main__":
    main()
