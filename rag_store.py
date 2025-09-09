# rag_store.py
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# Toggle to match whatever you used in rag_ingest
USE_CHROMA = False

if USE_CHROMA:
    from langchain_community.vectorstores import Chroma
else:
    from langchain_community.vectorstores import FAISS


load_dotenv()
STORE_DIR = Path("storage/faiss_edu")


def get_retriever(k: int = 4):
    if USE_CHROMA:
        vs = Chroma(
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory=str(STORE_DIR),
        )
    else:
        vs = FAISS.load_local(
            str(STORE_DIR),
            embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
            allow_dangerous_deserialization=True,
        )
    return vs.as_retriever(search_kwargs={"k": k})
