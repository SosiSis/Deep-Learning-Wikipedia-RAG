from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List
import os

from settings import EMBED_MODEL, INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def save_raw_page(title: str, content: str, raw_dir: str):
    os.makedirs(raw_dir, exist_ok=True)
    safe = title.replace("/", "_")
    path = os.path.join(raw_dir, f"{safe}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def build_faiss_from_docs(docs: List[Document]):
    embeddings = get_embeddings()
    return FAISS.from_documents(docs, embeddings)

def persist_faiss_index(index: FAISS, index_dir: str = INDEX_DIR):
    os.makedirs(index_dir, exist_ok=True)
    index.save_local(index_dir)

def load_faiss_index(index_dir: str = INDEX_DIR):
    embeddings = get_embeddings()
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
