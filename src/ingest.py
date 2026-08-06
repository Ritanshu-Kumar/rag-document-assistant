"""
Ingest documents from data/ into a persistent Chroma vector store.

Usage:
    python src/ingest.py

Add your own .txt or .pdf files to the data/ folder before running.
"""
import os
import glob
import uuid

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "documents"

# Chunking config — tune these based on your document type
CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # overlap between consecutive chunks


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf_file(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Simple sliding-window chunker. Good enough for a v1 RAG pipeline —
    swap in a smarter splitter (e.g. recursive/semantic chunking) later
    and you have a natural 'v2 improvement' talking point for interviews."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_documents():
    paths = glob.glob(os.path.join(DATA_DIR, "*.txt")) + glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not paths:
        raise FileNotFoundError(
            f"No .txt or .pdf files found in {DATA_DIR}. Add some documents first."
        )

    docs = []
    for path in paths:
        filename = os.path.basename(path)
        text = read_pdf_file(path) if path.endswith(".pdf") else read_text_file(path)
        for chunk in chunk_text(text):
            docs.append({"id": str(uuid.uuid4()), "text": chunk, "source": filename})
    return docs


def main():
    print("Loading documents from data/ ...")
    docs = load_documents()
    print(f"Loaded {len(docs)} chunks from source files.")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=DB_DIR)
    # Fresh build each run — simplest correct behavior for a v1 project
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

    batch_size = 64
    for i in tqdm(range(0, len(docs), batch_size), desc="Embedding + indexing"):
        batch = docs[i : i + batch_size]
        collection.add(
            ids=[d["id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[{"source": d["source"]} for d in batch],
        )

    print(f"Done. Indexed {len(docs)} chunks into '{COLLECTION_NAME}' at {DB_DIR}")


if __name__ == "__main__":
    main()
