import glob
import os
import re
import uuid

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from tqdm import tqdm


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "documents"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf_file(path: str) -> str:
    reader = PdfReader(path)

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return "\n\n".join(pages)


def clean_text(text: str) -> str:
    """Clean common PDF extraction artifacts without destroying structure."""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Fix excessive whitespace while preserving paragraph breaks.
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Join words broken across lines with a hyphen.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Join normal wrapped lines while preserving paragraph breaks.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    return text.strip()


def split_into_paragraphs(text: str) -> list[str]:
    """Split text using paragraph boundaries first."""

    paragraphs = re.split(r"\n\s*\n", text)

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:


    paragraphs = split_into_paragraphs(text)

    chunks = []
    current = ""

    for paragraph in paragraphs:

        # If the paragraph fits into the current chunk, append it.
        candidate = (
            f"{current}\n\n{paragraph}"
            if current
            else paragraph
        )

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        # Save the current chunk before starting a new one.
        if current:
            chunks.append(current.strip())

        # Handle unusually large paragraphs separately.
        if len(paragraph) > chunk_size:
            start = 0

            while start < len(paragraph):
                end = start + chunk_size
                piece = paragraph[start:end].strip()

                if piece:
                    chunks.append(piece)

                start += chunk_size - overlap

            current = ""
        else:
            current = paragraph

    if current:
        chunks.append(current.strip())

    return chunks


def load_documents():
    paths = sorted(
        glob.glob(os.path.join(DATA_DIR, "*.txt"))
        + glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    )

    if not paths:
        raise FileNotFoundError(
            f"No .txt or .pdf files found in {DATA_DIR}. "
            "Add some documents first."
        )

    docs = []

    for path in paths:
        filename = os.path.basename(path)

        print(f"Reading {filename}...")

        if path.lower().endswith(".pdf"):
            text = read_pdf_file(path)
        else:
            text = read_text_file(path)

        text = clean_text(text)

        chunks = chunk_text(text)

        print(f"  Created {len(chunks)} chunks.")

        for index, chunk in enumerate(chunks):
            docs.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "source": filename,
                    "chunk_index": index,
                }
            )

    return docs


def main():
    print("Loading documents from data/ ...")

    docs = load_documents()

    print(f"\nLoaded {len(docs)} total chunks.")

    embedding_fn = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )

    client = chromadb.PersistentClient(path=DB_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
        print("Deleted existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    batch_size = 64

    for i in tqdm(
        range(0, len(docs), batch_size),
        desc="Embedding + indexing",
    ):
        batch = docs[i : i + batch_size]

        collection.add(
            ids=[d["id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[
                {
                    "source": d["source"],
                    "chunk_index": d["chunk_index"],
                }
                for d in batch
            ],
        )

    print(
        f"\nDone. Indexed {len(docs)} chunks "
        f"into '{COLLECTION_NAME}'."
    )
    print(f"Vector store: {DB_DIR}")


if __name__ == "__main__":
    main()