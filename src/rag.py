"""RAG pipeline for retrieval and grounded answer generation."""

import os
import re

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq

from db import Timer, log_query

load_dotenv()

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "documents"

RETRIEVE_K = 8
FINAL_K = 4
GROQ_MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the provided context.

Rules:
- Use only information supported by the context.
- If the context does not contain the answer, say "I don't have enough information to answer that."
- Do not use outside knowledge or guess.
- Answer the question directly.
- Keep answers concise unless the question asks for an explanation or list.
"""


_client = None
_collection = None


def _get_collection():
    global _collection

    if _collection is None:
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        chroma_client = chromadb.PersistentClient(path=DB_DIR)

        _collection = chroma_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )

    return _collection


def _get_groq_client():
    global _client

    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError("Set GROQ_API_KEY in your .env file.")

        _client = Groq(api_key=api_key)

    return _client


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))


def _score_chunk(question: str, chunk: str) -> float:
    question_words = _tokenize(question)
    chunk_words = _tokenize(chunk)

    if not question_words or not chunk_words:
        return 0.0

    overlap = question_words & chunk_words

    return len(overlap) / len(question_words)


def retrieve(question: str, top_k: int = FINAL_K):
    collection = _get_collection()

    results = collection.query(
        query_texts=[question],
        n_results=RETRIEVE_K,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    candidates = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        lexical_score = _score_chunk(question, document)

        candidates.append(
            {
                "text": document,
                "source": metadata["source"],
                "chunk_index": metadata.get("chunk_index", 0),
                "distance": distance,
                "lexical_score": lexical_score,
            }
        )

    candidates.sort(
        key=lambda item: (
            item["lexical_score"],
            -item["distance"],
        ),
        reverse=True,
    )

    selected = candidates[:top_k]

    chunks = [item["text"] for item in selected]
    sources = [item["source"] for item in selected]

    return chunks, sources


def build_prompt(question: str, chunks: list[str]) -> str:
    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        context_parts.append(f"[Context {index}]\n{chunk}")

    context = "\n\n---\n\n".join(context_parts)

    return f"""Context:
{context}

Question:
{question}

Answer based only on the context above:"""


def answer_question(question: str, log: bool = True) -> dict:
    with Timer() as timer:
        chunks, sources = retrieve(question)
        prompt = build_prompt(question, chunks)

        client = _get_groq_client()

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=1000,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        answer = response.choices[0].message.content or ""

    result = {
        "question": question,
        "answer": answer,
        "sources": list(dict.fromkeys(sources)),
        "num_chunks_retrieved": len(chunks),
        "latency_ms": round(timer.elapsed_ms, 1),
    }

    if log:
        log_query(
            question,
            answer,
            result["sources"],
            result["num_chunks_retrieved"],
            result["latency_ms"],
        )

    return result


if __name__ == "__main__":
    question = input("Ask a question about your documents: ")

    result = answer_question(question)

    print(f"\nAnswer: {result['answer']}")
    print(f"Sources: {', '.join(result['sources'])}")
    print(f"Chunks: {result['num_chunks_retrieved']}")
    print(f"Latency: {result['latency_ms']} ms")