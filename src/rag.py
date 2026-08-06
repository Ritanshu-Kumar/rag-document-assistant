"""
Core RAG pipeline: retrieve relevant chunks, then generate a grounded answer.

The SYSTEM_PROMPT below is the "prompt engineering" layer — iterate on this
to reduce hallucination and improve answer quality. Keep notes on what you
change and why; that's your interview story for "tell me about your prompt
engineering experience."
"""
import os
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv

from db import log_query, Timer

load_dotenv()

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION_NAME = "documents"
TOP_K = 4  # number of chunks to retrieve per query
GROQ_MODEL = "llama-3.1-8b-instant"  # fast + free-tier friendly; swap to llama-3.1-70b-versatile for stronger answers

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context.

Rules:
- If the answer is not contained in the context, say "I don't have enough information to answer that" — do not guess or use outside knowledge.
- Quote or closely paraphrase the context rather than inventing details.
- Keep answers concise (2-4 sentences) unless the question asks for more detail.
- If relevant, mention which source the information came from.
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
        _collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
    return _collection


def _get_groq_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Set GROQ_API_KEY in your .env file (see .env.example).")
        _client = Groq(api_key=api_key)
    return _client


def retrieve(question: str, top_k: int = TOP_K):
    collection = _get_collection()
    results = collection.query(query_texts=[question], n_results=top_k)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return chunks, sources


def build_prompt(question: str, chunks: list) -> str:
    context = "\n\n---\n\n".join(chunks)
    return f"""Context:
{context}

Question: {question}

Answer based only on the context above:"""


def answer_question(question: str, log: bool = True) -> dict:
    """Full pipeline: retrieve -> prompt -> generate -> log. Returns a dict
    with the answer plus everything needed to inspect/debug/evaluate it."""
    with Timer() as t:
        chunks, sources = retrieve(question)
        prompt = build_prompt(question, chunks)

        client = _get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=500,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        answer = response.choices[0].message.content

    result = {
        "question": question,
        "answer": answer,
        "sources": list(set(sources)),
        "num_chunks_retrieved": len(chunks),
        "latency_ms": round(t.elapsed_ms, 1),
    }

    if log:
        log_query(question, answer, result["sources"], result["num_chunks_retrieved"], result["latency_ms"])

    return result


if __name__ == "__main__":
    q = input("Ask a question about your documents: ")
    result = answer_question(q)
    print(f"\nAnswer: {result['answer']}")
    print(f"Sources: {', '.join(result['sources'])}")
    print(f"Latency: {result['latency_ms']} ms")
