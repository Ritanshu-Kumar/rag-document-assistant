import os
import re
import sqlite3
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import CrossEncoder

from db import Timer, log_query


load_dotenv()


BASE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
)

DB_DIR = os.path.join(
    BASE_DIR,
    "chroma_db",
)

LEXICAL_DB_PATH = os.path.join(
    BASE_DIR,
    "lexical_index.db",
)

COLLECTION_NAME = "documents"

DENSE_K = 50
LEXICAL_K = 50
RRF_K = 30
RERANK_K = 30
FINAL_K = 6

MAX_CHUNKS_PER_SOURCE = 2

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "openai/gpt-oss-20b"


SYSTEM_PROMPT = """You are a helpful assistant answering questions from a user's
personal college notes.

Use ONLY the information contained in the supplied context.

Rules:
- Do not use outside knowledge.
- Do not guess or invent information.
- If the context does not contain enough information to answer the question,
  say exactly: "I don't have enough information to answer that."
- Answer the question directly.
- Explain concepts clearly.
- Prefer information directly supported by the retrieved notes.
- When multiple retrieved sections contribute to the answer, synthesize them.
- Keep the response reasonably concise unless the question asks for detail.
"""


STOP_WORDS = {
    "a",
    "about",
    "after",
    "again",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "more",
    "most",
    "of",
    "on",
    "or",
    "our",
    "should",
    "so",
    "some",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "under",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}


_chroma_collection = None
_groq_client = None
_reranker = None


def tokenize(text: str) -> list[str]:
    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower(),
    )

    return [
        word
        for word in words
        if word not in STOP_WORDS
        and len(word) > 1
    ]


def get_collection():
    global _chroma_collection

    if _chroma_collection is None:
        embedding_fn = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )

        client = chromadb.PersistentClient(
            path=DB_DIR,
        )

        _chroma_collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )

    return _chroma_collection


def get_groq_client():
    global _groq_client

    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "Set GROQ_API_KEY in your .env file."
            )

        _groq_client = Groq(
            api_key=api_key
        )

    return _groq_client


def get_reranker():
    global _reranker

    if _reranker is None:
        print(
            f"Loading reranker: {RERANKER_MODEL}"
        )

        _reranker = CrossEncoder(
            RERANKER_MODEL
        )

    return _reranker


def lexical_index_exists() -> bool:
    if not os.path.exists(
        LEXICAL_DB_PATH
    ):
        return False

    try:
        conn = sqlite3.connect(
            LEXICAL_DB_PATH
        )

        row = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'chunks'
            """
        ).fetchone()

        conn.close()

        return row is not None

    except sqlite3.Error:
        return False


def build_lexical_index():
    print(
        "\nBuilding lexical search index..."
    )

    collection = get_collection()

    data = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    documents = data.get(
        "documents",
        [],
    )

    metadatas = data.get(
        "metadatas",
        [],
    )

    ids = data.get(
        "ids",
        [],
    )

    if not documents:
        raise RuntimeError(
            "Chroma collection is empty."
        )

    if os.path.exists(
        LEXICAL_DB_PATH
    ):
        os.remove(
            LEXICAL_DB_PATH
        )

    conn = sqlite3.connect(
        LEXICAL_DB_PATH
    )

    conn.execute(
        """
        CREATE VIRTUAL TABLE chunks
        USING fts5(
            chunk_id UNINDEXED,
            text,
            source,
            semester,
            course,
            chunk_index UNINDEXED
        )
        """
    )

    batch = []

    for chunk_id, document, metadata in zip(
        ids,
        documents,
        metadatas,
    ):
        metadata = metadata or {}

        batch.append(
            (
                chunk_id,
                document or "",
                metadata.get(
                    "source",
                    "",
                ),
                metadata.get(
                    "semester",
                    "",
                ),
                metadata.get(
                    "course",
                    "",
                ),
                str(
                    metadata.get(
                        "chunk_index",
                        0,
                    )
                ),
            )
        )

        if len(batch) >= 1000:
            conn.executemany(
                """
                INSERT INTO chunks
                (
                    chunk_id,
                    text,
                    source,
                    semester,
                    course,
                    chunk_index
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                batch,
            )

            conn.commit()
            batch = []

    if batch:
        conn.executemany(
            """
            INSERT INTO chunks
            (
                chunk_id,
                text,
                source,
                semester,
                course,
                chunk_index
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            batch,
        )

    conn.commit()
    conn.close()

    print(
        f"Lexical index created with "
        f"{len(documents)} chunks."
    )


def get_lexical_connection():
    if not lexical_index_exists():
        build_lexical_index()

    return sqlite3.connect(
        LEXICAL_DB_PATH
    )


def dense_retrieve(
    question: str,
    k: int = DENSE_K,
) -> list[dict[str, Any]]:

    collection = get_collection()

    results = collection.query(
        query_texts=[question],
        n_results=k,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    candidates = []

    for rank, (
        document,
        metadata,
        distance,
    ) in enumerate(
        zip(
            documents,
            metadatas,
            distances,
        ),
        start=1,
    ):
        metadata = metadata or {}

        candidates.append(
            {
                "text": document,
                "source": metadata.get(
                    "source",
                    "",
                ),
                "relative_path": metadata.get(
                    "relative_path",
                    metadata.get(
                        "source",
                        "",
                    ),
                ),
                "semester": metadata.get(
                    "semester",
                    "Unknown",
                ),
                "course": metadata.get(
                    "course",
                    "Unknown",
                ),
                "file_type": metadata.get(
                    "file_type",
                    "",
                ),
                "chunk_index": metadata.get(
                    "chunk_index",
                    0,
                ),
                "distance": float(distance),
                "dense_rank": rank,
            }
        )

    return candidates


def build_fts_query(
    question: str,
) -> str:

    terms = tokenize(question)

    if not terms:
        return ""

    return " OR ".join(
        f'"{term}"'
        for term in terms
    )


def lexical_retrieve(
    question: str,
    k: int = LEXICAL_K,
) -> list[dict[str, Any]]:

    query = build_fts_query(
        question
    )

    if not query:
        return []

    conn = get_lexical_connection()

    rows = conn.execute(
        """
        SELECT
            chunk_id,
            text,
            source,
            semester,
            course,
            chunk_index,
            bm25(chunks) AS score
        FROM chunks
        WHERE chunks MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (
            query,
            k,
        ),
    ).fetchall()

    conn.close()

    candidates = []

    for rank, row in enumerate(
        rows,
        start=1,
    ):
        (
            chunk_id,
            text,
            source,
            semester,
            course,
            chunk_index,
            score,
        ) = row

        candidates.append(
            {
                "id": chunk_id,
                "text": text,
                "source": source,
                "relative_path": source,
                "semester": semester,
                "course": course,
                "chunk_index": chunk_index,
                "bm25_score": float(score),
                "lexical_rank": rank,
            }
        )

    return candidates


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    lexical_results: list[dict[str, Any]],
    k: int = RRF_K,
) -> list[dict[str, Any]]:

    rrf_constant = 60
    merged: dict[str, dict[str, Any]] = {}

    def make_key(
        item: dict[str, Any]
    ) -> str:
        return (
            f"{item.get('source', '')}"
            f"::{item.get('chunk_index', '')}"
            f"::{hash(item.get('text', ''))}"
        )

    for item in dense_results:
        key = make_key(item)

        if key not in merged:
            merged[key] = dict(item)

        merged[key]["rrf_score"] = (
            merged[key].get(
                "rrf_score",
                0.0,
            )
            + 1.0
            / (
                rrf_constant
                + item["dense_rank"]
            )
        )

    for item in lexical_results:
        key = make_key(item)

        if key not in merged:
            merged[key] = dict(item)

        merged[key]["rrf_score"] = (
            merged[key].get(
                "rrf_score",
                0.0,
            )
            + 1.0
            / (
                rrf_constant
                + item["lexical_rank"]
            )
        )

        if not merged[key].get(
            "source"
        ):
            merged[key]["source"] = (
                item["source"]
            )

        if not merged[key].get(
            "course"
        ):
            merged[key]["course"] = (
                item["course"]
            )

    candidates = list(
        merged.values()
    )

    candidates.sort(
        key=lambda item: item.get(
            "rrf_score",
            0.0,
        ),
        reverse=True,
    )

    return candidates[:k]


def rerank(
    question: str,
    candidates: list[dict[str, Any]],
    k: int = RERANK_K,
) -> list[dict[str, Any]]:

    if not candidates:
        return []

    candidates = candidates[:k]

    reranker = get_reranker()

    pairs = [
        (
            question,
            candidate["text"],
        )
        for candidate in candidates
    ]

    scores = reranker.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False,
    )

    reranked = []

    for candidate, score in zip(
        candidates,
        scores,
    ):
        item = dict(candidate)

        item["reranker_score"] = float(
            score
        )

        reranked.append(item)

    reranked.sort(
        key=lambda item: item[
            "reranker_score"
        ],
        reverse=True,
    )

    return reranked


def normalize_text(
    text: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        text.lower(),
    ).strip()


def select_final_context(
    candidates: list[dict[str, Any]],
    k: int = FINAL_K,
) -> list[dict[str, Any]]:

    selected = []
    seen_text = set()
    source_counts = {}

    for candidate in candidates:
        text_key = normalize_text(
            candidate["text"]
        )

        if text_key in seen_text:
            continue

        source = candidate.get(
            "source",
            "",
        )

        source_count = source_counts.get(
            source,
            0,
        )

        if source_count >= MAX_CHUNKS_PER_SOURCE:
            continue

        seen_text.add(text_key)
        source_counts[source] = (
            source_count + 1
        )

        selected.append(candidate)

        if len(selected) >= k:
            break

    return selected


def build_prompt(
    question: str,
    candidates: list[dict[str, Any]],
) -> str:

    context_parts = []

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        context_parts.append(
            f"""[Context {index}]
Course: {candidate.get("course", "Unknown")}
Semester: {candidate.get("semester", "Unknown")}
Source: {candidate.get("source", "Unknown")}

{candidate["text"]}"""
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    return f"""The following excerpts are taken from the user's college notes.

Context:
{context}

Question:
{question}

Answer the question using only the context above."""


def answer_question(
    question: str,
    log: bool = True,
) -> dict:

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    with Timer() as timer:
        dense_results = dense_retrieve(
            question,
            DENSE_K,
        )

        lexical_results = lexical_retrieve(
            question,
            LEXICAL_K,
        )

        fused_candidates = (
            reciprocal_rank_fusion(
                dense_results,
                lexical_results,
                RRF_K,
            )
        )

        reranked_candidates = rerank(
            question,
            fused_candidates,
            RERANK_K,
        )

        final_candidates = (
            select_final_context(
                reranked_candidates,
                FINAL_K,
            )
        )

        if not final_candidates:
            answer = (
                "I don't have enough information "
                "to answer that."
            )
        else:
            prompt = build_prompt(
                question,
                final_candidates,
            )

            client = get_groq_client()

            response = (
                client.chat.completions.create(
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
            )

            answer = (
                response.choices[0]
                .message.content
                or ""
            )

    sources = list(
        dict.fromkeys(
            candidate["source"]
            for candidate in final_candidates
        )
    )

    result = {
        "question": question,
        "answer": answer,
        "sources": sources,
        "num_chunks_retrieved": len(
            final_candidates
        ),
        "latency_ms": round(
            timer.elapsed_ms,
            1,
        ),
    }

    if log:
        log_query(
            question,
            answer,
            sources,
            len(final_candidates),
            result["latency_ms"],
        )

    return result


if __name__ == "__main__":
    question = input(
        "Ask a question about your documents: "
    )

    result = answer_question(
        question
    )

    print(
        f"\nAnswer: "
        f"{result['answer']}"
    )

    print(
        "\nSources: "
        f"{', '.join(result['sources'])}"
    )

    print(
        f"Chunks: "
        f"{result['num_chunks_retrieved']}"
    )

    print(
        f"Latency: "
        f"{result['latency_ms']} ms"
    )