import os
import re
import sqlite3
import time
from typing import Any, Iterator

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

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
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


SYSTEM_PROMPT = """You are a helpful assistant answering questions using the provided knowledge base.

Rules:
- Use ONLY information supported by the supplied context.
- Do not use outside knowledge.
- Do not guess or invent information.
- If the context does not contain enough information to answer the question, say exactly:
  "I don't have enough information to answer that."
- Answer the question directly and clearly.
- When using information from a context item, cite it using [1], [2], [3], etc.
- Place citations immediately after the relevant statement.
- Do not invent citation numbers.
- Prefer the smallest number of citations necessary.
- Keep the answer concise unless the question asks for detail.
"""


FOLLOW_UP_PROMPT = """The user is continuing the previous question.

Previous question:
{previous_question}

Follow-up instruction:
{instruction}

Use the same underlying subject and the supplied context to answer the follow-up.

Rules:
- Treat the follow-up as a continuation of the previous question.
- Do not search for the follow-up instruction itself.
- Use only the supplied context.
- Do not introduce outside information.
- Preserve the factual meaning of the retrieved information.
- Follow the requested style, depth, or format.
- Cite relevant context using [1], [2], etc.
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
        if word not in STOP_WORDS and len(word) > 1
    ]

FOLLOW_UP_PHRASES = {
    "explain it",
    "explain this",
    "explain that",
    "explain more",
    "explain further",
    "explain in simple terms",
    "explain simply",
    "simplify it",
    "simplify this",
    "make it simpler",
    "make this simpler",
    "tell me more",
    "go deeper",
    "elaborate",
    "elaborate on this",
    "give me an example",
    "give an example",
    "show me an example",
    "what are the key points",
    "key points",
    "summarize it",
    "summarize this",
    "summarise it",
    "summarise this",
    "like i'm five",
    "like im five",
    "explain like im five",
    "explain like i'm five",
    "explain it like im five",
    "explain it like i'm five",
}

FOLLOW_UP_WORDS = {
    "it",
    "this",
    "that",
    "these",
    "those",
    "above",
    "previous",
    "earlier",
}


def is_follow_up_query(
    query: str,
    previous_question: str | None,
) -> bool:
    if not previous_question:
        return False

    normalized = " ".join(
        query.lower().strip().split()
    )

    if not normalized:
        return False

    if normalized in FOLLOW_UP_PHRASES:
        return True

    if any(
        phrase in normalized
        for phrase in FOLLOW_UP_PHRASES
    ):
        return True

    words = set(
        tokenize(normalized)
    )

    if words & FOLLOW_UP_WORDS:
        return True

    short_follow_up_starts = (
        "explain ",
        "simplify ",
        "summarize ",
        "summarise ",
        "elaborate ",
        "compare ",
        "expand ",
    )

    if (
        len(normalized.split()) <= 8
        and normalized.startswith(
            short_follow_up_starts
        )
    ):
        return True

    return False


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
            api_key=api_key,
        )

    return _groq_client


def get_reranker():
    global _reranker

    if _reranker is None:
        _reranker = CrossEncoder(
            RERANKER_MODEL,
        )

    return _reranker


def lexical_index_exists() -> bool:
    if not os.path.exists(LEXICAL_DB_PATH):
        return False

    try:
        conn = sqlite3.connect(LEXICAL_DB_PATH)

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
    collection = get_collection()

    data = collection.get(
        include=[
            "documents",
            "metadatas",
        ]
    )

    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    ids = data.get("ids", [])

    if not documents:
        raise RuntimeError(
            "Chroma collection is empty."
        )

    if os.path.exists(LEXICAL_DB_PATH):
        os.remove(LEXICAL_DB_PATH)

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
                metadata.get("source", ""),
                metadata.get("semester", ""),
                metadata.get("course", ""),
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
                "semester": metadata.get(
                    "semester",
                    "Unknown",
                ),
                "course": metadata.get(
                    "course",
                    "Unknown",
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

    query = build_fts_query(question)

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
                "semester": semester,
                "course": course,
                "chunk_index": chunk_index,
                "bm25_score": float(score),
                "lexical_rank": rank,
            }
        )

    return candidates


def reciprocal_rank_fusion(
    dense_results,
    lexical_results,
    k: int = RRF_K,
) -> list[dict[str, Any]]:

    rrf_constant = 60
    merged = {}

    def make_key(item):
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
    candidates,
    k: int = RERANK_K,
):
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
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    return reranked


def normalize_text(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text.lower(),
    ).strip()


def select_final_context(
    candidates,
    k: int = FINAL_K,
):
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


def retrieve_context(
    retrieval_query: str,
):
    timings = {}

    start = time.perf_counter()

    dense_results = dense_retrieve(
        retrieval_query,
        DENSE_K,
    )

    timings["dense_latency_ms"] = (
        time.perf_counter() - start
    ) * 1000

    start = time.perf_counter()

    lexical_results = lexical_retrieve(
        retrieval_query,
        LEXICAL_K,
    )

    timings["lexical_latency_ms"] = (
        time.perf_counter() - start
    ) * 1000

    fused_candidates = reciprocal_rank_fusion(
        dense_results,
        lexical_results,
        RRF_K,
    )

    start = time.perf_counter()

    reranked_candidates = rerank(
        retrieval_query,
        fused_candidates,
        RERANK_K,
    )

    timings["rerank_latency_ms"] = (
        time.perf_counter() - start
    ) * 1000

    final_candidates = select_final_context(
        reranked_candidates,
        FINAL_K,
    )

    return final_candidates, timings


def build_prompt(
    retrieval_query: str,
    candidates,
    instruction: str | None = None,
    previous_question: str | None = None,
):
    context_parts = []

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        context_parts.append(
            f"""[Context {index}]
Source: {candidate.get("source", "Unknown")}

{candidate["text"]}"""
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    if instruction:
        task = FOLLOW_UP_PROMPT.format(
            previous_question=previous_question
            or retrieval_query,
            instruction=instruction,
        )
    else:
        task = f"""Question:
{retrieval_query}

Answer the question using only the supplied context.
Cite relevant context items using [1], [2], [3], etc."""

    return f"""The following excerpts are from the indexed knowledge base.

{context}

{task}"""


def build_result(
    question,
    answer,
    final_candidates,
    total_latency_ms,
    timings,
    generation_latency_ms,
):
    sources = list(
        dict.fromkeys(
            candidate["source"]
            for candidate in final_candidates
        )
    )

    contexts = []

    for index, candidate in enumerate(
        final_candidates,
        start=1,
    ):
        contexts.append(
            {
                "index": index,
                "source": candidate.get(
                    "source",
                    "Unknown",
                ),
                "score": round(
                    candidate.get(
                        "reranker_score",
                        0.0,
                    ),
                    4,
                ),
                "text": candidate.get(
                    "text",
                    "",
                ),
            }
        )

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "contexts": contexts,
        "num_chunks_retrieved": len(
            final_candidates
        ),
        "latency_ms": round(
            total_latency_ms,
            1,
        ),
        "dense_latency_ms": round(
            timings["dense_latency_ms"],
            1,
        ),
        "lexical_latency_ms": round(
            timings["lexical_latency_ms"],
            1,
        ),
        "rerank_latency_ms": round(
            timings["rerank_latency_ms"],
            1,
        ),
        "generation_latency_ms": round(
            generation_latency_ms,
            1,
        ),
    }


def _generate(
    prompt: str,
    stream: bool = False,
):
    client = get_groq_client()

    return client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=1000,
        temperature=0,
        stream=stream,
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


def answer_question(
    question: str,
    log: bool = True,
    retrieval_query: str | None = None,
    instruction: str | None = None,
    previous_question: str | None = None,
):
    question = question.strip()

    retrieval_query = (
        retrieval_query or question
    ).strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    with Timer() as timer:

        final_candidates, timings = (
            retrieve_context(
                retrieval_query
            )
        )

        if not final_candidates:
            answer = (
                "I don't have enough information "
                "to answer that."
            )

            generation_latency_ms = 0.0

        else:
            prompt = build_prompt(
                retrieval_query,
                final_candidates,
                instruction=instruction,
                previous_question=previous_question,
            )

            start = time.perf_counter()

            response = _generate(
                prompt,
                stream=False,
            )

            generation_latency_ms = (
                time.perf_counter() - start
            ) * 1000

            answer = (
                response.choices[0]
                .message.content
                or ""
            )

    result = build_result(
        question,
        answer,
        final_candidates,
        timer.elapsed_ms,
        timings,
        generation_latency_ms,
    )

    if log:
        log_query(
            question,
            answer,
            result["sources"],
            result["num_chunks_retrieved"],
            result["latency_ms"],
            result["dense_latency_ms"],
            result["lexical_latency_ms"],
            result["rerank_latency_ms"],
            result["generation_latency_ms"],
        )

    return result


def stream_answer(
    question: str,
    retrieval_query: str | None = None,
    instruction: str | None = None,
    previous_question: str | None = None,
) -> Iterator[dict]:

    question = question.strip()

    retrieval_query = (
        retrieval_query or question
    ).strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    with Timer() as timer:

        final_candidates, timings = (
            retrieve_context(
                retrieval_query
            )
        )

        if not final_candidates:
            yield {
                "type": "complete",
                "answer": (
                    "I don't have enough information "
                    "to answer that."
                ),
                "contexts": [],
                "sources": [],
                "num_chunks_retrieved": 0,
                "latency_ms": round(
                    timer.elapsed_ms,
                    1,
                ),
                "dense_latency_ms": round(
                    timings["dense_latency_ms"],
                    1,
                ),
                "lexical_latency_ms": round(
                    timings["lexical_latency_ms"],
                    1,
                ),
                "rerank_latency_ms": round(
                    timings["rerank_latency_ms"],
                    1,
                ),
                "generation_latency_ms": 0.0,
            }
            return

        prompt = build_prompt(
            retrieval_query,
            final_candidates,
            instruction=instruction,
            previous_question=previous_question,
        )

        start = time.perf_counter()

        stream = _generate(
            prompt,
            stream=True,
        )

        answer_parts = []

        for chunk in stream:
            delta = chunk.choices[0].delta.content

            if delta:
                answer_parts.append(delta)

                yield {
                    "type": "token",
                    "text": delta,
                }

        generation_latency_ms = (
            time.perf_counter() - start
        ) * 1000

    answer = "".join(
        answer_parts
    )

    result = build_result(
        question,
        answer,
        final_candidates,
        timer.elapsed_ms,
        timings,
        generation_latency_ms,
    )

    log_query(
        question,
        answer,
        result["sources"],
        result["num_chunks_retrieved"],
        result["latency_ms"],
        result["dense_latency_ms"],
        result["lexical_latency_ms"],
        result["rerank_latency_ms"],
        result["generation_latency_ms"],
    )

    yield {
        "type": "complete",
        **result,
    }