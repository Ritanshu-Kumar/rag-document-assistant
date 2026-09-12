import json
import os
import sqlite3
import time


DB_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "query_log.db",
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    retrieved_sources TEXT,
    num_chunks_retrieved INTEGER,
    latency_ms REAL,
    dense_latency_ms REAL,
    lexical_latency_ms REAL,
    rerank_latency_ms REAL,
    generation_latency_ms REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def get_connection():
    conn = sqlite3.connect(DB_PATH)

    conn.execute(SCHEMA)

    existing_columns = {
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(query_log)"
        ).fetchall()
    }

    required_columns = {
        "dense_latency_ms": "REAL",
        "lexical_latency_ms": "REAL",
        "rerank_latency_ms": "REAL",
        "generation_latency_ms": "REAL",
    }

    for column, column_type in required_columns.items():
        if column not in existing_columns:
            conn.execute(
                f"ALTER TABLE query_log ADD COLUMN "
                f"{column} {column_type}"
            )

    conn.commit()

    return conn


def log_query(
    question: str,
    answer: str,
    sources: list,
    num_chunks: int,
    latency_ms: float,
    dense_latency_ms: float = 0.0,
    lexical_latency_ms: float = 0.0,
    rerank_latency_ms: float = 0.0,
    generation_latency_ms: float = 0.0,
):
    conn = get_connection()

    conn.execute(
        """
        INSERT INTO query_log (
            question,
            answer,
            retrieved_sources,
            num_chunks_retrieved,
            latency_ms,
            dense_latency_ms,
            lexical_latency_ms,
            rerank_latency_ms,
            generation_latency_ms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            question,
            answer,
            json.dumps(sources),
            num_chunks,
            latency_ms,
            dense_latency_ms,
            lexical_latency_ms,
            rerank_latency_ms,
            generation_latency_ms,
        ),
    )

    conn.commit()
    conn.close()


def get_stats():
    conn = get_connection()

    cur = conn.execute(
        """
        SELECT
            COUNT(*),
            AVG(latency_ms),
            MIN(latency_ms),
            MAX(latency_ms),
            AVG(dense_latency_ms),
            AVG(lexical_latency_ms),
            AVG(rerank_latency_ms),
            AVG(generation_latency_ms)
        FROM query_log
        """
    )

    (
        count,
        avg_latency,
        min_latency,
        max_latency,
        avg_dense,
        avg_lexical,
        avg_rerank,
        avg_generation,
    ) = cur.fetchone()

    conn.close()

    return {
        "total_queries": count or 0,
        "avg_latency_ms": (
            round(avg_latency, 1)
            if avg_latency is not None
            else None
        ),
        "min_latency_ms": (
            round(min_latency, 1)
            if min_latency is not None
            else None
        ),
        "max_latency_ms": (
            round(max_latency, 1)
            if max_latency is not None
            else None
        ),
        "avg_dense_latency_ms": (
            round(avg_dense, 1)
            if avg_dense is not None
            else None
        ),
        "avg_lexical_latency_ms": (
            round(avg_lexical, 1)
            if avg_lexical is not None
            else None
        ),
        "avg_rerank_latency_ms": (
            round(avg_rerank, 1)
            if avg_rerank is not None
            else None
        ),
        "avg_generation_latency_ms": (
            round(avg_generation, 1)
            if avg_generation is not None
            else None
        ),
    }


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        self.elapsed_ms = 0.0
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (
            time.perf_counter() - self._start
        ) * 1000