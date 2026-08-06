"""
SQLite logging for every query the RAG system handles.

This is what turns your project into real, provable resume metrics:
after running queries or the eval set, you can pull average latency,
retrieval counts, etc. straight from this table instead of guessing.
"""
import os
import sqlite3
import time
import json

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "query_log.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    retrieved_sources TEXT,   -- JSON list of source filenames used
    num_chunks_retrieved INTEGER,
    latency_ms REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def log_query(question: str, answer: str, sources: list, num_chunks: int, latency_ms: float):
    conn = get_connection()
    conn.execute(
        """INSERT INTO query_log (question, answer, retrieved_sources, num_chunks_retrieved, latency_ms)
           VALUES (?, ?, ?, ?, ?)""",
        (question, answer, json.dumps(sources), num_chunks, latency_ms),
    )
    conn.commit()
    conn.close()


def get_stats():
    """Pull summary stats — use these numbers directly in your resume bullet."""
    conn = get_connection()
    cur = conn.execute(
        """SELECT COUNT(*), AVG(latency_ms), MIN(latency_ms), MAX(latency_ms)
           FROM query_log"""
    )
    count, avg_latency, min_latency, max_latency = cur.fetchone()
    conn.close()
    return {
        "total_queries": count or 0,
        "avg_latency_ms": round(avg_latency, 1) if avg_latency else None,
        "min_latency_ms": round(min_latency, 1) if min_latency else None,
        "max_latency_ms": round(max_latency, 1) if max_latency else None,
    }


class Timer:
    """Small helper: `with Timer() as t: ... ; t.elapsed_ms`"""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
