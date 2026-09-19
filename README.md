# RAG Knowledge Assistant

A retrieval-augmented generation (RAG) system that answers questions from a local document collection using a multi-stage retrieval pipeline, grounded generation, conversational follow-ups, evaluation, and query-level observability.

## Project at a glance

| Capability | Implementation |
|---|---|
| Document ingestion | PDF, DOCX, TXT; recursive discovery |
| Chunking | Paragraph-aware chunks with overlap |
| Semantic retrieval | `all-MiniLM-L6-v2` + ChromaDB |
| Lexical retrieval | SQLite FTS5 / BM25 |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | GPT-OSS 20B via Groq |
| Grounding | Context-only prompting + explicit refusal |
| Citations | Numbered source citations |
| Conversation | Follow-up detection and context reuse |
| Observability | SQLite query + stage latency logging |
| Interface | Streamlit UI + CLI |

## Evaluation snapshot

The repository includes a 15-question evaluation set covering data visualization concepts plus two out-of-scope questions. The current saved run achieved:

| Metric | Result |
|---|---:|
| Questions | 15 |
| Correct | 12 |
| Accuracy | 80.0% |
| Average latency | 5.70 s |

The evaluation uses a simple expected-substring check, so this is an **initial project benchmark**, not a standardized RAG benchmark. The saved results also show substantial latency variation between queries (roughly 0.8 s to 20.9 s), which is why the README reports both accuracy and latency rather than presenting the 80% figure in isolation.

The benchmark artifacts are versioned in `eval/` so the exact questions and recorded outputs can be inspected.

## Architecture

```text
                     Documents
                PDF / DOCX / TXT
                        |
                        v
                Text Extraction
                        |
                        v
                Cleaning + Chunking
                        |
                        v
                 Sentence Transformer
                    Embeddings
                        |
                        v
                    ChromaDB
                        |
                 +------+------+
                 |             |
                 v             v
          Dense Retrieval   BM25 Retrieval
             Top 50            Top 50
                 |             |
                 +------+------+
                        v
             Reciprocal Rank Fusion
                        |
                      Top 30
                        |
                        v
              Cross-Encoder Reranking
                        |
                      Top 6
                        |
                        v
                 Grounded Prompt
                        |
                        v
              GPT-OSS 20B via Groq
                        |
                        v
                     Answer
                        |
                        v
                 SQLite Logging
```

## Features

- **Recursive document ingestion** — Supports PDF, DOCX, and TXT files across nested directories.
- **Metadata-aware ingestion** — Preserves document hierarchy and source metadata during indexing.
- **Text chunking** — Splits extracted content into retrieval-friendly chunks with overlap.
- **Dense retrieval** — Uses Sentence Transformer embeddings with ChromaDB.
- **Lexical retrieval** — Uses SQLite FTS5 / BM25 for exact-term matching.
- **Hybrid retrieval** — Combines dense and lexical rankings with Reciprocal Rank Fusion.
- **Cross-encoder reranking** — Reranks the fused candidate set before generation.
- **Grounded generation** — Uses retrieved context rather than unrestricted model knowledge.
- **Grounded refusal** — Returns an explicit insufficient-information response when the retrieved context is insufficient.
- **Conversational follow-ups** — Reuses the previous retrieval query for supported follow-up instructions.
- **Streaming responses** — Streams generation into the Streamlit interface.
- **Source transparency** — Displays retrieved excerpts and relevance information.
- **Conversation management** — Supports multiple sessions, editing, regeneration, and export.
- **Feedback** — Supports positive and negative answer feedback.
- **Query observability** — Logs retrieval and generation timing to SQLite.
- **CLI + Web UI** — Supports both direct testing and interactive use.

## Conversational retrieval

The system distinguishes a new information request from a follow-up instruction.

Example:

**Question**
> What is an operating system?

**Follow-up**
> Explain it like I am five.

The follow-up reuses the previous retrieval query instead of searching for the wording of the instruction itself. The follow-up then becomes a generation instruction over the already retrieved context.

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB |
| Lexical Search | SQLite FTS5 / BM25 |
| Rank Fusion | Reciprocal Rank Fusion |
| Reranking | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| LLM | GPT-OSS 20B via Groq |
| Query Logging | SQLite |
| Web UI | Streamlit |
| PDF Parsing | pypdf |
| DOCX Parsing | python-docx |
| Progress Tracking | tqdm |

## Setup

### Requirements

- Python 3.11
- A Groq API key

### Clone the repository

```bash
git clone https://github.com/Ritanshu-Kumar/rag-document-assistant.git
cd rag-document-assistant
```

### Create a virtual environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure the API key

Create `.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

`.env.example` is included as the template. Never commit the real API key.

## Usage

### 1. Add documents

Place PDF, DOCX, or TXT files inside `data/`. Nested directories are supported.

```text
data/
├── collection_a/
│   ├── document1.pdf
│   └── document2.docx
└── collection_b/
    └── document3.txt
```

### 2. Build the knowledge index

```bash
python src/ingest.py
```

The ingestion pipeline discovers supported files, extracts text, cleans it, chunks it, creates embeddings, and stores the resulting metadata and vectors in ChromaDB.

The indexed benchmark collection used during development contained approximately 67,000 chunks. Because the source documents and generated indexes are intentionally excluded from Git, reproducing that exact benchmark requires the same source collection.

### 3. Test through the CLI

```bash
python src/rag.py
```

### 4. Launch the web interface

```bash
streamlit run src/app.py
```

## Retrieval pipeline

The pipeline deliberately uses multiple retrieval stages:

1. **Dense retrieval** — semantic search with Sentence Transformer embeddings.
2. **BM25 retrieval** — lexical search for exact terminology and identifiers.
3. **Reciprocal Rank Fusion** — combines the two rankings without assuming their score scales are directly comparable.
4. **Cross-encoder reranking** — scores the fused candidates using the question and passage together.
5. **Context selection** — keeps the strongest final chunks while limiting repeated source coverage.

Current retrieval configuration is defined in `src/rag.py` and includes 50 dense candidates, 50 lexical candidates, RRF top 30, reranking over 30 candidates, and 6 final context chunks.

## Grounded generation

The generation prompt instructs the model to:

- use only supplied context,
- avoid outside knowledge,
- avoid guessing,
- cite supporting context,
- explicitly refuse when the evidence is insufficient.

This makes retrieval quality part of the answer-generation contract rather than treating the LLM as an unrestricted knowledge source.

## Evaluation

Run the benchmark with:

```bash
python eval/evaluate.py
```

The evaluation set lives in `eval/eval_set.json`, and the latest saved results are in `eval/results.json`.

The current benchmark is intentionally lightweight: each answer is checked for an expected text fragment. It is useful for regression testing and quick project-level comparisons, but it should not be interpreted as a general measure of factual accuracy or RAG quality.

## Observability

Each logged query records:

- question
- answer
- retrieved sources
- number of retrieved chunks
- total latency
- dense retrieval latency
- BM25 latency
- reranking latency
- generation latency
- timestamp

This makes it possible to separate retrieval cost from model-generation cost rather than treating the end-to-end latency as a single opaque number.

## Project structure

```text
rag-document-assistant/
├── src/
│   ├── ingest.py
│   ├── rag.py
│   ├── db.py
│   └── app.py
│
├── eval/
│   ├── eval_set.json
│   ├── evaluate.py
│   └── results.json
│
├── tests/
│   ├── test_rag_utils.py
│   └── test_eval_assets.py
│
├── data/                 # local documents; ignored by Git
├── chroma_db/            # generated vector store; ignored by Git
├── lexical_index.db      # generated lexical index; ignored by Git
├── query_log.db          # generated query log; ignored by Git
├── requirements.txt
├── requirements-ci.txt
├── .env.example
├── TROUBLESHOOTING.md
├── LICENSE
└── README.md
```

## Reproducibility and limitations

- The exact 67,000-chunk benchmark depends on a local document collection that is not stored in Git.
- Groq API latency depends on the remote inference service and can vary between runs.
- The current evaluation set is small and uses substring matching rather than human or model-based grading.
- The project is designed as a focused RAG system rather than a production multi-tenant deployment.

## What I learned

This project was built to understand what happens inside a RAG system rather than treating RAG as a single vector-search API call.

The main areas explored were document preprocessing, chunking, metadata preservation, semantic retrieval, lexical retrieval, rank fusion, cross-encoder reranking, context selection, grounded generation, refusal behavior, conversational retrieval, latency instrumentation, and interactive RAG application design.

A major lesson from the project was that improving a RAG application often requires improving retrieval quality and evaluation methodology rather than simply changing the language model.

## Future improvements

- Build a larger, more representative evaluation benchmark.
- Add retrieval-specific metrics such as Recall@K and MRR.
- Measure retrieval and reranking latency across repeated runs.
- Explore stronger embedding models and metadata-aware retrieval.
- Improve document update and re-indexing workflows.
- Improve citation navigation and source highlighting.

## License

MIT. See `LICENSE`.
