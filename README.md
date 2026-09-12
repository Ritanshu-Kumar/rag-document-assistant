# RAG Knowledge Assistant

A retrieval-augmented generation (RAG) system that answers questions from a large local document collection.

The project implements the complete RAG pipeline from document ingestion and chunking through hybrid retrieval, reranking, grounded generation, conversational follow-ups, evaluation, and query-level observability.

## Features

- **Recursive document ingestion** — Supports PDF, DOCX, and TXT files across nested directories.
- **Metadata-aware ingestion** — Preserves document hierarchy and source metadata during indexing.
- **Text chunking** — Splits extracted content into retrieval-friendly chunks with overlap.
- **Dense retrieval** — Uses `all-MiniLM-L6-v2` embeddings with ChromaDB.
- **Lexical retrieval** — Uses SQLite FTS5 / BM25 for exact-term matching.
- **Hybrid retrieval** — Combines dense and lexical rankings using Reciprocal Rank Fusion (RRF).
- **Cross-encoder reranking** — Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank the strongest candidates.
- **Grounded generation** — Uses GPT-OSS 20B through Groq with retrieved context only.
- **Citation support** — Answers reference retrieved context using numbered citations.
- **Grounded refusal** — Returns an explicit insufficient-information response when the indexed data does not support an answer.
- **Conversational follow-ups** — Distinguishes new questions from follow-up instructions and reuses the previous retrieval context when appropriate.
- **Streaming responses** — Streams generated answers into the web interface.
- **Source transparency** — Displays retrieved source excerpts and relevance information.
- **Conversation management** — Supports multiple chat sessions, editing, regeneration, and conversation export.
- **Feedback** — Supports positive and negative answer feedback.
- **Query observability** — Logs questions, answers, sources, chunk counts, total latency, and retrieval/generation timing to SQLite.
- **Web UI** — Custom Streamlit interface for interactive use.
- **CLI** — Command-line interface for direct testing.

## Architecture

```text
                     Documents
                PDF / DOCX / TXT
                        │
                        ▼
                Text Extraction
                        │
                        ▼
                Cleaning + Chunking
                        │
                        ▼
                 Sentence Transformer
                    Embeddings
                        │
                        ▼
                    ChromaDB
                        │
                 ┌──────┴──────┐
                 │              │
                 ▼              ▼
          Dense Retrieval   BM25 Retrieval
             Top 50            Top 50
                 │              │
                 └──────┬──────┘
                        ▼
             Reciprocal Rank Fusion
                        │
                     Top 30
                        │
                        ▼
              Cross-Encoder Reranking
                        │
                      Top 6
                        │
                        ▼
                 Grounded Prompt
                        │
                        ▼
              GPT-OSS 20B via Groq
                        │
                        ▼
                     Answer
                        │
                        ▼
                 SQLite Logging
```

## Conversational Retrieval

The system distinguishes between a new information request and a follow-up instruction.

For example:

**Question:**
> What is an operating system?

**Follow-up:**
> Explain it like I am five.

The follow-up does not trigger retrieval for the phrase "explain it like I am five." Instead, the system reuses the previous retrieval query and applies the new message as a generation instruction.

This prevents conversational instructions from degrading retrieval quality.

## Tech Stack

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

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
```

## Usage

### 1. Add documents

Place PDF, DOCX, or TXT files inside the `data/` directory.

Nested directories are supported.

Example:

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

The ingestion pipeline:

1. Recursively discovers supported documents.
2. Extracts text.
3. Cleans and chunks the content.
4. Generates embeddings.
5. Stores the chunks and metadata in ChromaDB.

The current knowledge base contains approximately 67,000 indexed chunks.

### 3. Test through the CLI

```bash
python src/rag.py
```

### 4. Launch the web interface

```bash
streamlit run src/app.py
```

## Retrieval Pipeline

The retrieval system intentionally uses multiple stages.

**Dense retrieval**
ChromaDB performs semantic search using Sentence Transformer embeddings. This provides high recall for conceptually related content.

**BM25 retrieval**
SQLite FTS5 provides lexical search over the same chunk collection. This improves retrieval for exact technical terminology and identifiers.

**Reciprocal Rank Fusion**
Dense and lexical rankings are combined using Reciprocal Rank Fusion rather than directly mixing incompatible score scales.

**Cross-encoder reranking**
The fused candidate set is passed through a cross-encoder which evaluates the question and candidate passage together. Only the strongest final chunks are provided to the language model.

## Grounded Generation

The language model is explicitly instructed to:

- use only retrieved context,
- avoid outside knowledge,
- avoid guessing,
- cite supporting context,
- refuse when the retrieved evidence is insufficient.

This makes retrieval quality part of the answer-generation contract rather than treating the LLM as an unrestricted knowledge source.

## Evaluation and Testing

Testing covers:

- subject-specific questions,
- technical terminology,
- cross-domain retrieval,
- out-of-scope questions,
- conversational follow-ups,
- retrieval failures,
- latency across individual pipeline stages.

During development, a pure semantic retrieval approach was found to retrieve broad subject material for some specific technical questions. For example, a semaphore query could retrieve general operating-system chunks even when semaphore-specific content existed in the corpus.

That failure led to the hybrid dense + BM25 + RRF + cross-encoder architecture.

The system has also been tested with questions involving operating systems, databases, neural networks, networking, and out-of-domain queries.

## Observability

Each query is logged to SQLite with:

- question
- answer
- retrieved sources
- number of chunks
- total latency
- dense retrieval latency
- BM25 latency
- reranking latency
- generation latency
- timestamp

This allows retrieval and generation performance to be measured independently.

## Project Structure

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
├── data/
├── chroma_db/
├── lexical_index.db
├── query_log.db
├── requirements.txt
├── .gitignore
└── README.md
```

The data, vector store, lexical index, query database, environment variables, and other local runtime artifacts are excluded from Git.

## What I Learned

This project was built to understand what actually happens inside a RAG system rather than treating RAG as a single vector-search API call.

The main areas explored were:

- document preprocessing and recursive ingestion
- text extraction across document formats
- chunking and metadata preservation
- embedding-based semantic retrieval
- lexical retrieval with BM25
- hybrid retrieval and rank fusion
- cross-encoder reranking
- context selection and deduplication
- grounded LLM generation
- refusal behavior
- conversational retrieval
- latency instrumentation
- interactive RAG application design
- failure-driven retrieval improvements

One of the main lessons from the project was that improving RAG quality often requires improving the retrieval system rather than simply changing the language model.

## Future Improvements

- Build a larger automated evaluation benchmark
- Add retrieval-specific metrics such as Recall@K and MRR
- Optimize dense retrieval latency
- Optimize cross-encoder reranking latency
- Explore stronger embedding models
- Add metadata-aware retrieval filters
- Improve document ingestion and update workflows
- Add persistent user conversation storage
- Improve citation navigation and source highlighting

## License

Personal project. Currently unlicensed.