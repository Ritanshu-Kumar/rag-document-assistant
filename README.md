# Document Q&A — RAG Chatbot

A Retrieval-Augmented Generation (RAG) system that answers questions grounded in your own documents. Combines local vector search with an LLM, a hallucination-resistant prompt, and full query observability via SQL logging.

Built to explore the end-to-end RAG pattern: chunking → embedding → retrieval → grounded generation → evaluation.

## Features

- **Semantic retrieval** over your own PDF/text documents using local embeddings (no API cost for search)
- **Grounded generation** via Llama 3.1 (Groq API) with a system prompt engineered to reduce hallucination
- **Honest refusals** — correctly declines to answer questions outside the indexed documents instead of making something up
- **Full observability** — every query, answer, source, and latency logged to SQLite
- **Measured, not assumed** — includes an evaluation harness with a real, manually-reviewed accuracy score
- **Web UI** via Streamlit, alongside a CLI for quick testing

## Architecture


data/*.txt|*.pdf
      │
      ▼
  Chunking (sliding window)
      │
      ▼
  Embedding (sentence-transformers, local)
      │
      ▼
  ChromaDB vector store
      │
      │        user question
      │              │
      │              ▼
      └──────►  Retrieve top-k relevant chunks
                     │
                     ▼
          Build grounded prompt (context + question)
                     │
                     ▼
          Llama 3.1 generates answer (Groq API)
                     │
                     ▼
          Log to SQLite (question, answer, sources, latency)


## Tech Stack

| Layer | Tool |
|---|---|
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB |
| LLM | Llama 3.1 via Groq API |
| Query logging | SQLite |
| Web UI | Streamlit |
| PDF parsing | pypdf |

## Setup

**Requires Python 3.11 or 3.12** (see [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if you hit dependency install issues on newer Python versions).

# Clone and enter the project
git clone <your-repo-url>
cd rag-project

# Create and activate a virtual environment
python3.11 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Configure your API key
cp .env.example .env
# Edit .env and add your free key from https://console.groq.com/keys

## Usage

```bash
# 1. Add your documents (.txt or .pdf) to data/

# 2. Build the vector index
python src/ingest.py

# 3. Ask a question via CLI
python src/rag.py

# 4. Or launch the web UI
streamlit run src/app.py
```

## Evaluation

The project includes a lightweight evaluation harness (`eval/evaluate.py`) that runs a fixed set of test questions against the live pipeline and checks answers against expected keywords, logging results with real latency figures.

**Results from evaluation on a set of course-note PDFs:**

| Metric | Result |
|---|---|
| Accuracy (manually reviewed) | **80%** (12/15 test questions) |
| Automated keyword-match accuracy | 66.7% (10/15) — undercounts true accuracy due to answer paraphrasing |
| Best-case end-to-end latency | ~400ms |

The gap between automated and manually-reviewed accuracy is itself a finding: exact substring matching penalizes correct answers that are phrased differently than expected, which is a common pitfall in evaluating generative systems. One genuine retrieval limitation was identified and documented — a question about "data types" occasionally retrieved a semantically adjacent but incorrect section ("dataset types") from the source material.

Run it yourself:
```bash
python eval/evaluate.py
```

## Project Structure

```
rag-project/
├── src/
│   ├── ingest.py      # Document chunking + embedding + indexing
│   ├── rag.py          # Core retrieval + generation pipeline
│   ├── db.py            # SQLite query logging
│   └── app.py             # Streamlit web UI
├── eval/
│   ├── eval_set.json    # Test questions
│   └── evaluate.py       # Evaluation harness
├── data/                    # Your documents go here
└── requirements.txt
```

## Roadmap / Future Improvements

- [ ] Instrument retrieval-only latency separately from generation latency
- [ ] Semantic/recursive chunking instead of fixed sliding-window
- [ ] Re-ranking step after initial retrieval
- [ ] Multi-turn conversation memory
- [ ] Agentic routing — decide whether retrieval is needed per query (e.g. via LangGraph)
- [ ] Resolve the identified retrieval confusion between semantically similar terms via metadata tagging