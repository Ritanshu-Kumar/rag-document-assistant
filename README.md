# Document Q&A — RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions using information from your own PDF and text documents.

The project covers the full RAG pipeline: document ingestion, chunking, local embeddings, vector search, retrieval refinement, grounded generation, and evaluation.

## Features

- **Document ingestion** — Supports PDF and text documents.
- **Paragraph-aware chunking** — Splits documents while preserving paragraph boundaries where possible.
- **Semantic retrieval** — Uses `all-MiniLM-L6-v2` embeddings with ChromaDB for local vector search.
- **Retrieval refinement** — Combines semantic similarity with lexical relevance to select useful chunks.
- **Grounded generation** — Generates answers using only the retrieved document context.
- **Honest refusals** — Refuses to answer when the required information is not present in the indexed documents.
- **Query observability** — Logs questions, answers, sources, retrieved chunk counts, and latency to SQLite.
- **Evaluation harness** — Includes a fixed test set for measuring answer accuracy and latency.
- **Web UI** — Streamlit interface for interacting with the chatbot.
- **CLI** — Command-line interface for quick testing.

## Architecture

```text
PDF / TXT Documents
        │
        ▼
   Text Extraction
        │
        ▼
Paragraph-aware Chunking
        │
        ▼
Sentence Transformer
   Embeddings
        │
        ▼
     ChromaDB
        │
        ▼
 Semantic Retrieval
        │
        ▼
Lexical Relevance Scoring
        │
        ▼
  Relevant Context
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

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| LLM | GPT-OSS 20B via Groq API |
| Query Logging | SQLite |
| Web UI | Streamlit |
| PDF Parsing | pypdf |
| Progress Tracking | tqdm |

## Setup

### Requirements

- Python 3.11
- A Groq API key

### Clone the repository

```bash
git clone <your-repo-url>
cd rag-project
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

GROQ_API_KEY=your_groq_api_key


Add your API key from the Groq console.

## Usage

### 1. Add documents

Place your `.txt` or `.pdf` files inside the `data/` directory.

### 2. Build the vector index

```bash
python src/ingest.py
```

This extracts the documents, creates chunks, generates embeddings, and stores them in ChromaDB.

### 3. Ask questions through the CLI

```bash
python src/rag.py
```

### 4. Launch the web interface

```bash
streamlit run src/app.py
```

## Evaluation

The project includes an evaluation harness in `eval/evaluate.py`.

The current test set contains 15 questions covering concepts from the indexed course-note PDFs.

### Latest Evaluation

| Metric | Result |
|---|---|
| Test questions | 15 |
| Correct | 12 |
| Accuracy | 80.0% |
| Average latency | 5.7s |

The initial version achieved 66.7% accuracy with an average latency of 7.5s.

After improving document chunking and retrieval, the evaluation increased to 80.0% accuracy while reducing average latency to 5.7s.

The evaluation currently uses expected-answer keyword matching, so the score can penalize answers that are semantically correct but use different wording from the expected answer.

Run the evaluation with:

```bash
python eval/evaluate.py
```

Results are written to `eval/results.json`.

## Project Structure

rag-project/
├── src/
│ ├── ingest.py # Document processing and indexing
│ ├── rag.py # Retrieval and generation pipeline
│ ├── db.py # SQLite query logging
│ └── app.py # Streamlit interface
│
├── eval/
│ ├── eval_set.json # Evaluation questions
│ ├── evaluate.py # Evaluation harness
│ └── results.json # Evaluation results
│
├── data/ # Documents to index
├── chroma_db/ # Local vector database
├── requirements.txt
└── README.md


## What I Learned

This project was built to understand what actually happens inside a RAG system rather than treating it as a black-box API.

The main areas explored were:

- Document preprocessing and chunking
- Embedding-based semantic search
- Vector database indexing and retrieval
- Combining semantic and lexical relevance
- Grounding LLM responses in retrieved context
- Handling questions outside the document knowledge
- Measuring latency and retrieval behaviour
- Building an evaluation pipeline for generative answers

The evaluation also showed a limitation of simple keyword-based metrics: a generated answer can be correct while still being marked incorrect because it uses different wording from the expected answer.

## Future Improvements

- Improve evaluation using semantic answer matching
- Add retrieval-specific evaluation metrics
- Experiment with recursive or semantic chunking
- Add conversation memory
- Add metadata-based filtering for larger document collections
- Separate retrieval and generation latency in the evaluation
- Experiment with stronger reranking approaches

## License

Personal project. Currently unlicensed.