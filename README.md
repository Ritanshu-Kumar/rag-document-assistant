# rag-document-assistant

Retrieval-Augmented Generation (RAG) system for intelligent document Q&amp;A with semantic search, vector embeddings, ChromaDB, Groq LLM, SQL query logging, and AWS deployment.

# RAG Document Assistant

A Retrieval-Augmented Generation (RAG) application that answers questions from a collection of PDF documents using semantic search and a Large Language Model. The system retrieves relevant document chunks from a vector database and generates responses grounded in the retrieved context instead of relying solely on the model's knowledge.

## Features

- Semantic search over PDF documents
- Document chunking and embedding generation
- ChromaDB vector database for retrieval
- Grounded response generation using Groq (Llama 3.1)
- SQL logging for user queries and responses
- Evaluation framework for testing retrieval quality
- Modular codebase for easy extension and deployment

## Project Structure

```
rag-document-assistant/
├── data/              # Input PDF documents (ignored by Git)
├── chroma_db/         # Vector database (generated locally)
├── eval/              # Evaluation scripts and datasets
├── src/
│   ├── app.py
│   ├── db.py
│   ├── ingest.py
│   └── rag.py
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## How It Works

1. PDF documents are placed inside the `data/` directory.
2. The ingestion pipeline extracts text, splits it into chunks, and generates embeddings.
3. Embeddings are stored in ChromaDB.
4. When a user asks a question, the system retrieves the most relevant chunks using semantic search.
5. The retrieved context is sent to the LLM, which generates an answer based only on that information.

## Tech Stack

- Python
- ChromaDB
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Groq API
- Llama 3.1
- SQLite
- python-dotenv

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/rag-document-assistant.git
cd rag-document-assistant
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file using `.env.example` and add your Groq API key.

## Running the Project

### 1. Ingest Documents

Place your PDF files inside the `data/` folder and run:

```bash
python src/ingest.py
```

### 2. Ask Questions

```bash
python src/rag.py
```

Example:

```
Ask a question about your documents:
What are the different data types?
```

## Sample Output

```
Answer:
The data types mentioned in the provided documents are:
- Categorical
- Ordinal
- Quantitative

Sources:
2 Data Abstraction.pdf
5 Task_data_Analysis.pdf
```

## Future Improvements

- Hybrid search (BM25 + vector search)
- Cross-encoder reranking
- Metadata filtering
- Conversational memory
- Streamlit web interface
- AWS deployment
- Docker support

## License

This project is available under the MIT License.

