# Personal RAG Assistant

A simple, modular, and maintainable Personal RAG (Retrieval-Augmented Generation) application built with:
- **Streamlit**: Web chat interface with session management.
- **Jina AI API (`jina-embeddings-v3`)**: 1024-dimension cloud embeddings for document chunks and user queries.
- **MongoDB Atlas Local / Docker**: Vector search and conversation history persistence.
- **Groq Cloud API (`qwen/qwen3.8-27b`)**: Ultra-fast instruction-tuned cloud LLM for factual context-grounded answers.

---

## Architecture

```text
User
  │
  ▼
Streamlit UI (app.py)
  │
  ▼
RAG Pipeline (rag/pipeline.py)
  │
  ├──► Jina AI Cloud Embedding API (llm/providers.py)
  │       └─ Converts query to 1024-dim embedding vector
  │
  ├──► MongoDB Vector Search (retrieval/retriever.py)
  │       └─ Retrieves top matching chunks from `documents_jina` collection
  │
  ├──► MongoDB Session History (database/session.py)
  │       └─ Loads conversation context for pronoun & follow-up resolution
  │
  ├──► Groq Cloud LLM API (llm/providers.py)
  │       └─ Generates strictly grounded answer with zero hallucinations
  │
  ▼
Answer saved to MongoDB session & returned to Streamlit
```

---

## Prerequisites

1. **Python 3.10+** (tested on Python 3.14).
2. **Docker** running MongoDB with Atlas Search / Vector Search enabled (e.g. `mongodb/mongodb-atlas-local:preview`).
3. **Jina AI API Key** (Free 1M tokens from [jina.ai](https://jina.ai/embeddings/)).
4. **Groq API Key** (Free tier from [console.groq.com](https://console.groq.com/keys)).

---

## Setup & Configuration

1. **Clone or navigate to the repository**:
   ```bash
   cd D:\Personal_rag
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables in `.env`**:
   ```env
   JINA_API_KEY=your_jina_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   MONGODB_URI=mongodb://localhost:27017/?directConnection=true
   ```

---

## Running the Application

1. **Create the Vector Index**:
   ```bash
   python -m database.vector_index
   ```

2. **Ingest the Knowledge Base (Idempotent)**:
   ```bash
   python -m ingestion.ingest
   ```

3. **Launch the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

---

## Running the Automated Test Suite

Run all verification tests independently:

```bash
# 1. MongoDB Connection Test
python -m test.test_connection

# 2. Jina AI Cloud Embedding Test
python -m test.test_embedding

# 3. Groq Cloud Text Generation Test
python -m test.test_generation

# 4. Vector Search & Threshold Calibration Test
python -m test.test_retrieval

# 5. End-to-End RAG, Follow-ups, and Fallback Test
python -m test.test_rag_session

# 6. Multi-Session Isolation Test
python -m test.test_multi_session
```

---

## Rollback Procedure (Back to Gemini)

If you ever wish to roll back to the original Gemini setup:
1. In `config/settings.py`, change:
   - `COLLECTION_NAME = "documents"`
   - `VECTOR_INDEX_NAME = "vector_index"`
   - `EMBEDDING_DIMENSIONS = 3072`
2. In `rag/pipeline.py`, revert generation to `llm/gemini.py`.
3. In `retrieval/retriever.py`, revert embeddings to `gemini-embedding-001`.
*(The original `documents` collection and its 3072-dim `vector_index` remain 100% intact).*
