# ID-based RAG FastAPI

## Overview
This project integrates LangChain with FastAPI in an Asynchronous, Scalable manner, providing a lightweight framework for document indexing and retrieval using Google Gemini embeddings and a PostgreSQL/pgvector database.

Files are organized and queried by `file_id`. The primary use case is for integration with [LibreChat](https://librechat.ai), but this simple API can be used for any ID-based RAG application.

The core pipeline follows a strict, streamlined sequence:
**Upload/Ingest → Parse/Chunk → Embed (Gemini) → Store (pgvector) → Query by file_id (including query_multiple)**.

---

## Features
- **Document Management**: Methods for adding (`/embed`, `/embed-upload`, `/local/embed`), retrieving (`/documents`), and deleting (`/documents`) documents.
- **Text Extraction**: A `/text` endpoint for extracting raw text from uploaded files without creating database embeddings.
- **Vector Search**: Targeted similarity queries filtered by `file_id` (single file search via `/query`, or multiple files via `/query_multiple`).
- **Asynchronous Support**: Async database operations and concurrency utilizing a thread pool for parsing and embedding.

---

## Setup

### Getting Started

1. **Configure the `.env` file** based on the [Environment Variables](#environment-variables) section.
2. **Start the pgvector database**:
   - Run an existing PostgreSQL instance with the `vector` extension enabled, or
   - Start via Docker: `docker compose up -d db` (from the provided `docker-compose.yaml`).
3. **Run the API**:
   - Install dependencies and start local server:
     ```bash
     pip install -r requirements.txt
     uvicorn main:app --reload
     ```

### Environment Variables

Configure these variables in a `.env` file or within your environment:

#### Google Gemini Configuration
- `RAG_GOOGLE_API_KEY` (or `GOOGLE_API_KEY` / `GOOGLE_KEY`): The API key for Google Gemini Developer API.
- `EMBEDDINGS_MODEL`: The Gemini embedding model to use. Defaults to `gemini-embedding-001`.

#### Database Configuration
- `POSTGRES_DB`: The name of the PostgreSQL database. Defaults to `mydatabase`.
- `POSTGRES_USER`: The username for connecting to the database. Defaults to `myuser`.
- `POSTGRES_PASSWORD`: The password for connecting to the database. Defaults to `mypassword`.
- `DB_HOST`: The hostname or IP address of the database server. Defaults to `db`.
- `DB_PORT`: The port number of the database server. Defaults to `5432`.
- `POSTGRES_USE_UNIX_SOCKET`: Set to `True` to connect using Unix Sockets instead of TCP. Defaults to `False`.
- `PGVECTOR_CREATE_EXTENSION`: Set to `False` to skip the `CREATE EXTENSION IF NOT EXISTS vector` call on startup. Defaults to `True`.
- `POSTGRES_SCHEMA`: Comma-separated list of schemas to prepend to the connection search path.
- `COLLECTION_NAME`: The name of the collection in the vector store. Defaults to `testcollection`.

#### Connection Pool Settings
- `PG_POOL_PRE_PING`: Enables SQLAlchemy's pre-ping check to replace dropped connections. Defaults to `True`.
- `PG_POOL_RECYCLE`: Max connection age in seconds before recycling. Defaults to `-1` (disabled).

#### Server & Text Processing Configurations
- `RAG_HOST`: Host address where the API server runs. Defaults to `0.0.0.0`.
- `RAG_PORT`: Port where the API server runs. Defaults to `8000`.
- `RAG_UPLOAD_DIR`: Directory where uploaded files are temporarily stored. Defaults to `./uploads/`.
- `CHUNK_SIZE`: Size of the chunks for text splitting. Defaults to `1500`.
- `CHUNK_OVERLAP`: Overlap size between chunks. Defaults to `100`.
- `RAG_DISTANCE_THRESHOLD`: Filter out vector results whose distance exceeds this value. Defaults to `None`.
- `JWT_SECRET`: Secret key used to verify signed JWT request headers. Omit to run without authentication.

---

## Running Tests

### Install Test Dependencies
```bash
pip install -r test_requirements.txt
```

### Run the Test Suite
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v
```
