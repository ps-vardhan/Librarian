# IRA - ID based Retrival-Augmentation Generation API

A self-contained **Retrieval-Augmented Generation (RAG) API & UI** powered by FastAPI, PostgreSQL (pgvector), and Google Gemini.

By default, Librarian runs as a **headless API** designed for easy integration with downstream applications, with an optional sleek, dark-themed Single-Page Application (SPA) frontend.

---

## Features

- Offloaded Async I/O\*\*: Custom `AsyncPgVector` engine offloads blocking database operations to a CPU-bounded `ThreadPoolExecutor`, maximizing concurrent query throughput.
- Secure User Isolation\*\*: Comprehensive metadata-level multi-tenant mapping scopes all document ingestions, deletions, vector lookups, and AI queries to the specific User ID boundary.
- High-Performance Containment Queries\*\*: Automated database migrations convert cmetadata to Postgres `JSONB` and apply `jsonb_path_ops` GIN indexing to leverage faster containment query execution (`@>`).
- Multi-Format Ingestion**: Background parser processes **10+ formats\*\* (PDF, CSV, Docx, XML, PPTX, EPub, Markdown, JSON, etc.) asynchronously using FastAPI `BackgroundTasks`.
- Generative Q&A on Demand\*\*: Integrates Gemini (`gemini-1.5-flash`) via LangChain to synthesize contextual answers directly from retrieved document chunks.

---

## Prerequisites

- **Docker & Docker Desktop** (Recommended for containerized deployment)
- **Python 3.10+** (If running locally outside container)
- **Google Gemini API Key**

---

## Getting Started

### Option A: Running with Docker

Run the cross-platform orchestration script to boot the PostgreSQL database and FastAPI containers together:

- **Windows**:
  ```cmd
  run-docker.bat
  ```

### Option B: Running Locally

1. **Start the database**:
   ```bash
   docker compose up db -d
   ```
2. **Setup environment variables**:
   ```bash
   cp .env.example .env
   ```
   _Edit `.env` and paste your `RAG_GOOGLE_API_KEY`._
3. **Install dependencies and run**:
   - **Windows**: Double-click `run.bat`
   - **macOS/Linux**:
     ```bash
     pip install -r requirements.txt
     python main.py
     ```

---

## Environment Variables

| Variable             | Description                                                       | Default                         |
| :------------------- | :---------------------------------------------------------------- | :------------------------------ |
| `RAG_SERVE_UI`       | Serve the web-based SPA UI at root `/` (Set to `True` to enable). | `False` (Headless API)          |
| `RAG_GOOGLE_API_KEY` | Gemini API Studio credential key.                                 | `your_google_api_key`           |
| `RAG_PORT`           | Port binding for the FastAPI server container/process.            | `8000`                          |
| `DB_HOST`            | Host for the PostgreSQL instance.                                 | `localhost` (or `db` in Docker) |
| `DB_PORT`            | Port for the PostgreSQL instance.                                 | `5433` (or `5432` in Docker)    |
| `JWT_SECRET`         | Authentication secret. Bypasses token checks if undefined.        | _(None)_                        |
