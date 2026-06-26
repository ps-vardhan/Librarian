# ID-Rag/main.py
import os
import sys

# Ensure the ID-Rag folder is at the beginning of the Python search path.
# This guarantees that 'import app' resolves to ID-Rag/app even if executed from the parent workspace directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from app.config import (
    VectorDBType,
    debug_mode,
    RAG_HOST,
    RAG_PORT,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTOR_DB_TYPE,
    RAG_SERVE_UI,
    logger,
    vector_store,
)
from app.middleware import security_middleware, LogMiddleware
from app.routes import document_routes
from app.services.database import PSQLDatabase, ensure_vector_indexes
from app.services.vector_store.factory import close_vector_store_connections


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    # Create bounded thread pool executor based on CPU cores
    max_workers = min(
        int(os.getenv("RAG_THREAD_POOL_SIZE", str(os.cpu_count()))), 8
    )  # Cap at 8
    app.state.thread_pool = ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="rag-worker"
    )
    logger.info(
        f"Initialized thread pool with {max_workers} workers (CPU cores: {os.cpu_count()}) in ID-Rag"
    )

    if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
        try:
            await PSQLDatabase.get_pool()  # Initialize the pool
            await ensure_vector_indexes()
        except Exception as e:
            logger.warning(
                "Database connection or migration failed during startup. The application will continue running (mock/lazy connect). Error: %s",
                e,
            )

    yield

    # Cleanup logic
    if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
        try:
            logger.info("Closing asyncpg connection pool in ID-Rag")
            await PSQLDatabase.close_pool()
            logger.info("asyncpg connection pool closed in ID-Rag")
        except Exception as e:
            logger.warning("Failed to close asyncpg pool in ID-Rag: %s", e)

    # Drain in-flight work before closing backing resources
    logger.info("Shutting down thread pool in ID-Rag")
    app.state.thread_pool.shutdown(wait=True)
    logger.info("Thread pool shutdown complete in ID-Rag")

    # Close vector store connections (SQLAlchemy engine)
    try:
        close_vector_store_connections(vector_store)
    except Exception as e:
        logger.warning("Failed to close vector store connections in ID-Rag: %s", e)


app = FastAPI(lifespan=lifespan, debug=debug_mode)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LogMiddleware)

app.middleware("http")(security_middleware)

# Set state variables for use in routes
app.state.CHUNK_SIZE = CHUNK_SIZE
app.state.CHUNK_OVERLAP = CHUNK_OVERLAP

# Include routers
app.include_router(document_routes.router)


@app.get("/")
async def get_index():
    """Serves the frontend user interface file index.html directly from the root route if enabled."""
    if RAG_SERVE_UI:
        index_path = os.path.join(os.path.dirname(__file__), "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
    return JSONResponse(
        status_code=200,
        content={"status": "UP", "message": "Librarian RAG API is running"},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.debug("Validation error in ID-Rag: %s", exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "message": "Request validation failed"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host=RAG_HOST, port=RAG_PORT, log_config=None)
