# ID-Rag/app/routes/document_routes.py
import os
import uuid
from pathlib import Path
import hashlib
import traceback
import aiofiles
import aiofiles.os

from typing import List, Iterable, Optional, TYPE_CHECKING
from fastapi import (
    APIRouter,
    Request,
    UploadFile,
    HTTPException,
    File,
    Form,
    Body,
    Query,
    status,
    BackgroundTasks,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from functools import lru_cache
import asyncio

if TYPE_CHECKING:
    from app.services.vector_store.async_pg_vector import AsyncPgVector
    from langchain_community.vectorstores.pgvector import PGVector as PgVector

from app.config import (
    logger,
    vector_store,
    VECTOR_DB_TYPE,
    VectorDBType,
    RAG_UPLOAD_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RAG_DISTANCE_THRESHOLD,
)

from app.constants import ERROR_MESSAGES
from app.models import (
    StoreDocument,
    QueryRequestBody,
    DocumentResponse,
    QueryMultipleBody,
)
from app.services.vector_store.async_pg_vector import AsyncPgVector
from app.utils.document_loader import (
    get_loader,
    clean_text,
    process_documents,
    cleanup_temp_encoding_file,
)
from app.utils.health import is_health_ok

router = APIRouter()


def _apply_distance_threshold(documents):
    """Drop (doc, score) tuples whose distance exceeds RAG_DISTANCE_THRESHOLD.

    Only applied for pgvector, where similarity_search_with_score_by_vector
    returns a distance (lower = more similar).
    """
    if RAG_DISTANCE_THRESHOLD is None:
        return documents
    return [(doc, score) for doc, score in documents if score <= RAG_DISTANCE_THRESHOLD]


def get_user_id(request: Request, entity_id: str = None) -> str:
    """Extract user ID from request or entity_id."""
    if not hasattr(request.state, "user"):
        return entity_id if entity_id else "public"
    else:
        return entity_id if entity_id else request.state.user.get("id")


async def save_upload_file_async(file: UploadFile, temp_file_path: str) -> None:
    """Save uploaded file asynchronously."""
    try:
        async with aiofiles.open(temp_file_path, "wb") as temp_file:
            chunk_size = 64 * 1024  # 64 KB
            while content := await file.read(chunk_size):
                await temp_file.write(content)
    except Exception as e:
        logger.error(
            "Failed to save uploaded file | Path: %s | Error: %s | Traceback: %s",
            temp_file_path,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save the uploaded file. Error: {str(e)}",
        )


def validate_file_path(base_dir: str, file_path: str) -> Optional[str]:
    """Validate that file_path resolves within base_dir. Returns resolved absolute path or None."""
    if not file_path or not file_path.strip() or "\x00" in file_path:
        return None
    try:
        allowed = Path(base_dir).resolve()
        requested = Path(os.path.join(base_dir, file_path)).resolve()
        requested.relative_to(allowed)
        return str(requested)
    except (ValueError, RuntimeError, TypeError, OSError):
        return None


def _make_unique_temp_path(user_id: str, filename: str) -> Optional[str]:
    """Build a unique temp file path under RAG_UPLOAD_DIR/{user_id}/ to prevent
    concurrent upload collisions. Returns a validated absolute path, or None if
    the raw filename would escape RAG_UPLOAD_DIR (path traversal rejection)."""
    if validate_file_path(RAG_UPLOAD_DIR, os.path.join(user_id, filename)) is None:
        return None
    p = Path(filename)
    unique_name = f"{p.stem}_{uuid.uuid4().hex}{p.suffix}"
    return str(Path(RAG_UPLOAD_DIR, user_id, unique_name).resolve())


async def load_file_content(
    filename: str,
    content_type: str,
    file_path: str,
    executor,
    raw_text: bool = False,
) -> tuple:
    """Load file content using appropriate loader.

    Pass ``raw_text=True`` when the caller wants verbatim file contents (e.g.
    the ``/text`` endpoint) so text-formatted files are not semantically
    parsed.
    """
    loader = None
    try:
        loader, known_type, file_ext = get_loader(
            filename, content_type, file_path, raw_text=raw_text
        )
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(executor, lambda: list(loader.lazy_load()))
        return data, known_type, file_ext
    finally:
        if loader is not None:
            cleanup_temp_encoding_file(loader)


def extract_text_from_documents(documents: List[Document], file_ext: str) -> str:
    """Extract text content from loaded documents."""
    text_content = ""
    if documents:
        for doc in documents:
            if hasattr(doc, "page_content"):
                if file_ext == "pdf":
                    text_content += clean_text(doc.page_content) + "\n"
                else:
                    text_content += doc.page_content + "\n"
    return text_content.rstrip("\n")


async def cleanup_temp_file_async(file_path: str) -> None:
    """Clean up temporary file asynchronously."""
    try:
        await aiofiles.os.remove(file_path)
    except Exception as e:
        logger.error(
            "Failed to remove temporary file | Path: %s | Error: %s | Traceback: %s",
            file_path,
            str(e),
            traceback.format_exc(),
        )


async def process_and_store_file_background(
    filename: str,
    content_type: str,
    file_path: str,
    file_id: str,
    user_id: str,
    clean_content: bool,
    executor,
) -> None:
    try:
        data, known_type, file_ext = await load_file_content(
            filename,
            content_type,
            file_path,
            executor,
        )

        await store_data_in_vector_db(
            data=data,
            file_id=file_id,
            user_id=user_id,
            clean_content=clean_content,
            executor=executor,
        )
        logger.info(f"Background ingestion completed successfully for file: {filename}")
    except Exception as e:
        logger.error(
            "Background ingestion failed | File: %s | Error: %s | Traceback: %s",
            filename,
            str(e),
            traceback.format_exc(),
        )
    finally:
        await cleanup_temp_file_async(file_path)


@router.get("/ids")
async def get_all_ids(request: Request, entity_id: str = Query(None)):
    try:
        user_authorized = get_user_id(request, entity_id)
        if isinstance(vector_store, AsyncPgVector):
            ids = await vector_store.get_all_ids(user_id=user_authorized, executor=request.app.state.thread_pool)
        else:
            ids = vector_store.get_all_ids(user_id=user_authorized)
        return list(set(ids))
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in get_all_ids | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Failed to get all IDs | Error: %s | Traceback: %s",
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    try:
        if await is_health_ok():
            return {"status": "UP"}
        else:
            logger.error("Health check failed")
            return {"status": "DOWN"}, 503
    except Exception as e:
        logger.error(
            "Error during health check | Error: %s | Traceback: %s",
            str(e),
            traceback.format_exc(),
        )
        return {"status": "DOWN", "error": str(e)}, 503


@router.get("/documents", response_model=list[DocumentResponse])
async def get_documents_by_ids(request: Request, ids: list[str] = Query(...)):
    try:
        user_authorized = get_user_id(request)
        allowed_users = {user_authorized, "public"}

        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_filtered_ids(
                ids, executor=request.app.state.thread_pool
            )
            documents = await vector_store.get_documents_by_ids(
                ids, executor=request.app.state.thread_pool
            )
        else:
            existing_ids = vector_store.get_filtered_ids(ids)
            documents = vector_store.get_documents_by_ids(ids)

        if not all(id in existing_ids for id in ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        # Security check: verify ownership of each doc
        for doc in documents:
            if doc.metadata.get("user_id") not in allowed_users:
                raise HTTPException(status_code=403, detail="Permission denied to access these documents")

        if not documents:
            raise HTTPException(
                status_code=404, detail="No documents found for the given IDs"
            )

        return documents
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in get_documents_by_ids | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error getting documents by IDs | IDs: %s | Error: %s | Traceback: %s",
            ids,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents")
async def delete_documents(request: Request, document_ids: List[str] = Body(...), entity_id: str = Query(None)):
    try:
        user_authorized = get_user_id(request, entity_id)
        allowed_users = {user_authorized, "public"}

        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_filtered_ids(
                document_ids, executor=request.app.state.thread_pool
            )
            documents = await vector_store.get_documents_by_ids(
                document_ids, executor=request.app.state.thread_pool
            )
        else:
            existing_ids = vector_store.get_filtered_ids(document_ids)
            documents = vector_store.get_documents_by_ids(document_ids)

        if not all(id in existing_ids for id in document_ids):
            raise HTTPException(status_code=404, detail="One or more IDs not found")

        # Security check: verify ownership before deletion
        for doc in documents:
            if doc.metadata.get("user_id") not in allowed_users:
                raise HTTPException(status_code=403, detail="Permission denied to delete these documents")

        if isinstance(vector_store, AsyncPgVector):
            await vector_store.delete(
                ids=document_ids, executor=request.app.state.thread_pool
            )
        else:
            vector_store.delete(ids=document_ids)

        file_count = len(document_ids)
        return {
            "message": f"Documents for {file_count} file{'s' if file_count > 1 else ''} deleted successfully"
        }
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in delete_documents | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Failed to delete documents | IDs: %s | Error: %s | Traceback: %s",
            document_ids,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@lru_cache(maxsize=128)
def get_cached_query_embedding(query: str):
    return vector_store.embedding_function.embed_query(query)


@router.post("/query")
async def query_embeddings_by_file_id(
    body: QueryRequestBody,
    request: Request,
):
    if hasattr(request.state, "user"):
        allowed_users = [request.state.user.get("id"), "public"]
        if body.entity_id:
            if body.entity_id in allowed_users:
                allowed_users = [body.entity_id]
            else:
                allowed_users = ["public"]
    else:
        allowed_users = [body.entity_id] if body.entity_id else ["public"]

    try:
        embedding = get_cached_query_embedding(body.query)

        # SQL-level filter: file_id matches AND user_id matches allowed user(s)
        db_filter = {
            "file_id": {"$eq": body.file_id},
            "user_id": {"$in": allowed_users}
        }

        if isinstance(vector_store, AsyncPgVector):
            documents = await vector_store.asimilarity_search_with_score_by_vector(
                embedding,
                k=body.k,
                filter=db_filter,
                executor=request.app.state.thread_pool,
            )
        else:
            documents = vector_store.similarity_search_with_score_by_vector(
                embedding, k=body.k, filter=db_filter
            )

        documents = _apply_distance_threshold(documents)

        if not body.generate_answer:
            return documents

        answer = None
        if not documents:
            answer = "No relevant document chunks found to answer your query."
        else:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                from langchain_core.messages import HumanMessage
                from app.config import RAG_GOOGLE_API_KEY, GOOGLE_API_KEY
                
                api_key = RAG_GOOGLE_API_KEY or GOOGLE_API_KEY
                if not api_key:
                    answer = "Error: Google Gemini API key is missing. Please add it to your .env file."
                else:
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        google_api_key=api_key,
                    )
                    context_text = "\n---\n".join([doc.page_content for doc, _ in documents])
                    prompt = (
                        "You are a helpful assistant that answers questions based strictly on the provided context.\n"
                        "If the context does not contain enough information to answer the question, state that you do not know.\n\n"
                        f"Context:\n{context_text}\n\n"
                        f"Question: {body.query}\n\n"
                        "Answer:"
                    )
                    response = await llm.ainvoke([HumanMessage(content=prompt)])
                    answer = response.content
            except Exception as e:
                logger.error(f"Error generating AI answer: {e}")
                answer = f"Error generating AI answer: {str(e)}"

        return {
            "answer": answer,
            "documents": documents
        }

    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in query_embeddings_by_file_id | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error in query embeddings | File ID: %s | Query: %s | Error: %s | Traceback: %s",
            body.file_id,
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


def generate_digest(page_content: str) -> str:
    return hashlib.md5(page_content.encode("utf-8", "ignore")).hexdigest()


def _prepare_documents_sync(
    data: Iterable[Document],
    file_id: str,
    user_id: str,
    clean_content: bool,
) -> List[Document]:
    """
    Synchronous document preparation - runs in executor to avoid blocking event loop.
    Handles text splitting, cleaning, and metadata preparation.
    """
    processed_data = []
    if not clean_content:
        for doc in data:
            if isinstance(doc.page_content, str):
                lines = doc.page_content.split("\n")
                for line in lines:
                    stripped = line.strip()
                    if stripped:
                        meta = (doc.metadata or {}).copy()
                        processed_data.append(Document(page_content=stripped, metadata=meta))
            else:
                processed_data.append(doc)
    else:
        processed_data = list(data)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    documents = text_splitter.split_documents(processed_data)

    if clean_content:
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)

    return [
        Document(
            page_content=doc.page_content,
            metadata={
                "file_id": file_id,
                "user_id": user_id,
                "digest": generate_digest(doc.page_content),
                **(doc.metadata or {}),
            },
        )
        for doc in documents
    ]


async def store_data_in_vector_db(
    data: Iterable[Document],
    file_id: str,
    user_id: str = "",
    clean_content: bool = False,
    executor=None,
) -> bool:
    loop = asyncio.get_running_loop()
    docs = await loop.run_in_executor(
        executor,
        _prepare_documents_sync,
        data,
        file_id,
        user_id,
        clean_content,
    )

    try:
        if isinstance(vector_store, AsyncPgVector):
            ids = await vector_store.aadd_documents(
                docs, ids=[file_id] * len(docs), executor=executor
            )
        else:
            ids = vector_store.add_documents(docs, ids=[file_id] * len(docs))

        return {"message": "Documents added successfully", "ids": ids}

    except Exception as e:
        logger.error(
            "Failed to store data in vector DB | File ID: %s | User ID: %s | Error: %s | Traceback: %s",
            file_id,
            user_id,
            str(e),
            traceback.format_exc(),
        )
        return {"message": "An error occurred while adding documents.", "error": str(e)}


@router.post("/local/embed")
async def embed_local_file(
    document: StoreDocument, request: Request, entity_id: str = None
):
    file_path = validate_file_path(RAG_UPLOAD_DIR, document.filepath)

    if file_path is None or not os.path.exists(file_path):
        logger.warning("Path validation failed for local embed: %s", document.filepath)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.FILE_NOT_FOUND,
        )

    if not hasattr(request.state, "user"):
        user_id = entity_id if entity_id else "public"
    else:
        user_id = entity_id if entity_id else request.state.user.get("id")

    loader = None
    try:
        loader, known_type, file_ext = get_loader(
            document.filename, document.file_content_type, file_path
        )
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(
            request.app.state.thread_pool, lambda: list(loader.lazy_load())
        )

        result = await store_data_in_vector_db(
            data,
            document.file_id,
            user_id,
            clean_content=file_ext == "pdf",
            executor=request.app.state.thread_pool,
        )

        if result and "error" not in result:
            return {
                "status": True,
                "file_id": document.file_id,
                "filename": document.filename,
                "known_type": known_type,
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "An error occurred while adding documents."),
            )
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in embed_local_file | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(e)
        if "No pandoc was found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.PANDOC_NOT_INSTALLED,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.DEFAULT(e),
            )
    finally:
        if loader is not None:
            cleanup_temp_encoding_file(loader)


@router.post("/embed")
async def embed_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file_id: str = Form(...),
    file: UploadFile = File(...),
    entity_id: str = Form(None),
):
    user_id = get_user_id(request, entity_id)
    validated_file_path = _make_unique_temp_path(user_id, file.filename)

    if validated_file_path is None:
        logger.warning("Path validation failed for embed: %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Invalid request"),
        )

    try:
        os.makedirs(os.path.dirname(validated_file_path), exist_ok=True)
        await save_upload_file_async(file, validated_file_path)
        
        file_ext = file.filename.split(".")[-1].lower()

        background_tasks.add_task(
            process_and_store_file_background,
            file.filename,
            file.content_type,
            validated_file_path,
            file_id,
            user_id,
            file_ext == "pdf",
            request.app.state.thread_pool,
        )

        return {
            "status": True,
            "message": "File uploaded successfully. Ingestion started in background.",
            "file_id": file_id,
            "filename": file.filename,
            "known_type": None,
        }
    except Exception as e:
        logger.error(
            "Error starting file processing: %s\nTraceback: %s",
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting file processing: {str(e)}",
        )


@router.get("/documents/{id}/context")
async def load_document_context(request: Request, id: str):
    ids = [id]
    try:
        if isinstance(vector_store, AsyncPgVector):
            existing_ids = await vector_store.get_filtered_ids(
                ids, executor=request.app.state.thread_pool
            )
            documents = await vector_store.get_documents_by_ids(
                ids, executor=request.app.state.thread_pool
            )
        else:
            existing_ids = vector_store.get_filtered_ids(ids)
            documents = vector_store.get_documents_by_ids(ids)

        if not all(id in existing_ids for id in ids):
            raise HTTPException(
                status_code=404, detail="The specified file_id was not found"
            )

        if not documents:
            raise HTTPException(
                status_code=404, detail="No document found for the given ID"
            )

        return process_documents(documents)
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in load_document_context | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error loading document context | Document ID: %s | Error: %s | Traceback: %s",
            id,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT(e),
        )


@router.post("/embed-upload")
async def embed_file_upload(
    request: Request,
    background_tasks: BackgroundTasks,
    file_id: str = Form(...),
    uploaded_file: UploadFile = File(...),
    entity_id: str = Form(None),
):
    user_id = get_user_id(request, entity_id)
    validated_temp_file_path = _make_unique_temp_path(user_id, uploaded_file.filename)

    if validated_temp_file_path is None:
        logger.warning(
            "Path validation failed for embed-upload: %s", uploaded_file.filename
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Invalid request"),
        )

    try:
        os.makedirs(os.path.dirname(validated_temp_file_path), exist_ok=True)
        await save_upload_file_async(uploaded_file, validated_temp_file_path)
        
        file_ext = uploaded_file.filename.split(".")[-1].lower()

        background_tasks.add_task(
            process_and_store_file_background,
            uploaded_file.filename,
            uploaded_file.content_type,
            validated_temp_file_path,
            file_id,
            user_id,
            file_ext == "pdf",
            request.app.state.thread_pool,
        )

        return {
            "status": True,
            "message": "File processed successfully. Ingestion started in background.",
            "file_id": file_id,
            "filename": uploaded_file.filename,
            "known_type": None,
        }
    except Exception as e:
        logger.error(
            "Error during file processing | File: %s | Error: %s | Traceback: %s",
            uploaded_file.filename,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error during file processing: {str(e)}",
        )


@router.post("/query_multiple")
async def query_embeddings_by_file_ids(request: Request, body: QueryMultipleBody):
    if hasattr(request.state, "user"):
        allowed_users = [request.state.user.get("id"), "public"]
        if body.entity_id:
            if body.entity_id in allowed_users:
                allowed_users = [body.entity_id]
            else:
                allowed_users = ["public"]
    else:
        allowed_users = [body.entity_id] if body.entity_id else ["public"]

    try:
        embedding = get_cached_query_embedding(body.query)

        # SQL-level filter: file_id matches list AND user_id matches allowed user(s)
        db_filter = {
            "file_id": {"$in": body.file_ids},
            "user_id": {"$in": allowed_users}
        }

        if isinstance(vector_store, AsyncPgVector):
            documents = await vector_store.asimilarity_search_with_score_by_vector(
                embedding,
                k=body.k,
                filter=db_filter,
                executor=request.app.state.thread_pool,
            )
        else:
            documents = vector_store.similarity_search_with_score_by_vector(
                embedding, k=body.k, filter=db_filter
            )

        documents = _apply_distance_threshold(documents)

        if not body.generate_answer:
            if not documents:
                raise HTTPException(
                    status_code=404, detail="No documents found for the given query"
                )
            return documents

        answer = None
        if not documents:
            answer = "No relevant document chunks found to answer your query."
        else:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                from langchain_core.messages import HumanMessage
                from app.config import RAG_GOOGLE_API_KEY, GOOGLE_API_KEY
                
                api_key = RAG_GOOGLE_API_KEY or GOOGLE_API_KEY
                if not api_key:
                    answer = "Error: Google Gemini API key is missing. Please add it to your .env file."
                else:
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash",
                        google_api_key=api_key,
                    )
                    context_text = "\n---\n".join([doc.page_content for doc, _ in documents])
                    prompt = (
                        "You are a helpful assistant that answers questions based strictly on the provided context.\n"
                        "If the context does not contain enough information to answer the question, state that you do not know.\n\n"
                        f"Context:\n{context_text}\n\n"
                        f"Question: {body.query}\n\n"
                        "Answer:"
                    )
                    response = await llm.ainvoke([HumanMessage(content=prompt)])
                    answer = response.content
            except Exception as e:
                logger.error(f"Error generating AI answer: {e}")
                answer = f"Error generating AI answer: {str(e)}"

        return {
            "answer": answer,
            "documents": documents
        }
    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in query_embeddings_by_file_ids | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error in query multiple embeddings | File IDs: %s | Query: %s | Error: %s | Traceback: %s",
            body.file_ids,
            body.query,
            str(e),
            traceback.format_exc(),
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/text")
async def extract_text_from_file(
    request: Request,
    file_id: str = Form(...),
    file: UploadFile = File(...),
    entity_id: str = Form(None),
):
    """
    Extract text content from an uploaded file without creating embeddings.
    Returns the raw text content for text parsing purposes.
    """
    user_id = get_user_id(request, entity_id)
    validated_temp_file_path = _make_unique_temp_path(user_id, file.filename)

    if validated_temp_file_path is None:
        logger.warning("Path validation failed for text extraction: %s", file.filename)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ERROR_MESSAGES.DEFAULT("Invalid request"),
        )

    try:
        os.makedirs(os.path.dirname(validated_temp_file_path), exist_ok=True)
        await save_upload_file_async(file, validated_temp_file_path)
        data, known_type, file_ext = await load_file_content(
            file.filename,
            file.content_type,
            validated_temp_file_path,
            request.app.state.thread_pool,
            raw_text=True,
        )

        text_content = extract_text_from_documents(data, file_ext)

        return {
            "text": text_content,
            "file_id": file_id,
            "filename": file.filename,
            "known_type": known_type,
        }

    except HTTPException as http_exc:
        logger.error(
            "HTTP Exception in extract_text_from_file | Status: %d | Detail: %s",
            http_exc.status_code,
            http_exc.detail,
        )
        raise http_exc
    except Exception as e:
        logger.error(
            "Error during text extraction | File: %s | Error: %s | Traceback: %s",
            file.filename,
            str(e),
            traceback.format_exc(),
        )
        if "No pandoc was found" in str(e):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ERROR_MESSAGES.PANDOC_NOT_INSTALLED,
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error during text extraction: {str(e)}",
            )
    finally:
        await cleanup_temp_file_async(validated_temp_file_path)
