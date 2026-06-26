# ID-Rag/app/services/vector_store/factory.py
import logging
from typing import Any, List, Optional

from langchain_core.embeddings import Embeddings

from .async_pg_vector import AsyncPgVector
from .extended_pg_vector import ExtendedPgVector

logger = logging.getLogger(__name__)


def _parse_schemas(schema: str) -> List[str]:
    """Split POSTGRES_SCHEMA's comma-separated value into a clean list."""
    return [s.strip() for s in schema.split(",") if s.strip()]


def _build_search_path(schemas: List[str]) -> str:
    """Build a Postgres search_path value that includes every requested schema plus public."""
    parts = list(schemas)
    if "public" not in parts:
        parts.append("public")
    return ",".join(parts)


def _verify_schemas_exist(connection_string: str, schemas: List[str]) -> None:
    """Raise if the POSTGRES_SCHEMA config won't let the app write to the target schema or read types from its fallbacks."""
    from sqlalchemy import create_engine, text

    engine = create_engine(connection_string)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT s.schema_name, "
                    "       has_schema_privilege(s.schema_name, 'USAGE') AS has_usage, "
                    "       has_schema_privilege(s.schema_name, 'CREATE') AS has_create "
                    "FROM information_schema.schemata s "
                    "WHERE s.schema_name = ANY(:names)"
                ),
                {"names": schemas},
            ).fetchall()
            found = {row[0]: (row[1], row[2]) for row in rows}

            target_schema = schemas[0]
            fallback_schemas = schemas[1:]

            missing = [s for s in schemas if s not in found]
            no_usage = [s for s in schemas if s in found and not found[s][0]]
            no_create_target = (
                [target_schema]
                if target_schema in found and not found[target_schema][1]
                else []
            )

            problems = []
            if missing:
                problems.append(f"does not exist: {missing!r}")
            if no_usage:
                problems.append(f"role lacks USAGE on: {no_usage!r}")
            if no_create_target:
                problems.append(f"role lacks CREATE on target: {no_create_target!r}")

            if problems:
                hint_target = (missing or no_usage or no_create_target)[0]
                raise ValueError(
                    "POSTGRES_SCHEMA: " + "; ".join(problems) + ". "
                    "Create/grant out-of-band first."
                )
    finally:
        engine.dispose()


def get_vector_store(
    connection_string: str,
    embeddings: Embeddings,
    collection_name: str,
    mode: str = "sync",
    search_index: Optional[str] = None,
    create_extension: bool = True,
    pool_pre_ping: bool = True,
    pool_recycle: int = -1,
    schema: Optional[str] = None,
):
    """Create a vector store instance for the given mode."""
    engine_args: dict = {"pool_pre_ping": pool_pre_ping}
    if pool_recycle > 0:
        engine_args["pool_recycle"] = pool_recycle
    if schema:
        schemas = _parse_schemas(schema)
        if schemas:
            _verify_schemas_exist(connection_string, schemas)
            search_path = _build_search_path(schemas)
            engine_args["connect_args"] = {"options": f"-csearch_path={search_path}"}

    if mode == "sync":
        return ExtendedPgVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name=collection_name,
            use_jsonb=True,
            create_extension=create_extension,
            engine_args=engine_args,
        )
    elif mode == "async":
        return AsyncPgVector(
            connection_string=connection_string,
            embedding_function=embeddings,
            collection_name=collection_name,
            use_jsonb=True,
            create_extension=create_extension,
            engine_args=engine_args,
        )
    else:
        raise ValueError(
            "Invalid mode specified. Choose 'sync' or 'async'."
        )


def close_vector_store_connections(vector_store: Any) -> None:
    """Close connections held by the vector store and its backing clients."""
    engine = getattr(vector_store, "_bind", None)
    if engine is not None and hasattr(engine, "dispose"):
        try:
            engine.dispose()
            logger.info("SQLAlchemy engine disposed in ID-Rag")
        except Exception as e:
            logger.warning("Failed to dispose SQLAlchemy engine in ID-Rag: %s", e)
