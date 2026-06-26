# ID-Rag/app/utils/health.py
from app.config import VECTOR_DB_TYPE, VectorDBType
from app.services.database import pg_health_check


async def is_health_ok():
    if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
        return await pg_health_check()
    else:
        return True
