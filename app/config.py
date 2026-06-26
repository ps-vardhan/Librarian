# ID-Rag/app/config.py
import os
import logging
import urllib.parse
from enum import Enum
from dotenv import find_dotenv, load_dotenv

from app.services.vector_store.factory import get_vector_store

load_dotenv(find_dotenv())

class VectorDBType(Enum):
    PGVECTOR = "pgvector"

class EmbeddingsProvider(Enum):
    GOOGLE_GENAI = "google_genai"

def get_env_variable(
    var_name: str, default_value: str = None, required: bool = False
) -> str:
    value = os.getenv(var_name)
    if value is None:
        if default_value is None and required:
            raise ValueError(f"Environment variable '{var_name}' not found.")
        return default_value
    return value

# App Configuration
RAG_HOST = os.getenv("RAG_HOST", "0.0.0.0")
RAG_PORT = int(os.getenv("RAG_PORT", 8000))
RAG_UPLOAD_DIR = get_env_variable("RAG_UPLOAD_DIR", "./uploads/")
RAG_SERVE_UI = get_env_variable("RAG_SERVE_UI", "False").lower() in ("true", "1", "yes", "on")
if not os.path.exists(RAG_UPLOAD_DIR):
    os.makedirs(RAG_UPLOAD_DIR, exist_ok=True)

# Database Configuration
VECTOR_DB_TYPE = VectorDBType.PGVECTOR
POSTGRES_USE_UNIX_SOCKET = (
    get_env_variable("POSTGRES_USE_UNIX_SOCKET", "False").lower() == "true"
)
POSTGRES_DB = get_env_variable("POSTGRES_DB", "mydatabase")
POSTGRES_USER = get_env_variable("POSTGRES_USER", "myuser")
POSTGRES_PASSWORD = get_env_variable("POSTGRES_PASSWORD", "mypassword")
DB_HOST = get_env_variable("DB_HOST", "db")
DB_PORT = get_env_variable("DB_PORT", "5432")

# Smart fallback: if running locally outside Docker (detected via /.dockerenv existence),
# default "db" host falls back to "localhost" and maps to compose port "5433".
if DB_HOST == "db" and not os.path.exists("/.dockerenv"):
    DB_HOST = "localhost"
    if DB_PORT == "5432":
        DB_PORT = "5433"
PGVECTOR_CREATE_EXTENSION = get_env_variable(
    "PGVECTOR_CREATE_EXTENSION", "True"
).lower() in ("true", "1", "yes", "on")
PG_POOL_PRE_PING = get_env_variable("PG_POOL_PRE_PING", "True").lower() in (
    "true",
    "1",
    "yes",
    "on",
)
PG_POOL_RECYCLE = int(get_env_variable("PG_POOL_RECYCLE", "-1"))
POSTGRES_SCHEMA = get_env_variable("POSTGRES_SCHEMA", None) or None
COLLECTION_NAME = get_env_variable("COLLECTION_NAME", "testcollection")

CHUNK_SIZE = int(get_env_variable("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(get_env_variable("CHUNK_OVERLAP", "100"))

# Build DSN and SQLAlchemy connection string
if POSTGRES_USE_UNIX_SOCKET:
    connection_suffix = f"{urllib.parse.quote_plus(POSTGRES_USER)}:{urllib.parse.quote_plus(POSTGRES_PASSWORD)}@/{urllib.parse.quote_plus(POSTGRES_DB)}?host={urllib.parse.quote_plus(DB_HOST)}"
else:
    connection_suffix = f"{urllib.parse.quote_plus(POSTGRES_USER)}:{urllib.parse.quote_plus(POSTGRES_PASSWORD)}@{DB_HOST}:{DB_PORT}/{urllib.parse.quote_plus(POSTGRES_DB)}"

CONNECTION_STRING = f"postgresql+psycopg2://{connection_suffix}"
DSN = f"postgresql://{connection_suffix}"

# Logger Configuration
logger = logging.getLogger("ID-Rag")
debug_mode = os.getenv("DEBUG_RAG_API", "False").lower() in ("true", "1", "yes", "y", "t", "m")
if debug_mode:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Credentials
GOOGLE_API_KEY = get_env_variable("GOOGLE_API_KEY", "")
GOOGLE_KEY = get_env_variable("GOOGLE_KEY", GOOGLE_API_KEY)
RAG_GOOGLE_API_KEY = get_env_variable("RAG_GOOGLE_API_KEY", GOOGLE_KEY)

# Relevance Threshold
RAG_DISTANCE_THRESHOLD = 0.36
_distance_threshold_raw = get_env_variable("RAG_DISTANCE_THRESHOLD", None)
if _distance_threshold_raw not in (None, ""):
    RAG_DISTANCE_THRESHOLD = float(_distance_threshold_raw)

# Embeddings
def init_embeddings(model):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    api_key = RAG_GOOGLE_API_KEY
    if not api_key:
        api_key = "dummy_key"
    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=api_key,
    )

EMBEDDINGS_PROVIDER = EmbeddingsProvider.GOOGLE_GENAI
EMBEDDINGS_MODEL = get_env_variable("EMBEDDINGS_MODEL", "gemini-embedding-001")

embeddings = init_embeddings(EMBEDDINGS_MODEL)
logger.info(f"Initialized embeddings of type: {type(embeddings)}")

class LazyVectorStoreProxy:
    def __init__(self):
        self._instance = None

    def _get_instance(self):
        if self._instance is None:
            self._instance = get_vector_store(
                connection_string=CONNECTION_STRING,
                embeddings=embeddings,
                collection_name=COLLECTION_NAME,
                mode="async",
                create_extension=PGVECTOR_CREATE_EXTENSION,
                pool_pre_ping=PG_POOL_PRE_PING,
                pool_recycle=PG_POOL_RECYCLE,
                schema=POSTGRES_SCHEMA,
            )
        return self._instance

    @property
    def __class__(self):
        try:
            return self._get_instance().__class__
        except Exception:
            from app.services.vector_store.async_pg_vector import AsyncPgVector
            return AsyncPgVector

    def __getattr__(self, name):
        return getattr(self._get_instance(), name)


class LazyRetrieverProxy:
    def __init__(self, proxy):
        self._proxy = proxy
        self._instance = None

    def _get_instance(self):
        if self._instance is None:
            self._instance = self._proxy.as_retriever()
        return self._instance

    def __getattr__(self, name):
        return getattr(self._get_instance(), name)


# Lazy Vector Store & Retriever Proxy initialization
vector_store = LazyVectorStoreProxy()
retriever = LazyRetrieverProxy(vector_store)
