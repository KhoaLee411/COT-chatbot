import os
from dotenv import load_dotenv

load_dotenv()


# ── AWS / S3 Vectors ────────────────────────────────────────────
AWS_REGION: str = os.getenv("S3_VECTOR_REGION", "us-east-1")
AWS_ACCESS_KEY: str | None = os.getenv("S3_VECTOR_ACCESS_KEY")
AWS_SECRET_KEY: str | None = os.getenv("S3_VECTOR_SECRET_KEY")

S3_VECTOR_BUCKET: str = os.getenv("S3_VECTOR_BUCKET", "cot-chatbot-vectors")
S3_VECTOR_INDEX: str = os.getenv("S3_VECTOR_INDEX", "chatbot-index")
S3_CONTENT_BUCKET: str = os.getenv("S3_CONTENT_BUCKET", "cot-ai-datalake")
S3_CONTENT_PREFIX: str = os.getenv("S3_CONTENT_PREFIX", "chunks/")

# ── S3 Data Lake (watcher) ───────────────────────────────────────
DATA_LAKE_BUCKET: str = os.getenv(
    "DATA_LAKE_BUCKET",
    "cot-ai-datalake",
)
DATA_LAKE_PREFIX: str = os.getenv("DATA_LAKE_PREFIX", "admin/")

# ── Gemini ───────────────────────────────────────────────────────
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
GEMINI_CHAT_MODEL: str = os.getenv("GEMINI_CHAT_MODEL", "models/gemini-2.5-flash")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
EMBED_DIM: int = int(os.getenv("EMBED_DIM", "1536"))

# ── LlamaParse ───────────────────────────────────────────────────
LLAMAPARSE_API_KEY: str | None = os.getenv("LLAMAPARSE_API_KEY")

# ── Chunking ─────────────────────────────────────────────────────
CHUNK_SIZE: int = 4_000
CHUNK_OVERLAP: int = 400
EMBED_BATCH: int = 5
PUT_BATCH: int = 500

# ── Retrieval ────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = 5
RETRIEVAL_DISTANCE_THRESHOLD: float = 0.4

# ── API ──────────────────────────────────────────────────────────
API_URL: str = os.getenv("API_URL", "http://localhost:8000")

# ── Backend ───────────────────────────────────────────────────────
BACKEND_HOST: str = os.getenv("BACKEND_HOST", "http://localhost:5726/api/v3/en")
FAQ_PATH: str = "/faq/public"
FAQ_S3_PREFIX: str = "faq/"