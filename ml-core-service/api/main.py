from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, AsyncGenerator
import asyncio

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from loguru import logger
logger.add("api_error.log", backtrace=True, diagnose=True)
from pydantic import BaseModel

from config.settings import (
    AWS_REGION,
    AWS_ACCESS_KEY,
    AWS_SECRET_KEY,
    S3_VECTOR_BUCKET,
    S3_VECTOR_INDEX,
    GEMINI_API_KEY,
)
from source.agent.graph import build_graph
from source.rag_core.retriever import S3Retriever
from source.rag_core.generator import GeminiGenerator
from langsmith import traceable
from langsmith.run_trees import RunTree


# ── App state ─────────────────────────────────────────────────────────────────

class AppState:
    retriever: S3Retriever
    generator: GeminiGenerator
    agent: Any


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing retriever and generator...")
    app_state.retriever = S3Retriever(
        region_name=AWS_REGION,
        access_key=AWS_ACCESS_KEY,
        secret_key=AWS_SECRET_KEY,
        bucket_name=S3_VECTOR_BUCKET,
        index_name=S3_VECTOR_INDEX,
        gemini_api_key=GEMINI_API_KEY,
    )
    app_state.generator = GeminiGenerator()
    app_state.agent = build_graph(app_state.retriever, app_state.generator)
    logger.info("Agent ready.")
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Coach On Tap AI Agent API", lifespan=lifespan)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict[str, Any]]] = []
    user_id: Optional[str] = "anonymous"
    session_id: Optional[str] = "session_default"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(request: ChatRequest):
    @traceable(name="ProcessChat", run_type="chain")
    async def event_generator(query: str, chat_history: list):
        try:
            # Bước 1: Chạy graph (guardrails → analyzer → retriever → generator)
            state = await asyncio.to_thread(
                app_state.agent.invoke,
                {
                    "query": query,
                    "chat_history": chat_history,
                    "chunks": [],
                    "is_malicious": False,
                    "is_nonsense": False,
                    "is_greeting": False,
                    "is_out_of_scope": False,
                },
                config={
                    "metadata": {
                        "user_id": request.user_id,
                        "session_id": request.user_id, # 1 user = 1 conversation
                    },
                    "tags": ["chatbot_api", f"user:{request.user_id}"]
                }
            )

            # Bước 2: Nếu graph đã xử lý và set answer (blocked cases)
            # malicious / nonsense / out_of_scope / no chunks đều đã có answer
            blocked = (
                state.get("is_malicious")
                or state.get("is_nonsense")
                or state.get("is_greeting")
                or state.get("is_out_of_scope")
                or not state.get("chunks")
            )
            if blocked:
                yield state.get("answer", "")
                return

            # Bước 3: In-scope + có chunks → stream từ generator
            for token in app_state.generator.generate_stream(
                query=state.get("standalone_query") or query,
                context=state["chunks"],
                chat_history=chat_history,
                langsmith_extra={
                    "tags": ["chatbot_api", f"user:{request.user_id}"],
                    "metadata": {
                        "user_id": request.user_id,
                        "session_id": request.user_id,
                    }
                }
            ):
                yield token

        except Exception as e:
            logger.error(f"Chat error: {e}")
            yield "An error occurred. Please try again or contact support@coachontap.co"

    return StreamingResponse(
        event_generator(
            request.query, 
            request.chat_history,
            langsmith_extra={
                "tags": ["chatbot_api", f"user:{request.user_id}"],
                "metadata": {
                    "user_id": request.user_id,
                    "session_id": request.user_id,
                }
            }
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@app.post("/faq")
async def ingest_faq():
    try:
        from source.ingestion.ingest_faq import FAQIngester
        summary = await asyncio.to_thread(FAQIngester().ingest)
        return {"status": "success", **summary}
    except Exception as e:
        logger.error(f"FAQ ingestion failed: {e}")
        return {"status": "error", "detail": str(e)}
 
@app.post("/sync-document")
async def sync_document(request: dict):
    file_key = request.get("file_key")
    if not file_key:
        return {"status": "error", "detail": "file_key is required"}
    try:
        from source.ingestion.processor import IngestionProcessor
        import boto3
        import os
        from config.settings import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION, DATA_LAKE_BUCKET
        
        s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION,
        )
        
        local_path = f"/tmp/{os.path.basename(file_key)}"
        logger.info(f"Downloading {file_key} from {DATA_LAKE_BUCKET} to {local_path}")
        s3.download_file(DATA_LAKE_BUCKET, file_key, local_path)
        
        processor = IngestionProcessor()
        await asyncio.to_thread(processor.process_file, local_path)
        
        # Clean up
        if os.path.exists(local_path):
            os.remove(local_path)
            
        return {"status": "success", "file_key": file_key}
    except Exception as e:
        logger.error(f"Document sync failed: {e}")
        return {"status": "error", "detail": str(e)}

@app.delete("/sync-document")
async def unsync_document(request: dict):
    file_key = request.get("file_key")
    if not file_key:
        return {"status": "error", "detail": "file_key is required"}
    try:
        import os
        from source.ingestion.processor import IngestionProcessor
        
        filename = os.path.basename(file_key)
        processor = IngestionProcessor()
        await asyncio.to_thread(processor.delete_file, filename)
        
        return {"status": "success", "file_key": file_key, "deleted": filename}
    except Exception as e:
        logger.error(f"Document unsync failed: {e}")
        return {"status": "error", "detail": str(e)}

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)