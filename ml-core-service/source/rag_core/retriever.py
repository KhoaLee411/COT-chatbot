"""
S3Retriever: embed query → tìm vector gần nhất → fetch nội dung từ S3.
"""
from __future__ import annotations

import boto3
import google.generativeai as genai
from loguru import logger
from typing import List, Dict, Any

from config.settings import (
    EMBEDDING_MODEL,
    EMBED_DIM,
    RETRIEVAL_TOP_K,
    RETRIEVAL_DISTANCE_THRESHOLD,
    S3_CONTENT_BUCKET,
    S3_CONTENT_PREFIX
)
from langsmith import traceable


class S3Retriever:
    def __init__(
        self,
        region_name: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        index_name: str,
        gemini_api_key: str,
        content_bucket: str = S3_CONTENT_BUCKET,
    ):
        self.bucket_name = bucket_name
        self.index_name = index_name
        self.content_bucket = content_bucket

        self.vector_client = boto3.client(
            "s3vectors",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name,
        )
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name,
        )

        genai.configure(api_key=gemini_api_key)
        self.embedding_model = EMBEDDING_MODEL

    # ── Embedding ────────────────────────────────────────────────

    @traceable(run_type="llm", name="GeminiEmbedQuery")
    def _embed_query(self, text: str) -> List[float]:
        from langsmith import get_current_run_tree

        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_query",
            output_dimensionality=EMBED_DIM,
        )

        current_run = get_current_run_tree()
        if current_run:
            current_run.add_metadata({
                "ls_provider": "google",
                "ls_model_name": self.embedding_model.replace("models/", "")
            })

        return result["embedding"]

    # ── Fetch content ────────────────────────────────────────────

    def _fetch_chunk_content(self, chunk_key: str, filename: str = None) -> str | None:
        try:
            if filename:
                try:
                    obj = self.s3_client.get_object(
                        Bucket=self.content_bucket,
                        Key=f"{S3_CONTENT_PREFIX}{filename}/{chunk_key}.txt",
                    )
                    return obj["Body"].read().decode("utf-8")
                except self.s3_client.exceptions.NoSuchKey:
                    pass
            
            obj = self.s3_client.get_object(
                Bucket=self.content_bucket,
                Key=f"{S3_CONTENT_PREFIX}{chunk_key}.txt",
            )
            return obj["Body"].read().decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to fetch chunk {chunk_key!r}: {e}")
            return None

    # ── Public API ───────────────────────────────────────────────

    @traceable(run_type="retriever", name="S3VectorSearch")
    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[Dict[str, Any]]:
        query_vector = self._embed_query(query)

        try:
            response = self.vector_client.query_vectors(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                queryVector={"float32": query_vector},
                topK=top_k,
                returnMetadata=True,
                returnDistance=True,
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

        hits = response.get("vectors", response.get("matches", []))

        results: List[Dict[str, Any]] = []
        for hit in hits:
            distance = hit.get("distance", 1.0)
            if distance > RETRIEVAL_DISTANCE_THRESHOLD:
                continue  

            metadata = hit.get("metadata", {})
            chunk_key = metadata.get("chunk_key")
            filename = metadata.get("filename")

            content = ""
            if chunk_key:
                content = self._fetch_chunk_content(chunk_key, filename) or ""
            else:
                content = metadata.get("content", "")

            results.append({
                "content": content,
                "metadata": metadata,
                "score": distance,
            })

        logger.debug(f"Retrieval: {len(results)}/{len(hits)} hits passed threshold.")
        return results
