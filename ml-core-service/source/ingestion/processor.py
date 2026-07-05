"""
IngestionProcessor: parse → chunk → embed → upsert vào S3 Vectors.
"""
from __future__ import annotations

import os
import re
import uuid
from typing import List, Dict, Any

import boto3
import google.generativeai as genai
from llama_parse import LlamaParse
from loguru import logger

from config.settings import (
    AWS_ACCESS_KEY,
    AWS_SECRET_KEY,
    AWS_REGION,
    S3_VECTOR_BUCKET,
    S3_VECTOR_INDEX,
    S3_CONTENT_BUCKET,
    S3_CONTENT_PREFIX,
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    EMBED_DIM,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_BATCH,
    PUT_BATCH,
    LLAMAPARSE_API_KEY,
)


class IngestionProcessor:
    def __init__(self):
        self.vector_bucket = S3_VECTOR_BUCKET
        self.vector_index = S3_VECTOR_INDEX
        self.content_bucket = S3_CONTENT_BUCKET

        self.vector_client = boto3.client(
            "s3vectors",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION,
        )
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION,
        )

        self._ensure_vector_store()

        self.parser = LlamaParse(
            api_key=LLAMAPARSE_API_KEY,
            result_type="markdown",
        )

        genai.configure(api_key=GEMINI_API_KEY)
        logger.info(f"Embedding model: {EMBEDDING_MODEL} | dim={EMBED_DIM}")

    # ── Setup ────────────────────────────────────────────────────

    def _ensure_vector_store(self) -> None:
        try:
            self.vector_client.create_vector_bucket(vectorBucketName=self.vector_bucket)
            logger.info(f"Vector bucket ready: {self.vector_bucket}")
        except Exception as e:
            logger.debug(f"Vector bucket already exists or error: {e}")

        try:
            self.vector_client.create_index(
                vectorBucketName=self.vector_bucket,
                indexName=self.vector_index,
                dimension=EMBED_DIM,
                distanceMetric="cosine",
                dataType="float32",
            )
            logger.info(f"Vector index ready: {self.vector_index}")
        except Exception as e:
            logger.debug(f"Vector index already exists or error: {e}")

    # ── Public entry point ───────────────────────────────────────

    def process_file(self, file_path: str) -> None:
        """Parse → chunk → embed → upsert vector + lưu content vào S3."""
        logger.info(f"Processing: {file_path}")

        text = self._parse(file_path)
        if not text:
            logger.warning(f"No text extracted from {file_path}")
            return

        chunks = self._smart_chunk(text)
        logger.info(f"{len(chunks)} chunks from {os.path.basename(file_path)}")

        records = self._embed_chunks(chunks, source=file_path)
        self._put_vectors(records)
        logger.success(f"Done: {file_path} → {len(records)} vectors indexed.")

    def process_text(self, text: str, source: str) -> int:
        if not text or not text.strip():
            logger.warning(f"Empty text for source: {source!r}")
            return 0
 
        chunks = self._smart_chunk(text)
        logger.info(f"{len(chunks)} chunks from {source!r}")
 
        records = self._embed_chunks(chunks, source=source)
        self._put_vectors(records)
        logger.success(f"Done: {source!r} → {len(records)} vectors indexed.")
        return len(records)

    # ── Parse ────────────────────────────────────────────────────

    def _parse(self, file_path: str) -> str:
        if file_path.endswith(".pdf"):
            docs = self.parser.load_data(file_path)
            return "\n\n".join(d.text for d in docs)
        if file_path.endswith(".md") or file_path.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        logger.warning(f"Unsupported format: {file_path}")
        return ""

    # ── Chunking ─────────────────────────────────────────────────

    def _smart_chunk(self, text: str) -> List[str]:
        paragraphs = re.split(r"\n(?=#{1,3} )|\n{2,}", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks: List[str] = []
        current = ""

        for para in paragraphs:
            if len(para) > CHUNK_SIZE:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(self._split_by_sentence(para))
                continue

            if len(current) + len(para) + 2 > CHUNK_SIZE:
                chunks.append(current.strip())
                current = current[-CHUNK_OVERLAP:] + "\n\n" + para
            else:
                current = (current + "\n\n" + para).strip()

        if current:
            chunks.append(current.strip())

        return [c for c in chunks if len(c) > 50]

    def _split_by_sentence(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, current = [], ""
        for sent in sentences:
            if len(current) + len(sent) > CHUNK_SIZE:
                if current:
                    chunks.append(current.strip())
                current = current[-CHUNK_OVERLAP:] + " " + sent
            else:
                current = (current + " " + sent).strip()
        if current:
            chunks.append(current.strip())
        return chunks

    # ── Embedding ────────────────────────────────────────────────

    def _embed_chunks(self, chunks: List[str], source: str) -> List[Dict[str, Any]]:
        total = len(chunks)
        records: List[Dict[str, Any]] = []

        for batch_start in range(0, total, EMBED_BATCH):
            batch = chunks[batch_start : batch_start + EMBED_BATCH]
            for i, chunk in enumerate(batch):
                chunk_idx = batch_start + i
                chunk_key = str(uuid.uuid4())
                try:
                    filename = os.path.basename(source)
                    self.s3_client.put_object(
                        Bucket=self.content_bucket,
                        Key=f"{S3_CONTENT_PREFIX}{filename}/{chunk_key}.txt",
                        Body=chunk.encode("utf-8"),
                        ContentType="text/plain; charset=utf-8",
                    )
                    result = genai.embed_content(
                        model=EMBEDDING_MODEL,
                        content=chunk,
                        task_type="retrieval_document",
                        output_dimensionality=EMBED_DIM,
                    )
                    records.append({
                        "key": chunk_key,
                        "data": {"float32": result["embedding"]},
                        "metadata": {
                            "chunk_key": chunk_key,
                            "filename": os.path.basename(source),
                            "chunk_index": chunk_idx,
                            "chunk_total": total,
                        },
                    })
                except Exception as e:
                    logger.error(f"Embed failed for chunk {chunk_idx}: {e}")

            logger.debug(f"Embedded {min(batch_start + EMBED_BATCH, total)}/{total}")

        return records

    # ── Upsert ───────────────────────────────────────────────────

    def _put_vectors(self, records: List[Dict[str, Any]]) -> None:
        for i in range(0, len(records), PUT_BATCH):
            batch = records[i : i + PUT_BATCH]
            self.vector_client.put_vectors(
                vectorBucketName=self.vector_bucket,
                indexName=self.vector_index,
                vectors=batch,
            )
            logger.debug(f"Upserted {i + len(batch)}/{len(records)} vectors")

    # ── Delete ───────────────────────────────────────────────────

    def delete_file(self, filename: str) -> None:
        """Xóa toàn bộ vector và chunk content liên quan đến một filename."""
        try:
            # 1. Delete chunk contents in S3
            prefix = f"{S3_CONTENT_PREFIX}{filename}/"
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.content_bucket, Prefix=prefix)
            
            delete_us = {"Objects": []}
            for item in pages.search("Contents"):
                if item:
                    delete_us["Objects"].append({"Key": item["Key"]})
                    if len(delete_us["Objects"]) >= 1000:
                        self.s3_client.delete_objects(Bucket=self.content_bucket, Delete=delete_us)
                        delete_us = {"Objects": []}
            
            if len(delete_us["Objects"]) > 0:
                self.s3_client.delete_objects(Bucket=self.content_bucket, Delete=delete_us)
            logger.success(f"Deleted S3 chunk contents for {filename}")
        except Exception as e:
            logger.error(f"Failed to delete S3 chunk contents for {filename}: {e}")

        # 2. Delete vectors in VectorDB
        try:
            self.vector_client.delete_vectors(
                vectorBucketName=self.vector_bucket,
                indexName=self.vector_index,
                filter={"filename": filename}
            )
            logger.success(f"Deleted vectors for {filename}")
        except Exception as e:
            logger.error(f"Failed to delete vectors for {filename}: {e}")

