"""
Ingest tất cả PDF và Markdown từ một thư mục local.
"""
from __future__ import annotations

import glob
import os

from dotenv import load_dotenv
from loguru import logger

from source.ingestion.processor import IngestionProcessor

load_dotenv()

SUPPORTED_PATTERNS = ("*.pdf", "*.md")


def ingest_local_directory(directory_path: str) -> None:
    processor = IngestionProcessor()

    files: list[str] = []
    for pattern in SUPPORTED_PATTERNS:
        files.extend(glob.glob(os.path.join(directory_path, pattern)))

    if not files:
        logger.warning(f"No supported files found in {directory_path!r}")
        return

    logger.info(f"Found {len(files)} file(s) to ingest from {directory_path!r}")
    for file_path in files:
        try:
            processor.process_file(file_path)
        except Exception as e:
            logger.error(f"Failed to ingest {file_path!r}: {e}")


if __name__ == "__main__":
    DATA_DIR = os.getenv("INGEST_DATA_DIR", "./data")
    ingest_local_directory(DATA_DIR)
