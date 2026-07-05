from __future__ import annotations

import time
import boto3
from loguru import logger

from config.settings import (
    AWS_ACCESS_KEY,
    AWS_SECRET_KEY,
    AWS_REGION,
    DATA_LAKE_BUCKET,
    DATA_LAKE_PREFIX,
)
from source.ingestion.processor import IngestionProcessor


class S3Watcher:
    def __init__(
        self,
        bucket_name: str = DATA_LAKE_BUCKET,
        prefix: str = DATA_LAKE_PREFIX,
    ):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.processor = IngestionProcessor()
        self.processed_keys: set[str] = set()

        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION,
        )

    def watch(self, interval: int = 60) -> None:
        logger.info(f"S3 Watcher started — bucket={self.bucket_name}, prefix={self.prefix!r}")
        while True:
            self._poll()
            time.sleep(interval)

    def _poll(self) -> None:
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.prefix,
            )
            for obj in response.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".pdf") and key not in self.processed_keys:
                    logger.info(f"New file detected: {key}")
                    self.processor.process_file(key)
                    self.processed_keys.add(key)
        except Exception as e:
            logger.error(f"S3 polling error: {e}")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    S3Watcher().watch()
