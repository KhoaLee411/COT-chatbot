from __future__ import annotations

import os
import hashlib
import json
import re
from typing import List, Dict, Any

import requests
from loguru import logger

from config.settings import BACKEND_HOST, FAQ_PATH
from source.ingestion.processor import IngestionProcessor


# ── Junk filter ───────────────────────────────────────────────────────────────

# Các pattern question/answer là placeholder — bỏ qua khi ingest
_JUNK_PATTERNS = re.compile(
    r"^(question|answer|questionaire|test|123|sample|placeholder|n/a|tbd|xxx)$",
    re.IGNORECASE,
)


def _is_junk_item(item: Dict[str, Any]) -> bool:
    """Trả về True nếu item là placeholder / rác."""
    q = str(item.get("question") or "").strip()
    a = str(item.get("answer") or "").strip()

    if not q or not a:
        return True
    if _JUNK_PATTERNS.match(q) or _JUNK_PATTERNS.match(a):
        return True
    # answer chưa điền (giống hệt question hoặc quá ngắn)
    if a == q or len(a) < 10:
        return True

    return False


# ── Formatter ─────────────────────────────────────────────────────────────────


def _format_topic(topic: Dict[str, Any]) -> str | None:
    """
    Format 1 topic thành markdown text để đưa vào chunker.
    """
    title = str(topic.get("title") or "").strip()
    topic_type = str(topic.get("topicType") or "").strip()
    items = topic.get("items") or []

    valid_items = [item for item in items if not _is_junk_item(item)]

    if not valid_items:
        logger.debug(f"Skipping empty/junk topic: {title!r} ({topic_type})")
        return None

    lines = [f"# {title} ({topic_type})\n"]
    for item in valid_items:
        q = str(item["question"]).strip()
        a = str(item["answer"]).strip()
        lines.append(f"Q: {q}\nA: {a}\n")

    return "\n".join(lines)


# ── Main ingester ─────────────────────────────────────────────────────────────


class FAQIngester:
    def __init__(self):
        self.processor = IngestionProcessor()
        self.faq_url = f"{BACKEND_HOST}{FAQ_PATH}"
        self.state_file = ".faq_ingestion_state.json"
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, str]:
        """Load state chứa mã hash của các topic đã ingest."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load state file: {e}")
        return {}

    def _save_state(self):
        """Lưu lại mã hash của các topic để đối chiếu cho lần sau."""
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state file: {e}")

    def fetch(self) -> List[Dict[str, Any]]:
        """Fetch FAQ data từ backend API."""
        logger.info(f"Fetching FAQ from {self.faq_url}")
        try:
            response = requests.get(self.faq_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Fetched {len(data)} topics from FAQ API.")
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch FAQ: {e}")
            raise

    def ingest(self) -> Dict[str, Any]:
        """
        Fetch → filter → format → ingest toàn bộ FAQ (chỉ ingest những topic có thay đổi).

        Returns: summary dict với số topic và vector đã index.
        """
        topics = self.fetch()

        total_vectors = 0
        ingested_topics = 0
        skipped_topics = 0
        unchanged_topics = 0

        for topic in topics:
            title = topic.get("title", "unknown")
            topic_type = topic.get("topicType", "unknown")
            source = f"faq/{topic_type}/{title}"

            text = _format_topic(topic)
            if text is None:
                skipped_topics += 1
                continue

            # Tính MD5 hash của đoạn text markdown
            text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
            
            # Nếu hash không đổi -> Không làm gì cả
            if self.state.get(source) == text_hash:
                logger.debug(f"Skipping unchanged topic: {source!r}")
                unchanged_topics += 1
                continue

            try:
                count = self.processor.process_text(text, source=source)
                total_vectors += count
                ingested_topics += 1
                # Cập nhật hash mới vào state sau khi thành công
                self.state[source] = text_hash
            except Exception as e:
                logger.error(f"Failed to ingest topic {source!r}: {e}")
                skipped_topics += 1

        # Lưu state ra file sau khi xử lý xong
        self._save_state()

        summary = {
            "total_topics": len(topics),
            "ingested_topics": ingested_topics,
            "unchanged_topics": unchanged_topics,
            "skipped_topics": skipped_topics,
            "total_vectors": total_vectors,
        }
        logger.success(f"FAQ ingestion complete: {summary}")
        return summary