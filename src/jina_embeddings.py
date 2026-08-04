"""Shared client for the Jina Embeddings API."""

import os
from pathlib import Path
import time

import requests
from dotenv import load_dotenv


JINA_EMBEDDINGS_URL = "https://api.jina.ai/v1/embeddings"


def embed_texts(
    texts: list[str],
    *,
    model: str,
    task: str,
    batch_size: int = 64,
) -> list[list[float]]:
    """Embed texts through Jina and preserve the caller's input order."""
    if not texts:
        return []

    load_dotenv(Path(__file__).parent.parent / ".env")
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise RuntimeError("Thiếu JINA_API_KEY. Hãy thêm key vào file .env.")

    embeddings: list[list[float]] = []
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        payload = {
            "model": model,
            "input": batch,
            "task": task,
            "normalized": True,
            "embedding_type": "float",
        }
        response = None
        for attempt in range(3):
            try:
                response = requests.post(
                    JINA_EMBEDDINGS_URL,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt == 2:
                    raise RuntimeError("Không thể gọi Jina Embeddings API sau 3 lần thử.") from exc
                time.sleep(2**attempt)
        try:
            data = response.json()["data"]
            data.sort(key=lambda item: item["index"])
            embeddings.extend(item["embedding"] for item in data)
        except (KeyError, ValueError) as exc:
            detail = response.text[:500]
            raise RuntimeError(f"Jina Embeddings API lỗi: {detail}") from exc

    if len(embeddings) != len(texts):
        raise RuntimeError("Jina API trả về thiếu embeddings.")
    return embeddings
