"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from .task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL
from .jina_embeddings import embed_texts


def _get_collection():
    """Mở Chroma collection; trả về None nếu Task 4 chưa index dữ liệu."""
    if not CHROMA_DIR.exists():
        return None

    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return client.get_collection(COLLECTION_NAME)
    except ValueError:
        return None


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not query or not query.strip() or top_k <= 0:
        return []

    collection = _get_collection()
    if collection is None:
        return []

    result_count = min(top_k, collection.count())
    if result_count == 0:
        return []

    query_vector = embed_texts(
        [query],
        model=EMBEDDING_MODEL,
        task="retrieval.query",
    )[0]
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=result_count,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for content, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # With Chroma's cosine space, distance = 1 - cosine similarity.
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        output.append(
            {
                "content": content,
                "score": round(score, 4),
                "metadata": metadata or {},
            }
        )

    output.sort(key=lambda item: item["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("what is the tuition fee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
