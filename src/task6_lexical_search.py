"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
_BM25_INDEX = None


def load_corpus() -> list[dict]:
    """
    Load toàn bộ markdown documents từ data/standardized/ và tách thành các chunks.
    """
    global CORPUS
    if CORPUS:
        return CORPUS

    corpus = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"

        # Tách nhỏ file theo paragraph để làm chunk BM25
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        for idx, p in enumerate(paragraphs):
            corpus.append({
                "content": p,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                    "chunk_id": idx
                }
            })

    CORPUS = corpus
    return CORPUS


def build_bm25_index(corpus: list[dict] = None):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    global _BM25_INDEX, CORPUS
    if corpus is None:
        corpus = load_corpus()

    CORPUS = corpus
    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    _BM25_INDEX = BM25Okapi(tokenized_corpus)
    return _BM25_INDEX


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    global _BM25_INDEX, CORPUS
    if _BM25_INDEX is None or not CORPUS:
        load_corpus()
        build_bm25_index(CORPUS)

    if not CORPUS or _BM25_INDEX is None:
        return []

    tokenized_query = query.lower().split()
    scores = _BM25_INDEX.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(scores[idx]),
                "metadata": CORPUS[idx]["metadata"]
            })
    return results


if __name__ == "__main__":
    print("Building BM25 index...")
    load_corpus()
    build_bm25_index()
    results = lexical_search("học phí", top_k=5)
    print(f"Found {len(results)} results:")
    for r in results:
        safe_content = r['content'][:100].replace('\n', ' ')
        print(f"[{r['score']:.3f}] Source: {r['metadata']['source']} | {safe_content}".encode("ascii", "replace").decode("ascii"))
