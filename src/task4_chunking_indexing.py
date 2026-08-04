"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho cả tiếng Việt lẫn tiếng Anh)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# TODO: Chọn chunking strategy và giải thích vì sao
CHUNK_SIZE = 500        # 500 ký tự/chunk giúp cân bằng giữa độ đầy đủ ngữ cảnh và tốc độ truy vấn.
CHUNK_OVERLAP = 50      # Giữ khoảng 10% nội dung giữa hai chunk để tránh mất thông tin ở ranh giới.
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"


# TODO: Chọn embedding model và giải thích
EMBEDDING_MODEL = "BAAI/bge-m3"  # bge-m3 hỗ trợ đa ngôn ngữ, đặc biệt tiếng Việt và tiếng Anh, chất lượng retrieval cao.
EMBEDDING_DIM = 1024

# TODO: Chọn vector store
VECTOR_STORE = "chromadb"  # ChromaDB là vector database local, persistent, không cần Docker và dễ tích hợp với LangChain.
COLLECTION_NAME = "university_services_docs"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    # TODO: Iterate qua STANDARDIZED_DIR, đọc .md files
    # documents = []
    # for md_file in STANDARDIZED_DIR.rglob("*.md"):
    #     content = md_file.read_text(encoding="utf-8")
    #     doc_type = "legal" if "legal" in str(md_file) else "news"
    #     documents.append({
    #         "content": content,
    #         "metadata": {"source": md_file.name, "type": doc_type}
    #     })
    # return documents

    documents = []

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")

        doc_type = "legal"
        if "news" in str(md_file).lower():
            doc_type = "news"

        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                },
            }
        )

    return documents



def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    # TODO: Implement chunking
    #
    # Ví dụ với RecursiveCharacterTextSplitter:
    # from langchain_text_splitters import RecursiveCharacterTextSplitter
    #
    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=CHUNK_SIZE,
    #     chunk_overlap=CHUNK_OVERLAP,
    #     separators=["\n\n", "\n", ". ", " ", ""]
    # )
    # chunks = []
    # for doc in documents:
    #     splits = splitter.split_text(doc["content"])
    #     for i, chunk_text in enumerate(splits):
    #         chunks.append({
    #             "content": chunk_text,
    #             "metadata": {**doc["metadata"], "chunk_index": i}
    #         })
    # return chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = []

    for doc in documents:

        splits = splitter.split_text(doc["content"])

        for idx, text in enumerate(splits):

            chunks.append(
                {
                    "content": text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": idx,
                    },
                }
            )

    return chunks



def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    # TODO: Implement embedding
    #
    # Ví dụ với sentence-transformers:
    # from sentence_transformers import SentenceTransformer
    #
    # model = SentenceTransformer(EMBEDDING_MODEL)
    # texts = [c["content"] for c in chunks]
    # embeddings = model.encode(texts, show_progress_bar=True)
    # for chunk, emb in zip(chunks, embeddings):
    #     chunk["embedding"] = emb.tolist()
    # return chunks
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["content"] for c in chunks]

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    return chunks

import chromadb

def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    # TODO: Implement indexing
    #
    # Ví dụ với ChromaDB:
    # import chromadb
    #
    # CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    # client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # collection = client.get_or_create_collection(
    #     name=COLLECTION_NAME,
    #     metadata={"hnsw:space": "cosine"},
    # )
    #
    # ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in chunks]
    # collection.upsert(
    #     ids=ids,
    #     documents=[c["content"] for c in chunks],
    #     embeddings=[c["embedding"] for c in chunks],
    #     metadatas=[c["metadata"] for c in chunks],
    # )
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
        },
    )

    ids = [
        f"{c['metadata']['source']}_{c['metadata']['chunk_index']}"
        for c in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=[c["content"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )

    print(f"Indexed {len(chunks)} chunks.")

def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n[OK] Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"[OK] Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"[OK] Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("[OK] Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
