"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
DOC_IDS_FILE = Path(__file__).parent.parent / "pageindex_doc_ids.json"


def upload_documents() -> list[str]:
    """
    Upload toàn bộ PDF documents từ data/landing/legal/ lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        print("[WARNING] PAGEINDEX_API_KEY missing in .env")
        return []

    try:
        # PageIndex SDK >= 0.2 exports the client at package level.
        from pageindex import PageIndexClient
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    except Exception as e:
        print(f"[ERROR] PageIndex import failed: {e}")
        return []

    doc_ids = []
    legal_dir = LANDING_DIR / "legal"
    if legal_dir.exists():
        for pdf_file in legal_dir.glob("*.pdf"):
            try:
                resp = client.submit_document(str(pdf_file))
                doc_id = resp.get("doc_id") or resp.get("id")
                if doc_id:
                    doc_ids.append(doc_id)
                    print(f"  [OK] Uploaded: {pdf_file.name} -> {doc_id}")
            except Exception as e:
                print(f"  [ERROR] Upload {pdf_file.name}: {e}")

    if doc_ids:
        DOC_IDS_FILE.write_text(json.dumps(doc_ids, indent=2), encoding="utf-8")

    return doc_ids


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex làm fallback.
    """
    if not PAGEINDEX_API_KEY:
        return []

    try:
        # PageIndex SDK >= 0.2 exports the client at package level.
        from pageindex import PageIndexClient
        client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    except Exception:
        return []

    doc_ids = []
    if DOC_IDS_FILE.exists():
        try:
            doc_ids = json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not doc_ids:
        doc_ids = upload_documents()

    if not doc_ids:
        return []

    results = []
    for doc_id in doc_ids[:2]:
        try:
            resp = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = resp.get("retrieval_id") or resp.get("id")
            if not retrieval_id:
                continue

            for _ in range(10):
                retrieval = client.get_retrieval(retrieval_id)
                if retrieval.get("status") == "completed":
                    break
                time.sleep(1)

            rank = 1
            for node in retrieval.get("retrieved_nodes", []):
                for group in node.get("relevant_contents", []):
                    for item in group:
                        results.append({
                            "content": item.get("relevant_content", ""),
                            "score": float(1.0 / rank),
                            "metadata": {"section": item.get("section_title", "N/A"), "doc_id": doc_id},
                            "source": "pageindex",
                        })
                        rank += 1
        except Exception as e:
            print(f"  [ERROR] Querying PageIndex doc {doc_id}: {e}")

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("[INFO] PAGEINDEX_API_KEY not set in .env. Safe fallback enabled.")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("tuition fee payment methods", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
