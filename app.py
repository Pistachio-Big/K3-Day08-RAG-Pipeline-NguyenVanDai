"""Interactive UI for the University Services RAG pipeline.

Run with: ``.venv/bin/streamlit run app.py``
"""

from pathlib import Path
import sys

import streamlit as st
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(page_title="University Services RAG", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner=False)
def get_generator():
    """Import once per Streamlit server; no answer is mocked in the UI."""
    from src.task10_generation import generate_with_citation

    return generate_with_citation


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} chunks)"):
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata", {}) or {}
            st.markdown(
                f"**[{index}] {metadata.get('source', 'Unknown')}** "
                f"`{metadata.get('type', 'unknown')}` · score `{source.get('score', 0):.4f}`"
            )
            st.write(source.get("content", "")[:500])


def query_with_history(question: str, messages: list[dict]) -> str:
    """Provide a small amount of context for clear follow-up questions."""
    follow_up_markers = ("còn", "vậy", "thế", "nó", "điều đó", "họ", "trường đó")
    if not question.lower().strip().startswith(follow_up_markers) or not messages:
        return question
    recent = messages[-4:]
    conversation = "\n".join(f"{msg['role']}: {msg['content']}" for msg in recent)
    return f"Ngữ cảnh hội thoại gần đây:\n{conversation}\n\nCâu hỏi tiếp theo: {question}"


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

with st.sidebar:
    st.title("🎓 University Services RAG")
    st.caption("Hỏi đáp về điểm chuẩn, tuyển sinh và chính sách đại học.")
    top_k = st.slider("Số chunks dùng làm nguồn", min_value=3, max_value=10, value=5)
    if st.button("Xóa lịch sử hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("Gợi ý")
    suggestions = [
        "Điểm chuẩn ngành Công nghệ thông tin năm 2025 là bao nhiêu?",
        "Đại học Bách khoa Hà Nội có những phương thức xét tuyển nào?",
        "Điều kiện xét tuyển bằng chứng chỉ IELTS là gì?",
        "Học phí chương trình chuẩn của Bách khoa Hà Nội là bao nhiêu?",
    ]
    for index, suggestion in enumerate(suggestions):
        if st.button(suggestion, key=f"suggestion_{index}", use_container_width=True):
            st.session_state.pending_query = suggestion
            st.rerun()
    st.divider()
    st.caption("Semantic search + BM25 → RRF → PageIndex fallback → LLM có citation")

st.title("University Services RAG Chatbot")
st.caption("Câu trả lời chỉ được tổng hợp từ các tài liệu đã index.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

typed_query = st.chat_input("Nhập câu hỏi về tuyển sinh hoặc điểm chuẩn...")
question = typed_query or st.session_state.pending_query

if question:
    st.session_state.pending_query = None
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất tài liệu và tạo câu trả lời..."):
            try:
                generate = get_generator()
                response = generate(
                    query_with_history(question, st.session_state.messages[:-1]), top_k=top_k
                )
                answer = response.get("answer") or "Tôi không thể xác minh thông tin này từ nguồn hiện có."
                sources = response.get("sources", [])
            except Exception as exc:
                answer = f"⚠️ Không thể chạy pipeline: `{type(exc).__name__}`. Kiểm tra ChromaDB và các API key trong `.env`."
                sources = []
            st.markdown(answer)
            render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
