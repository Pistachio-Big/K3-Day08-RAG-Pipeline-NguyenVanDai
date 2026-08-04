"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trang công khai của một trường đại học.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: thông báo tuyển sinh, sự kiện, dịch vụ thư viện, hỗ trợ sinh viên, học bổng.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài viết cần crawl
ARTICLE_URLS = [
    # Ví dụ (trang công khai RMIT Vietnam):
    "https://xaydungchinhsach.chinhphu.vn/diem-chuan-truong-dai-hoc-cong-nghe-dhqg-ha-noi-nam-2025-119250822151451285.htm",
    "https://xaydungchinhsach.chinhphu.vn/diem-chuan-dai-hoc-bach-khoa-ha-noi-nam-2025-119250822165346027.htm",
    "https://xaydungchinhsach.chinhphu.vn/diem-chuan-truong-dai-hoc-bach-khoa-tphcm-dhqg-tphcm-nam-2025-11925082220080899.htm",
    "https://xaydungchinhsach.chinhphu.vn/diem-chuan-truong-dai-hoc-khoa-hoc-tu-nhien-hus-dhqg-ha-noi-nam-2925-119250822173424001.htm",
    "https://xaydungchinhsach.chinhphu.vn/diem-chuan-truong-dai-hoc-ngoai-thuong-ftu-nam-2025-119250823073938701.htm",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=url)

        title = "Unknown"
        if getattr(result, "metadata", None):
            title = result.metadata.get("title", "Unknown") or "Unknown"

        content = getattr(result, "markdown", None) or ""
        if not content and getattr(result, "raw_html", None):
            content = result.raw_html

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": content or f"Không thể trích xuất nội dung từ: {url}",
        }
    except Exception:
        import re
        import urllib.request
        from html import unescape

        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                html = response.read().decode("utf-8", errors="ignore")
        except Exception:
            html = ""

        title_match = re.search(r"<title>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        title = unescape(title_match.group(1)).strip() if title_match else "Unknown"

        text = re.sub(r"<style.*?</style>", " ", html, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<script.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<[^>]+>", "\n", text)
        text = unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        content = text or f"Không thể trích xuất nội dung từ: {url}"

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": content,
        }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except Exception as exc:
            article = {
                "url": url,
                "title": "Unknown",
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": f"Crawl failed: {exc}",
            }

        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        # filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang thông báo/sự kiện trên trang chính thức của trường đại học")
    else:
        asyncio.run(crawl_all())
