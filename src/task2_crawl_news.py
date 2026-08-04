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

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://xaydungchinhsach.chinhphu.vn/diem-chuan-truong-dai-hoc-cong-nghe-dhqg-ha-noi-nam-2025-119250822151451285.htm",
    "https://xaydungchinhsach.chinhphu.vn/diem-chuan-dai-hoc-bach-khoa-ha-noi-nam-2025-119250822165346027.htm",
    "https://xaydungchinhsach.chinhphu.vn/diem-chuan-truong-dai-hoc-bach-khoa-tphcm-dhqg-tphcm-nam-2025-11925082220080899.htm",
    "https://xaydungchinhsach.chinhphu.vn/diem-chuan-truong-dai-hoc-khoa-hoc-tu-nhien-hus-dhqg-ha-noi-nam-2925-119250822173424001.htm",
    "https://xaydungchinhsach.chinhphu.vn/diem-chuan-truong-dai-hoc-ngoai-thuong-ftu-nam-2025-119250823073938701.htm",
]

# A concise service directory can be shorter than a news article; 300 still
# rejects error pages while retaining legitimate service pages.
MIN_CONTENT_CHARS = 300


def _extract_article_from_html(url: str) -> dict:
    """Fetch and extract readable text; requests transparently decompresses gzip/br."""
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; UniversityServicesRAG/1.0)"},
        timeout=30,
    )
    response.raise_for_status()
    response.encoding = response.encoding or response.apparent_encoding

    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
        element.decompose()

    title = soup.find("h1")
    title_text = title.get_text(" ", strip=True) if title else ""
    if not title_text:
        title_text = soup.title.get_text(" ", strip=True) if soup.title else "Unknown"

    main_content = soup.find("main") or soup.find("article") or soup.body
    content = main_content.get_text("\n", strip=True) if main_content else ""
    if len(content) < MIN_CONTENT_CHARS:
        raise ValueError(f"Nội dung trích xuất quá ngắn ({len(content)} ký tự)")

    return {
        "url": url,
        "title": title_text,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content,
    }


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

        content = (getattr(result, "markdown", None) or "").strip()
        if len(content) < MIN_CONTENT_CHARS:
            raise ValueError("Crawl4AI trả về nội dung quá ngắn")

        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": content,
        }
    except Exception as crawl_error:
        print(f"  ⚠ Crawl4AI failed ({crawl_error}); using requests fallback")
        return await asyncio.to_thread(_extract_article_from_html, url)


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

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
