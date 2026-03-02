"""
scraper.py
ดึงข้อมูลหนังสือจาก naiin.com
ใช้ URL รูปแบบ: /category?category_1_code=XX&product_type_id=1
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from tqdm import tqdm

BASE_URL = "https://www.naiin.com"
OUTPUT_FILE = "books.csv"
DELAY = 2.0

# category_1_code จริงของ naiin.com
CATEGORIES = {
    "คอมพิวเตอร์":          16,
    "นิยาย":                2,
    "วรรณกรรมและบันเทิงคดี": 33,
    "ภาษาต่างประเทศ":        20,
    "การ์ตูนความรู้":         14,
}


def make_session() -> requests.Session:
    """สร้าง session พร้อม headers ที่ทำให้ดูเหมือน browser จริง"""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "th-TH,th;q=0.9,en-US;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.naiin.com/",
        "Connection": "keep-alive",
    })
    # เปิดหน้าหลักก่อนเพื่อรับ cookies
    try:
        session.get(BASE_URL, timeout=15)
        time.sleep(1)
    except Exception:
        pass
    return session


def get_book_links(session: requests.Session, cat_code: int, max_pages: int = 5) -> list:
    """ดึง URL หนังสือจากหน้า category"""
    links = set()

    for page in range(1, max_pages + 1):
        url = (
            f"{BASE_URL}/category"
            f"?category_1_code={cat_code}"
            f"&product_type_id=1"
            f"&pageNo={page}"
        )
        try:
            r = session.get(url, timeout=20)
            print(f"    หน้า {page}: HTTP {r.status_code} | {len(r.text)} chars")

            if r.status_code != 200:
                break

            soup = BeautifulSoup(r.text, "html.parser")

            # ลองหลาย selector
            anchors = (
                soup.select("a.product-item-link")
                or soup.select("a[href*='/product/detail/']")
                or soup.select(".product-name a")
                or soup.select("h2.product-name a")
            )

            if not anchors:
                if page == 1:
                    with open(f"debug_cat{cat_code}.html", "w", encoding="utf-8") as f:
                        f.write(r.text)
                    print(f"    [DEBUG] บันทึก HTML → debug_cat{cat_code}.html (ให้เปิดดู selector จริง)")
                break

            before = len(links)
            for a in anchors:
                href = a.get("href", "")
                if href:
                    full = href if href.startswith("http") else BASE_URL + href
                    links.add(full)

            added = len(links) - before
            print(f"    พบลิงก์ใหม่ {added} รายการ (รวม {len(links)})")
            if added == 0:
                break

            time.sleep(DELAY)

        except Exception as e:
            print(f"    [!] Error: {e}")
            break

    return list(links)


def scrape_book_detail(session: requests.Session, url: str) -> dict | None:
    """ดึงรายละเอียดหนังสือจาก URL"""
    try:
        r = session.get(url, timeout=20)
        if r.status_code != 200:
            return None

        soup = BeautifulSoup(r.text, "html.parser")

        def get_text(*selectors):
            for sel in selectors:
                tag = soup.select_one(sel)
                if tag:
                    return tag.get_text(strip=True)
            return ""

        title = get_text("h1.page-title span", "h1.product-name", "span[itemprop='name']", "h1")
        author = get_text("td.author-name a", ".product-author a", "a[href*='author']")
        publisher = get_text("td.publisher-name a", ".product-publisher a", "a[href*='publisher']")
        isbn = get_text("td.isbn", "[itemprop='isbn']")
        price = get_text("span.price", "[itemprop='price']", ".price-box .price")

        crumbs = soup.select("ul.items li.item a, .breadcrumbs a")
        categories = " > ".join(c.get_text(strip=True) for c in crumbs if c.get_text(strip=True))

        desc_tag = soup.select_one(
            "#description .value, "
            ".product.attribute.description .value, "
            "[itemprop='description'], "
            "#product-detail-synopsis"
        )
        description = desc_tag.get_text(separator=" ", strip=True) if desc_tag else ""

        tag_tags = soup.select(".product-tags a, .tags a")
        tags = ", ".join(t.get_text(strip=True) for t in tag_tags)

        if not title:
            return None

        return {
            "title": title, "author": author, "publisher": publisher,
            "isbn": isbn, "price": price, "categories": categories,
            "tags": tags, "description": description, "url": url,
        }

    except Exception as e:
        print(f"  [!] Error {url}: {e}")
        return None


def run_scraper(max_books_per_category: int = 50, max_pages: int = 3):
    session = make_session()
    all_books = []

    for cat_name, cat_code in CATEGORIES.items():
        print(f"\n{'='*55}")
        print(f"📚 หมวด: {cat_name}  (code={cat_code})")
        print(f"{'='*55}")

        links = get_book_links(session, cat_code, max_pages)
        links = links[:max_books_per_category]

        if not links:
            print(f"  [!] ไม่พบลิงก์ — ดูไฟล์ debug_cat{cat_code}.html เพื่อเช็ค HTML จริง")
            continue

        for link in tqdm(links, desc=f"  {cat_name}"):
            book = scrape_book_detail(session, link)
            if book:
                book["main_category"] = cat_name
                all_books.append(book)
            time.sleep(DELAY)

    if not all_books:
        print("\n❌ ไม่ได้ข้อมูล — เปิดไฟล์ debug_cat*.html แล้ว Inspect หา CSS selector จริง")
        return pd.DataFrame()

    df = pd.DataFrame(all_books).drop_duplicates(subset="title")
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n✅ บันทึก {len(df)} เล่ม → {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    run_scraper(max_books_per_category=50, max_pages=3)
