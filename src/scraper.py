from playwright.sync_api import sync_playwright
from seleniumbase import sb_cdp
import builtins
import pandas as pd
import time
import os


def get_all_book_links(page, category_code):
    links = []
    page_no = 1

    while True:
        url = f"https://www.naiin.com/category?category_1_code={category_code}&product_type_id=3&pageNo={page_no}"
        builtins.print(f"กำลังดึงหน้า {page_no}...")

        page.goto(url)
        page.wait_for_load_state("domcontentloaded")

        # ✅ รอจนกว่า element จะโหลดจริงๆ ไม่ใช่รอตายตัว
        try:
            page.wait_for_selector("a.item-img-block", timeout=10000)
        except:
            builtins.print("  รอ 10 วิแล้วยังไม่เจอ element หยุด")
            break

        items = page.query_selector_all("a.item-img-block")
        if not items:
            builtins.print("ไม่พบหนังสือ หยุด")
            break

        for item in items:
            href = item.get_attribute("href")
            if href:
                full_url = (
                    f"https://www.naiin.com{href}" if href.startswith("/") else href
                )
                links.append(full_url)

        builtins.print(f"  หน้า {page_no}: พบ {len(items)} เล่ม (รวม {len(links)})")

        next_btn = page.query_selector(".nav-pag.pag-next")
        if not next_btn:
            builtins.print("ถึงหน้าสุดท้ายแล้ว")
            break

        page_no += 1
        page.wait_for_timeout(1000)  # หน่วงนิดนึงก่อนขึ้นหน้าถัดไป

    return links


def scrape_book_detail(page, url):
    page.goto(url)
    page.wait_for_load_state("domcontentloaded")

    try:
        page.wait_for_selector("h1.title-topic", timeout=10000)
    except:
        builtins.print(f"  โหลดไม่ได้: {url}")
        return None

    # ชื่อหนังสือ
    title = ""
    el = page.query_selector("h1.title-topic")
    if el:
        title = el.inner_text().strip()

    # ผู้แต่ง และ สำนักพิมพ์ — class เดียวกัน
    author = ""
    publisher = ""
    links = page.query_selector_all("a.inline-block.link-book-detail")
    if len(links) > 0:
        author = links[0].inner_text().strip()
    if len(links) > 1:
        publisher = links[1].inner_text().strip()

    # ราคา
    price = ""
    el = page.query_selector("span#discount-price")
    if el:
        price = el.inner_text().strip()

    # คำอธิบาย
    description = ""
    el = page.query_selector("div.book-decription")
    if el:
        raw = el.inner_text().strip()
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("รายละเอียด")]
        description = "\n".join(lines).strip()

    result = {
        "title": title,
        "author": author,
        "publisher": publisher,
        "price": price,
        "description": description,
        "url": url,
    }
    return result


# --- Main ---
# setup selenium
sb = sb_cdp.Chrome()
sb.get("https://www.naiin.com")

endpoint_url = sb.get_endpoint_url()
playwright = sync_playwright().start()
browser = playwright.chromium.connect_over_cdp(endpoint_url)
context = browser.contexts[0]
page = context.pages[0]

# get all book link
file = os.path.join("data", "computer_ebook_links.txt")
if os.path.exists(file):
    with open(file, "r") as f:
        computer_ebook_links = [line.strip() for line in f.readlines() if line.strip()]
    builtins.print(f"Links loading from file {file}")
else:
    computer_ebook_links = get_all_book_links(page, category_code=16)
    with open(file, "w") as f:
        for link in computer_ebook_links:
            f.write(link + "\n")
    builtins.print(f"Creating file '{file}'")

builtins.print(f"\nรวม {len(computer_ebook_links)} ลิงก์ กำลังเริ่ม scraping นะจ้ะ")

# scraping book
all_books_details = []
for i, url in enumerate(computer_ebook_links):
    builtins.print(f"[{i + 1} from {len(computer_ebook_links)}] {url}")
    book = scrape_book_detail(page, url)
    if book:
        all_books_details.append(book)
    page.wait_for_timeout(1000)

df = pd.DataFrame(all_books_details)
# df.to_csv("data/books.csv", index=False, encoding="utf-8-sig")
df.to_csv(os.path.join("data", "books.csv"), index=False)
builtins.print(f"\nบันทึกแล้ว {len(df)} เล่ม --> books.csv")

playwright.stop()
sb.driver.quit()
