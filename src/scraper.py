from playwright.sync_api import sync_playwright
from seleniumbase import sb_cdp
import builtins
import pandas as pd
import time
import os
import subprocess


# reconnect wifi (bash command)
def reconnect_wifi():
    print("ทำการ reconnet wifi")
    subprocess.run(["nmcli", "radio", "wifi", "off"], capture_output=True)
    time.sleep(2)
    subprocess.run(["nmcli", "radio", "wifi", "on"], capture_output=True)
    time.sleep(5)  # รอให้เชื่อมต่อใหม่
    print("reconnect wifi เรียบร้อย")


def get_all_book_links(page, category_list):
    all_links = []

    for category_code in category_list:
        builtins.print(f"กำลังดึง: {category_code}")
        page_no = 1
        retry = 0

        while True:
            url = f"https://www.naiin.com/category?category_1_code={category_code}&product_type_id=3&pageNo={page_no}"
            builtins.print(f"กำลังดึงหน้า: {page_no}...")

            page.goto(url)
            page.wait_for_load_state("domcontentloaded")

            # ✅ รอจนกว่า element จะโหลดจริงๆ ไม่ใช่รอตายตัว
            try:
                page.wait_for_selector("a.item-img-block", timeout=60000)
            except:
                retry += 1
                builtins.print("รอ 60 วิแล้วยังไม่เจอ element หยุด")
                if retry >= 20:
                    builtins.print("wifi หรืออะไรของมึงเนี่ย")
                    break
                reconnect_wifi()
                continue

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
                    if full_url not in all_links:
                        all_links.append(full_url)

            builtins.print(
                f"หมวด {category_code}:  หน้า {page_no}: พบ {len(items)} เล่ม (รวม {len(all_links)})"
            )

            next_btn = page.query_selector(".nav-pag.pag-next")
            if not next_btn:
                builtins.print("ถึงหน้าสุดท้ายแล้ว")
                break

            page_no += 1
            page.wait_for_timeout(1000)  # หน่วงนิดนึงก่อนขึ้นหน้าถัดไป

    return all_links


def scrape_book_detail(page, url):
    retry = 0
    while True:
        try:
            page.goto(url)
            page.wait_for_load_state("domcontentloaded")
            page.wait_for_selector("h1.title-topic", timeout=60000)
        except:
            retry += 1
            builtins.print(f"รอ 60 วินาทีแล้วโหลดไม่ได้: {url}")
            if retry >= 20:
                builtins.print("wifi หรืออะไรของมึงเนี่ย")
                return None
            reconnect_wifi()
            continue

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
if __name__ == "__main__":
    # setup selenium
    sb = sb_cdp.Chrome()
    sb.get("https://www.naiin.com")

    endpoint_url = sb.get_endpoint_url()
    playwright = sync_playwright().start()
    browser = playwright.chromium.connect_over_cdp(endpoint_url)
    context = browser.contexts[0]
    page = context.pages[0]

    # get all book link
    file = os.path.join("data", "ebook_links.txt")
    category_list = ["16", "2", "15", "11"]

    if os.path.exists(file):
        builtins.print(f"Links loading from file {file}")
        with open(file, "r") as f:
            ebook_links = [line.strip() for line in f.readlines() if line.strip()]
    else:
        ebook_links = get_all_book_links(page, category_list)
        with open(file, "w") as f:
            for link in ebook_links:
                f.write(link + "\n")
        builtins.print(f"Creating file '{file}'")

    builtins.print(f"\nรวม {len(ebook_links)} ลิงก์ เริ่ม scraping เลยมั้ย?")
    choice = input("Y/n: ")
    if choice.lower() == "y":
        # scraping book
        all_books_details = []
        for i, url in enumerate(ebook_links):
            builtins.print(f"[{i + 1} from {len(ebook_links)}] {url}")
            book = scrape_book_detail(page, url)
            if book:
                all_books_details.append(book)
            page.wait_for_timeout(1000)
    else:
        exit()

    df = pd.DataFrame(all_books_details)
    # df.to_csv("data/books.csv", index=False, encoding="utf-8-sig")
    df.to_csv(os.path.join("data", "ebooks.csv"), index=False)
    builtins.print(f"\nบันทึกแล้ว {len(df)} เล่ม --> books.csv")

    playwright.stop()
    sb.driver.quit()
