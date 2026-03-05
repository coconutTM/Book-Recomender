from scraper import get_all_book_links, scrape_book_detail
from recommender import (
    recommend_by_interest,
    recommend_by_title,
    print_results,
    search_book,
)
from cleanup import clean

import os, time
import glob
import pandas as pd
from playwright.sync_api import sync_playwright
from seleniumbase import sb_cdp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


if __name__ == "__main__":
    print("-" * 50)
    print("ระบบแนะนำหนังสือจากเว็บไซต์ 'naiin.com' ~~~")
    print("-" * 50)
    print()

    while True:

        print("เลือกโหมด: ")
        print("  1. รันระบบแนะนำหนังสือ")
        print("  2. Scrape ข้อมูลใหม่")
        print("  0. ปิดโปรแกรม")

        mode = input("เลือก (1/2 or 0): ").strip()
        if mode == "1":
            try:
                csv_files = glob.glob(os.path.join("data", "*.csv"))
            except:
                print("ไม่พบไฟล์ csv ต้อง Scrape ข้อมูลก่อนครับ")
                continue

            print("\nเลือกไฟล์ csv ที่คุณต้องการ:")
            for i, file in enumerate(csv_files):
                file = os.path.basename(file)
                print(f"{i + 1}. {file}")

            sel = int(input("เลือกไฟล์ csv: "))
            select_csv = os.path.basename(csv_files[sel - 1])
            print(f"เลือก: '{select_csv}'")

            df = pd.read_csv(os.path.join("data", select_csv))
            # รวม title + description เป็น text เดียว
            df["content"] = df["title"] + " " + df["description"]
            df["content"] = df["content"].fillna("")

            # สร้าง TF-IDF Metrix
            vectorizer = TfidfVectorizer()
            tfidf_metrix = vectorizer.fit_transform(df["content"])

            print("-" * 50)
            print(f"\nโหลดข้อมูลหนังสือทั้งหมด {len(df)} เล่ม!")
            print("เลือก mode การแนะนำหนังสือ: ")
            print("  1. แนะนำตามความสนใจ")
            print("  2. แนะนำตามหนังสือที่สนใจ")
            recommend_mode = input("(1 or 2): ")

            if recommend_mode == "1":
                query = input("พิมพ์ความสนใจของคุณ: ")
                results = recommend_by_interest(query, df, vectorizer, tfidf_metrix)
                print(f"หนังสือแนะนำสำหรับ '{query}' 10 อันดับได้แก่")
                print("-" * 50)
                print_results(results)
            if recommend_mode == "2":
                while True:
                    title = input("พิมพ์ชื่อหนังสือ (บางส่วนก็ได้): ").strip()
                    if not title:
                        print("กรุณาพิมพ์ชื่อหนังสือของคุณที่นี่ก่อนนะครับ!")
                        continue

                    # ค้นหาหนังสือที่ชื่อคล้ายกัน
                    matches = search_book(title, df)
                    if matches.empty:
                        print(f"ไม่พบหนังสือชื่อ '{title}' กรุณาลองใหม่")
                        continue

                    print("\n-" * 50)
                    print(f"พบหนังสือที่ตรงกับ '{title}' ทั้งหมด {len(matches)} เล่ม")
                    for i, row in matches.iterrows():
                        print(f"{i+1:2}. {row["title"]}")

                    pick = input(
                        f"\nเลือกเล่มที่ต้องการ (1-{len(matches)}, n = พิมพ์ชื่อหนังสือใหม่): "
                    ).strip()
                    if pick.lower() == "n":
                        continue
                    elif not pick.isdigit() or not (1 <= int(pick) <= len(matches)):
                        print("เลือกไม่ถูกต้องครับ")
                        continue

                    else:
                        select_title = matches.iloc[int(pick) - 1]["title"]
                        found_title, results = recommend_by_title(
                            select_title, df, vectorizer, tfidf_metrix
                        )
                        print(f"หนังสือแนะนำสำหรับ '{found_title}' 10 อันดับได้แก่")
                        print_results(results)
                        break
        elif mode == "2":
            print("\nเลือก mode การ Scrape: ")
            print("  1. สร้างไฟล์ links.txt")
            print("  2. รัน Scrape สร้างไฟล์ .csv")
            print("  3. ทำความสะอาดไฟล์ .csv")
            scrape_mode = input("(1, 2 or 3): ")
            if scrape_mode == "1":
                print("เริ่มสร้างไฟล์ .txt")
                txt_name = input("ตั้งชื่อไฟล์ (ไม่ต้องใส่ .txt): ")
                file = os.path.join("data", f"{txt_name}.txt")
                
                sb = sb_cdp.Chrome()
                sb.get("https://www.naiin.com")

                endpoint_url = sb.get_endpoint_url()
                playwright = sync_playwright().start()
                browser = playwright.chromium.connect_over_cdp(endpoint_url)
                context = browser.contexts[0]
                page = context.pages[0]

                # get all book link
                category_list = ["16", "17"] # computer, หนังสือเตรียมสอบ
                all_ebook_links = get_all_book_links(page, category_list)
                with open(file, "w") as f:
                    for link in all_ebook_links:
                        f.write(link + "\n")
                print(f"สร้างไฟล์ '{file}' เสร็จแล้ว")
                playwright.stop()
                sb.driver.quit()

            elif scrape_mode == "2":
                try:
                    all_txt_files = glob.glob(os.path.join("data", "*.txt"))
                except:
                    print("ไม่พบไฟล์ txt ต้อง Scrape links ก่อนครับ")
                    continue
                print(f"ไฟล์ .txt ทั้งหมด")
                for i, file in enumerate(all_txt_files):
                    print(f"{i + 1}. {os.path.basename(file)}")
                sel_txt = int(input("เลือกไฟล์ที่: "))
                file_name = os.path.basename(all_txt_files[sel_txt - 1])
                print(f"เลือก '{file_name}'")
                links_file = os.path.join("data", file_name)
                with open(links_file, "r") as f:
                    ebook_links = [line.strip() for line in f.readlines() if line.strip()]
  
                sb = sb_cdp.Chrome()
                sb.get("https://www.naiin.com")

                endpoint_url = sb.get_endpoint_url()
                playwright = sync_playwright().start()
                browser = playwright.chromium.connect_over_cdp(endpoint_url)
                context = browser.contexts[0]
                page = context.pages[0]

                # scraping book
                all_books_details = []
                for i, url in enumerate(ebook_links):
                    print(f"[{i + 1} from {len(ebook_links)}] {url}")
                    book = scrape_book_detail(page, url)
                    if book:
                        all_books_details.append(book)
                    # page.wait_for_timeout(1000)
                    time.sleep(1)
                
                df = pd.DataFrame(all_books_details)
                # df.to_csv("data/books.csv", index=False, encoding="utf-8-sig")
                csv_files = f"{file_name.split(".")[0]}.csv"
                df.to_csv(os.path.join("data", csv_files), index=False)
                print(f"\nบันทึกแล้ว {len(df)} เล่ม --> {csv_files}")

                playwright.stop()
                sb.driver.quit()

            elif scrape_mode == "3":
                try:
                    all_txt_files = glob.glob(os.path.join("data", "*.csv"))
                except:
                    print("ไม่พบไฟล์ .csv ต้อง Scrape ก่อนครับ")
                    continue
                print(f"ไฟล์ .csv ทั้งหมด")
                for i, file in enumerate(all_txt_files):
                    print(f"{i + 1}. {os.path.basename(file)}")
                sel_txt = int(input("เลือกไฟล์ที่: "))
                file_name = os.path.basename(all_txt_files[sel_txt - 1])
                clean_file = clean(file_name) 


        elif mode == "0" or mode.lower() == "exit":
            exit()
        else:
            print("**เลือก (1/2 or 0) เท่านั้น!**\n")
            continue
