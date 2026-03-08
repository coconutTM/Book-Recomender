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
            # ค้นไฟล์ .csv ทั้งหมดที่มีใน "data/"
            csv_files = glob.glob(os.path.join("data", "*.csv"))
            # ถ้าไม่เจอ 
            if not csv_files:
                print("ไม่พบไฟล์ csv ต้อง Scrape ข้อมูลก่อนครับ")
                continue # กลับไปเลือก mode ใหม่

            print("\nเลือกไฟล์ csv ที่คุณต้องการ:")
            # ทำการลูปเพื่อแสดงชื่อไฟล์ที่พบทั้งหมด
            for i, file in enumerate(csv_files):  # ตัวอย่าง file = 'data/xxxx.csv'
                file = os.path.basename(file) # file = "xxxx.csv" 
                print(f"{i + 1}. {file}")

            sel = int(input("เลือกไฟล์ csv: "))
            select_csv = os.path.basename(csv_files[sel - 1])
            print(f"เลือก: '{select_csv}'")

            # อ่านไฟล์ที่เลือก
            df = pd.read_csv(os.path.join("data", select_csv))

            # รวม title + description เป็น text เดียว
            df["content"] = df["title"] + " " + df["description"]
            df["content"] = df["content"].fillna("") # ถ้าว่าง ให้เป็น string เปล่าๆ เพื่อกัน error

            # สร้าง TF-IDF Metrix
            vectorizer = TfidfVectorizer()
            tfidf_metrix = vectorizer.fit_transform(df["content"]) 
            # 1. fit() = เรียนรู้ว่ามีคำอะไรบ้างใน data frame ตัวนี้ 
            # 2. transform = แปลงทุกเล่มเป็นตัวเลข

            print("-" * 50)
            print(f"\nโหลดข้อมูลหนังสือทั้งหมด {len(df)} เล่ม!")
            print("เลือก mode การแนะนำหนังสือ: ")
            print("  1. แนะนำตามความสนใจ")
            print("  2. แนะนำตามหนังสือที่สนใจ")
            recommend_mode = input("(1 or 2): ")

            if recommend_mode == "1": # แนะนำตามความสนใจของเรา
                query = input("พิมพ์ความสนใจของคุณ: ")
                results = recommend_by_interest(query, df, vectorizer, tfidf_metrix)
                print(f"หนังสือแนะนำสำหรับ '{query}' 10 อันดับได้แก่")
                print("-" * 50)
                print_results(results)
            elif recommend_mode == "2":
                while True:
                    title = input("พิมพ์ชื่อหนังสือ (บางส่วนก็ได้): ").strip()
                    if not title:
                        print("กรุณาพิมพ์ชื่อหนังสือของคุณที่นี่ก่อนนะครับ!")
                        continue

                    # ค้นหาหนังสือที่ชื่อคล้ายกัน
                    matches = search_book(title, df)
                    if matches.empty: # ถ้าไม่เจอ 
                        print(f"ไม่พบหนังสือชื่อ '{title}' กรุณาลองใหม่")
                        continue # กลับไปพิมพ์ชื่อหนังสือใหม่

                    # ถ้าเจอหนังสือ 
                    print("\n")
                    print("-" * 50)
                    print(f"พบหนังสือที่ตรงกับ '{title}' ทั้งหมด {len(matches)} เล่ม")
                    for i, row in matches.iterrows():
                        print(f"{i+1:2}. {row["title"]}")

                    pick = input(
                        f"\nเลือกเล่มที่ต้องการ (1 - {len(matches)}, n = พิมพ์ชื่อหนังสือใหม่): "
                    ).strip()
                    if pick.lower() == "n":
                        continue # กลับไปพิมพ์ชื่อหนังสือใหม่
                    elif not pick.isdigit() or not (1 <= int(pick) <= len(matches)):
                        print("เลือกไม่ถูกต้องครับ")
                        continue 

                    else:
                        select_title = matches.iloc[int(pick) - 1]["title"] 
                        # .iloc = เลือกหนังสือใน data frame ตาม index 
                        # ["title"] = เอาแค่ title 
                        found_title, results = recommend_by_title(
                            select_title, df, vectorizer, tfidf_metrix
                        )
                        print(f"หนังสือแนะนำสำหรับ '{found_title}' 10 อันดับได้แก่")
                        print_results(results)
                        break # ออกจาก loop
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
                category_list = ["16"]  # computer ebook
                all_ebook_links = get_all_book_links(page, category_list)
                with open(file, "w") as f:
                    for link in all_ebook_links:
                        f.write(link + "\n") # เขียน url ต่อทีละบรรทัด
                print(f"สร้างไฟล์ '{file}' เสร็จแล้ว")
                playwright.stop()
                sb.driver.quit()

            elif scrape_mode == "2":
                all_txt_files = glob.glob(os.path.join("data", "*.txt"))
                # ค้นหา .txt file ทั้งหมดใน "data/"
                if not all_txt_files:
                    print("ไม่พบไฟล์ txt ต้อง Scrape links ก่อนครับ")
                    continue

                print(f"ไฟล์ .txt ทั้งหมด")
                # แสดงผลการค้นหา
                for i, file in enumerate(all_txt_files):
                    print(f"{i + 1}. {os.path.basename(file)}")
                sel_txt = int(input("เลือกไฟล์ที่: "))
                file_name = os.path.basename(all_txt_files[sel_txt - 1]) 
                # เอาแค่ชื่อไฟล์ไม่เอาเป็น path
                print(f"เลือก '{file_name}'")
                links_file = os.path.join("data", file_name) # path links file 
                # อ่านไฟล์นั้น 
                with open(links_file, "r") as f:
                    # สร้าง list ของ links book ขึ้นมา
                    ebook_links = [
                        line.strip() for line in f.readlines() if line.strip()
                    ]

                sb = sb_cdp.Chrome()
                sb.get("https://www.naiin.com")

                endpoint_url = sb.get_endpoint_url()
                playwright = sync_playwright().start()
                browser = playwright.chromium.connect_over_cdp(endpoint_url)
                context = browser.contexts[0]
                page = context.pages[0]

                # scraping book
                all_books_details = [] 
                csv_name = f"{file_name.split(".")[0]}.csv"
                csv_file = os.path.join("data", csv_name)

                for i, url in enumerate(ebook_links):
                    print(f"[{i + 1} from {len(ebook_links)}] {url}")
                    book = scrape_book_detail(page, url)
                    if book: # ถ้า book มีข้อมูล ให้เพิ่มเข้า list
                        all_books_details.append(book)
                        # ถ้าครบ 100 เล่ม ทำการ save 
                        if len(all_books_details) % 100 == 0:
                            df = pd.DataFrame(
                                all_books_details[-100:]
                            )  # เอา 100 เล่มหลัง
                            if os.path.exists(csv_file): # ถ้ามีไฟล์แล้ว ทำการ append
                                df.to_csv(csv_file, mode="a", header=False, index=False)
                            else: # ถ้ายังไม่มีไฟล์ ให้สร้างก่อน
                                df.to_csv(csv_file, index=False)
                            print(f"save --> {csv_name}: {len(all_books_details)}")
                    time.sleep(1) # หน่วง 1 วิ

                # เพิ่มหนังสือที่เหลือ  
                remain_book = len(all_books_details) % 100
                if remain_book > 0:
                    df = pd.DataFrame(all_books_details[-remain_book:])
                    if os.path.exists(csv_file):
                        df.to_csv(csv_file, mode="a", header=False, index=False)
                    else:
                        df.to_csv(csv_file, index=False)

                print(f"save --> {csv_name}: {len(all_books_details)}")

                playwright.stop()
                sb.driver.quit()

            elif scrape_mode == "3":
                # ค้นหาไฟล์ .csv ทั้งหมดใน "data/"
                all_csv_files = glob.glob(os.path.join("data", "*.csv"))
                if not all_csv_files:
                    print("ไม่พบไฟล์ .csv ต้อง Scrape ก่อนครับ")
                    continue # ถ้าไม่เจอ ให้กลับไปเลือก mode ใหม่

                print(f"ไฟล์ .csv ทั้งหมด")
                # แสดงผลที่ค้นมา
                for i, file in enumerate(all_csv_files):
                    print(f"{i + 1}. {os.path.basename(file)}")
                sel_csv = int(input("เลือกไฟล์ที่: "))
                file_name = os.path.basename(all_csv_files[sel_csv - 1])
                clean_file = clean(file_name) # เรียก function ทำความสะอาบข้อมูล

        elif mode == "0" or mode.lower() == "exit":
            exit()
        else:
            print("**เลือก (1/2 or 0) เท่านั้น!**\n")
            continue
