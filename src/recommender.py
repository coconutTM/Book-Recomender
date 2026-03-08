import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Function recommend!!
# แนะนำตามความสนใจ
def recommend_by_interest(query, df, vectorizer, tfidf_metrix, top_n=10):
    # แปลงเป็น vector เพื่อคำนวณ  ใช้ vocabulary เดิมที่ fit ไว้ก่อนหน้า
    query_vec = vectorizer.transform([query])

    # คำนวณ cosine-similarity กับหนังสือทุกเล่ม ได้ array ของคะแนน
    scores = cosine_similarity(query_vec, tfidf_metrix).flatten() # .flatten() = แปลงเป็น array 1 มิติ 

    top_indices = scores.argsort()[::-1][:top_n]
    # argsort() = เรียง index คะแนนจากน้อยไปมาก [2, 0, 5, 3]
    # [::-1] = กลับเป็นมากไปน้อย [3, 5, 0, 2]
    # [:top_n] = เอาแค่ top_n เล่ม ก็คือ 10 เล่ม

    results = df.iloc[top_indices][
        ["title", "author", "publisher", "price", "url"]
    ].copy() # .copy() = ทำสำเนาใหม่ ไม่ได้แก้ df ต้นฉบับ
    results["score"] = scores[top_indices].round(4) # ใส่เลขกลมๆ 4 ตำแหน่ง 
    results = results[results["score"] > 0]  # ตัดเล่มที่ไม่เกี่ยวข้องออก

    return results.reset_index(drop=True) # รีเซ็ต index แล้วตัด column index ทิ้ง 


# แนะนำตามหนังสือที่สนใจ
def recommend_by_title(title, df, vectorizer, tfidf_metrix, top_n=10):
    # หาเล่มที่ตรงกับเล่มที่เราสนใจ
    matches = df[df["title"].str.contains(title, case=False, na=False, regex=False)]

    # ถ้าไม่เจอ 
    if matches.empty:
        return None, None # return 2 ค่า เพราะ caller รับ 2 ค่า

    # เอาเล่มแรกที่เจอ
    book = matches.iloc[0] # เอา row แรก
    book_idx = matches.index[0] # เอา index ของเล่มนั้นจริงใน df (ถ้าผ่าน filter มาจะไม่ใช่ 0 เสมอไป)

    # เอา vector หนังสือเล่มนั้นไปเทียบกับทุกเล่ม
    book_vec = tfidf_metrix[book_idx] 
    scores = cosine_similarity(book_vec, tfidf_metrix).flatten() # แปลงเป็น array 1 มิติ

    # เรียงลำดับจากมากไปน้อย แล้วตัดตัวเองออก
    top_indices = scores.argsort()[::-1]

    # เอามาแค่ 10 เล่ม
    top_indices = [i for i in top_indices if i != book_idx][:top_n] 

    results = df.iloc[top_indices][
        ["title", "author", "publisher", "price", "url"]
    ].copy()
    results["score"] = scores[top_indices].round(4)
    results = results[results["score"] > 0]  # ตัดเล่มที่ไม่เกี่ยวข้องออก

    return book["title"], results.reset_index(drop=True) # รีเซ็ต index แล้วตัด column index ทิ้ง
    # return ชื่อหนังสือ กับ dataframe ผลลัพธ์


# Print data
def print_results(results):
    # ถ้าไม่มี
    if results is None or results.empty:
        print("ไม่พบหนังสือที่เกี่ยวข้อง ลองพิมพ์คำอื่นดูครับ")
        return

    print("-" * 50)
    for i, row in results.iterrows():
        print(f"{i+1:2}. {row["title"]}")
        print(f"         ผู้เขียน: {row["author"]}")
        print(f"         สำนักพิมพ์: {row["publisher"]}")
        print(f"         ราคา: {row["price"]} บาท")
        print(f"         คะแนนความเกี่ยวข้อง {row["score"]}")
        print(f"         url: {row["url"]}")
        print()

    print("-" * 50 + "\n")


def search_book(title, df):
    # ค้นหาหนังสือจากชื่อที่คล้ายกัน return dataframe
    matches = df[df["title"].str.contains(title, case=False, na=False, regex=False)]
    # case=False: ไม่สนตัวพิมพ์เล็กหรือตัวพิมพ์ใหญ่
    # na=False: ถ้า title ว่างให้ถือว่าเป็น False ไปเลย
    # regex=False: ปกติแล้วถ้า title มี () [] . จะถือว่าเป็น regex, ตั้งเป็น False ให้มันมองว่าเป็น string ปกติแทน
    return matches.reset_index() # reset index แต่เก็บ index เดิมไว้เผื่อใช้ 




# -- Main --
if __name__ == "__main__":
    # โหลดข้อมูล
    df = pd.read_csv(os.path.join("data", "ebooks_cleaned.csv"))
    # รวม title + description เป็น text เดียว
    df["content"] = df["title"] + " " + df["description"]
    df["content"] = df["content"].fillna("")

    # สร้าง TF-IDF Metrix
    vectorizer = TfidfVectorizer()
    tfidf_metrix = vectorizer.fit_transform(df["content"])

    print(f"\nโหลดข้อมูลหนังสือทั้งหมด {len(df)} เล่ม!")

    print("-" * 50)
    print("ระบบแนะนำหนังสือจากเว็บไซต์ 'naiin.com' ~~~")
    print("-" * 50)
    print()

    while True:
        print("\nเลือกวิธีการค้นหา: ")
        print("  1. พิมพ์ความสนใจ")
        print("  2. พิมพ์ชื่อหนังสือ แล้วหาเล่มที่คล้ายกัน")
        print("  0. ปิดโปรแกรม")

        choice = input("\n(1, 2 or 0): ").strip()

        if choice == "0":
            print("กำลังปิดโปรแกรม!")
            exit()
        elif choice == "1":
            query = input("พิมพ์ความสนใจของคุณ: ").strip()
            if not query:
                print("กรุณาพิมพ์ความสนใจของคุณที่นี่ก่อนนะครับ >_< \n")
                continue
            results = recommend_by_interest(query)

            print(f"หนังสือแนะนำสำหรับ '{query}' 10 อันดับได้แก่")
            print("-" * 50)
            print_results(results)
        elif choice == "2":
            while True:
                title = input("พิมพ์ชื่อหนังสือ (บางส่วนก็ได้): ").strip()
                if not title:
                    print("กรุณาพิมพ์ชื่อหนังสือของคุณที่นี่ก่อนนะครับ!")
                    continue

                # ค้นหาหนังสือที่ชื่อคล้ายกัน
                matches = search_book(title)
                if matches.empty:
                    print(f"ไม่พบหนังสือชื่อ '{title}' กรุณาลองใหม่")
                    continue

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
                    found_title, results = recommend_by_title(select_title)
                    print(f"หนังสือแนะนำสำหรับ '{found_title}' 10 อันดับได้แก่")
                    print_results(results)
                    break
