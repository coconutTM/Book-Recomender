import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# โหลดข้อมูล
# df = pd.read_csv("data/books_cleaned.csv")
df = pd.read_csv(os.path.join("data", "books_cleaned.csv"))
# รวม title + description เป็น text เดียว
df["content"] = df["title"] + " " + df["description"]
df["content"] = df["content"].fillna("")

# สร้าง TF-IDF Metrix
vectorizer = TfidfVectorizer()
tfidf_metrix = vectorizer.fit_transform(df["content"])

print(f"\nโหลดข้อมูลหนังสือเกี่ยวกับ computer ทั้งหมด {len(df)} เล่ม!")


# Function recommend!!


# แนะนำตามความสนใจ
def recommend_by_interest(query, top_n=10):
    # แปลงเป็น vector เพื่อคำนวณ
    query_vec = vectorizer.transform([query])

    # คำนวณ cosine-similarity กับหนังสือทุกเล่ม
    scores = cosine_similarity(query_vec, tfidf_metrix).flatten()

    # เรียงคะแนนจากมากที่สุด แล้วเอาแค่ top 10
    top_indices = scores.argsort()[::-1][:top_n]

    results = df.iloc[top_indices][
        ["title", "author", "publisher", "price", "url"]
    ].copy()
    results["score"] = scores[top_indices].round(4)
    results = results[results["score"] > 0]  # ตัดเล่มที่ไม่เกี่ยวข้องออก

    return results.reset_index(drop=True)


# แนะนำตามหนังสือที่สนใจ
def recommend_by_title(title, top_n=10):
    matches = df[df["title"].str.contains(title, case=False, na=False)]

    if matches.empty:
        return None, None

    # เอาเล่มแรกที่เจอ
    book = matches.iloc[0]
    book_idx = matches.index[0]

    # เอา vector หนังสือเล่มนั้นไปเทียบกับทุกเล่ม
    book_vec = tfidf_metrix[book_idx]
    scores = cosine_similarity(book_vec, tfidf_metrix).flatten()

    # เรียงลำดับจากมากไปน้อย แล้วตัดตัวเองออก
    top_indices = scores.argsort()[::-1]
    top_indices = [i for i in top_indices if i != book_idx][:top_n]

    results = df.iloc[top_indices][
        ["title", "author", "publisher", "price", "url"]
    ].copy()
    results["score"] = scores[top_indices].round(4)
    results = results[results["score"] > 0]  # ตัดเล่มที่ไม่เกี่ยวข้องออก

    return book["title"], results.reset_index(drop=True)


# Print data
def print_results(results):
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


def search_book(title):
    # ค้นหาหนังสือจากชื่อที่คล้ายกัน return dataframe
    matches = df[df["title"].str.contains(title, case=False, na=False)]
    return matches.reset_index()


print("-" * 50)
print("ระบบแนะนำหนังสือจากเว็บไซต์ 'naiin.com' ~~~")
print("-" * 50)
print("** พิมพ์ 'exit' เพื่อออก **")
print()

while True:
    print("\nเลือกวิธีการค้นหา: ")
    print("  1. พิมพ์ความสนใจ")
    print("  2. พิมพ์ชื่อหนังสือ แล้วหาเล่มที่คล้ายกัน")
    print("  0. ปิดโปรแกรม")

    choice = input("\nเลือก (0/1/2): ").strip()

    if choice == "0":
        print("กำลังปิดโปรแกรม!")
        break
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
