import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# โหลดข้อมูล
df = pd.read_csv("books_cleaned.csv")
# รวม title + description เป็น text เดียว
df["content"] = df["title"] + " " + df["description"]
df["content"] = df["content"].fillna("")

# สร้าง TF-IDF Metrix
vectorizer = TfidfVectorizer()
tfidf_metrix = vectorizer.fit_transform(df["content"])

print(f"\nโหลดหนังสือ {len(df)} เล่ม!")


# Function recommend!!
def recommend(query, top_n=10):
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


print("-" * 50)
print("ระบบแนะนำหนังสือจากเว็บไซต์ 'naiin.com'")
print("-" * 50)
print("พิมพ์ 'exit' เพื่อออก")
print()

while True:
    query = input("พิมพ์ความสนใจของคุณ: ").strip()

    if query.lower() == "exit":
        print("กำลังปิดโปรแกรม... แล้วพบกันใหม่\n")
        break
    if not query:
        print("พิมพ์ความสนใจของคุณที่นี่ก่อนนะครับ >_< \n")
        continue

    results = recommend(query=query, top_n=10)
    if results.empty:
        print("เสียใจด้วย ไม่พบหนังสือเลย กรุณาลองใหม่... \n")
        continue

    print(f"\nหนังสือแนะนำสำหรับ '{query}' 10 อันดับได้แก่")
    print("-" * 50)

    for i, row in results.iterrows():
        print(f"{i+1:2}. {row["title"]}")
        print(f"         ผู้เขียน: {row["author"]}")
        print(f"         สำนักพิมพ์: {row["publisher"]}")
        print(f"         ราคา: {row["price"]} บาท")
        print(f"         คะแนนความเกี่ยวข้อง {row["score"]}")
        print(f"         {row["url"]}")
        print()

    print("-" * 50 + "\n")
