import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

df = pd.read_csv(os.path.join("data", "tmp_cleaned.csv"))
df["content"] = (df["title"] + " " + df["description"]).fillna("")

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["content"])

# กำหนด query ทดสอบ
test_queries = [
    "python programming",
    "machine learning",
    "network security",
    "database sql",
    "web development",
    "data science",
    "java object oriented",
    "linux operating system",
    "artificial intelligence",
    "software engineering",
]

# รันทดสอบ
all_results = []

for query in test_queries:
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_scores = sorted(scores, reverse=True)[:10]  # top 10

    all_results.append(
        {
            "query": query,
            "top1_score": round(top_scores[0], 4),  # อันดับ 1
            "top5_avg": round(np.mean(top_scores[:5]), 4),  # เฉลี่ย top 5
            "top10_avg": round(np.mean(top_scores), 4),  # เฉลี่ย top 10
            "found_books": sum(1 for s in top_scores if s > 0),  # เล่มที่เกี่ยวข้อง
        }
    )

# แสดงผล
result_df = pd.DataFrame(all_results)
print(result_df.to_string(index=False))
print()
print("ภาพรวมทั้งหมด")
print(f"Top-1 Score เฉลี่ย  : {result_df['top1_score'].mean():.4f}")
print(f"Top-5 Score เฉลี่ย  : {result_df['top5_avg'].mean():.4f}")
print(f"Top-10 Score เฉลี่ย : {result_df['top10_avg'].mean():.4f}")
print(f"Query ที่หาเล่มได้    : {(result_df['found_books'] > 0).sum()}/{len(test_queries)}")

# บันทึกลง csv
result_df.to_csv("data/evaluation_results.csv", index=False)
print("\nบันทึกแล้ว --> evaluation_results.csv")
