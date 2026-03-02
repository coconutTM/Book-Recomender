"""
evaluate.py
ประเมินประสิทธิภาพของระบบแนะนำหนังสือ
วัดค่า Precision, Recall และ NDCG@K
"""

import pandas as pd
import numpy as np
from recommender import BookRecommender


# =========================================================
# ชุดทดสอบ (Ground Truth)
# =========================================================

# นิยามชุดคำถาม + หมวดหมู่ที่ "ถูกต้อง"
TEST_CASES = [
    {
        "query":           "Python machine learning data science",
        "relevant_cats":   {"การเขียนโปรแกรม", "คอมพิวเตอร์และเทคโนโลยี"},
        "required_keywords": ["python", "machine learning", "data", "AI"],
    },
    {
        "query":           "การลงทุนหุ้น กองทุน การเงินส่วนบุคคล",
        "relevant_cats":   {"ธุรกิจและการบริหาร"},
        "required_keywords": ["การเงิน", "ลงทุน", "หุ้น"],
    },
    {
        "query":           "web development JavaScript React frontend",
        "relevant_cats":   {"การเขียนโปรแกรม"},
        "required_keywords": ["javascript", "react", "web"],
    },
    {
        "query":           "คณิตศาสตร์ สถิติ ฟิสิกส์",
        "relevant_cats":   {"วิทยาศาสตร์และคณิตศาสตร์"},
        "required_keywords": ["คณิตศาสตร์", "สถิติ", "ฟิสิกส์", "วิทยาศาสตร์"],
    },
    {
        "query":           "startup entrepreneur business innovation",
        "relevant_cats":   {"ธุรกิจและการบริหาร"},
        "required_keywords": ["startup", "ธุรกิจ", "นวัตกรรม"],
    },
    {
        "query":           "ภาษาอังกฤษ IELTS TOEIC communication",
        "relevant_cats":   {"ภาษาและการสื่อสาร"},
        "required_keywords": ["english", "ภาษา"],
    },
    {
        "query":           "พัฒนาตนเอง นิสัย productivity ทักษะ",
        "relevant_cats":   {"พัฒนาตนเอง"},
        "required_keywords": ["นิสัย", "productivity", "ทักษะ", "self"],
    },
    {
        "query":           "cybersecurity network security hacking",
        "relevant_cats":   {"คอมพิวเตอร์และเทคโนโลยี"},
        "required_keywords": ["security", "hacking", "network"],
    },
]


# =========================================================
# Metric Functions
# =========================================================

def precision_at_k(recommended: list[str], relevant_cats: set, k: int) -> float:
    """Precision@K: สัดส่วนผลลัพธ์ที่ถูกต้องใน top-K"""
    top_k = recommended[:k]
    hits = sum(1 for cat in top_k if cat in relevant_cats)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended: list[str], relevant_cats: set, k: int, total_relevant: int) -> float:
    """Recall@K: สัดส่วนของ relevant items ที่ถูกดึงขึ้นมา"""
    if total_relevant == 0:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for cat in top_k if cat in relevant_cats)
    return hits / total_relevant


def ndcg_at_k(recommended: list[str], relevant_cats: set, k: int) -> float:
    """NDCG@K: วัดคุณภาพการจัดอันดับ (ผลที่ถูกต้องควรอยู่ด้านบน)"""
    top_k = recommended[:k]
    dcg = sum(
        (1 / np.log2(i + 2)) if cat in relevant_cats else 0
        for i, cat in enumerate(top_k)
    )
    ideal = sorted([1 if cat in relevant_cats else 0 for cat in top_k], reverse=True)
    idcg = sum(
        (ideal[i] / np.log2(i + 2)) for i in range(len(ideal))
    )
    return dcg / idcg if idcg > 0 else 0.0


def keyword_hit_rate(results_df: pd.DataFrame, keywords: list[str]) -> float:
    """ตรวจสอบว่าผลลัพธ์มี keyword ที่เกี่ยวข้องอยู่กี่ %"""
    if results_df.empty:
        return 0.0
    combined = " ".join(
        (results_df["title"] + " " + results_df["tags"] + " " + results_df["description"])
        .fillna("").str.lower().tolist()
    )
    hits = sum(1 for kw in keywords if kw.lower() in combined)
    return hits / len(keywords) if keywords else 0.0


# =========================================================
# รัน Evaluation
# =========================================================

def run_evaluation(rec: BookRecommender, k: int = 5):
    print("=" * 65)
    print(f"  ประเมินประสิทธิภาพ Book Recommender System (K={k})")
    print("=" * 65)

    p_scores, r_scores, ndcg_scores, kw_scores = [], [], [], []

    for i, test in enumerate(TEST_CASES, 1):
        results = rec.recommend_by_interest(test["query"], n=k)
        rec_cats = results["main_category"].tolist() if not results.empty else []

        p = precision_at_k(rec_cats, test["relevant_cats"], k)
        r = recall_at_k(rec_cats, test["relevant_cats"], k, total_relevant=k)
        ndcg = ndcg_at_k(rec_cats, test["relevant_cats"], k)
        kw_hit = keyword_hit_rate(results, test["required_keywords"])

        p_scores.append(p)
        r_scores.append(r)
        ndcg_scores.append(ndcg)
        kw_scores.append(kw_hit)

        print(f"\n[{i}] Query: \"{test['query'][:50]}\"")
        print(f"    Precision@{k}  : {p:.3f}")
        print(f"    Recall@{k}     : {r:.3f}")
        print(f"    NDCG@{k}       : {ndcg:.3f}")
        print(f"    Keyword Hit   : {kw_hit:.3f}")
        if not results.empty:
            for j, row in results.iterrows():
                mark = "✅" if row["main_category"] in test["relevant_cats"] else "❌"
                print(f"      {mark} {j+1}. {row['title'][:45]} [{row['main_category']}]")

    print("\n" + "=" * 65)
    print("  สรุปผลการประเมิน (ค่าเฉลี่ย)")
    print("=" * 65)
    print(f"  Mean Precision@{k}  : {np.mean(p_scores):.4f}")
    print(f"  Mean Recall@{k}     : {np.mean(r_scores):.4f}")
    print(f"  Mean NDCG@{k}       : {np.mean(ndcg_scores):.4f}")
    print(f"  Mean Keyword Hit  : {np.mean(kw_scores):.4f}")
    print("=" * 65)

    # สร้าง summary DataFrame
    summary = pd.DataFrame({
        "Query": [t["query"][:40] for t in TEST_CASES],
        f"Precision@{k}": [round(v, 3) for v in p_scores],
        f"Recall@{k}": [round(v, 3) for v in r_scores],
        f"NDCG@{k}": [round(v, 3) for v in ndcg_scores],
        "Keyword Hit": [round(v, 3) for v in kw_scores],
    })

    avg_row = pd.DataFrame([{
        "Query": "📊 ค่าเฉลี่ย",
        f"Precision@{k}": round(np.mean(p_scores), 3),
        f"Recall@{k}": round(np.mean(r_scores), 3),
        f"NDCG@{k}": round(np.mean(ndcg_scores), 3),
        "Keyword Hit": round(np.mean(kw_scores), 3),
    }])
    summary = pd.concat([summary, avg_row], ignore_index=True)

    summary.to_csv("evaluation_results.csv", index=False, encoding="utf-8-sig")
    print("\n✅ บันทึกผลการประเมิน → evaluation_results.csv")
    return summary


if __name__ == "__main__":
    rec = BookRecommender("books.csv", n_recommendations=10)
    rec.load_data().build_index()
    run_evaluation(rec, k=5)
