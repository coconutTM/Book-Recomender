"""
recommender.py
ระบบแนะนำหนังสือแบบ Content-Based Filtering
ใช้ TF-IDF และ Cosine Similarity ในการคำนวณความคล้ายคลึง
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import os
import re

try:
    from pythainlp.tokenize import word_tokenize
    from pythainlp.corpus import thai_stopwords
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False


# =========================================================
# ยูทิลิตี้ข้อความภาษาไทย
# =========================================================

THAI_STOP_WORDS = set()
if PYTHAINLP_AVAILABLE:
    THAI_STOP_WORDS = set(thai_stopwords())


def preprocess_text(text: str) -> str:
    """
    ทำความสะอาดและ tokenize ข้อความภาษาไทย/อังกฤษ
    - ลบอักขระพิเศษ
    - tokenize ด้วย pythainlp ถ้ามี
    - ลบ stopword
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # ลบอักขระพิเศษที่ไม่จำเป็น
    text = re.sub(r"[^\u0E00-\u0E7Fa-zA-Z0-9\s,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if PYTHAINLP_AVAILABLE:
        tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
        tokens = [t for t in tokens if t not in THAI_STOP_WORDS and len(t.strip()) > 1]
        return " ".join(tokens)
    else:
        # fallback: ใช้ whitespace tokenization
        return text.lower()


# =========================================================
# คลาสหลัก
# =========================================================

class BookRecommender:
    """
    ระบบแนะนำหนังสือแบบ Content-Based Filtering

    Algorithm:
    1. รวม features ของหนังสือ (title, description, tags, categories)
    2. แปลงเป็น TF-IDF vector
    3. คำนวณ Cosine Similarity ระหว่างความสนใจของผู้ใช้กับหนังสือแต่ละเล่ม
    4. จัดอันดับและ return ผลลัพธ์
    """

    def __init__(
        self,
        data_path: str = "books.csv",
        n_recommendations: int = 10,
        tfidf_params: dict | None = None,
    ):
        self.data_path        = data_path
        self.n_recommendations = n_recommendations
        self.df               = None
        self.tfidf_matrix     = None
        self.vectorizer       = None

        default_tfidf = dict(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            analyzer="word",
        )
        if tfidf_params:
            default_tfidf.update(tfidf_params)
        self.tfidf_params = default_tfidf

    # --------------------------------------------------
    # โหลดและเตรียมข้อมูล
    # --------------------------------------------------

    def load_data(self, df: pd.DataFrame | None = None) -> "BookRecommender":
        """โหลดข้อมูลหนังสือจาก CSV หรือ DataFrame ที่ส่งเข้ามาโดยตรง"""
        if df is not None:
            self.df = df.copy()
        elif os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path, encoding="utf-8-sig")
        else:
            raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูล: {self.data_path}")

        # เติมค่าว่างด้วย string เปล่า
        for col in ["title", "author", "description", "tags", "categories", "main_category"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("").astype(str)

        print(f"✅ โหลดข้อมูลสำเร็จ: {len(self.df)} รายการ")
        return self

    def _build_feature_text(self, row: pd.Series) -> str:
        """
        รวม feature ทุก field ให้เป็น string เดียว
        ให้น้ำหนักที่แตกต่างกันโดยการ repeat
        """
        parts = []

        # ชื่อหนังสือ (น้ำหนักสูงสุด x3)
        if row.get("title"):
            parts.extend([preprocess_text(row["title"])] * 3)

        # tags (น้ำหนักรอง x2)
        if row.get("tags"):
            parts.extend([preprocess_text(row["tags"])] * 2)

        # หมวดหมู่ (x2)
        for cat_col in ["categories", "main_category"]:
            if row.get(cat_col):
                parts.extend([preprocess_text(row[cat_col])] * 2)

        # คำอธิบาย (x1)
        if row.get("description"):
            parts.append(preprocess_text(row["description"]))

        # ผู้แต่ง สำนักพิมพ์ (x1)
        for col in ["author", "publisher"]:
            if row.get(col):
                parts.append(preprocess_text(row[col]))

        return " ".join(parts)

    def build_index(self) -> "BookRecommender":
        """สร้าง TF-IDF matrix จากข้อมูลหนังสือทั้งหมด"""
        if self.df is None:
            raise RuntimeError("กรุณาเรียก load_data() ก่อน")

        print("🔨 กำลังสร้าง TF-IDF index...")
        self.df["_feature_text"] = self.df.apply(self._build_feature_text, axis=1)

        self.vectorizer = TfidfVectorizer(**self.tfidf_params)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["_feature_text"])
        print(f"✅ สร้าง index สำเร็จ: {self.tfidf_matrix.shape}")
        return self

    # --------------------------------------------------
    # ฟังก์ชันแนะนำ
    # --------------------------------------------------

    def _query_to_vector(self, query: str):
        """แปลง query string เป็น TF-IDF vector"""
        query_processed = preprocess_text(query)
        return self.vectorizer.transform([query_processed])

    def recommend_by_interest(
        self,
        interest_text: str,
        n: int | None = None,
        category_filter: str | None = None,
    ) -> pd.DataFrame:
        """
        แนะนำหนังสือจากข้อความความสนใจของผู้ใช้

        Parameters
        ----------
        interest_text  : ข้อความอธิบายความสนใจ เช่น "python machine learning data science"
        n              : จำนวนหนังสือที่ต้องการแนะนำ (default: self.n_recommendations)
        category_filter: กรองเฉพาะหมวดหมู่ที่ระบุ (None = ทุกหมวด)
        """
        if self.tfidf_matrix is None:
            raise RuntimeError("กรุณาเรียก build_index() ก่อน")

        n = n or self.n_recommendations
        query_vec = self._query_to_vector(interest_text)
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        result_df = self.df.copy()
        result_df["similarity_score"] = scores

        if category_filter:
            mask = (
                result_df["main_category"].str.contains(category_filter, na=False)
                | result_df["categories"].str.contains(category_filter, na=False)
            )
            result_df = result_df[mask]

        result_df = result_df[result_df["similarity_score"] > 0]
        result_df = result_df.sort_values("similarity_score", ascending=False).head(n)

        # แปลง score เป็น % เพื่อความสวยงาม
        if len(result_df) > 0:
            scaler = MinMaxScaler(feature_range=(0, 100))
            result_df["score_pct"] = scaler.fit_transform(
                result_df[["similarity_score"]]
            ).round(1)

        return result_df[
            ["title", "author", "publisher", "main_category",
             "tags", "similarity_score", "score_pct", "url", "description"]
        ].reset_index(drop=True)

    def recommend_similar(
        self,
        book_title: str,
        n: int | None = None,
    ) -> pd.DataFrame:
        """
        แนะนำหนังสือที่คล้ายกับหนังสือที่ระบุ (item-to-item)

        Parameters
        ----------
        book_title : ชื่อหนังสือต้นทาง (ค้นหาแบบ partial match)
        n          : จำนวนหนังสือที่ต้องการ
        """
        if self.tfidf_matrix is None:
            raise RuntimeError("กรุณาเรียก build_index() ก่อน")

        n = n or self.n_recommendations

        # ค้นหาหนังสือต้นทาง
        matches = self.df[self.df["title"].str.contains(book_title, case=False, na=False)]
        if matches.empty:
            print(f"❌ ไม่พบหนังสือชื่อ '{book_title}'")
            return pd.DataFrame()

        idx = matches.index[0]
        book_vec = self.tfidf_matrix[idx]
        scores = cosine_similarity(book_vec, self.tfidf_matrix).flatten()

        result_df = self.df.copy()
        result_df["similarity_score"] = scores
        result_df = result_df.drop(index=idx)  # ไม่รวมหนังสือตัวเอง
        result_df = result_df.sort_values("similarity_score", ascending=False).head(n)

        if len(result_df) > 0:
            scaler = MinMaxScaler(feature_range=(0, 100))
            result_df["score_pct"] = scaler.fit_transform(
                result_df[["similarity_score"]]
            ).round(1)

        return result_df[
            ["title", "author", "publisher", "main_category",
             "tags", "similarity_score", "score_pct", "url", "description"]
        ].reset_index(drop=True)

    def get_categories(self) -> list[str]:
        """ดึงรายการหมวดหมู่ทั้งหมด"""
        if self.df is None:
            return []
        return sorted(self.df["main_category"].dropna().unique().tolist())

    def search_books(self, keyword: str) -> pd.DataFrame:
        """ค้นหาหนังสือด้วย keyword"""
        if self.df is None:
            return pd.DataFrame()
        mask = (
            self.df["title"].str.contains(keyword, case=False, na=False)
            | self.df["description"].str.contains(keyword, case=False, na=False)
            | self.df["tags"].str.contains(keyword, case=False, na=False)
        )
        return self.df[mask][["title", "author", "main_category", "tags"]].reset_index(drop=True)


# =========================================================
# ทดสอบ standalone
# =========================================================

if __name__ == "__main__":
    rec = BookRecommender("books.csv", n_recommendations=5)
    rec.load_data().build_index()

    print("\n" + "="*60)
    print("🔍 ทดสอบ: แนะนำหนังสือตามความสนใจ 'Python machine learning data science'")
    print("="*60)
    results = rec.recommend_by_interest("Python machine learning data science")
    for i, row in results.iterrows():
        print(f"{i+1}. {row['title']} ({row['main_category']}) — ความคล้าย: {row['score_pct']}%")

    print("\n" + "="*60)
    print("🔍 ทดสอบ: หนังสือคล้ายกับ 'Python สำหรับ Data Science'")
    print("="*60)
    results2 = rec.recommend_similar("Python สำหรับ Data Science")
    for i, row in results2.iterrows():
        print(f"{i+1}. {row['title']} ({row['main_category']}) — ความคล้าย: {row['score_pct']}%")
