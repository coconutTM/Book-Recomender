from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# =========================================================
# โหลดข้อมูลตอนเริ่ม server
# =========================================================
df = pd.read_csv(os.path.join("data", "tmp_cleaned.csv"))
df["content"] = (df["title"] + " " + df["description"]).fillna("")

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["content"])

print(f"✅ โหลดหนังสือ {len(df)} เล่ม พร้อมแนะนำแล้ว!")


# =========================================================
# Models
# =========================================================
class QueryRequest(BaseModel):
    query: str
    top_n: int = 10


class TitleRequest(BaseModel):
    title: str
    top_n: int = 10


# =========================================================
# API Routes
# =========================================================


@app.get("/")
def index():
    return FileResponse("src/static/index.html")


@app.post("/recommend/interest")
def recommend_by_interest(req: QueryRequest):
    """แนะนำหนังสือจากความสนใจ"""
    query_vec = vectorizer.transform([req.query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][: req.top_n]

    results = []
    for idx in top_indices:
        score = round(float(scores[idx]), 4)
        if score <= 0:
            continue
        row = df.iloc[idx]
        results.append(
            {
                "title": row["title"],
                "author": row["author"],
                "publisher": row["publisher"],
                "price": row["price"],
                "url": row["url"],
                "img_url": row["img_url"] if pd.notna(row["img_url"]) else "",
                "score": score,
            }
        )

    return {"query": req.query, "results": results}


@app.post("/recommend/title")
def recommend_by_title(req: TitleRequest):
    """แนะนำหนังสือจากชื่อหนังสือ"""
    matches = df[df["title"].str.contains(req.title, case=False, na=False, regex=False)]
    if matches.empty:
        return {"error": f"ไม่พบหนังสือชื่อ '{req.title}'"}

    book_idx = matches.index[0]
    found_title = df.loc[book_idx, "title"]

    book_vec = tfidf_matrix[book_idx]
    scores = cosine_similarity(book_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1]
    top_indices = [i for i in top_indices if i != book_idx][: req.top_n]

    results = []
    for idx in top_indices:
        score = round(float(scores[idx]), 4)
        if score <= 0:
            continue
        row = df.iloc[idx]
        results.append(
            {
                "title": row["title"],
                "author": row["author"],
                "publisher": row["publisher"],
                "price": row["price"],
                "url": row["url"],
                "img_url": row["img_url"] if pd.notna(row["img_url"]) else "",
                "score": score,
            }
        )

    return {"found_title": found_title, "results": results}


@app.get("/search")
def search_books(q: str):
    matches = df[df["title"].str.contains(q, case=False, na=False, regex=False)]
    books = matches[["title", "img_url"]].head(20).fillna("").to_dict("records")
    return {"books": books}


# =========================================================
# Static files
# =========================================================
app.mount("/static", StaticFiles(directory="src/static"), name="static")
