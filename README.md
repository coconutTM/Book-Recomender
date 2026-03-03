## ติดตั้ง

```bash
# 1. clone โปรเจกต์
git clone https://github.com/coconutTM/book_recommender.git
cd book_recommender

# 2. สร้าง virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows

# 3. ติดตั้ง dependencies
pip install -r requirements.txt

# 4. ติดตั้ง Playwright browser
playwright install chromium
```
---
## วิธีใช้งาน

### ขั้นที่ 1 — ดึงข้อมูลหนังสือ
```bash
python src/scraper.py
```
จะได้ไฟล์ `data/books.csv`

### ขั้นที่ 2 — ทำความสะอาดข้อมูล
```bash
python src/cleaner.py
```
จะได้ไฟล์ `data/books_cleaned.csv`

### ขั้นที่ 3 — รันระบบแนะนำหนังสือ
```bash
python src/recommender.py
```
