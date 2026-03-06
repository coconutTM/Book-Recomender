# Book Content-Based Recommender System
ระบบแนะนำหนังสือตามความสนใจของผู้ใช้
ข้อมูลหนังสือจาก [naiin.com](https://www.naiin.com/)

## หลักการทำงาน 

```
naiin.com --> scraper.py --> cleanup.py --> recommender.py --> ebooks.csv 
          --> main.py

|                                        
|                                        

recommender.py         

|
|

ผู้ใช้เลือก : แนะนำตามความสนใจของผู้ใช้ 
         เลือกหนังสือที่สนใจ แล้วแนะนำตามเล่มนั้น

|
|

TF-IDF + Cosine Similarity

|
|

แนะนำหนังสือ Top 10 ที่ใกล้เคียงที่สุด 
```

**TF-IDF** — แปลง title + description ของหนังสือแต่ละเล่มให้เป็น vector ตัวเลข  
**Cosine Similarity** — วัดความคล้ายระหว่าง vector ของผู้ใช้กับหนังสือแต่ละเล่ม แล้วเรียงอันดับ

---
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

### สามารถรัน main.py ได้เลย 
```bash
python src/main.py
```

### ขั้นที่ 1: ดึงข้อมูลหนังสือ
```bash
python src/scraper.py
```
จะได้ไฟล์ `data/ebook_links.txt`, `data/ebooks.csv`

### ขั้นที่ 2: ทำความสะอาดข้อมูล
```bash
python src/cleaner.py
```
จะได้ไฟล์ `data/ebooks_cleaned.csv`

### ขั้นที่ 3: รันระบบแนะนำหนังสือ
**3.1** CLI
```bash
python src/recommender.py
```
**3.2** website
```
uvicorn src.api:app --reload
```
จากนั้นเปิด http://localhost:8000 
---
