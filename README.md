# Book Content-Based Recommender System
ระบบแนะนำหนังสือตามความสนใจของผู้ใช้
ข้อมูลหนังสือจาก [naiin.com](https://www.naiin.com/)

## หลักการทำงาน 

```
naiin.com --> scraper.py --> cleanup.py --> recommender.py --> ebooks.csv 
          --> main.py
          --> api.py, index.html (vibe code 110%)

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

### รัน Script 
```bash
python src/main.py
```
### สามารถรันระบบแนะนำหนังสือแบบ localhost
```bash
uvicorn src.api:app --reload
```
จากนั้นเปิด http://localhost:8000 

---
## ผู้จัดทำ

- นาย ชวกร ทองอินทร์ รหัสนักศึกษา 683380069-9
- นาย ธนกร มีฤทธิ์ รหัสนักศึกษา 683380294-2

วิทยาลัยการคอมพิวเตอร์ สาขาวิทยาการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น  
โครงงานรายวิชาวิทยาการคำนวณ ภาคเรียนที่ 2 ปีการศึกษา 2568