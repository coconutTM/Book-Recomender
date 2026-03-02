# 📚 Book Content-Based Recommender System

ระบบแนะนำหนังสือแบบอิงเนื้อหา (Content-Based Filtering)  
โครงงานรายวิชา Computational Science | วิทยาลัยการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น ปีการศึกษา 2568

---

## โครงสร้างโปรเจกต์

```
book_recommender/
├── scraper.py         ← ดึงข้อมูลหนังสือจาก naiin.com
├── recommender.py     ← ระบบแนะนำหนังสือ (TF-IDF + Cosine Similarity)
├── app.py             ← Web Application (Streamlit)
├── evaluate.py        ← ประเมินประสิทธิภาพระบบ
├── books.csv          ← ข้อมูลหนังสือตัวอย่าง (50 เล่ม)
├── requirements.txt   ← Python dependencies
└── README.md
```

---

## วิธีติดตั้ง

```bash
# 1. สร้าง virtual environment (แนะนำ)
python -m venv venv
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

# 2. ติดตั้ง dependencies
pip install -r requirements.txt

# 3. ติดตั้ง PyThaiNLP (สำหรับตัด word ภาษาไทย)
pip install pythainlp
python -m pythainlp.tools download
```

---

## วิธีใช้งาน

### ขั้นที่ 1: ดึงข้อมูลจาก naiin.com (ต้องเชื่อมต่ออินเทอร์เน็ต)

```bash
python scraper.py
```

ระบบจะดึงข้อมูลหนังสือจากทุกหมวดหมู่และบันทึกเป็นไฟล์ `books.csv`  
(ถ้าต้องการใช้ข้อมูลตัวอย่างที่มีให้แล้ว ข้ามขั้นนี้ได้เลย)

### ขั้นที่ 2: รัน Web Application

```bash
streamlit run app.py
```

เปิดเบราว์เซอร์ที่ `http://localhost:8501`

### ขั้นที่ 3: ประเมินประสิทธิภาพ

```bash
python evaluate.py
```

---

## หลักการทำงาน

```
Input: ความสนใจของผู้ใช้ (text)
          │
          ▼
    ┌─────────────────┐
    │  Text Preprocessing  │   ← ตัดคำ ลบ stop words (PyThaiNLP)
    └─────────────────┘
          │
          ▼
    ┌─────────────────┐
    │  TF-IDF Vectorizer   │   ← แปลงข้อความเป็น numerical vector
    └─────────────────┘
          │
          ▼
    ┌─────────────────┐
    │  Cosine Similarity   │   ← เปรียบเทียบ query กับหนังสือทุกเล่ม
    └─────────────────┘
          │
          ▼
Output: หนังสือที่แนะนำ เรียงตาม similarity score
```

### TF-IDF (Term Frequency–Inverse Document Frequency)
- **TF**: ความถี่ที่คำปรากฏในเอกสาร
- **IDF**: ลดน้ำหนักคำที่ปรากฏในเอกสารจำนวนมาก (คำทั่วไป)
- ผลลัพธ์คือ vector ที่บ่งบอกความสำคัญของแต่ละคำ

### Cosine Similarity
```
similarity = cos(θ) = (A · B) / (|A| × |B|)
```
- วัดมุมระหว่าง vector ของ query กับหนังสือแต่ละเล่ม
- ค่าใกล้ 1 = คล้ายกันมาก, ค่าใกล้ 0 = ไม่เกี่ยวข้องกัน

---

## Metrics ที่ใช้ประเมิน

| Metric | ความหมาย |
|--------|----------|
| Precision@K | สัดส่วนหนังสือที่ถูกต้องใน top-K |
| Recall@K | สัดส่วนของหนังสือที่เกี่ยวข้องที่ถูกดึงขึ้นมา |
| NDCG@K | คุณภาพการจัดอันดับ (ผลที่ถูกควรอยู่บน) |
| Keyword Hit Rate | % ของ keyword ที่ปรากฏในผลลัพธ์ |

---

## ผู้จัดทำ

- นาย ชวกร ทองอินทร์ รหัสนักศึกษา 683380069-9
- นาย ธนกร มีฤทธิ์ รหัสนักศึกษา 683380294-2 

วิทยาลัยการคอมพิวเตอร์ สาขาวิทยาการคอมพิวเตอร์ มหาวิทยาลัยขอนแก่น
