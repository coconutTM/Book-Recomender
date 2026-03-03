import pandas as pd
import os
import re

# df = pd.read_csv("books.csv")
df = pd.read_csv(os.path.join("data", "books.csv"))
print(f"ก่อน clean: {len(df)} แถว")

# 1. ลบแถวที่ title ซ้ำ
df = df.drop_duplicates(subset="title")

# 2. ตัด "บาท" ออกจากราคา → เหลือแค่ตัวเลข
df["price"] = df["price"].str.replace(" บาท", "", regex=False).str.strip()

# 3. ตัด URL ออกจาก description
df["description"] = (
    df["description"].str.replace(r"https?://\S+", "", regex=True).str.strip()
)

# 4. ลบแถวที่ description ว่าง หรือสั้นกว่า 20 ตัวอักษร
df = df[df["description"].notna()]
df = df[df["description"].str.len() >= 20]

print(f"หลัง clean: {len(df)} แถว")
print(f"ลบออกไป: {725 - len(df)} แถว")

# df.to_csv("books_cleaned.csv", index=False, encoding="utf-8-sig")
df.to_csv(os.path.join("data", "books_cleaned.csv"), index=False)
print("✅ บันทึกแล้ว → books_cleaned.csv")
