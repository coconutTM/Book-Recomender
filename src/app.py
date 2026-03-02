"""
app.py
Web Application สำหรับระบบแนะนำหนังสือ
รันด้วย: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os
from recommender import BookRecommender

# =========================================================
# ตั้งค่า Page
# =========================================================

st.set_page_config(
    page_title="📚 ระบบแนะนำหนังสือ | Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CSS แต่งหน้า
# =========================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #4a4a6a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .book-card {
        background: linear-gradient(135deg, #f8f9ff 0%, #eef1ff 100%);
        border: 1px solid #d0d7ff;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: box-shadow 0.2s;
    }
    .book-card:hover {
        box-shadow: 0 4px 16px rgba(100,100,255,0.15);
    }
    .book-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .score-badge {
        display: inline-block;
        background: #4361ee;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .category-badge {
        display: inline-block;
        background: #e0e7ff;
        color: #3730a3;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.78rem;
        margin-right: 4px;
    }
    .tag-text {
        color: #6b7280;
        font-size: 0.82rem;
    }
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a1a2e;
        border-left: 4px solid #4361ee;
        padding-left: 10px;
        margin: 20px 0 12px 0;
    }
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# โหลด Model (Cache)
# =========================================================

DATA_FILE = "books.csv"

@st.cache_resource(show_spinner="กำลังโหลดระบบแนะนำหนังสือ...")
def load_recommender():
    rec = BookRecommender(DATA_FILE, n_recommendations=10)
    rec.load_data().build_index()
    return rec

try:
    rec = load_recommender()
    model_ready = True
except Exception as e:
    st.error(f"❌ โหลดระบบไม่สำเร็จ: {e}")
    model_ready = False


# =========================================================
# Helper Functions
# =========================================================

def render_book_card(rank: int, row: pd.Series):
    score = row.get("score_pct", 0)
    title = row.get("title", "-")
    author = row.get("author", "-")
    publisher = row.get("publisher", "-")
    category = row.get("main_category", "")
    tags = row.get("tags", "")
    desc = row.get("description", "")
    url = row.get("url", "")

    # สีของ score badge
    if score >= 80:
        badge_color = "#16a34a"
    elif score >= 50:
        badge_color = "#4361ee"
    else:
        badge_color = "#9ca3af"

    desc_short = (desc[:200] + "...") if len(desc) > 200 else desc

    link_html = f'<a href="{url}" target="_blank" style="color:#4361ee;font-size:0.8rem;">🔗 ดูที่นายอินทร์</a>' if url else ""

    st.markdown(f"""
    <div class="book-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:8px;">
            <div class="book-title">#{rank} {title}</div>
            <span style="background:{badge_color};color:white;padding:2px 12px;border-radius:20px;font-size:0.82rem;font-weight:700;">
                คะแนน {score:.1f}%
            </span>
        </div>
        <div style="margin-top:6px;">
            <span class="category-badge">{category}</span>
        </div>
        <div style="margin-top:8px; color:#374151; font-size:0.9rem;">
            ✍️ <b>{author}</b> &nbsp;|&nbsp; 🏢 {publisher}
        </div>
        <div class="tag-text" style="margin-top:4px;">🏷️ {tags}</div>
        <div style="margin-top:8px; color:#4b5563; font-size:0.85rem;">{desc_short}</div>
        <div style="margin-top:8px;">{link_html}</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# Header
# =========================================================

st.markdown('<div class="main-header">📚 ระบบแนะนำหนังสือ</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Book Content-Based Recommender System | ข้อมูลจาก naiin.com</div>',
    unsafe_allow_html=True
)

# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.image("https://www.naiin.com/images/logo-naiin.svg", width=140, use_container_width=False)
    st.markdown("---")

    st.markdown("### ⚙️ ตั้งค่าการแนะนำ")
    n_results = st.slider("จำนวนผลลัพธ์", min_value=3, max_value=20, value=8)

    categories = ["ทุกหมวดหมู่"] + (rec.get_categories() if model_ready else [])
    selected_cat = st.selectbox("กรองหมวดหมู่", categories)
    cat_filter = None if selected_cat == "ทุกหมวดหมู่" else selected_cat

    st.markdown("---")
    st.markdown("### ℹ️ เกี่ยวกับระบบ")
    st.markdown("""
    ระบบใช้ **TF-IDF** แปลงข้อความเนื้อหาหนังสือให้เป็น vector
    แล้วคำนวณ **Cosine Similarity** เพื่อเปรียบเทียบกับความสนใจของผู้ใช้
    """)

    if model_ready:
        st.metric("จำนวนหนังสือในระบบ", f"{len(rec.df):,} เล่ม")

    st.markdown("---")
    st.caption("โครงงานรายวิชา Computational Science | ม.ขอนแก่น ปีการศึกษา 2568")


# =========================================================
# Main Tabs
# =========================================================

if not model_ready:
    st.stop()

tab1, tab2, tab3 = st.tabs(["🎯 แนะนำตามความสนใจ", "📖 หนังสือที่คล้ายกัน", "🔍 ค้นหาหนังสือ"])


# --------------------------------------------------
# Tab 1: แนะนำตามความสนใจ
# --------------------------------------------------
with tab1:
    st.markdown('<div class="section-title">กรอกความสนใจของคุณ</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    💡 <b>วิธีใช้:</b> พิมพ์หัวข้อ คำศัพท์ หรือประโยคที่อธิบายสิ่งที่สนใจ
    เช่น <i>"Python machine learning data science"</i> หรือ <i>"การลงทุนหุ้น กองทุน การเงิน"</i>
    </div>
    """, unsafe_allow_html=True)

    # ตัวอย่าง preset
    presets = {
        "": "— เลือกตัวอย่าง —",
        "Python machine learning data science AI": "🤖 AI & Machine Learning",
        "การลงทุนหุ้น กองทุน การเงินส่วนบุคคล": "💰 การเงินและการลงทุน",
        "web development JavaScript React frontend": "🌐 Web Development",
        "การพัฒนาตนเอง productivity นิสัย ทักษะ": "🌱 พัฒนาตนเอง",
        "database SQL cloud AWS DevOps": "☁️ Cloud & Database",
        "คณิตศาสตร์ สถิติ ฟิสิกส์ วิทยาศาสตร์": "🔬 วิทยาศาสตร์",
    }

    col1, col2 = st.columns([2, 1])
    with col1:
        interest_input = st.text_area(
            "ความสนใจ",
            placeholder="เช่น: Python data science machine learning visualization",
            height=100,
            label_visibility="collapsed",
        )
    with col2:
        preset_label = st.selectbox(
            "หรือเลือกหัวข้อตัวอย่าง",
            options=list(presets.values()),
            label_visibility="visible",
        )
        if preset_label != "— เลือกตัวอย่าง —":
            preset_text = [k for k, v in presets.items() if v == preset_label][0]
            if preset_text:
                interest_input = preset_text

    search_btn = st.button("🔍 แนะนำหนังสือ", type="primary", use_container_width=True)

    if search_btn:
        if not interest_input.strip():
            st.warning("กรุณากรอกความสนใจก่อนค้นหา")
        else:
            with st.spinner("กำลังประมวลผล..."):
                results = rec.recommend_by_interest(
                    interest_input.strip(),
                    n=n_results,
                    category_filter=cat_filter,
                )

            if results.empty:
                st.info("ไม่พบหนังสือที่ตรงกับความสนใจในหมวดหมู่ที่เลือก")
            else:
                st.markdown(
                    f'<div class="section-title">📋 ผลการแนะนำ ({len(results)} เล่ม)</div>',
                    unsafe_allow_html=True
                )
                for i, row in results.iterrows():
                    render_book_card(i + 1, row)

                # Download CSV
                csv = results[["title", "author", "main_category", "score_pct", "url"]].to_csv(
                    index=False, encoding="utf-8-sig"
                )
                st.download_button(
                    "📥 ดาวน์โหลดผลลัพธ์ CSV",
                    data=csv,
                    file_name="book_recommendations.csv",
                    mime="text/csv",
                )


# --------------------------------------------------
# Tab 2: หนังสือที่คล้ายกัน (Item-based)
# --------------------------------------------------
with tab2:
    st.markdown('<div class="section-title">ค้นหาหนังสือที่คล้ายกัน</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    💡 พิมพ์ชื่อหนังสือ (บางส่วนก็ได้) เพื่อดูหนังสือที่มีเนื้อหาใกล้เคียงกัน
    </div>
    """, unsafe_allow_html=True)

    all_titles = rec.df["title"].tolist()
    book_search_term = st.text_input("ชื่อหนังสือ", placeholder="เช่น: Python, Machine Learning, React...")

    if book_search_term:
        matches = rec.df[rec.df["title"].str.contains(book_search_term, case=False, na=False)]
        if matches.empty:
            st.warning("ไม่พบหนังสือที่ตรงกัน กรุณาลองคำอื่น")
        else:
            selected_title = st.selectbox(
                "เลือกหนังสือ",
                options=matches["title"].tolist(),
            )
            find_similar_btn = st.button("🔍 หาหนังสือที่คล้ายกัน", type="primary")

            if find_similar_btn:
                with st.spinner("กำลังคำนวณ..."):
                    similar = rec.recommend_similar(selected_title, n=n_results)

                if similar.empty:
                    st.info("ไม่พบหนังสือที่คล้ายกัน")
                else:
                    st.markdown(
                        f'<div class="section-title">📋 หนังสือที่คล้ายกับ "{selected_title}"</div>',
                        unsafe_allow_html=True
                    )
                    for i, row in similar.iterrows():
                        render_book_card(i + 1, row)


# --------------------------------------------------
# Tab 3: ค้นหาหนังสือ
# --------------------------------------------------
with tab3:
    st.markdown('<div class="section-title">ค้นหาหนังสือในฐานข้อมูล</div>', unsafe_allow_html=True)

    search_kw = st.text_input("🔍 คำค้นหา", placeholder="ชื่อหนังสือ, หัวข้อ, หมวดหมู่...")
    if search_kw:
        found = rec.search_books(search_kw)
        if found.empty:
            st.info(f"ไม่พบหนังสือที่เกี่ยวกับ '{search_kw}'")
        else:
            st.success(f"พบ {len(found)} รายการ")
            st.dataframe(found, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">📊 ข้อมูลทั้งหมดในระบบ</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        cat_counts = rec.df["main_category"].value_counts().reset_index()
        cat_counts.columns = ["หมวดหมู่", "จำนวนหนังสือ"]
        st.dataframe(cat_counts, use_container_width=True, hide_index=True)

    with col2:
        st.bar_chart(cat_counts.set_index("หมวดหมู่"))
