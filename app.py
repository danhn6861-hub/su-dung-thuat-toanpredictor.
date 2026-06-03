import streamlit as st
import time
import random
from datetime import datetime

# 1. Thiết lập cấu hình chuẩn Web Application
st.set_page_config(
    page_title="Aesthetic Workspace — Tối Giản & Hiệu Suất",
    page_icon="🕊️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Khởi tạo Session State cho các tính năng nâng cao
if 'todos' not in st.session_state:
    st.session_state.todos = []
if 'journal' not in st.session_state:
    st.session_state.journal = []
if 'brain_dump' not in st.session_state:
    st.session_state.brain_dump = ""
if 'energy_log' not in st.session_state:
    st.session_state.energy_log = []
if 'xp' not in st.session_state:
    st.session_state.xp = 0

# 3. Inject CSS để tùy biến giao diện thành một Premium Minimalist Website
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <style>
    /* Reset & Toàn bộ font chữ hệ thống */
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .stApp {
        background-color: #FAF8F5; /* Màu nền kem Minimalist */
        color: #2D2D2D;
    }
    
    /* Thiết kế Header Website */
    .web-header {
        padding: 2rem 0 1rem 0;
        border-bottom: 1px solid #EAE5DC;
        margin-bottom: 2rem;
    }
    .web-logo {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #1A1A1A;
    }
    .web-subtitle {
        font-size: 0.95rem;
        color: #7A756B;
        margin-top: 0.2rem;
    }
    
    /* Thẻ Container cao cấp (Bento Grid Layout) */
    .card-premium {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 14px;
        border: 1px solid #EFEBE4;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.015);
        margin-bottom: 1.2rem;
        transition: all 0.3s ease;
    }
    .card-premium:hover {
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.03);
        border-color: #DFDAD0;
    }
    
    /* Định dạng Typography các thẻ tiêu đề */
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1A1A1A;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Nhãn tiến trình tối giản */
    .metric-num {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2D2D2D;
        line-height: 1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8A8477;
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Custom thanh Tabs giống thanh điều hướng website thực tế */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: transparent;
        border-bottom: 1px solid #EAE5DC;
        padding-bottom: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        white-space: pre;
        background-color: transparent;
        border-radius: 6px;
        color: #7A756B;
        font-weight: 500;
        font-size: 0.95rem;
        padding: 0 16px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #1A1A1A;
        background-color: #F1ECE3;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
    }
    
    /* Huy hiệu phân loại công việc */
    .badge-high { background: #FCE8E6; color: #A8201A; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
    .badge-medium { background: #FEF3D6; color: #8F6B00; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
    .badge-low { background: #E6F4EA; color: #137333; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# 4. Web App Header Layout
st.markdown("""
    <div class="web-header">
        <div class="web-logo">STUDIO WORKSPACE</div>
        <div class="web-subtitle">Hệ thống quản trị mục tiêu và tối ưu hóa không gian tập trung cá nhân.</div>
    </div>
""", unsafe_allow_html=True)

# Khởi tạo thanh điều hướng Tab cao cấp
tab_dashboard, tab_focus, tab_planner, tab_workspace = st.tabs([
    "📊 Bảng điều khiển", 
    "⏱️ Không gian Pomodoro", 
    "🎯 Trình quản lý mục tiêu", 
    "✍️ Góc nháp & Lưu trữ"
])

# ==========================================
# TAB 1: BẢNG ĐIỀU KHIỂN & ĐỘNG LỰC TRƯỞNG THÀNH
# ==========================================
with tab_dashboard:
    col_dash_left, col_dash_right = st.columns([7, 3])
    
    with col_dash_left:
        # Lời đề tựa tối giản thay thế các câu nói sến súa cũ
        st.markdown(f"""
            <div class="card-premium" style="background: linear-gradient(135deg, #FDFBF7 0%, #F5EFEB 100%); border-left: 4px solid #C4A484;">
                <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; color: #A09485; font-weight: 600;">Ý niệm hôm nay</div>
                <div style="font-size: 1.25rem; font-weight: 500; color: #2D2D2D; margin-top: 0.6rem; line-height: 1.6; font-style: italic;">
                    "Sự tập trung sâu sắc là nghệ thuật loại bỏ những điều thừa thãi. Hãy hoàn thành tốt một việc nhỏ trước khi nghĩ tới những điều lớn lao."
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Grid thống kê hiệu suất dạng số tối giản
        st.markdown('<div class="section-title">📉 Chỉ số tiến độ cá nhân</div>', unsafe_allow_html=True)
        total_tasks = len(st.session_state.todos)
        done_tasks = sum(1 for t in st.session_state.todos if t['done'])
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f'<div class="card-premium"><div class="metric-num">{total_tasks}</div><div class="metric-label">Tổng mục tiêu</div></div>', unsafe_allow_html=True)
        with m_col2:
            st.markdown(f'<div class="card-premium"><div class="metric-num">{done_tasks}</div><div class="metric-label">Đã xử lý</div></div>', unsafe_allow_html=True)
        with m_col3:
            rate = int(done_tasks / total_tasks * 100) if total_tasks > 0 else 0
            st.markdown(f'<div class="card-premium"><div class="metric-num">{rate}%</div><div class="metric-label">Tỷ lệ hoàn thành</div></div>', unsafe_allow_html=True)
            
        if total_tasks > 0:
            st.progress(rate / 100)
        else:
            st.info("Hệ thống chưa ghi nhận kế hoạch nào cho ngày hôm nay. Vui lòng chuyển sang tab 'Trình quản lý mục tiêu'.")

    with col_dash_right:
        # Tính năng mới: Mood & Energy Tracker (Phân tích năng lượng làm việc)
        st.markdown('<div class="card-premium">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔋 Trạng thái năng lượng</div>', unsafe_allow_html=True)
        energy_level = st.slider("Mức độ sẵn sàng làm việc của não bộ:", 0, 100, 80, help="Kéo thanh để tự đánh giá năng lượng hiện tại của bạn.")
        
        if st.button("Lưu trạng thái hôm nay", use_container_width=True):
            st.session_state.energy_log.append({"time": datetime.now().strftime("%H:%M"), "level": energy_level})
            st.toast("Đã ghi nhận chỉ số sinh học vào hệ thống.")
            
        if st.session_state.energy_log:
            st.caption("Nhật ký năng lượng gần nhất:")
            for log in st.session_state.energy_log[-2:]:
                st.markdown(f"⏱️ **{log['time']}** — Sức bền đạt **{log['level']}%**")
        st.markdown('</div>', unsafe_allow_html=True)

        # Trình phát âm thanh nền nhúng tối giản (Sử dụng playlist lofi không lời cao cấp)
        st.markdown('<div class="card-premium">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎧 Không gian âm thanh</div>', unsafe_allow_html=True)
        st.caption("Âm thanh sóng não giúp duy trì trạng thái tập trung sâu (Deep Work):")
        st.video("https://www.youtube.com/watch?v=jfKfPfyJRdk")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 2: ĐỒNG HỒ POMODORO ĐỊNH VỊ CÔNG VIỆC
# ==========================================
with tab_focus:
    st.markdown('<div class="card-premium">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⏱️ Chu kỳ tập trung Pomodoro</div>', unsafe_allow_html=True)
    
    f_col1, f_col2 = st.columns([4, 6])
    
    with f_col1:
        # Tính năng kết nối Pomodoro trực tiếp với danh sách việc cần làm thực tế
        undone_tasks = [t["task"] for t in st.session_state.todos if not t["done"]]
        if undone_tasks:
            target_task = st.selectbox("Chọn mục tiêu đang thực hiện trong phiên này:", undone_tasks)
        else:
            target_task = st.selectbox("Chọn mục tiêu đang thực hiện trong phiên này:", ["Học tập tự do / Nghiên cứu tài liệu"])
            
        focus_type = st.radio("Cấu hình phiên:", ["Phiên làm việc (25 phút)", "Nghỉ ngắn (5 phút)", "Nghỉ dài (15 phút)"])
        duration = 25 if "25" in focus_type else (5 if "5" in focus_type else 15)
        
        btn_start = st.button("Kích hoạt phiên đếm ngược", use_container_width=True, type="primary
