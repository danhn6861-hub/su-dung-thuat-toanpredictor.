import streamlit as st

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Quà Tặng Từ Vũ Trụ",
    page_icon="🌌",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. KHU VỰC THAY ĐỔI LINK ẢNH (ĐÃ THAY BẰNG CODELINK TRONG SUỐT 100%) ---
# Mẹo: Các link dưới đây lấy từ Giphy Sticker nên đảm bảo xóa phông hoàn toàn.
# Nếu muốn tự thay, hãy lên giphy.com/stickers (phải chọn mục STICKERS thì nền mới trong suốt).
LINK_ANH_HOP_QUA = "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExM3Z0NTA4YnFwYmN4Yms3YnY0NTA2bms0Ym15NDFpYmVidG4xdTFvcyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/vX79B48fKgnW9YosvO/giphy.gif" 
LINK_ANH_DOA_HOA = "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbW90YzAwbnd6ZHp4N2pndjZ6YTJ5YXBwYmZ4MXB5M29tN3k0ZXFxeSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/8g759bSgK9AOkbYvL9/giphy.gif"

# --- 3. CSS CUSTOM: VŨ TRỤ & GLASSMORPHISM ---
st.markdown("""
<style>
    /* Nền dải ngân hà sâu thẳm */
    .stApp {
        background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%);
        overflow-x: hidden;
        color: white;
    }

    /* Phủ một lớp sao lấp lánh */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background: transparent url('https://www.transparenttextures.com/patterns/stardust.png') repeat;
        opacity: 0.4;
        pointer-events: none;
    }

    /* Thẻ Glassmorphism chứa toàn bộ nội dung (bao gồm cả nút) */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 35px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.7);
        margin: 30px auto;
        max-width: 480px;
        animation: slideUp 1s cubic-bezier(0.19, 1, 0.22, 1);
    }

    /* Chữ tiêu đề Gradient ánh kim vũ trụ */
    .title-text {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        background: linear-gradient(135deg, #fad0c4 0%, #ffd1ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.2rem;
        margin-bottom: 15px;
        letter-spacing: -0.5px;
    }

    .sub-text {
        font-size: 1.1rem;
        color: #e0e0e0;
        line-height: 1.6;
        margin-bottom: 10px;
    }

    /* Định dạng ảnh hộp quà bay lơ lửng ma mị */
    .gift-img {
        width: 180px;
        height: 180px;
        object-fit: contain;
        margin: 15px auto;
        filter: drop-shadow(0 0 20px rgba(255, 209, 255, 0.6));
        animation: float 3s ease-in-out infinite;
    }
    
    /* Định dạng đóa hoa chui ra từ hộp quà bùng nổ */
    .flower-img {
        width: 220px;
        height: 220px;
        object-fit: contain;
        margin: 15px auto;
        filter: drop-shadow(0 0 25px rgba(250, 208, 196, 0.8));
        animation: popAndBloom 1.2s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
    }

    /* --- Hệ thống Animation chuyển động mượt mà --- */
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-15px) rotate(3deg); }
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(40px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes popAndBloom {
        0% { transform: scale(0) translateY(50px); opacity: 0; }
        60% { transform: scale(1.1) translateY(-10px); opacity: 1; }
        100% { transform: scale(1) translateY(0); }
    }

    /* Tối ưu Responsive tuyệt đối cho thiết bị di động nhỏ */
    @media (max-width: 480px) {
        .glass-card { padding: 20px; margin: 15px; }
        .title-text { font-size: 1.7rem; }
        .sub-text { font-size: 1rem; }
        .gift-img { width: 140px; height: 140px; }
        .flower-img { width: 180px; height: 180px; }
    }
    
    /* Thiết kế lại nút bấm Streamlit lồng bên trong thẻ kính */
    div.stButton > button {
        background: linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%);
        color: #111 !important;
        border: none;
        padding: 12px 20px;
        border-radius: 50px;
        transition: all 0.3s ease;
        font-weight: bold;
        font-size: 1rem;
        width: 100%;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(251, 194, 235, 0.4);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(251, 194, 235, 0.7);
        color: #000 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. QUẢN LÝ TRẠNG THÁI (SESSION STATE) ---
if "step" not in st.session_state:
    st.session_state.step = 1
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# --- 5. ĐIỀU HƯỚNG GIAO DIỆN ---
st.write("##") 

# BƯỚC 1: Nhập tên
if st.session_state.step == 1:
    st.markdown("""
    <div class="glass-card">
        <div class="title-text">Tín Hiệu Vũ Trụ...</div>
        <p class="sub-text">Có một món quà từ vì sao đang tìm kiếm chủ nhân của nó. Cho hỏi tên bạn là gì?</p>
    """, unsafe_allow_html=True)
    
    name = st.text_input("", placeholder="Nhập tên của bạn vào đây...", key="name_input")
    
    if st.button("Kết nối tín hiệu ✨", use_container_width=True):
        if name.strip():
            st.session_state.user_name = name.strip()
            st.session_state.step = 2
            st.rerun()
        else:
            st.warning("Vui lòng nhập tên trước khi nhận tín hiệu nhé!")
            
    st.markdown("</div>", unsafe_allow_html=True)

# BƯỚC 2: Hộp quà lơ lửng chờ mở
elif st.session_state.step == 2:
    st.markdown(f"""
    <div class="glass-card">
        <div class="title-text">Xin chào, {st.session_state.user_name}!</div>
        <p class="sub-text">Giữa ngân hà bao la rộng lớn, nhà phát triển có một món quà nhỏ muốn gửi tặng riêng cho bạn.</p>
        <img src="{LINK_ANH_HOP_QUA}" class="gift-img">
    """, unsafe_allow_html=True)
    
    # Nút bấm đã được lồng ghép chuẩn chỉnh vào trong khối DIV kính mờ
    if st.button("Mở món quà từ vì sao 🎁", use_container_width=True):
        st.session_state.step = 3
        st.rerun()
        
    st.markdown("</div>", unsafe_allow_html=True)

# BƯỚC 3: Hộp quà biến mất, hoa bay vút lên nở rộ
elif st.session_state.step == 3:
    st.markdown(f"""
    <div class="glass-card">
        <div class="title-text">Dành Cho {st.session_state.user_name}</div>
        <p class="sub-text">Chúc bạn luôn rạng rỡ, hạnh phúc và tỏa sáng rực rỡ như những vì tinh tú!</p>
        <img src="{LINK_ANH_DOA_HOA}" class="flower-img">
        <p style="margin-top: 20px; font-style: italic; color: #ffd1ff; font-size: 0.85rem; opacity: 0.8;">--- From Developer with Love ---</p>
    """, unsafe_allow_html=True)
    
    st.balloons() # Hiệu ứng bóng bay phụ họa thêm sinh động
    
    if st.button("Quay lại vũ trụ 🔄", use_container_width=True):
        st.session_state.step = 1
        st.session_state.user_name = ""
        st.rerun()
        
    st.markdown("</div>", unsafe_allow_html=True)
