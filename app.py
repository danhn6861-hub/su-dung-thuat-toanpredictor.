import streamlit as st

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Quà Tặng Từ Vũ Trụ",
    page_icon="🌌",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. KHU VỰC THAY ĐỔI LINK ẢNH ---
# Bạn có thể tự do thay đổi hai link ảnh/GIF dưới này theo ý muốn nhé!
LINK_ANH_HOP_QUA = "https://i.pinimg.com/736x/3f/49/93/3f4993bff0712b5bc855e6ecb77d3dd9.jpg" 
LINK_ANH_DOA_HOA = "https://i.pinimg.com/originals/91/97/8e/91978e87493a595ec7695325785a9df4.gif"

# --- 3. CSS CUSTOM: VŨ TRỤ & GLASSMORPHISM ---
# Đã xóa chữ 'f' ở đầu chuỗi để loại bỏ hoàn toàn lỗi SyntaxError f-string
st.markdown("""
<style>
    /* Nền dải ngân hà động */
    .stApp {
        background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%);
        overflow: hidden;
        color: white;
    }

    /* Tạo hiệu ứng sao lấp lánh nhẹ phía sau */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background: transparent url('https://www.transparenttextures.com/patterns/stardust.png') repeat;
        opacity: 0.5;
        pointer-events: none;
    }

    /* Thẻ Glassmorphism (Hiệu ứng kính mờ xịn xò) */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 40px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.6);
        margin: auto;
        max-width: 500px;
        animation: slideUp 1s ease-out;
    }

    /* Định dạng và màu sắc chữ tiêu đề chuyển sắc (Gradient) */
    .title-text {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(to right, #a18cd1 0%, #fbc2eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        font-size: 2rem;
        margin-bottom: 20px;
    }

    .sub-text {
        font-size: 1.1rem;
        color: #e0e0e0;
        line-height: 1.6;
    }

    /* Hiệu ứng hộp quà bay lơ lửng */
    .gift-img {
        width: 200px;
        margin: 20px auto;
        filter: drop-shadow(0 0 15px rgba(161, 140, 209, 0.8));
        animation: float 3s ease-in-out infinite;
    }
    
    /* Hiệu ứng đóa hoa xuất hiện bung nở */
    .flower-img {
        width: 280px;
        border-radius: 15px;
        filter: drop-shadow(0 0 20px rgba(251, 194, 235, 0.9));
        animation: bloom 1.5s ease-out;
    }

    /* Các kịch bản chuyển động Animation */
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }

    @keyframes slideUp {
        from { opacity: 0; transform: translateY(50px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes bloom {
        from { transform: scale(0.5); opacity: 0; }
        to { transform: scale(1); opacity: 1; }
    }

    /* Tối ưu hiển thị Responsive mượt mà trên Điện thoại di động */
    @media (max-width: 640px) {
        .glass-card { padding: 25px; margin: 10px; }
        .title-text { font-size: 1.6rem; }
        .gift-img { width: 160px; }
        .flower-img { width: 220px; }
    }
    
    /* Làm đẹp lại các nút bấm mặc định của Streamlit */
    div.stButton > button {
        background: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        padding: 12px 25px;
        border-radius: 50px;
        transition: 0.3s ease;
        font-weight: bold;
        width: 100%;
        margin-top: 15px;
    }
    div.stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 0 20px rgba(37, 117, 252, 0.5);
        color: white;
    }
    div.stButton > button:active {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- 4. QUẢN LÝ TRẠNG THÁI (SESSION STATE) ---
if "step" not in st.session_state:
    st.session_state.step = 1
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# --- 5. ĐIỀU HƯỚNG GIAO DIỆN ---
st.write("##")  # Tạo một khoảng cách nhỏ đầu trang cho cân đối

# BƯỚC 1: Form nhập tên
if st.session_state.step == 1:
    st.markdown("""
    <div class="glass-card">
        <div class="title-text">Gửi Tới Tương Lai...</div>
        <p class="sub-text">Có một tín hiệu từ vũ trụ đang kết nối với bạn. Hãy cho chúng tôi biết tên bạn?</p>
    </div>
    """, unsafe_allow_html=True)
    
    name = st.text_input("", placeholder="Nhập tên của bạn vào đây...", key="name_input")
    
    if st.button("Nhận tín hiệu ✨"):
        if name.strip():
            st.session_state.user_name = name.strip()
            st.session_state.step = 2
            st.rerun()
        else:
            st.warning("Vui lòng nhập tên trước khi nhận tín hiệu nhé!")

# BƯỚC 2: Hiển thị hộp quà lơ lửng
elif st.session_state.step == 2:
    st.markdown(f"""
    <div class="glass-card">
        <div class="title-text">Xin chào, {st.session_state.user_name}!</div>
        <p class="sub-text">Giữa ngân hà bao la rộng lớn, nhà phát triển có một món quà nhỏ muốn gửi tặng riêng cho bạn.</p>
        <img src="{LINK_ANH_HOP_QUA}" class="gift-img">
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Mở món quà từ vì sao 🎁"):
        st.session_state.step = 3
        st.rerun()

# BƯỚC 3: Mở quà và hiện đóa hoa bung nở
elif st.session_state.step == 3:
    st.markdown(f"""
    <div class="glass-card">
        <div class="title-text">Dành Cho {st.session_state.user_name}</div>
        <p class="sub-text">Chúc bạn luôn luôn rạng rỡ, hạnh phúc và tỏa sáng như những vì tinh tú trên bầu trời!</p>
        <img src="{LINK_ANH_DOA_HOA}" class="flower-img">
        <p style="margin-top: 25px; font-style: italic; color: #fbc2eb; font-size: 0.9rem;">--- From Developer with Love ---</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.balloons()  # Thả bong bóng chúc mừng sinh động
    
    if st.button("Quay lại vũ trụ 🔄"):
        st.session_state.step = 1
        st.session_state.user_name = ""
        st.rerun()
