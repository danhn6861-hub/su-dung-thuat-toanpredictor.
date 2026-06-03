import streamlit as st

# 1. Cấu hình trang (Tối ưu hiển thị và thanh tiêu đề)
st.set_page_config(
    page_title="Món Quà Bất Ngờ",
    page_icon="🎁",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Nhúng CSS Custom để tối ưu giao diện Responsive & tạo hiệu ứng chuyển động
st.markdown("""
<style>
    /* Ẩn các thành phần mặc định không cần thiết của Streamlit để tăng thẩm mỹ */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Định dạng nền Gradient nhẹ nhàng, lãng mạn */
    .stApp {
        background: linear-gradient(135deg, #ffe5ec 0%, #ffc2d1 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Khung chứa chính giữa màn hình, tự động co giãn theo thiết bị (Responsive) */
    .main-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 20px;
        margin-top: 5vh;
    }
    
    /* Định dạng chữ thông điệp */
    .message-text {
        font-size: 1.6rem;
        color: #ff4d6d;
        font-weight: bold;
        margin: 20px 0;
        line-height: 1.5;
        animation: fadeIn 1.2s ease-in-out;
    }
    
    /* Hiệu ứng Hộp quà lắc lư (Tạo sự tò mò) */
    .gift-box {
        font-size: 7rem;
        display: inline-block;
        cursor: pointer;
        animation: shake 0.6s infinite;
        user-select: none;
    }
    
    /* Hiệu ứng Đóa hoa nở rộ ra từ tâm */
    .flower {
        font-size: 8rem;
        display: inline-block;
        animation: bloom 1.8s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
        user-select: none;
    }

    /* Các kịch bản chuyển động (Keyframes) */
    @keyframes shake {
        0% { transform: translate(1px, 1px) rotate(0deg); }
        10% { transform: translate(-1px, -2px) rotate(-1deg); }
        20% { transform: translate(-3px, 0px) rotate(1deg); }
        30% { transform: translate(0px, 2px) rotate(0deg); }
        40% { transform: translate(1px, -1px) rotate(1deg); }
        50% { transform: translate(-1px, 2px) rotate(-1deg); }
        60% { transform: translate(-3px, 1px) rotate(0deg); }
        70% { transform: translate(2px, 1px) rotate(-1deg); }
        80% { transform: translate(-1px, -1px) rotate(1deg); }
        90% { transform: translate(2px, 2px) rotate(0deg); }
        100% { transform: translate(1px, -2px) rotate(0deg); }
    }
    
    @keyframes bloom {
        0% { transform: scale(0) rotate(-45deg); opacity: 0; }
        70% { transform: scale(1.2) rotate(20deg); }
        100% { transform: scale(1) rotate(0deg); opacity: 1; }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* CSS Responsive: Tự động tối ưu kích thước chữ/hình trên Điện thoại */
    @media (max-width: 640px) {
        .message-text { font-size: 1.25rem; padding: 0 10px; }
        .gift-box { font-size: 5rem; }
        .flower { font-size: 5.5rem; }
        .main-container { margin-top: 2vh; }
    }
</style>
""", unsafe_allow_html=True)

# 3. Quản lý trạng thái màn hình bằng Session State
if "step" not in st.session_state:
    st.session_state.step = 1
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

# Toàn bộ nội dung đặt trong div bọc để nhận CSS responsive
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# --- BƯỚC 1: Hỏi tên ---
if st.session_state.step == 1:
    st.markdown("<h2 style='color: #ff4d6d; margin-bottom: 25px;'>Chào bạn! Mình có một điều bất ngờ... ✨</h2>", unsafe_allow_html=True)
    
    # Ô nhập tên thân thiện
    name_input = st.text_input("Bạn tên gì thế nhỉ?", placeholder="Nhập tên của bạn vào đây...", key="name_field")
    
    # Nút xác nhận phủ kín chiều rộng trên mobile (use_container_width)
    if st.button("Xác nhận để tiếp tục ➔", use_container_width=True):
        if name_input.strip():
            st.session_state.user_name = name_input.strip()
            st.session_state.step = 2
            st.rerun()
        else:
            st.error("Bạn vui lòng nhập tên trước khi tiếp tục nhé! 😊")

# --- BƯỚC 2: Hiển thị lời nhắn & Hộp quà lắc lư ---
elif st.session_state.step == 2:
    st.markdown(
        f'<div class="message-text">Nhà phát triển muốn tặng món quà nhỏ này cho bạn, {st.session_state.user_name}! ❤️</div>', 
        unsafe_allow_html=True
    )
    
    # Hiển thị hiệu ứng hộp quà chuyển động bằng CSS div
    st.markdown('<div style="margin: 30px 0;"><span class="gift-box">🎁</span></div>', unsafe_allow_html=True)
    
    if st.button("Bấm vào đây để mở hộp quà 🌟", use_container_width=True):
        st.session_state.step = 3
        st.rerun()

# --- BƯỚC 3: Mở hộp quà ra đóa hoa ---
elif st.session_state.step == 3:
    st.markdown(
        f'<div class="message-text">Tèn ten! Một đóa hoa tươi thắm dành riêng cho {st.session_state.user_name} nhé! 🌸✨</div>', 
        unsafe_allow_html=True
    )
    
    # Hiện hiệu ứng đóa hoa nở bung ra ngoài
    st.markdown('<div style="margin: 30px 0;"><span class="flower">🌹</span></div>', unsafe_allow_html=True)
    
    # Tạo hiệu ứng pháo hoa ăn mừng của Streamlit giúp trang web sống động hơn
    st.balloons()
    
    # Nút tùy chọn để chơi lại từ đầu
    if st.button("Mở lại quà một lần nữa 🔄", use_container_width=True):
        st.session_state.step = 1
        st.session_state.user_name = ""
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
