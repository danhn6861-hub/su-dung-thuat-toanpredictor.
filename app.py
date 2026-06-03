import streamlit as st

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Quà Tặng Từ Vũ Trụ",
    page_icon="🌌",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- KHU VỰC THAY ĐỔI LINK ẢNH (TẠI ĐÂY) ---
# Mẹo: Nên tìm các ảnh PNG tách nền hoặc ảnh GIF tại GIPHY.com để đẹp nhất
LINK_ANH_HOP_QUA = "https://cdn.pixabay.com/animation/2023/03/19/02/45/02-45-42-263_512.gif" 
LINK_ANH_DOA_HOA = "https://i.pinimg.com/originals/91/97/8e/91978e87493a595ec7695325785a9df4.gif"

# --- CSS CUSTOM: VŨ TRỤ & GLASSMORPHISM ---
st.markdown(f"""
<style>
    /* 1. Nền dải ngân hà động */
    .stApp {{
        background: radial-gradient(ellipse at bottom, #1B2735 0%, #090A0F 100%);
        overflow: hidden;
        color: white;
    }}

    /* Tạo hiệu ứng sao lấp lánh bằng shadow */
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background: transparent url('https://www.transparenttextures.com/patterns/stardust.png') repeat;
        opacity: 0.5;
    }}

    /* 2. Thẻ Glassmorphism (Kính mờ) */
    .glass-card {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 40px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        margin: auto;
        max-width: 500px;
        animation: slideUp 1s ease-out;
    }}

    /* 3. Tối ưu chữ */
    .title-text {{
        font-family: 'Courier New', Courier, monospace;
        background: linear-gradient(to right, #a18cd1 0%, #fbc2eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        font-size: 2rem;
        margin-bottom: 20px;
    }}

    .sub-text {{
        font-size: 1.1rem;
        color: #e0e0e0;
        line-height: 1.6;
    }}

    /* 4. Hiệu ứng ảnh */
    .gift-img {{
        width: 200px;
        margin: 20px auto;
        filter: drop-shadow(0 0 15px rgba(161, 140, 209, 0.8));
        animation: float 3s ease-in-out infinite;
    }}
    
    .flower-img {{
        width: 280px;
        border-radius: 15px;
        filter: drop-shadow(0 0 20px rgba(251, 194, 235, 0.9));
        animation: bloom 1.5s ease-out;
    }}

    /* 5. Hiệu ứng Animation */
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-20px); }}
    }}

    @keyframes slideUp {{
        from {{ opacity: 0; transform: translateY(50px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes bloom {{
        from {{ transform: scale(0.5); opacity: 0; }}
        to {{ transform: scale(1); opacity: 1; }}
    }}

    /* 6. Tối ưu Responsive cho điện thoại */
    @media (max-width: 640px) {{
        .glass-card {{ padding: 25px; margin: 10px; }}
        .title-text {{ font-size:
