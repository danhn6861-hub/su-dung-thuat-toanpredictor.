import streamlit as st
import time
import random
from datetime import datetime

# 1. Cấu hình trang web (Bắt buộc đặt ở đầu)
st.set_page_config(
    page_title="CozyStudy - Góc Học Tập Nhỏ 🌸",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed" # Tự động thu gọn sidebar trên điện thoại cho gọn gọn
)

# 2. Khởi tạo dữ liệu Session State
if 'todos' not in st.session_state:
    st.session_state.todos = []
if 'journal' not in st.session_state:
    st.session_state.journal = []
if 'xp' not in st.session_state:
    st.session_state.xp = 0

# Hệ thống cấp độ dễ thương dành cho phái nữ
def get_level_title(xp):
    lvl = (xp // 50) + 1
    if lvl == 1: return "🌱 Mầm Nhỏ Chăm Chỉ"
    elif lvl == 2: return "🌿 Lá Xanh Tự Tin"
    elif lvl == 3: return "🌸 Hoa Thắm Trưởng Thành"
    else: return "👑 Nữ Hoàng Kỷ Luật"

user_level = (st.session_state.xp // 50) + 1
xp_next_level = 50 - (st.session_state.xp % 50)
level_title = get_level_title(st.session_state.xp)

# 3. Tùy biến CSS giao diện Pastel Pink mềm mại, tối ưu cho Mobile
st.markdown("""
    <style>
    /* Nền kem sữa ấm áp */
    .stApp {
        background-color: #FFFDF9;
    }
    
    /* Banner chính bo góc mềm, màu hồng pastel gradient */
    .cozy-banner {
        background: linear-gradient(135deg, #FFB7B2 0%, #FFDAC1 100%);
        color: #4A4A4A;
        padding: 1.8rem;
        border-radius: 20px;
        box-shadow: 0 8px 20px rgba(255, 183, 178, 0.2);
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    /* Thẻ tính năng bo tròn, viền hồng nhạt */
    .cozy-card {
        background: white;
        padding: 1.2rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02);
        border: 1px solid #FFEBEB;
        margin-bottom: 1rem;
    }
    
    /* Badge hiển thị danh hiệu */
    .girl-badge {
        background: #FF7B94;
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        font-weight: bold;
        font-size: 0.85rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    /* Tiêu đề chính tối ưu kích thước không bị tràn màn hình điện thoại */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(to right, #FF7B94, #FFAAA6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    /* Tinh chỉnh khoảng cách hiển thị tab trên mobile */
    .stTabs [data-baseweb="tab"] {
        padding-left: 8px;
        padding-right: 8px;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Danh sách lời nhắn nhủ ngọt ngào, tạo động lực
QUOTES = [
    {"text": "Cứ đi từng bước nhỏ một thôi cô gái nhé, bông hoa đẹp luôn cần thời gian để nở rộ.", "author": "Lời nhắn từ tương lai"},
    {"text": "Học tập không phải là gánh nặng, đó là cách em tự tô màu cho ước mơ của chính mình.", "author": "Góc bình yên"},
    {"text": "Hôm nay em đã rất cố gắng rồi. Hãy mỉm cười và tiếp tục tiến lên nào!", "author": "Trái tim ấm áp"},
    {"text": "Đừng so sánh chương 1 của mình với chương 20 của người khác. Hãy tập trung vào phiên bản tốt hơn của em.", "author": "Keep Growing"},
    {"text": "Một chút chăm chỉ mỗi ngày sẽ tích tiểu thành đại thành một tương lai rực rỡ.", "author": "Cố lên nhé"}
]

# 4. Khu vực thông tin cá nhân (Sidebar gọn gàng)
with st.sidebar:
    st.markdown("### 🎀 Góc Nhỏ Của Em")
    st.markdown(f"<span class='girl-badge'>{level_title}</span>", unsafe_allow_html=True)
    st.write(f"**Điểm chăm chỉ (XP):** {st.session_state.xp} pts")
    
    # Tiến trình thăng cấp
    progress_val = (st.session_state.xp % 50) / 50
    st.progress(progress_val)
    st.caption(f"Còn **{xp_next_level} XP** nữa để nâng mầm cây học tập!")
    
    st.markdown("---")
    st.markdown("### 🎵 Nhạc Lofi Học Tập")
    st.caption("Bật một chút giai điệu nhẹ nhàng để thư giãn não bộ:")
    # Nhúng danh sách phát lofi phong cách cottagecore/cute vô cùng hợp chủ đề
    st.video("https://www.youtube.com/watch?v=jfKfPfyJRdk")

# Giao diện chính trên Web / Điện thoại
st.markdown("<h1 class='main-title'>CozyStudy 🌸</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #8A8A8A; font-size: 0.95rem; margin-bottom: 1rem;'>Nơi tưới tắm cho mầm mống tri thức và nuôi dưỡng sự kiên trì của em.</p>", unsafe_allow_html=True)

# 5. Hệ thống Tabs điều hướng (Thân thiện với màn hình cảm ứng di động)
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Trang Chủ", 
    "⏱️ Pomodoro", 
    "🎯 Việc Nhỏ", 
    "🏆 Khen Mình"
])

# ==========================================
# TAB 1: TRANG CHỦ & ĐỘNG LỰC
# ==========================================
with tab1:
    random_quote = random.choice(QUOTES)
    st.markdown(f"""
        <div class='cozy-banner'>
            <h2 style='color: #4A4A4A; font-style: italic; font-size: 1.3rem; font-weight: 600; line-height: 1.5;'>"{random_quote['text']}"</h2>
            <p style='color: #FF7B94; margin-top: 0.8rem; font-weight: bold; font-size: 0.9rem;'>— {random_quote['author']} —</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='cozy-card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Nhật Ký Chăm Chỉ Ngày Hôm Nay")
    total_tasks = len(st.session_state.todos)
    done_tasks = sum(1 for t in st.session_state.todos if t['done'])
    
    c1, c2 = st.columns(2)
    c1.metric(label="Mục tiêu đề ra", value=total_tasks)
    c2.metric(label="Đã làm xong", value=f"{done_tasks} việc")
    
    if total_tasks > 0:
        rate = int(done_tasks / total_tasks * 100)
        st.progress(rate / 100)
        st.caption(f"Em đã hoàn thành xuất sắc **{rate}%** chặng đường rồi á!")
    else:
        st.info("Em chưa lên kế hoạch gì nè. Hãy chuyển sang tab 'Việc Nhỏ' để ghi ra vài việc nha!")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='cozy-card'>", unsafe_allow_html=True)
    # HOÀN TOÀN KHÔNG DÙNG DẤU NGOẶC KÉP TRÙNG NHAU Ở ĐÂY ĐỂ TRÁNH LỖI SYNTAX
    st.markdown("### ⚡ Chế độ nạp năng lượng")
    st.write("Hôm nay em muốn đón nhận ngày mới với trạng thái tâm lý thế nào?")
    
    mindset = st.radio(
        "Tâm trạng em chọn chọn:",
        ["Nhẹ nhàng tập trung, tránh xa điện thoại 📱", 
         "Không áp lực, học tới đâu vui tới đó 🧠", 
         "Nỗ lực hết mình vì một phiên bản rực rỡ hơn ✨"]
    )
    if st.button("Kích hoạt tâm trạng tích cực", use_container_width=True):
        st.balloons()
        st.success("Tâm trạng ngọt ngào đã được nạp vào ngày mới của em!")
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 2: ĐỒNG HỒ POMODORO
# ==========================================
with tab2:
    st.markdown("<div class='cozy-card'>", unsafe_allow_html=True)
    st.markdown("### ⏱️ Đồng Hồ Cà Chua Tập Trung")
    st.write("Em hãy chọn thời gian, sau đó úp điện thoại xuống và tập trung nhé.")
    
    duration_type = st.selectbox("Chọn mốc thời gian nghỉ/học:", ["Tập trung cao độ (25 phút)", "Nghỉ ngơi ngắn (5 phút)", "Nghỉ ngơi dài (15 phút)"])
    duration_minutes = 25 if "Tập trung" in duration_type else (5 if "ngắn" in duration_type else 15)
    
    start_btn = st.button("🎀 Bắt đầu đếm giờ", use_container_width=True, type="primary")
    
    timer_placeholder = st.empty()
    timer_placeholder.markdown(f"<h1 style='text-align: center; font-size: 4rem; color: #FF7B94; font-family: monospace;'>{duration_minutes:02d}:00</h1>", unsafe_allow_html=True)
    
    if start_btn:
        total_seconds = duration_minutes * 60
        progress_bar = st.progress(0)
        
        for remaining in range(total_seconds, -1, -1):
            mins, secs = divmod(remaining, 60)
            timer_placeholder.markdown(f"<h1 style='text-align: center; font-size: 4rem; color: #FF4B4B; font-family: monospace;'>{mins:02d}:{secs:02d}</h1>", unsafe_allow_html=True)
            
            progress = (total_seconds - remaining) / total_seconds
            progress_bar.progress(progress)
            time.sleep(1)
            
        if "Tập trung" in duration_type:
            st.session_state.xp += 25
            st.success("🎉 Siêu quá đi! Em đã hoàn thành trọn vẹn và nhận được +25 điểm chăm chỉ!")
        else:
            st.success("☕ Hết giờ thư giãn rồi, cùng quay lại học tập thôi nào.")
        st.snow()
    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 3: VIỆC CẦN LÀM (TO-DO LIST)
# ==========================================
with tab3:
    st.markdown("### 🎯 Danh sách việc nhỏ cần làm")
    st.caption("Chia nhỏ bài học giúp em đỡ ngợp hơn đó. Làm xong mỗi việc được nhận ngay +10 XP.")
    
    with st.form("todo_form", clear_on_submit=True):
        new_task = st.text_input("Ghi việc cần làm (Ví dụ: Chép 10 từ vựng, Đọc xong chương 2 lý,...)")
        submit_task = st.form_submit_button("Thêm vào danh sách")
        
        if submit_task and new_task:
            st.session_state.todos.append({"task": new_task, "done": False, "claimed": False})
            st.rerun()

    if st.session_state.todos:
        for idx, item in enumerate(st.session_state.todos):
            st.markdown("<div class='cozy-card'>", unsafe_allow_html=True)
            
            # Tối ưu hóa To-do cho Mobile bằng cách gom Checkbox và Text chung một dòng, nút Xóa nằm gọn bên phải
            col_left, col_right = st.columns([8, 2])
            
            is_done = col_left.checkbox(item["task"], value=item["done"], key=f"check_{idx}")
            if is_done != item["done"]:
                st.session_state.todos[idx]["done"] = is_done
                if is_done and not item.get("claimed", False):
                    st.session_state.xp += 10
                    st.session_state.todos[idx]["claimed"] = True
                    st.toast("⚡ Thưởng +10 XP chăm chỉ đã nạp!")
                st.rerun()
                
            if item["done"]:
                col_left.caption("✨ *Tuyệt vời! Đã hoàn thành*")
                
            if col_right.button("🗑️", key=f"del_{idx}"):
                if item.get("claimed", False):
                    st.session_state.xp -= 10
                st.session_state.todos.pop(idx)
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Hôm nay chưa có việc gì cần làm hết á. Thêm một vài việc nhỏ đi em!")

# ==========================================
# TAB 4: NHẬT KÝ KHEN MÌNH (Đã sửa lỗi Tab thụt lề)
# ==========================================
with tab4:
    st.markdown("### 🏆 Nhật Ký Tự Hào Của Bản Thân")
    st.caption("Hãy viết lại những việc em thấy mình làm tốt hôm nay nhé. Học cách yêu thương và công nhận chính mình nha.")
    
    with st.form("journal_form", clear_on_submit=True):
        achievement = st.text_area("Hôm nay em tự hào nhất về điều gì ở bản thân mình?")
        submit_journal = st.form_submit_button("Ghi lại dấu ấn")
        
        if submit_journal and achievement:
            now = datetime.now().strftime("%d/%m/%Y - %H:%M")
            st.session_state.journal.insert(0, {"time": now, "content": achievement})
            st.session_state.xp += 5
            st.toast("🏆 Đã lưu vào cuốn sổ nhỏ! +5 XP.")
            st.rerun()
            
    # ĐÃ ĐƯỢC ĐẶT CHÍNH XÁC BÊN TRONG KHỐI 'WITH TAB4' - SỬA LỖI INDENTATION CŨ
    if st.session_state.journal:
        st.write("---")
        for item in st.session_state.journal:
            st.markdown(f"""
                <div class='cozy-card' style='border-left: 4px solid #FF7B94;'>
                    <span style='color: #A3A3A3; font-size: 0.8rem;'>⏱️ Lúc: {item['time']}</span>
                    <p style='margin-top: 0.4rem; font-size: 1rem; color: #4A4A4A; font-weight: 500;'>🌸 {item['content']}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Chưa có dòng nhật ký nào hết nè. Hãy viết một dòng để tự khen thưởng bản thân nha!")

# Chân trang nhẹ nhàng
st.markdown("<br>---", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #B5B5B5; font-size: 0.85rem;'>CozyStudy Pro © 2026 | Thương gửi những cô gái đang kiên trì vì ước mơ của mình.</p>", unsafe_allow_html=True)
