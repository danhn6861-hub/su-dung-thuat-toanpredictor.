import streamlit as st
import time
import random
from datetime import datetime

# 1. Cấu hình trang web (Phải đặt ở đầu tiên)
st.set_page_config(
    page_title="EduSpark - Động Lực Học Tập Mỗi Ngày",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Tùy biến giao diện bằng CSS (Giao diện chuyên nghiệp, bo góc, đổ bóng)
st.markdown("""
    <style>
    /* Nền tảng và font chữ chung */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Thiết kế các thẻ Card phong cách Glassmorphism nhẹ */
    .motivational-card {
        background: linear-gradient(135deg, #6B73FF 10%, #000DFF 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .feature-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 5px solid #6B73FF;
        margin-bottom: 1rem;
    }
    
    /* Tùy chỉnh tiêu đề */
    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        color: #1E1E24;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Khởi tạo dữ liệu Session State (Giữ dữ liệu không bị mất khi reload)
if 'todos' not in st.session_state:
    st.session_state.todos = []
if 'journal' not in st.session_state:
    st.session_state.journal = []

# Danh sách câu nói truyền cảm hứng
QUOTES = [
    {"text": "Đường tuy ngắn không đi không đến, việc tuy nhỏ không làm không nên.", "author": "Tục ngữ"},
    {"text": "Cách tốt nhất để dự đoán tương lai là tự kiến tạo ra nó.", "author": "Abraham Lincoln"},
    {"text": "Thành công là tổng hợp của những nỗ lực nhỏ bé được lặp lại ngày qua ngày.", "author": "Robert Collier"},
    {"text": "Đừng đợi thời điểm hoàn hảo, hãy chọn một thời điểm và làm cho nó trở nên hoàn hảo.", "author": "Khuyết danh"},
    {"text": "Thiên tài chỉ có 1% là tài năng bẩm sinh, 99% còn lại là mồ hôi và nước mắt.", "author": "Thomas Edison"}
]

# 4. Giao diện Header
st.markdown("<h1 class='main-title'>🎓 EduSpark</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Không gian đánh thức tiềm năng và duy trì kỷ luật học tập của bạn.</p>", unsafe_allow_html=True)

# 5. Hệ thống Tabs điều hướng (Hiện đại hơn menu dọc truyền thống)
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Trang Chủ & Động Lực", 
    "⏱️ Kỷ Luật Pomodoro", 
    "📝 Mục Tiêu Hôm Nay", 
    "🏆 Nhật Ký Thành Tựu"
])

# ==========================================
# TAB 1: TRANG CHỦ & ĐỘNG LỰC
# ==========================================
with tab1:
    # Banner động lực lớn
    random_quote = random.choice(QUOTES)
    st.markdown(f"""
        <div class='motivational-card'>
            <h2 style='color: white; font-style: italic; font-size: 1.8rem;'>"{random_quote['text']}"</h2>
            <p style='color: #E0E7FF; margin-top: 1rem; font-weight: bold;'>— {random_quote['author']}</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Trạng thái học tập của bạn")
        # Thống kê nhanh dưới dạng Widget chuyên nghiệp
        total_tasks = len(st.session_state.todos)
        done_tasks = sum(1 for t in st.session_state.todos if t['done'])
        
        c1, c2, c3 = st.columns(3)
        c1.metric(label="Mục tiêu đã lập", value=total_tasks)
        c2.metric(label="Đã hoàn thành", value=done_tasks)
        c3.metric(label="Tỷ lệ", value=f"{int(done_tasks/total_tasks*100)}%" if total_tasks > 0 else "0%")
        
        st.info("💡 **Lời khuyên:** Hãy chia nhỏ lộ trình học tập thành các phiên Pomodoro 25 phút để đạt hiệu quả tập trung cao nhất mà không bị quá tải.")

    with col2:
        st.markdown("### ⚡ Thử thách "Nhân đôi hiệu suất"")
        st.write("Hôm nay bạn đã sẵn sàng vượt qua giới hạn của bản thân chưa? Chọn một tư duy tích cực để bắt đầu ngày mới:")
        mindset = st.radio(
            "Tư duy hôm nay của tôi là:",
            ["Tập trung tuyệt đối, không màng điện thoại 📱", 
             "Sai lầm là cơ hội để học hỏi và tiến bộ 🚀", 
             "Học hết mình, chơi nhiệt tình 🔥"]
        )
        if st.button("Kích hoạt năng lượng"):
            st.balloons()
            st.success(f"Tuyệt vời! Bạn đã chọn: **{mindset}**. Hãy giữ vững tinh thần này suốt ngày hôm nay!")

# ==========================================
# TAB 2: ĐỒNG HỒ POMODORO
# ==========================================
with tab2:
    st.markdown("### ⏱️ Phương pháp quả cà chua (Pomodoro)")
    st.caption("Tập trung cao độ trong 25 phút, sau đó nghỉ ngơi 5 phút. Giúp não bộ luôn ở trạng thái tối ưu.")
    
    p_col1, p_col2 = st.columns([1, 2])
    
    with p_col1:
        duration_type = st.selectbox("Chọn chế độ:", ["Tập trung (25 phút)", "Nghỉ ngắn (5 phút)", "Nghỉ dài (15 phút)"])
        duration_minutes = 25 if "Tập trung" in duration_type else (5 if "Nghỉ ngắn" in duration_type else 15)
        
        start_btn = st.button("🔥 Bắt đầu ngay", use_container_width=True)
        
    with p_col2:
        timer_placeholder = st.empty()
        # Hiển thị đồng hồ ban đầu
        timer_placeholder.markdown(f"<h1 style='text-align: center; font-size: 5rem; color: #6B73FF;'>{duration_minutes:02d}:00</h1>", unsafe_allow_html=True)
        
        if start_btn:
            total_seconds = duration_minutes * 60
            progress_bar = st.progress(0)
            
            for remaining in range(total_seconds, -1, -1):
                mins, secs = divmod(remaining, 60)
                timer_placeholder.markdown(f"<h1 style='text-align: center; font-size: 5rem; color: #FF4B4B;'>{mins:02d}:{secs:02d}</h1>", unsafe_allow_html=True)
                
                # Cập nhật thanh tiến trình
                progress = (total_seconds - remaining) / total_seconds
                progress_bar.progress(progress)
                
                time.sleep(1)
                
            st.success("🎉 Xuất sắc! Bạn đã hoàn thành phiên làm việc cực kỳ chất lượng!")
            st.balloons()

# ==========================================
# TAB 3: MỤC TIÊU HÔM NAY (TO-DO LIST)
# ==========================================
with tab3:
    st.markdown("### 📝 Quản lý mục tiêu học tập")
    
    # Form thêm task mới
    with st.form("todo_form", clear_on_submit=True):
        new_task = st.text_input("Nhập nhiệm vụ học tập mới (Ví dụ: Giải 3 bài toán hình học, Đọc 5 trang sách tiếng Anh...):")
        submit_task = st.form_submit_button("Thêm nhiệm vụ")
        
        if submit_task and new_task:
            st.session_state.todos.append({"task": new_task, "done": False})
            st.rerun()

    # Hiển thị danh sách task
    if st.session_state.todos:
        st.write("---")
        for idx, item in enumerate(st.session_state.todos):
            col_check, col_text, col_del = st.columns([1, 10, 1])
            
            # Checkbox để đánh dấu hoàn thành
            is_done = col_check.checkbox("", value=item["done"], key=f"check_{idx}")
            if is_done != item["done"]:
                st.session_state.todos[idx]["done"] = is_done
                st.rerun()
                
            # Định dạng chữ (gạch ngang nếu đã xong)
            if item["done"]:
                col_text.markdown(f"<del style='color: #94A3B8;'>{item['task']}</del> ✅", unsafe_allow_html=True)
            else:
                col_text.markdown(f"**{item['task']}**", unsafe_allow_html=True)
                
            # Nút xóa task
            if col_del.button("🗑️", key=f"del_{idx}"):
                st.session_state.todos.pop(idx)
                st.rerun()
    else:
        st.info("Hiện chưa có mục tiêu nào được lập. Hãy viết xuống những việc cần làm để giải phóng tâm trí nhé!")

# ==========================================
# TAB 4: NHẬT KÝ THÀNH TỰU (PRIDE JOURNAL)
# ==========================================
with tab4:
    st.markdown("### 🏆 Nhật ký tự hào (Dopamine Boost)")
    st.caption("Ghi lại những gì bạn đã làm tốt hôm nay, dù là nhỏ nhất. Việc công nhận bản thân giúp sản sinh Dopamine - tăng cường động lực học lâu dài.")
    
    with st.form("journal_form", clear_on_submit=True):
        achievement = st.text_area("Hôm nay bạn tự hào về điều gì nhất ở bản thân?")
        submit_journal = st.form_submit_button("Lưu vào nhật ký")
        
        if submit_journal and achievement:
            now = datetime.now().strftime("%d/%m/%Y - %H:%M")
            st.session_state.journal.insert(0, {"time": now, "content": achievement})
            st.rerun()
            
    # Hiển thị lịch sử nhật ký thành tựu
    if st.session_state.journal:
        st.write("---")
        for item in st.session_state.journal:
            st.markdown(f"""
                <div class='feature-card'>
                    <small style='color: #64748B;'>⏱️ Lên lịch lúc: {item['time']}</small>
                    <p style='margin-top: 0.5rem; font-size: 1.05rem; color: #1E1E24; font-weight: 500;'>{item['content']}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.write("Chưa có dòng nhật ký nào. Đừng ngần ngại khen ngợi chính mình, bạn đã cố gắng rất nhiều!")

# Footer chân trang chuyên nghiệp
st.markdown("---")
st.markdown("<p style='text-align: center; color: #94A3B8;'>EduSpark App © 2026 | Được thiết kế để đồng hành cùng GenZ trên con đường chinh phục tri thức.</p>", unsafe_allow_html=True)
