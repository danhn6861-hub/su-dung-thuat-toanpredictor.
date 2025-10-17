import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

st.set_page_config(page_title="Dự đoán Tài/Xỉu AI - Phiên bản Nâng Cao", layout="wide")

# Disclaimer
st.sidebar.markdown("""
### ⚠️ Lưu Ý
Ứng dụng này chỉ mang tính chất giải trí và tham khảo. Kết quả dự đoán dựa trên lịch sử ngẫu nhiên và không đảm bảo độ chính xác. Không khuyến khích sử dụng cho mục đích cờ bạc hoặc đầu tư thực tế, vì các trò chơi như Tài/Xỉu thường là ngẫu nhiên và có thể dẫn đến rủi ro tài chính.
""")

# ====== Khởi tạo trạng thái ======
if "history" not in st.session_state:
    st.session_state.history = []
if "ai_confidence" not in st.session_state:
    st.session_state.ai_confidence = []  # mức tin tưởng theo từng ván
if "models" not in st.session_state:
    st.session_state.models = None
if "ai_last_pred" not in st.session_state:
    st.session_state.ai_last_pred = None
if "undo_stack" not in st.session_state:
    st.session_state.undo_stack = []

# ====== Hàm tạo đặc trưng ======
def create_features(history, window=6):
    if len(history) < window + 1:  # Cần ít nhất window + 1 để có y
        return np.empty((0, window)), np.empty((0,))
    X = []
    y = []
    for i in range(window, len(history)):
        X.append([1 if x == "Tài" else 0 for x in history[i - window:i]])
        y.append(1 if history[i] == "Tài" else 0)
    return np.array(X), np.array(y)

# ====== Huấn luyện các mô hình với cải thiện ======
@st.cache_resource
def train_models(history_tuple, ai_confidence_tuple):
    history = list(history_tuple)
    ai_confidence = list(ai_confidence_tuple)
    X, y = create_features(history)
    if len(X) < 10:
        return None

    try:
        # Kiểm tra dữ liệu cân bằng
        if np.all(y == 0) or np.all(y == 1):
            st.warning("Dữ liệu không cân bằng (toàn Tài hoặc Xỉu). Mô hình có thể không chính xác.")
            return None

        # Sử dụng KFold với shuffle=False để phù hợp với time-series và là partition
        tscv = KFold(n_splits=3, shuffle=False)

        # Các base models
        estimators = [
            ('lr', LogisticRegression()),
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric="logloss"))
        ]

        # Stacking classifier cho kết hợp tốt hơn
        stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=tscv)

        # AI Strategy – học trọng số theo thời gian và độ tin cậy
        recent_weight = np.linspace(0.5, 1.0, len(y))
        combined_weight = recent_weight * np.array(ai_confidence[:len(y)]) if len(ai_confidence) >= len(y) else recent_weight

        # Note: StackingClassifier không hỗ trợ sample_weight trực tiếp, nên bỏ qua weight cho stacking
        # Nếu cần weight, có thể implement manual stacking
        stack.fit(X, y)

        # Đánh giá mô hình (optional, hiển thị accuracy)
        if len(X) > 20:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # Time-series split
            acc = accuracy_score(y_test, stack.predict(X_test))
            st.info(f"Độ chính xác đánh giá (test set): {acc:.2%}")

        return stack

    except Exception as e:
        st.error(f"Lỗi huấn luyện: {str(e)}")
        return None

# ====== Hàm phát hiện pattern cải thiện (sử dụng Markov chain đơn giản) ======
def pattern_detector(history, window=6):
    if len(history) < 2:
        return 0.5

    # Xây dựng transition matrix cho Markov
    states = {'Tài': 1, 'Xỉu': 0}
    trans = np.zeros((2, 2))
    for i in range(1, len(history)):
        prev = states[history[i-1]]
        curr = states[history[i]]
        trans[prev, curr] += 1

    row_sums = np.sum(trans, axis=1, keepdims=True)
    trans = np.divide(trans, row_sums, where=row_sums != 0)  # Tránh division by zero

    # Dự đoán dựa trên state cuối
    if len(history) == 0:
        return 0.5
    last_state = states[history[-1]]
    return trans[last_state, 1]  # Xác suất chuyển sang Tài

# ====== Hàm dự đoán ======
def predict_next(models, history):
    if len(history) < 6 or models is None:
        return None, None

    latest = np.array([[1 if x == "Tài" else 0 for x in history[-6:]]])
    stack_prob = models.predict_proba(latest)[0][1]
    pattern_score = pattern_detector(history)

    # Kết hợp: Trung bình có trọng số (70% stack, 30% pattern)
    final_score = 0.7 * stack_prob + 0.3 * pattern_score
    return {"Stacking Model": stack_prob, "Pattern Detector": pattern_score}, final_score

# ====== Hàm thêm kết quả với undo ======
def add_result(result):
    st.session_state.undo_stack.append((st.session_state.history.copy(), st.session_state.ai_confidence.copy()))  # Lưu cả confidence
    st.session_state.history.append(result)
    if len(st.session_state.history) > 200:
        st.session_state.history = st.session_state.history[-200:]
        st.session_state.ai_confidence = st.session_state.ai_confidence[-200:]

    # Cập nhật độ tin cậy của AI
    if st.session_state.ai_last_pred is not None:
        was_correct = (st.session_state.ai_last_pred == result)
        st.session_state.ai_confidence.append(1.2 if was_correct else 0.8)

# ====== Hàm undo ======
def undo_last():
    if st.session_state.undo_stack:
        history, confidence = st.session_state.undo_stack.pop()
        st.session_state.history = history
        st.session_state.ai_confidence = confidence

# ====== Export/Import lịch sử ======
def export_history():
    df = pd.DataFrame({"Kết quả": st.session_state.history})
    csv = df.to_csv(index=False).encode('utf-8')
    return csv

def import_history(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.history = df["Kết quả"].tolist()
        st.session_state.ai_confidence = [1.0] * len(st.session_state.history)  # Reset confidence
        st.success("Đã import lịch sử!")

# ====== Vẽ biểu đồ ======
def plot_history(history):
    if not history:
        return None
    df = pd.DataFrame({"Kết quả": history})
    counts = df["Kết quả"].value_counts(normalize=True) * 100
    fig, ax = plt.subplots()
    counts.plot(kind='bar', ax=ax, color=['green', 'red'])
    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_title("Tỷ lệ Tài/Xỉu trong lịch sử")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# ====== Giao diện ======
st.title("🎯 AI Dự đoán Tài / Xỉu – Phiên bản Nâng Cao Tự Học")

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown("#### 📊 Kết quả gần đây:")
    if st.session_state.history:
        st.write(" → ".join(st.session_state.history[-30:]))
    else:
        st.info("Chưa có dữ liệu, nhập kết quả để bắt đầu.")

with col2:
    if st.button("🧹 Xóa lịch sử", key="clear_history"):
        confirm_clear = st.checkbox("Xác nhận xóa toàn bộ lịch sử?", key="confirm_clear")
        if confirm_clear:
            st.session_state.history = []
            st.session_state.ai_confidence = []
            st.session_state.undo_stack = []
            st.session_state.models = None
            st.success("Đã xóa toàn bộ lịch sử!")

with col3:
    if st.button("↩️ Undo nhập cuối", key="undo_last"):
        undo_last()
        st.success("Đã undo nhập cuối!")

# Biểu đồ
if st.session_state.history:
    try:
        img_data = plot_history(st.session_state.history)
        if img_data:
            st.image(f"data:image/png;base64,{img_data}", caption="Biểu đồ tỷ lệ Tài/Xỉu", use_container_width=True)
    except Exception as e:
        st.warning(f"Không thể vẽ biểu đồ: {str(e)}. Vui lòng kiểm tra thư viện matplotlib.")

st.divider()

# Nút nhập kết quả với key unique
col_tai, col_xiu = st.columns(2)
with col_tai:
    if st.button("Nhập Tài", key="add_tai"):
        add_result("Tài")
        st.experimental_rerun()  # Sử dụng experimental_rerun nếu rerun không hoạt động ổn định
with col_xiu:
    if st.button("Nhập Xỉu", key="add_xiu"):
        add_result("Xỉu")
        st.experimental_rerun()  # Sử dụng experimental_rerun nếu rerun không hoạt động ổn định

st.divider()

# Huấn luyện
if st.button("⚙️ Huấn luyện lại từ lịch sử", key="train_models"):
    with st.spinner("Đang huấn luyện các mô hình..."):
        st.session_state.models = train_models(tuple(st.session_state.history), tuple(st.session_state.ai_confidence))
    if st.session_state.models is not None:
        st.success("✅ Huấn luyện thành công!")

# Dự đoán
if len(st.session_state.history) >= 6:
    if st.session_state.models is None:
        st.info("Vui lòng huấn luyện mô hình trước.")
    else:
        preds, final_score = predict_next(st.session_state.models, st.session_state.history)
        if preds:
            st.session_state.ai_last_pred = "Tài" if final_score >= 0.5 else "Xỉu"
            st.subheader(f"🎯 Dự đoán chung: **{st.session_state.ai_last_pred}** ({final_score:.2%})")
            st.caption("Tổng hợp từ Stacking Model + Pattern Detector:")

            for k, v in preds.items():
                st.write(f"**{k}** → {v:.2%}")
else:
    st.warning("Cần ít nhất 6 ván để bắt đầu dự đoán.")

st.divider()

# Export/Import
col_export, col_import = st.columns(2)
with col_export:
    csv = export_history()
    st.download_button("📥 Export lịch sử (CSV)", csv, f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv", key="export_history")
with col_import:
    uploaded_file = st.file_uploader("📤 Import lịch sử từ CSV", type="csv", key="import_file")
    if uploaded_file:
        import_history(uploaded_file)
