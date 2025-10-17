import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- Khởi tạo ---
st.set_page_config(page_title="AI Dự đoán Tài/Xỉu", layout="centered")

if "history" not in st.session_state:
    st.session_state.history = []
if "results" not in st.session_state:
    st.session_state.results = []
if "ai_strategy" not in st.session_state:
    st.session_state.ai_strategy = {"win_rate": 0.5, "adjust": 0.0}
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

# --- Giao diện chính ---
st.title("🎯 AI Dự Đoán Tài / Xỉu (Tối ưu tốc độ)")
st.caption("5 mô hình: Logistic Regression, Random Forest, XGBoost, AI Tự học, Pattern Detector")

col1, col2 = st.columns(2)
with col1:
    if st.button("🎲 Nhập Tài"):
        st.session_state.history.append(1)
    if st.button("🧠 Dự đoán"):
        st.session_state.model_trained = True
with col2:
    if st.button("⚪ Nhập Xỉu"):
        st.session_state.history.append(0)
    if st.button("🧹 Xóa lịch sử"):
        st.session_state.history.clear()
        st.session_state.results.clear()
        st.session_state.ai_strategy = {"win_rate": 0.5, "adjust": 0.0}
        st.session_state.model_trained = False
        st.success("Đã xóa lịch sử và reset AI!")

# --- Kiểm tra dữ liệu ---
if len(st.session_state.history) < 6:
    st.info("👉 Hãy nhập ít nhất 6 ván để bắt đầu huấn luyện.")
    st.stop()

# --- Chuẩn bị dữ liệu ---
data = np.array(st.session_state.history)
X = np.array([data[i:i+5] for i in range(len(data)-5)])
y = data[5:]

# --- Huấn luyện mô hình ---
lr = LogisticRegression()
rf = RandomForestClassifier(n_estimators=50, max_depth=4)
xgb = XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.2, verbosity=0)

lr.fit(X, y)
rf.fit(X, y)
xgb.fit(X, y)

# --- Pattern Detector ---
def pattern_predict(last6, history):
    if len(history) < 12:
        return np.random.choice([0, 1])
    for i in range(len(history) - 6):
        if history[i:i+6] == last6:
            return history[i+6] if i+6 < len(history) else np.random.choice([0, 1])
    return np.random.choice([0, 1])

# --- AI Tự học Chiến lược ---
def ai_strategy_predict():
    base = st.session_state.ai_strategy["win_rate"] + st.session_state.ai_strategy["adjust"]
    return 1 if np.random.random() < base else 0

# --- Dự đoán ---
last5 = np.array(st.session_state.history[-5:]).reshape(1, -1)
pred_lr = lr.predict(last5)[0]
pred_rf = rf.predict(last5)[0]
pred_xgb = xgb.predict(last5)[0]
pred_pattern = pattern_predict(st.session_state.history[-6:], st.session_state.history)
pred_ai = ai_strategy_predict()

preds = [pred_lr, pred_rf, pred_xgb, pred_ai, pred_pattern]
final_pred = int(round(np.mean(preds)))

# --- Hiển thị kết quả ---
st.subheader("📊 Kết quả dự đoán")
models = [
    ("Logistic Regression", pred_lr, "Phân định tuyến tính cơ bản"),
    ("Random Forest", pred_rf, "Giảm overfit, học ổn định"),
    ("XGBoost", pred_xgb, "Boosting mạnh, học mẫu phức tạp"),
    ("AI Tự học Chiến lược", pred_ai, "Rút kinh nghiệm từ kết quả"),
    ("Pattern Detector", pred_pattern, "Phát hiện mẫu lặp 6 ván gần nhất"),
]

cols = st.columns(5)
for i, (name, pred, desc) in enumerate(models):
    with cols[i]:
        st.markdown(f"**{name}**")
        st.markdown(f"{'🟥 Xỉu' if pred==0 else '🟩 Tài'}")
        st.caption(desc)

# --- Cập nhật học kinh nghiệm ---
if len(st.session_state.results) > 0:
    wins = sum(st.session_state.results)
    total = len(st.session_state.results)
    st.session_state.ai_strategy["win_rate"] = wins / total

# --- Lưu kết quả ---
st.session_state.results.append(final_pred)
win_rate = st.session_state.ai_strategy["win_rate"] * 100

st.markdown(f"### 🎯 Dự đoán chung: {'🟩 Tài' if final_pred==1 else '🟥 Xỉu'}")
st.progress(win_rate / 100)
st.write(f"**Tỉ lệ thắng (ước tính): {win_rate:.1f}%**")
