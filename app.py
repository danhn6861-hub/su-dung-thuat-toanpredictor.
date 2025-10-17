import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

st.set_page_config(page_title="Dự đoán Tài/Xỉu AI", layout="wide")

# ====== Khởi tạo trạng thái ======
if "history" not in st.session_state:
    st.session_state.history = []
if "features" not in st.session_state:
    st.session_state.features = []
if "labels" not in st.session_state:
    st.session_state.labels = []
if "ai_confidence" not in st.session_state:
    st.session_state.ai_confidence = []  # mức tin tưởng theo từng ván

# ====== Hàm tạo đặc trưng ======
def create_features(history, window=6):
    if len(history) < window:
        return np.empty((0, window))
    X = []
    y = []
    for i in range(window, len(history)):
        X.append([1 if x == "Tài" else 0 for x in history[i - window:i]])
        y.append(1 if history[i] == "Tài" else 0)
    return np.array(X), np.array(y)

# ====== Huấn luyện các mô hình ======
def train_models():
    X, y = create_features(st.session_state.history)
    if len(X) < 10:
        return None, None, None, None

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X, y)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)

    # XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xgb.fit(X, y)

    # AI Strategy – học trọng số theo thời gian và độ tin cậy
    ai = LogisticRegression()
    recent_weight = np.linspace(0.5, 1.0, len(y))

    # Nếu đã có độ tin cậy trước đó, nhân thêm để tự học tốt hơn
    if len(st.session_state.ai_confidence) == len(y):
        combined_weight = recent_weight * np.array(st.session_state.ai_confidence)
    else:
        combined_weight = recent_weight

    ai.fit(X, y, sample_weight=combined_weight)

    return lr, rf, xgb, ai

# ====== Hàm dự đoán ======
def predict_next(lr, rf, xgb, ai):
    history = st.session_state.history
    if len(history) < 6:
        return None

    latest = np.array([[1 if x == "Tài" else 0 for x in history[-6:]]])
    preds = {}

    preds["Logistic Regression"] = lr.predict_proba(latest)[0][1]
    preds["Random Forest"] = rf.predict_proba(latest)[0][1]
    preds["XGBoost"] = xgb.predict_proba(latest)[0][1]
    preds["AI Strategy"] = ai.predict_proba(latest)[0][1]

    # Pattern Detector
    pattern_score = 0.5
    if len(history) >= 12:
        recent = history[-6:]
        for i in range(len(history) - 12):
            if history[i:i+6] == recent:
                pattern_score = 1.0 if history[i+6] == "Tài" else 0.0
                break
    preds["Pattern Detector"] = pattern_score

    # Trung bình có trọng số
    final_score = np.mean(list(preds.values()))
    return preds, final_score

# ====== Hàm thêm kết quả ======
def add_result(result):
    st.session_state.history.append(result)
    if len(st.session_state.history) > 200:
        st.session_state.history = st.session_state.history[-200:]

    # Cập nhật độ tin cậy của AI
    if "ai_last_pred" in st.session_state:
        was_correct = (st.session_state.ai_last_pred == result)
        st.session_state.ai_confidence.append(1.2 if was_correct else 0.8)
        if len(st.session_state.ai_confidence) > len(st.session_state.history):
            st.session_state.ai_confidence = st.session_state.ai_confidence[-len(st.session_state.history):]

# ====== Giao diện ======
st.title("🎯 AI Dự đoán Tài / Xỉu – Phiên bản Tự Học Nâng Cấp")

col1, col2 = st.columns([2,1])
with col1:
    st.markdown("#### 📊 Kết quả gần đây:")
    if st.session_state.history:
        st.write(" → ".join(st.session_state.history[-30:]))
    else:
        st.info("Chưa có dữ liệu, nhập kết quả để bắt đầu.")

with col2:
    if st.button("🧹 Xóa lịch sử"):
        st.session_state.history.clear()
        st.session_state.ai_confidence.clear()
        st.success("Đã xóa toàn bộ lịch sử!")

st.divider()

# Nút nhập kết quả
col_tai, col_xiu = st.columns(2)
with col_tai:
    if st.button("Nhập Tài"):
        add_result("Tài")
with col_xiu:
    if st.button("Nhập Xỉu"):
        add_result("Xỉu")

st.divider()

# Huấn luyện
if st.button("⚙️ Huấn luyện lại từ lịch sử"):
    with st.spinner("Đang huấn luyện các mô hình..."):
        models = train_models()
    if models[0] is not None:
        st.success("✅ Huấn luyện thành công!")
    else:
        st.warning("❗ Cần ít nhất 10 ván để huấn luyện.")

# Dự đoán
if len(st.session_state.history) >= 6:
    models = train_models()
    if models and models[0]:
        preds, final_score = predict_next(*models)
        if preds:
            st.session_state.ai_last_pred = "Tài" if final_score >= 0.5 else "Xỉu"
            st.subheader(f"🎯 Dự đoán chung: **{st.session_state.ai_last_pred}** ({final_score:.2%})")
            st.caption("Tổng hợp từ 4 mô hình + phát hiện mẫu gần nhất:")

            for k, v in preds.items():
                st.write(f"**{k}** → {v:.2%}")
    else:
        st.info("Huấn luyện mô hình trước khi dự đoán.")

else:
    st.warning("Cần ít nhất 6 ván để bắt đầu dự đoán.")
