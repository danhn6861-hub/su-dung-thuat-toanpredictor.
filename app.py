import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, brier_score_loss
from datetime import datetime
import matplotlib.pyplot as plt
import joblib
import io
import os

# ====== Optional: XGBoost ======
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ====== Streamlit Config ======
st.set_page_config(page_title="AI Dự đoán Tài/Xỉu", layout="wide")
st.title("🎯 DỰ ĐOÁN TÀI/XỈU BẰNG TRÍ TUỆ NHÂN TẠO (AI)")
st.caption("⚠️ Ứng dụng phục vụ MỤC ĐÍCH NGHIÊN CỨU – KHÔNG khuyến khích cờ bạc.")

# ====== Constants ======
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ====== Feature Generator ======
def create_features(history, window=6):
    X, y = [], []
    if len(history) <= window:
        return np.empty((0, window)), np.array([])
    for i in range(window, len(history)):
        seq = [1 if h == "Tài" else 0 for h in history[i - window:i]]
        X.append(seq)
        y.append(1 if history[i] == "Tài" else 0)
    return np.array(X), np.array(y)

def plot_history_pie(history):
    df = pd.DataFrame(history, columns=["Kết quả"])
    counts = df["Kết quả"].value_counts()
    st.subheader("📊 Thống kê kết quả:")
    st.bar_chart(counts)

# ====== Train Models ======
@st.cache_resource
def train_models_hybrid(history_tuple, ai_conf_tuple, use_xgb=True):
    history = list(history_tuple)
    ai_conf = list(ai_conf_tuple)
    xgb = None

    if len(history) < 8:
        return None

    X, y = create_features(history, window=6)
    if X.shape[0] < 6 or len(np.unique(y)) < 2:
        return None

    recent_weight = np.linspace(0.4, 1.0, len(y))
    if ai_conf and len(ai_conf) >= len(y):
        combined = recent_weight * np.array(ai_conf[-len(y):], dtype=float)
    elif ai_conf and len(ai_conf) < len(y):
        pad = np.ones(len(y) - len(ai_conf))
        combined = recent_weight * np.concatenate([pad, np.array(ai_conf, dtype=float)])
    else:
        combined = recent_weight
    combined = np.clip(combined, 0.2, 2.0)

    n_splits = min(4, max(2, X.shape[0] // 6))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    lr = LogisticRegressionCV(cv=tscv, max_iter=1000, class_weight='balanced',
                              scoring='accuracy', random_state=RANDOM_SEED)
    rf = RandomForestClassifier(n_estimators=60, max_depth=6, min_samples_split=8,
                                class_weight='balanced', n_jobs=-1, random_state=RANDOM_SEED)

    learners = [('lr', lr), ('rf', rf)]

    if use_xgb and HAS_XGB:
        try:
            xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                n_estimators=50, random_state=RANDOM_SEED, n_jobs=1)
            learners.append(('xgb', xgb))
        except Exception:
            xgb = None

    try:
        lr.fit(X, y, sample_weight=combined)
    except Exception:
        lr.fit(X, y)
    rf.fit(X, y, sample_weight=combined)
    if use_xgb and HAS_XGB and xgb is not None:
        try:
            xgb.fit(X, y, sample_weight=combined)
        except Exception:
            xgb.fit(X, y)

    try:
        calibrated_rf = CalibratedClassifierCV(base_estimator=rf, cv='prefit').fit(X, y)
    except Exception:
        calibrated_rf = rf

    estimators_voting = [('lr', lr), ('rf', calibrated_rf)]
    if xgb is not None:
        estimators_voting.append(('xgb', xgb))

    voting = VotingClassifier(estimators=estimators_voting, voting='soft', n_jobs=-1)
    voting.fit(X, y)

    try:
        stack_estimators = [('lr', lr), ('rf', rf)]
        if xgb is not None:
            stack_estimators.append(('xgb', xgb))
        stack = StackingClassifier(
            estimators=stack_estimators,
            final_estimator=LogisticRegression(max_iter=500),
            passthrough=True,
            n_jobs=-1
        ).fit(X, y)
    except Exception:
        stack = voting

    metrics = {}
    if X.shape[0] > 12:
        split = int(0.8 * len(X))
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]
        try:
            pred = voting.predict(X_te)
            metrics['voting_acc'] = float(accuracy_score(y_te, pred))
            pprob = voting.predict_proba(X_te)[:, 1]
            metrics['voting_brier'] = float(brier_score_loss(y_te, pprob))
        except Exception:
            metrics['voting_acc'] = None
            metrics['voting_brier'] = None

    return {
        "voting": voting,
        "stacking": stack,
        "lr": lr,
        "rf": rf,
        "xgb": xgb,
        "metrics": metrics
    }

# ====== Prediction ======
def predict_next(model_dict, history):
    if model_dict is None or len(history) < 6:
        return None, None
    X, _ = create_features(history, window=6)
    last = X[-1].reshape(1, -1)
    model = model_dict["voting"]
    pred = model.predict(last)[0]
    proba = model.predict_proba(last)[0][pred]
    return ("Tài" if pred == 1 else "Xỉu"), round(proba, 3)

# ====== Streamlit UI ======
st.sidebar.header("⚙️ Tuỳ chọn")
st.sidebar.markdown("Nhập kết quả gần đây (Tài/Xỉu).")

if "history" not in st.session_state:
    st.session_state.history = []
if "ai_conf" not in st.session_state:
    st.session_state.ai_conf = []

col1, col2 = st.columns(2)
with col1:
    if st.button("➕ Thêm kết quả TÀI"):
        st.session_state.history.append("Tài")
        st.rerun()
with col2:
    if st.button("➖ Thêm kết quả XỈU"):
        st.session_state.history.append("Xỉu")
        st.rerun()

if st.button("↩️ Xoá kết quả cuối"):
    if st.session_state.history:
        st.session_state.history.pop()
        st.rerun()

# Hiển thị lịch sử
if st.session_state.history:
    st.write("🧾 **Chuỗi kết quả:**", " → ".join(st.session_state.history))
    plot_history_pie(st.session_state.history)

# Huấn luyện mô hình
if st.button("🚀 Huấn luyện AI"):
    with st.spinner("Đang huấn luyện mô hình..."):
        models = train_models_hybrid(
            tuple(st.session_state.history),
            tuple(st.session_state.ai_conf),
            use_xgb=True
        )
    if models:
        st.session_state.models = models
        st.success("✅ Huấn luyện thành công!")
        st.json(models["metrics"])
    else:
        st.error("❌ Không đủ dữ liệu hoặc chỉ có 1 loại kết quả!")

# Dự đoán
if "models" in st.session_state:
    if st.button("🤖 Dự đoán kết quả tiếp theo"):
        pred, conf = predict_next(st.session_state.models, st.session_state.history)
        if pred:
            st.success(f"AI dự đoán: **{pred}** (Độ tin cậy: {conf*100:.2f}%)")
            st.session_state.ai_conf.append(conf)
        else:
            st.warning("Không đủ dữ liệu để dự đoán!")

# Xuất / Nhập dữ liệu
st.divider()
st.subheader("📦 Lưu / Nạp dữ liệu")
col3, col4 = st.columns(2)
with col3:
    if st.button("💾 Xuất CSV"):
        df = pd.DataFrame(st.session_state.history, columns=["Kết quả"])
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Tải file CSV", csv, "history.csv", "text/csv")
with col4:
    uploaded = st.file_uploader("Tải file CSV có cột 'Kết quả'", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if "Kết quả" in df.columns:
            st.session_state.history = df["Kết quả"].tolist()
            st.success("✅ Đã tải dữ liệu thành công!")
        else:
            st.error("⚠️ File không có cột 'Kết quả' hợp lệ!")

st.markdown("---")
st.caption("© 2025 | Ứng dụng AI Dự đoán Tài/Xỉu – chỉ dùng cho mục đích nghiên cứu học thuật.")
