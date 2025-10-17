import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import io, base64
from datetime import datetime
import matplotlib.pyplot as plt
import random

st.set_page_config(page_title="AI Chiến Lược Tự Tiến Hóa – Cấp 5", layout="wide")

# ===========================
# ⚙️ KHỞI TẠO TRẠNG THÁI
# ===========================
if "history" not in st.session_state:
    st.session_state.history = []
if "ai_confidence" not in st.session_state:
    st.session_state.ai_confidence = []
if "models" not in st.session_state:
    st.session_state.models = None
if "strategies" not in st.session_state:
    st.session_state.strategies = {}
if "best_strategy" not in st.session_state:
    st.session_state.best_strategy = None
if "ai_last_pred" not in st.session_state:
    st.session_state.ai_last_pred = None

# ===========================
# 🧩 HÀM TẠO ĐẶC TRƯNG
# ===========================
def create_features(history, window=6):
    if len(history) < window + 1:
        return np.empty((0, window + 2)), np.empty((0,))
    X, y = [], []
    for i in range(window, len(history)):
        base = [1 if x == "Tài" else 0 for x in history[i-window:i]]
        tai_ratio = sum(base) / window
        flip = sum(base[j] != base[j-1] for j in range(1, len(base))) / (window - 1)
        X.append(base + [tai_ratio, flip])
        y.append(1 if history[i] == "Tài" else 0)
    return np.array(X), np.array(y)

# ===========================
# 🧠 MÔ HÌNH HỌC CƠ BẢN
# ===========================
def train_base_models(history, ai_confidence):
    X, y = create_features(history)
    if len(X) < 20:
        return None
    recent_weight = np.linspace(0.5, 1.0, len(y))
    if len(ai_confidence) == len(y):
        recent_weight *= np.array(ai_confidence)
    lr = LogisticRegression(max_iter=500)
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=40)
    lr.fit(X, y, sample_weight=recent_weight)
    rf.fit(X, y, sample_weight=recent_weight)
    xgb.fit(X, y, sample_weight=recent_weight)
    return lr, rf, xgb

# ===========================
# 🎯 CHIẾN LƯỢC TIẾN HÓA
# ===========================
def init_strategies():
    return {
        "pattern_reversal": {"score": 1.0, "desc": "Đảo chiều khi chuỗi dài"},
        "trend_follow": {"score": 1.0, "desc": "Theo xu hướng gần nhất"},
        "balanced_model": {"score": 1.0, "desc": "Kết hợp 3 model chính"},
        "random_check": {"score": 1.0, "desc": "Ngẫu nhiên kiểm tra đối chứng"},
        "meta_hybrid": {"score": 1.0, "desc": "Tự cân bằng giữa model & pattern"}
    }

def update_strategy_performance(result):
    if "ai_last_pred" not in st.session_state or st.session_state.ai_last_pred is None:
        return
    last = st.session_state.best_strategy
    if last not in st.session_state.strategies:
        return
    was_correct = (st.session_state.ai_last_pred == result)
    st.session_state.strategies[last]["score"] *= (1.1 if was_correct else 0.9)

def evolve_strategies():
    s = st.session_state.strategies
    scores = {k: v["score"] for k, v in s.items()}
    best = max(scores, key=scores.get)
    st.session_state.best_strategy = best
    return best, scores

# ===========================
# 🧩 PATTERN PHÂN TÍCH
# ===========================
def pattern_detector(history, lookback=6):
    if len(history) < lookback:
        return 0.5
    recent = history[-lookback:]
    same = sum(recent[i]==recent[i-1] for i in range(1,len(recent)))
    streak = same / (lookback - 1)
    return 1 - streak if streak > 0.6 else 0.5

# ===========================
# 🔮 DỰ ĐOÁN TỔNG HỢP
# ===========================
def predict_next(models, history):
    if models is None or len(history) < 6:
        return None, None
    lr, rf, xgb = models
    X, _ = create_features(history)
    latest = X[-1:]
    prob_lr = lr.predict_proba(latest)[0][1]
    prob_rf = rf.predict_proba(latest)[0][1]
    prob_xgb = xgb.predict_proba(latest)[0][1]
    model_prob = np.mean([prob_lr, prob_rf, prob_xgb])
    pattern_prob = pattern_detector(history)
    recent_ratio = sum(1 for x in history[-5:] if x == "Tài") / 5

    # CHIẾN LƯỢC HIỆN HÀNH
    strategy = st.session_state.best_strategy or "balanced_model"
    if strategy == "pattern_reversal":
        final = 0.7 * pattern_prob + 0.3 * (1 - model_prob)
    elif strategy == "trend_follow":
        final = 0.7 * model_prob + 0.3 * recent_ratio
    elif strategy == "balanced_model":
        final = (model_prob + pattern_prob) / 2
    elif strategy == "random_check":
        final = random.uniform(0.3, 0.7)
    elif strategy == "meta_hybrid":
        adapt = 0.6 if abs(recent_ratio - 0.5) > 0.3 else 0.4
        final = adapt * model_prob + (1 - adapt) * pattern_prob
    else:
        final = model_prob

    preds = {
        "Logistic": prob_lr,
        "RandomForest": prob_rf,
        "XGBoost": prob_xgb,
        "Pattern": pattern_prob
    }
    return preds, final

# ===========================
# ⚙️ HÀM THÊM KẾT QUẢ
# ===========================
def add_result(result):
    st.session_state.history.append(result)
    if len(st.session_state.history) > 300:
        st.session_state.history = st.session_state.history[-300:]
    update_strategy_performance(result)

# ===========================
# 💻 GIAO DIỆN
# ===========================
st.title("🧠 AI Dự Đoán Tài/Xỉu – Cấp 5: Chiến Lược Tự Tiến Hóa")

col1, col2 = st.columns([2,1])
with col1:
    st.write("📜 Lịch sử gần đây:")
    st.write(" → ".join(st.session_state.history[-30:]) if st.session_state.history else "Chưa có dữ liệu.")
with col2:
    if st.button("🧹 Xóa lịch sử"):
        st.session_state.history.clear()
        st.session_state.strategies = init_strategies()
        st.session_state.best_strategy = None
        st.success("Đã xóa toàn bộ lịch sử!")

st.divider()

# Nút nhập
col_tai, col_xiu = st.columns(2)
with col_tai:
    if st.button("Nhập Tài"):
        add_result("Tài")
        st.rerun()
with col_xiu:
    if st.button("Nhập Xỉu"):
        add_result("Xỉu")
        st.rerun()

st.divider()

# Huấn luyện
if st.button("⚙️ Huấn luyện AI"):
    st.session_state.models = train_base_models(st.session_state.history, st.session_state.ai_confidence)
    if st.session_state.models:
        if not st.session_state.strategies:
            st.session_state.strategies = init_strategies()
        st.success("✅ AI đã huấn luyện xong và lưu chiến lược!")
        st.rerun()

# Dự đoán
if st.session_state.models and len(st.session_state.history) >= 6:
    best, scores = evolve_strategies()
    preds, final = predict_next(st.session_state.models, st.session_state.history)
    if preds:
        st.session_state.ai_last_pred = "Tài" if final >= 0.5 else "Xỉu"
        st.subheader(f"🎯 Dự đoán: **{st.session_state.ai_last_pred}** ({final:.2%})")
        st.caption(f"Chiến lược hiện tại: **{best}** – điểm {scores[best]:.2f}")
        st.write("Chi tiết từng mô hình:")
        for k, v in preds.items():
            st.write(f"- {k}: {v:.2%}")
        st.progress(final)
else:
    st.info("Huấn luyện AI và nhập đủ 6 ván để bắt đầu dự đoán.")

st.sidebar.markdown("""
### 🧬 Cấp 5 – Evolutionary AI
- Tự sinh & tiến hóa chiến lược
- Tự đánh giá hiệu suất từng mô hình
- Cơ chế thưởng/phạt sau mỗi lần dự đoán
- Tự điều chỉnh hướng học (Meta-learning)
""")
