# app.py (Cấp 1.1 - Ổn định & Tối ưu tốc độ)
import streamlit as st
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -----------------------
# CONFIG
# -----------------------
WINDOW = 6
RANDOM_STATE = 42

# -----------------------
# HELPERS
# -----------------------
def encode_history(history):
    return [1 if x == "Tài" else 0 for x in history]

def create_features(history, window=WINDOW):
    H = encode_history(history)
    X, y = [], []
    for i in range(len(H) - window):
        X.append(H[i:i+window])
        y.append(H[i+window])
    return np.array(X), np.array(y)

def pattern_detector_predict(history, window=WINDOW):
    if len(history) < window + 1:
        return None, 0.5
    pattern = history[-window:]
    matches = []
    for i in range(len(history) - window):
        if history[i:i+window] == pattern:
            if i + window < len(history):
                matches.append(history[i + window])
    if not matches:
        return None, 0.5
    cnt = Counter(matches)
    pred = max(cnt.items(), key=lambda x: x[1])[0]
    prob = cnt[pred] / len(matches)
    return pred, prob

def safe_predict(model, feats):
    try:
        probs = model.predict_proba([feats])[0]
        return ("Tài", float(probs[1])) if probs[1] >= probs[0] else ("Xỉu", float(probs[0]))
    except Exception:
        try:
            p = model.predict([feats])[0]
            return ("Tài" if int(p) == 1 else "Xỉu", 0.5)
        except Exception:
            return None, 0.5

def normalize_weights(w):
    s = sum(w.values())
    if s == 0:
        n = len(w)
        for k in w:
            w[k] = 1 / n
    else:
        for k in w:
            w[k] = w[k] / s
    return w

# -----------------------
# INIT SESSION
# -----------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "weights" not in st.session_state:
    st.session_state.weights = {"LR": 1, "RF": 1, "XGB": 1, "PD": 1}
    normalize_weights(st.session_state.weights)

if "models" not in st.session_state:
    st.session_state.models = {"LR": None, "RF": None, "XGB": None}

if "stats" not in st.session_state:
    st.session_state.stats = {k: {"correct": 0, "total": 0} for k in ["LR", "RF", "XGB", "PD", "AI"]}

if "preds" not in st.session_state:
    st.session_state.preds = {}

if "probs" not in st.session_state:
    st.session_state.probs = {}

if "ai_history" not in st.session_state:
    st.session_state.ai_history = []

# -----------------------
# STYLING
# -----------------------
st.set_page_config(page_title="AI Tài/Xỉu - Cấp 1.1", page_icon="🎯", layout="centered")
st.markdown("""
<style>
.stApp { background-color:#071029; color:#e6eef8; }
.card { background-color:#0a1b2a; padding:14px; border-radius:12px; box-shadow:0 3px 8px rgba(0,0,0,0.4); margin:6px; }
.model-name { font-weight:700; font-size:16px; color:#8ab4f8; }
.pred { font-size:20px; font-weight:700; margin-top:4px; }
.small { font-size:13px; color:#9fb0c9; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 AI Dự đoán Tài/Xỉu — Hệ thống Cấp 1.1")
st.write("Tối ưu tốc độ huấn luyện, AI tự học rút kinh nghiệm, dự đoán 5 mô hình song song.")

# -----------------------
# CORE FUNCTIONS
# -----------------------
def train_models():
    hist = st.session_state.history
    if len(hist) <= WINDOW:
        return

    X, y = create_features(hist)
    feats = encode_history(hist[-WINDOW:])

    # Logistic Regression
    lr = LogisticRegression(max_iter=120, random_state=RANDOM_STATE)
    lr.fit(X, y)
    p_lr, pr_lr = safe_predict(lr, feats)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=40, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)
    p_rf, pr_rf = safe_predict(rf, feats)

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=40,
        learning_rate=0.3,
        max_depth=3,
        verbosity=0,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE
    )
    xgb.fit(X, y)
    p_xgb, pr_xgb = safe_predict(xgb, feats)

    # Pattern Detector
    p_pd, pr_pd = pattern_detector_predict(hist, window=WINDOW)

    st.session_state.models.update({"LR": lr, "RF": rf, "XGB": xgb})
    st.session_state.preds = {"LR": p_lr, "RF": p_rf, "XGB": p_xgb, "PD": p_pd}
    st.session_state.probs = {"LR": pr_lr, "RF": pr_rf, "XGB": pr_xgb, "PD": pr_pd}

    # AI Meta Strategy
    w = st.session_state.weights
    score_tai = sum(w[m] * st.session_state.probs[m] for m in w)
    score_xiu = sum(w[m] * (1 - st.session_state.probs[m]) for m in w)
    ai_pred = "Tài" if score_tai >= score_xiu else "Xỉu"
    ai_prob = max(score_tai, score_xiu) / (score_tai + score_xiu)
    st.session_state.preds["AI"], st.session_state.probs["AI"] = ai_pred, ai_prob

def update_ai(result):
    preds = st.session_state.preds
    w = st.session_state.weights
    for m in ["LR", "RF", "XGB", "PD"]:
        if preds.get(m) == result:
            w[m] *= 1.05
        else:
            w[m] *= 0.95
    normalize_weights(w)
    st.session_state.ai_history.append({"real": result, "weights": w.copy()})
    if len(st.session_state.ai_history) > 30:
        st.session_state.ai_history.pop(0)

def update_stats(result):
    for m, pred in st.session_state.preds.items():
        if pred is None:
            continue
        st.session_state.stats[m]["total"] += 1
        if pred == result:
            st.session_state.stats[m]["correct"] += 1

def add_result(result):
    update_stats(result)
    update_ai(result)
    st.session_state.history.append(result)
    train_models()

def reset_all():
    for key in ["history", "models", "weights", "stats", "preds", "probs", "ai_history"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# -----------------------
# BUTTONS
# -----------------------
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔴 TÀI"):
        add_result("Tài")
with col2:
    if st.button("🔵 XỈU"):
        add_result("Xỉu")
with col3:
    if st.button("🧹 Xóa lịch sử"):
        reset_all()

# -----------------------
# DISPLAY
# -----------------------
if not st.session_state.history:
    st.info("Bấm TÀI hoặc XỈU để bắt đầu huấn luyện.")
else:
    st.markdown("### 🧾 Lịch sử:")
    safe_history = [str(x) for x in st.session_state.history[-40:] if x is not None]
    st.write(" → ".join(safe_history))

st.markdown("---")
st.markdown("## ⚡ Kết quả dự đoán")

cols = st.columns(3)
models = ["LR", "RF", "XGB", "PD", "AI"]
for i, m in enumerate(models):
    with cols[i % 3]:
        pred = st.session_state.preds.get(m)
        prob = st.session_state.probs.get(m, 0.5)
        stats = st.session_state.stats.get(m, {"correct": 0, "total": 0})
        total = stats["total"]
        win = stats["correct"]
        rate = win / total if total else 0
        name = {
            "LR": "Logistic Regression",
            "RF": "Random Forest",
            "XGB": "XGBoost",
            "PD": "Pattern Detector",
            "AI": "AI Strategy"
        }[m]
        st.markdown(f"""
        <div class="card">
            <div class="model-name">{name}</div>
            <div class="small">Dự đoán:</div>
            <div class="pred">{pred if pred else 'Chưa đủ dữ liệu'}</div>
            <div class="small">Xác suất: {prob:.1%}</div>
            <div class="small">Tỉ lệ thắng: {rate:.1%} ({win}/{total})</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.write("🧠 **Trọng số học hiện tại:**")
st.write(st.session_state.weights)
