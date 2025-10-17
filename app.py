# app.py
import streamlit as st
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ========================
# CONFIG
# ========================
WINDOW = 6
RANDOM_STATE = 42
MAX_HISTORY = 200  # giới hạn lịch sử để tránh chậm dần

# ========================
# HELPERS
# ========================
def encode_history(history):
    """Map ['Tài'/'Xỉu'] -> [1/0]"""
    return [1 if x == "Tài" else 0 for x in history]

def create_features(history, window=WINDOW):
    """Return X (n-window, window), y (n-window)"""
    H = encode_history(history)
    X, y = [], []
    for i in range(len(H) - window):
        X.append(H[i:i+window])
        y.append(H[i+window])
    return np.array(X), np.array(y)

def pattern_detector_predict(history, window=WINDOW):
    """Pattern detector: find occurrences of last `window` pattern and return majority next outcome and probability."""
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

def safe_predict_proba(model, feats):
    """Return (label_string, prob_of_label). feats is list/array of 0/1 length WINDOW."""
    try:
        probs = model.predict_proba([feats])[0]
        # probs[1] prob of 1 => "Tài"
        if probs[1] >= probs[0]:
            return "Tài", float(probs[1])
        else:
            return "Xỉu", float(probs[0])
    except Exception:
        try:
            p = model.predict([feats])[0]
            return ("Tài" if int(p) == 1 else "Xỉu"), 0.5
        except Exception:
            return None, 0.5

def normalize_dict(d):
    s = sum(d.values())
    if s == 0:
        n = len(d)
        for k in d: d[k] = 1.0 / n
    else:
        for k in d: d[k] = d[k] / s
    return d

# ========================
# SESSION INIT
# ========================
if "history" not in st.session_state:
    st.session_state.history = []  # values: "Tài" / "Xỉu"

# Store base models (fitted objects). They are only updated on manual training.
if "models" not in st.session_state:
    st.session_state.models = {
        "LR": None,   # Logistic Regression (base)
        "RF": None,   # Random Forest
        "XGB": None,  # XGBoost
        "META": None  # AI Strategy (meta LogisticRegression trained on base probs with sample_weight)
    }

# Store last predictions (from current models) for display
if "preds" not in st.session_state:
    st.session_state.preds = {"LR": None, "RF": None, "XGB": None, "PD": None, "AI": None}

# Store last probabilities for display
if "probs" not in st.session_state:
    st.session_state.probs = {"LR": 0.5, "RF": 0.5, "XGB": 0.5, "PD": 0.5, "AI": 0.5}

# Stats for win rate
if "stats" not in st.session_state:
    st.session_state.stats = {k: {"correct": 0, "total": 0} for k in ["LR", "RF", "XGB", "PD", "AI"]}

# AI weights history (optional), keep short
if "ai_history" not in st.session_state:
    st.session_state.ai_history = []

# ========================
# STYLING
# ========================
st.set_page_config(page_title="AI Tài/Xỉu - Manual Train + Weighted Meta", page_icon="🎯", layout="centered")
st.markdown("""
<style>
.stApp { background-color:#071029; color:#e6eef8; }
.card { background-color:#0a1b2a; padding:14px; border-radius:12px; box-shadow:0 3px 8px rgba(0,0,0,0.4); margin:6px; }
.model-name { font-weight:700; font-size:16px; color:#8ab4f8; }
.pred { font-size:20px; font-weight:700; margin-top:4px; }
.small { font-size:13px; color:#9fb0c9; }
.btn { width:100%; padding:8px; border-radius:8px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 AI Dự đoán Tài/Xỉu — Manual Train (Weighted Meta)")
st.write("Dữ liệu lưu trong session. Chỉ huấn luyện khi bạn bấm **Huấn luyện lại**. AI Self-Learn sử dụng sample_weight ưu tiên mẫu gần đây.")

# ========================
# CORE: Manual training with weighted meta
# ========================
def train_models_manual():
    """Train base models and meta model. Meta model trained with sample_weight emphasizing recent samples."""
    hist = st.session_state.history

    # ensure history length limit
    if len(hist) > MAX_HISTORY:
        st.session_state.history = st.session_state.history[-MAX_HISTORY:]
        hist = st.session_state.history

    if len(hist) <= WINDOW:
        st.warning("Chưa đủ dữ liệu để huấn luyện (cần > WINDOW).")
        return

    # build X,y
    X, y = create_features(hist)  # X shape (n-WINDOW, WINDOW)
    if X.size == 0 or len(set(y)) < 2:
        st.warning("Dữ liệu chưa đủ đa dạng (cần cả Tài và Xỉu) để huấn luyện.")
        return

    # ---- train base models (light configs) ----
    # Base Logistic Regression
    lr = LogisticRegression(max_iter=200, solver="liblinear", random_state=RANDOM_STATE)
    lr.fit(X, y)

    # Random Forest (light)
    rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)

    # XGBoost (light)
    xgb = XGBClassifier(n_estimators=50, learning_rate=0.2, max_depth=3,
                        verbosity=0, use_label_encoder=False, eval_metric="logloss",
                        random_state=RANDOM_STATE)
    xgb.fit(X, y)

    # ---- build meta features: for each training sample, get base models' probs ----
    meta_X = []
    for i in range(len(X)):
        row = []
        # probs for class 1 ("Tài")
        try:
            p_lr = lr.predict_proba([X[i]])[0][1]
        except Exception:
            p_lr = 0.5
        try:
            p_rf = rf.predict_proba([X[i]])[0][1]
        except Exception:
            p_rf = 0.5
        try:
            p_xgb = xgb.predict_proba([X[i]])[0][1]
        except Exception:
            p_xgb = 0.5
        # add them as features, plus recent frequency
        row.extend([p_lr, p_rf, p_xgb])
        row.append(np.mean(X[i]))  # recent frequency of "Tài" in that window
        meta_X.append(row)
    meta_X = np.array(meta_X)

    # ---- sample_weight: give more weight to recent samples ----
    sample_weight = np.linspace(0.5, 1.0, len(y))  # earlier samples smaller weight, recent larger
    # normalize to reasonable scale (optional)
    sample_weight = sample_weight / np.mean(sample_weight)

    # ---- train meta LogisticRegression with sample_weight ----
    meta = LogisticRegression(max_iter=200, solver="liblinear", random_state=RANDOM_STATE)
    try:
        meta.fit(meta_X, y, sample_weight=sample_weight)
    except TypeError:
        # sklearn version may not accept sample_weight in this way; fallback to unweighted fit
        meta.fit(meta_X, y)

    # ---- save models ----
    st.session_state.models["LR"] = lr
    st.session_state.models["RF"] = rf
    st.session_state.models["XGB"] = xgb
    st.session_state.models["META"] = meta

    # ---- compute predictions for last window to display immediately ----
    last_window = encode_history(hist[-WINDOW:])
    # base probs for last_window
    try:
        p_lr_label, p_lr_prob = safe_predict_proba(lr, last_window)
    except Exception:
        p_lr_label, p_lr_prob = None, 0.5
    try:
        p_rf_label, p_rf_prob = safe_predict_proba(rf, last_window)
    except Exception:
        p_rf_label, p_rf_prob = None, 0.5
    try:
        p_xgb_label, p_xgb_prob = safe_predict_proba(xgb, last_window)
    except Exception:
        p_xgb_label, p_xgb_prob = None, 0.5

    pd_label, pd_prob = pattern_detector_predict(hist, window=WINDOW)

    # meta feature for last window
    meta_row = np.array([[p_lr_prob, p_rf_prob, p_xgb_prob, np.mean(last_window)]])
    try:
        meta_probs = meta.predict_proba(meta_row)[0]
        meta_label = "Tài" if meta_probs[1] >= meta_probs[0] else "Xỉu"
        meta_prob = float(max(meta_probs[1], meta_probs[0]))
    except Exception:
        meta_label, meta_prob = None, 0.5

    # write into session preds/probs
    st.session_state.preds["LR"] = p_lr_label
    st.session_state.probs["LR"] = p_lr_prob
    st.session_state.preds["RF"] = p_rf_label
    st.session_state.probs["RF"] = p_rf_prob
    st.session_state.preds["XGB"] = p_xgb_label
    st.session_state.probs["XGB"] = p_xgb_prob
    st.session_state.preds["PD"] = pd_label
    st.session_state.probs["PD"] = pd_prob
    st.session_state.preds["AI"] = meta_label
    st.session_state.probs["AI"] = meta_prob

    st.success("✅ Huấn luyện hoàn tất — mô hình đã cập nhật (meta dùng sample_weight tăng dần).")

# ========================
# RECORD ACTUAL RESULT (no auto-train)
# ========================
def record_result(real_result):
    # keep history values normalized and limited
    if real_result not in ("Tài", "Xỉu"):
        return
    st.session_state.history.append(real_result)
    # cap length
    if len(st.session_state.history) > MAX_HISTORY:
        st.session_state.history = st.session_state.history[-MAX_HISTORY:]
    # update stats comparing current displayed preds (if available)
    for name in ["LR", "RF", "XGB", "PD", "AI"]:
        pred = st.session_state.preds.get(name)
        if pred is None:
            continue
        st.session_state.stats[name]["total"] += 1
        if pred == real_result:
            st.session_state.stats[name]["correct"] += 1
    # we DO NOT retrain here (manual mode). Optionally append to ai_history log
    st.session_state.ai_history.append({"real": real_result, "preds": st.session_state.preds.copy()})
    if len(st.session_state.ai_history) > 200:
        st.session_state.ai_history.pop(0)

# ========================
# RESET
# ========================
def reset_all():
    st.session_state.history = []
    st.session_state.models = {"LR": None, "RF": None, "XGB": None, "META": None}
    st.session_state.preds = {"LR": None, "RF": None, "XGB": None, "PD": None, "AI": None}
    st.session_state.probs = {"LR": 0.5, "RF": 0.5, "XGB": 0.5, "PD": 0.5, "AI": 0.5}
    st.session_state.stats = {k: {"correct": 0, "total": 0} for k in ["LR", "RF", "XGB", "PD", "AI"]}
    st.session_state.ai_history = []
    st.success("Đã xóa lịch sử và reset mô hình.")

# ========================
# UI: Controls
# ========================
col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    if st.button("🔴 TÀI"):
        record_result("Tài")
with col2:
    if st.button("🔵 XỈU"):
        record_result("Xỉu")
with col3:
    if st.button("⚙️ Huấn luyện lại"):
        with st.spinner("⏳ Đang huấn luyện... (meta sẽ dùng sample_weight ưu tiên mẫu gần đây)"):
            train_models_manual()
with col4:
    if st.button("🧹 Xóa lịch sử"):
        reset_all()

st.markdown("---")

# ========================
# DISPLAY: history and model cards
# ========================
st.markdown("### 🧾 Lịch sử (mới nhất bên phải)")
if st.session_state.history:
    safe_history = [str(x) for x in st.session_state.history[-40:] if x is not None]
    st.write(" → ".join(safe_history))
else:
    st.info("Chưa có dữ liệu. Nhấn 'TÀI' hoặc 'XỈU' để tạo lịch sử, sau đó nhấn 'Huấn luyện lại' để train mô hình.")

st.markdown("---")
st.markdown("## 🔍 Dự đoán hiện tại từ 5 mô hình")

cols = st.columns(3)
display_models = [("LR","Logistic Regression"), ("RF","Random Forest"), ("XGB","XGBoost"),
                  ("PD","Pattern Detector"), ("AI","AI Strategy (meta)")]
for i, (key, title) in enumerate(display_models):
    with cols[i % 3]:
        pred = st.session_state.preds.get(key)
        prob = st.session_state.probs.get(key, 0.5)
        stats = st.session_state.stats.get(key, {"correct":0,"total":0})
        total = stats["total"]; correct = stats["correct"]
        rate = (correct/total) if total>0 else 0.0
        st.markdown(f"""
        <div class="card">
            <div class="model-name">{title}</div>
            <div class="small">Dự đoán:</div>
            <div class="pred">{pred if pred else 'Chưa huấn luyện'}</div>
            <div class="small">Xác suất: {prob:.1%}</div>
            <div class="small">Tỉ lệ thắng: {rate:.1%} ({correct}/{total})</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🧠 Thông tin meta & log")
st.write("Meta model (Logistic) được huấn luyện trên xác suất của 3 base models + tần suất ngắn hạn; sample_weight = np.linspace(0.5,1.0,len(y)) (ưu tiên mẫu gần đây).")
st.write("Tổng số bản ghi lịch sử:", len(st.session_state.history))
if st.button("Hiển thị log AI history (mới nhất 50)"):
    st.write(st.session_state.ai_history[-50:])
