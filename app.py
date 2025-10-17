# app.py
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
    """Map ["Tài"/"Xỉu"] -> [1/0]"""
    return [1 if x == "Tài" else 0 for x in history]

def create_features(history, window=WINDOW):
    """Sliding windows -> X, y for supervised training"""
    H = encode_history(history)
    X, y = [], []
    for i in range(len(H) - window):
        X.append(H[i:i+window])
        y.append(H[i+window])
    if len(X) == 0:
        return np.array([]), np.array([])
    return np.array(X), np.array(y)

def pattern_detector_predict(history, window=WINDOW):
    """Return prediction 'Tài'/'Xỉu' and probability from pattern matching last `window`."""
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
    """Return (label_string, prob_of_that_label) for given model and features (0/1 list)."""
    try:
        probs = model.predict_proba([feats])[0]
        # probs[1] -> prob of 1 ("Tài"), probs[0] -> prob of 0 ("Xỉu")
        if probs[1] >= probs[0]:
            return "Tài", float(probs[1])
        else:
            return "Xỉu", float(probs[0])
    except Exception:
        # fallback to predict()
        try:
            p = model.predict([feats])[0]
            return ("Tài" if int(p) == 1 else "Xỉu"), 0.5
        except Exception:
            return None, 0.5

def normalize_weights(wdict):
    s = sum(wdict.values())
    if s == 0:
        # reset to uniform
        n = len(wdict)
        for k in wdict:
            wdict[k] = 1.0 / n
        return wdict
    for k in wdict:
        wdict[k] = float(wdict[k]) / s
    return wdict

# -----------------------
# SESSION INIT
# -----------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of "Tài"/"Xỉu"

# models store
if "models" not in st.session_state:
    st.session_state.models = {
        "Logistic Regression": None,
        "Random Forest": None,
        "XGBoost": None,
        "Pattern Detector": None,  # placeholder
        "AI Strategy (meta)": None  # meta: will not retrain base models
    }

# last predictions shown (these were predictions for the round that just finished)
if "last_preds" not in st.session_state:
    st.session_state.last_preds = {name: None for name in st.session_state.models.keys()}

# model statistics for win rates
if "model_stats" not in st.session_state:
    st.session_state.model_stats = {name: {"correct": 0, "total": 0} for name in st.session_state.models.keys()}

# AI self-learning memory: weights for base models and history log
if "ai_memory" not in st.session_state:
    st.session_state.ai_memory = {
        "weights": {
            "Logistic Regression": 1.0,
            "Random Forest": 1.0,
            "XGBoost": 1.0,
            "Pattern Detector": 1.0
        },
        "history": []  # list of dicts {pred_meta, real, weights_before}
    }
    normalize_weights(st.session_state.ai_memory["weights"])

# -----------------------
# STYLING (dark dashboard)
# -----------------------
st.set_page_config(page_title="AI Tài/Xỉu Dashboard", page_icon="🎯", layout="centered")
st.markdown("""
    <style>
    .stApp { background-color:#071029; color: #e6eef8; }
    .card {
        background-color:#071a2a;
        padding:14px;
        border-radius:10px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.6);
        color: #e6eef8;
        margin-bottom:12px;
    }
    .model-name {font-weight:700; font-size:16px; margin-bottom:6px;}
    .pred {font-size:20px; font-weight:700; margin-top:6px;}
    .small {font-size:13px; color:#9fb0c9;}
    .btn { width:100%; padding:10px; border-radius:8px; font-weight:700; }
    </style>
""", unsafe_allow_html=True)

st.title("🎯 AI Dự đoán Tài/Xỉu — Dashboard (5 mô hình)")
st.write("Nhập kết quả thực tế bằng nút TÀI / XỈU. Hệ thống lưu session, huấn luyện nhanh, AI tự học rút kinh nghiệm theo thắng/thua.")

# -----------------------
# ACTIONS: add result / reset
# -----------------------
def train_and_predict_all_models():
    """Train base models if enough data, compute predictions for next round and compute meta prediction using ai weights."""
    history = st.session_state.history
    n = len(history)
    preds_next = {name: None for name in st.session_state.models.keys()}

    # if insufficient data, clear models/preds
    if n < WINDOW + 1:
        st.session_state.models = {name: st.session_state.models[name] if name == "Pattern Detector" else None for name in st.session_state.models}
        st.session_state.last_preds = {name: None for name in st.session_state.models.keys()}
        return

    # prepare X,y
    X, y = create_features(history, window=WINDOW)  # shape (n-WINDOW, WINDOW)
    if X.size == 0:
        st.session_state.last_preds = {name: None for name in st.session_state.models.keys()}
        return

    # Train base models (fast configs)
    # Logistic Regression
    try:
        lr = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)
        lr.fit(X, y)
        st.session_state.models["Logistic Regression"] = lr
    except Exception:
        st.session_state.models["Logistic Regression"] = None

    # Random Forest
    try:
        rf = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=RANDOM_STATE)
        rf.fit(X, y)
        st.session_state.models["Random Forest"] = rf
    except Exception:
        st.session_state.models["Random Forest"] = None

    # XGBoost (light)
    try:
        xgb = XGBClassifier(n_estimators=80, use_label_encoder=False, eval_metric="logloss", verbosity=0, random_state=RANDOM_STATE)
        xgb.fit(X, y)
        st.session_state.models["XGBoost"] = xgb
    except Exception:
        st.session_state.models["XGBoost"] = None

    # Pattern Detector placeholder (no training)
    st.session_state.models["Pattern Detector"] = "pattern_ok"

    # Prepare last window features for prediction
    last_window = encode_history(history[-WINDOW:])

    # compute base model predictions & probabilities
    base_info = {}
    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        model = st.session_state.models.get(name)
        if model is None:
            base_info[name] = {"pred": None, "prob": 0.5}
        else:
            try:
                pred_label, prob = safe_predict_proba(model, last_window)
                base_info[name] = {"pred": pred_label, "prob": prob}
            except Exception:
                base_info[name] = {"pred": None, "prob": 0.5}

    # pattern detector
    pd_pred, pd_prob = pattern_detector_predict(history, window=WINDOW)
    base_info["Pattern Detector"] = {"pred": pd_pred, "prob": pd_prob}

    # store individual predictions for display (these are predictions for NEXT round)
    for name in st.session_state.models.keys():
        preds_next[name] = base_info.get(name, {"pred": None})["pred"]

    # AI Strategy (meta) uses weighted combination of model probabilities (from base models + pattern)
    weights = st.session_state.ai_memory["weights"]
    # compute scores for "Tài" and "Xỉu"
    score_tai = 0.0
    score_xiu = 0.0
    for name, info in base_info.items():
        w = weights.get(name, 1.0)
        p_tai = info.get("prob", 0.5)
        p_xiu = 1.0 - p_tai
        # accumulate weighted scores
        score_tai += w * p_tai
        score_xiu += w * p_xiu

    if score_tai >= score_xiu:
        preds_next["AI Strategy (meta)"] = "Tài"
        # estimate meta prob as normalized score
        meta_prob = score_tai / (score_tai + score_xiu) if (score_tai + score_xiu) > 0 else 0.5
    else:
        preds_next["AI Strategy (meta)"] = "Xỉu"
        meta_prob = score_xiu / (score_tai + score_xiu) if (score_tai + score_xiu) > 0 else 0.5

    # Save last_preds (predictions for NEXT round) in session
    st.session_state.last_preds = preds_next
    # also save last computed probabilities for display
    st.session_state._last_probs = {name: base_info.get(name, {}).get("prob", 0.5) for name in base_info.keys()}
    st.session_state._last_probs["AI Strategy (meta)"] = meta_prob

def handle_new_result(result):
    """Process new real result (result = 'Tài'/'Xỉu'):
       - update model_stats using previous st.session_state.last_preds (these were predictions for the round that just finished)
       - update ai weights (rút kinh nghiệm)
       - append result to history
       - retrain and predict next
    """
    # 1) update stats for models based on last_preds (predictions made before this real result)
    prev_preds = st.session_state.last_preds.copy() if "last_preds" in st.session_state else {}
    if prev_preds:
        for name, pred in prev_preds.items():
            if pred is None:
                continue
            st.session_state.model_stats[name]["total"] += 1
            if pred == result:
                st.session_state.model_stats[name]["correct"] += 1

    # 2) AI self-learning: rút kinh nghiệm (dựa trên who predicted correctly this round)
    # We'll increase weights of models that predicted correctly, decrease those incorrect.
    # Use modest update factors for stability.
    weights = st.session_state.ai_memory["weights"]
    weight_increase = 1.05
    weight_decrease = 0.95
    updated = False
    if prev_preds:
        for name in weights.keys():
            pred = prev_preds.get(name)
            if pred is None:
                continue
            if pred == result:
                weights[name] *= weight_increase
                updated = True
            else:
                weights[name] *= weight_decrease
                updated = True
    if updated:
        normalize_weights(weights)
        # log history
        st.session_state.ai_memory["history"].append({
            "preds": prev_preds,
            "real": result,
            "weights": weights.copy()
        })

    # 3) append real result to history
    st.session_state.history.append(result)

    # 4) retrain & predict next
    train_and_predict_all_models()

def reset_history():
    st.session_state.history = []
    st.session_state.models = {name: None for name in st.session_state.models.keys()}
    st.session_state.last_preds = {name: None for name in st.session_state.models.keys()}
    st.session_state.model_stats = {name: {"correct": 0, "total": 0} for name in st.session_state.models.keys()}
    st.session_state.ai_memory = {
        "weights": {
            "Logistic Regression": 1.0,
            "Random Forest": 1.0,
            "XGBoost": 1.0,
            "Pattern Detector": 1.0
        },
        "history": []
    }
    normalize_weights(st.session_state.ai_memory["weights"])
    st.session_state._last_probs = {}
    st.success("Đã xóa lịch sử và reset mô hình/AI memory.")

# UI Buttons
col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("🔴 TÀI", key="btn_tai"):
        handle_new_result("Tài")
with col2:
    if st.button("🔵 XỈU", key="btn_xiu"):
        handle_new_result("Xỉu")
with col3:
    if st.button("🧹 Xóa lịch sử", key="btn_reset"):
        reset_history()

st.markdown("---")

# Ensure models/predictions are ready on load
if "last_preds" not in st.session_state or st.session_state.last_preds is None:
    st.session_state.last_preds = {name: None for name in st.session_state.models.keys()}
train_and_predict_all_models()

# -----------------------
# UI: History and cards
# -----------------------
st.markdown("### 🧾 Lịch sử (mới nhất bên phải):")
if st.session_state.history:
    st.write(" → ".join(st.session_state.history))
else:
    st.info("Chưa có dữ liệu. Bấm 'TÀI' hoặc 'XỈU' để bắt đầu.")

st.markdown("---")
st.markdown("## 🔍 Dự đoán 5 mô hình (mỗi card = 1 mô hình)")

model_names = list(st.session_state.models.keys())
cols = st.columns(3)

for idx, name in enumerate(model_names):
    col = cols[idx % 3]
    with col:
        pred = st.session_state.last_preds.get(name)
        stats = st.session_state.model_stats.get(name, {"correct": 0, "total": 0})
        total = stats["total"]
        correct = stats["correct"]
        win_rate = (correct / total) if total > 0 else 0.0
        # probability display
        prob_disp = "—"
        if "_last_probs" in st.session_state:
            prob = st.session_state._last_probs.get(name)
            if prob is not None:
                prob_disp = f"{prob:.1%}"
        # ai weight display (for base models & pattern)
        weight_disp = ""
        if name in st.session_state.ai_memory["weights"]:
            weight_disp = f"{st.session_state.ai_memory['weights'][name]:.2f}"

        st.markdown(f"""
            <div class="card">
                <div class="model-name">{name}</div>
                <div class="small">Dự đoán ván tiếp theo:</div>
                <div class="pred">{pred if pred is not None else 'Chưa đủ dữ liệu'}</div>
                <div class="small">Xác suất (ước lượng): {prob_disp}</div>
                <div class="small">Tỉ lệ thắng: {win_rate:.1%} ({correct}/{total})</div>
                <div class="small">Trọng số AI tin tưởng: {weight_disp}</div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 📋 Tóm tắt tỉ lệ thắng")
summary = []
for name, stt in st.session_state.model_stats.items():
    total = stt["total"]
    correct = stt["correct"]
    rate = (correct / total) if total > 0 else 0.0
    summary.append({"Model": name, "Correct": correct, "Total": total, "Win rate": f"{rate:.1%}"})
st.table(summary)

st.markdown("---")
st.markdown("### 🧠 AI Self-Learning (weights & history)")
st.write("Trọng số hiện tại (tổng = 1):")
st.write(st.session_state.ai_memory["weights"])
if st.button("Hiển thị log AI history"):
    st.write(st.session_state.ai_memory["history"][-20:])  # show last 20 entries

st.markdown("""
---
**Ghi chú**
- WINDOW = 6: Pattern Detector tìm các chuỗi 6 ván giống 6 ván gần nhất để dự đoán.
- AI Strategy là hệ thống meta dùng trọng số (tự điều chỉnh) dựa trên thắng/thua để kết hợp các mô hình.
- Dữ liệu lưu trong session; reload trình duyệt sẽ mất dữ liệu (nếu muốn lưu lâu dài, có thể export).
""")
