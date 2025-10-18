# app.py - Fusion Pro (Hybrid + Improved) - Streamlit-ready (2025)
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io, os, joblib
from datetime import datetime

# sklearn
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix

# optional xgboost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ----------------- Page config -----------------
st.set_page_config(page_title="Fusion Pro - AI Tài/Xỉu", layout="wide")
st.title("🔮 Fusion Pro — AI Dự đoán Tài / Xỉu (Hybrid + Improved)")
st.markdown("**Giao diện:** đẹp + biểu đồ & metrics • **Huấn luyện:** chỉ khi bạn nhấn nút • **Chạy mượt trên Streamlit Cloud**")
st.caption("⚠️ Ứng dụng chỉ phục vụ nghiên cứu/học tập. Không khuyến khích cờ bạc.")

# ----------------- Constants & init -----------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
MODEL_PATH = "/tmp/fusion_pro_model.joblib"  # fallback path
HISTORY_PATH = "/tmp/fusion_pro_history.csv"

if "history" not in st.session_state:
    st.session_state.history = []
if "ai_conf" not in st.session_state:
    st.session_state.ai_conf = []
if "models" not in st.session_state:
    st.session_state.models = None
if "ai_last_pred" not in st.session_state:
    st.session_state.ai_last_pred = None
if "undo_stack" not in st.session_state:
    st.session_state.undo_stack = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# ----------------- Utilities -----------------
def save_state_model(models, path=MODEL_PATH):
    try:
        joblib.dump(models, path)
        return True
    except Exception:
        return False

def load_state_model(path=MODEL_PATH):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None

def export_history_csv_bytes():
    df = pd.DataFrame({"Kết quả": st.session_state.history})
    return df.to_csv(index=False).encode("utf-8")

# ----------------- Feature engineering -----------------
def create_features(history, window=6):
    if len(history) <= window:
        return np.empty((0, window + 2)), np.empty((0,))
    X = []
    y = []
    for i in range(window, len(history)):
        window_slice = history[i-window:i]
        base = [1 if x == "Tài" else 0 for x in window_slice]
        tai_ratio = sum(base) / window
        changes = sum(base[j] != base[j-1] for j in range(1,len(base)))
        change_ratio = changes / max(1, (window-1))
        # current streak
        last = base[-1]
        streak = 1
        for j in range(len(base)-2, -1, -1):
            if base[j] == last:
                streak += 1
            else:
                break
        # last3 count
        last3 = sum(base[-3:]) if len(base) >=3 else sum(base)
        features = base + [tai_ratio, change_ratio, streak, last3]
        X.append(features)
        y.append(1 if history[i] == "Tài" else 0)
    return np.array(X, dtype=float), np.array(y, dtype=int)

# ----------------- Pattern detector -----------------
def pattern_detector(history, lookback=8):
    if len(history) < 3:
        return 0.5
    recent = history[-lookback:] if len(history) >= lookback else history[:]
    tai_recent = sum(1 for x in recent if x == "Tài")
    base_prob = tai_recent / len(recent)
    # streak detection
    max_streak = 1
    cur = 1
    for i in range(1, len(recent)):
        if recent[i] == recent[i-1]:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 1
    streak_factor = min(max_streak / 4.0, 1.0)
    if streak_factor > 0.5:
        adjusted = 1.0 - base_prob
    else:
        adjusted = 0.5
    return float(np.clip(adjusted, 0.1, 0.9))

# ----------------- Model training (Fusion) -----------------
@st.cache_resource
def train_fusion(history_tuple, ai_conf_tuple, use_xgb=True, cache_key=None):
    # history & weights
    history = list(history_tuple)
    ai_conf = list(ai_conf_tuple)
    xgb = None

    if len(history) < 12:  # require more data for robust hybrid
        return None

    X, y = create_features(history, window=6)
    if X.shape[0] < 8 or len(np.unique(y)) < 2:
        return None

    # noise injection to reduce exact memorization
    X = X + np.random.normal(0, 0.03, X.shape)

    # sample weights prefer recent but moderate
    recent_weight = np.linspace(0.3, 1.0, len(y))
    if ai_conf and len(ai_conf) >= len(y):
        combined = recent_weight * np.array(ai_conf[-len(y):], dtype=float)
    else:
        combined = recent_weight
    combined = np.clip(combined, 0.2, 2.0)

    # TimeSeries CV
    n_splits = min(4, max(2, X.shape[0] // 8))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Logistic with regularization
    lr = LogisticRegressionCV(cv=tscv, max_iter=1200, class_weight='balanced', scoring='accuracy', random_state=RANDOM_SEED)

    # RandomForest tuned to avoid overfit
    rf = RandomForestClassifier(n_estimators=80, max_depth=6, min_samples_split=6, min_samples_leaf=2,
                                class_weight='balanced', n_jobs=1, random_state=RANDOM_SEED)

    learners = [('lr', lr), ('rf', rf)]

    # Optional XGBoost with fallback
    if use_xgb and HAS_XGB:
        try:
            xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=80, random_state=RANDOM_SEED, n_jobs=1)
            learners.append(('xgb', xgb))
        except Exception:
            xgb = None

    # Fit models
    try:
        lr.fit(X, y, sample_weight=combined)
    except Exception:
        lr.fit(X, y)
    rf.fit(X, y, sample_weight=combined)
    if xgb is not None:
        try:
            xgb.fit(X, y, sample_weight=combined)
        except Exception:
            try:
                xgb.fit(X, y)
            except Exception:
                xgb = None

    # Calibrate RF prob if possible
    try:
        calibrated_rf = CalibratedClassifierCV(base_estimator=rf, cv='prefit').fit(X, y)
    except Exception:
        calibrated_rf = rf

    estimators_voting = [('lr', lr), ('rf', calibrated_rf)]
    if xgb is not None:
        estimators_voting.append(('xgb', xgb))

    voting = VotingClassifier(estimators=estimators_voting, voting='soft', n_jobs=1)
    voting.fit(X, y)

    # Stacking (fallback to voting on exception)
    try:
        stack_estimators = [('lr', lr), ('rf', rf)]
        if xgb is not None:
            stack_estimators.append(('xgb', xgb))
        stack = StackingClassifier(estimators=stack_estimators, final_estimator=LogisticRegression(max_iter=800), passthrough=True, n_jobs=1)
        stack.fit(X, y)
    except Exception:
        stack = voting

    # quick metrics
    metrics = {}
    if X.shape[0] > 20:
        split = int(0.8 * len(X))
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]
        try:
            pred = voting.predict(X_te)
            metrics['voting_acc'] = float(accuracy_score(y_te, pred))
        except Exception:
            metrics['voting_acc'] = None
        try:
            pprob = voting.predict_proba(X_te)[:, 1]
            metrics['voting_brier'] = float(brier_score_loss(y_te, pprob))
        except Exception:
            metrics['voting_brier'] = None

    return {
        "voting": voting,
        "stacking": stack,
        "lr": lr,
        "rf": rf,
        "xgb": xgb,
        "metrics": metrics
    }

# ----------------- Prediction combining model + pattern + rolling stats -----------------
def predict_fusion(models, history, adjust_strength=0.45, recent_n=20):
    if models is None or len(history) < 6:
        return None, None
    X, _ = create_features(history, window=6)
    if X.shape[0] == 0:
        return None, None
    latest = X[-1:].astype(float)
    # model probs
    try:
        prob_voting = models['voting'].predict_proba(latest)[0][1]
    except Exception:
        prob_voting = 0.5
    try:
        prob_stack = models['stacking'].predict_proba(latest)[0][1]
    except Exception:
        prob_stack = prob_voting
    model_prob = float(np.mean([prob_voting, prob_stack]))
    pattern_prob = pattern_detector(history)
    # rolling recent ratio
    n = min(len(history), recent_n)
    recent_ratio = sum(1 for x in history[-n:] if x == "Tài") / n

    # combine: allow user-tunable adjust_strength (0..1)
    # stronger adjust_strength -> more weight to pattern/recent ratio
    adapt = adjust_strength
    final = (1 - adapt) * model_prob + adapt * (0.5 * pattern_prob + 0.5 * recent_ratio)

    # smoothing
    final = float(np.clip(0.9 * final + 0.1 * 0.5, 0.01, 0.99))
    preds = {
        "VotingProb": prob_voting,
        "StackingProb": prob_stack,
        "ModelAvg": model_prob,
        "PatternProb": pattern_prob,
        "RecentRatio": recent_ratio
    }
    return preds, final

# ----------------- Add / undo / import / export -----------------
def add_result(result):
    if result not in ("Tài","Xỉu"):
        return
    st.session_state.undo_stack.append((st.session_state.history.copy(), st.session_state.ai_conf.copy(), st.session_state.models))
    st.session_state.history.append(result)
    # clamp size
    if len(st.session_state.history) > 1000:
        st.session_state.history = st.session_state.history[-1000:]
        st.session_state.ai_conf = st.session_state.ai_conf[-1000:]
    # update ai_conf based on last pred
    if st.session_state.ai_last_pred is not None:
        was_correct = (st.session_state.ai_last_pred == result)
        st.session_state.ai_conf.append(1.15 if was_correct else 0.85)

def undo():
    if st.session_state.undo_stack:
        history, conf, models = st.session_state.undo_stack.pop()
        st.session_state.history = history
        st.session_state.ai_conf = conf
        st.session_state.models = models

def import_history_file(uploaded):
    if uploaded is None:
        return
    try:
        df = pd.read_csv(uploaded)
        if "Kết quả" not in df.columns:
            st.error("CSV cần cột 'Kết quả'")
            return
        vals = df["Kết quả"].tolist()
        if not all(v in ["Tài","Xỉu"] for v in vals):
            st.error("CSV chứa giá trị lạ. Chỉ 'Tài' hoặc 'Xỉu' được chấp nhận.")
            return
        st.session_state.undo_stack.append((st.session_state.history.copy(), st.session_state.ai_conf.copy(), st.session_state.models))
        st.session_state.history = vals
        st.session_state.ai_conf = [1.0]*len(vals)
        st.success("Import thành công.")
    except Exception as e:
        st.error(f"Import lỗi: {e}")

# ----------------- Plot helpers -----------------
def plot_history_bar(history):
    if not history:
        return None
    df = pd.Series(history).value_counts(normalize=True) * 100
    fig, ax = plt.subplots()
    ax.bar(df.index, df.values)
    ax.set_ylim(0,100)
    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_title("Tỷ lệ Tài / Xỉu")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

def plot_confidence_trend(conf_list):
    if not conf_list:
        return None
    fig, ax = plt.subplots(figsize=(6,2.5))
    ax.plot(conf_list[-100:], marker='o', linewidth=1)
    ax.set_title("Trend: Độ tin cậy AI (lần gần nhất → cuối)")
    ax.set_ylim(0,1)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

# ----------------- Load saved model/history if available -----------------
loaded = load_state_model(MODEL_PATH)
if loaded is not None and st.session_state.models is None:
    st.session_state.models = loaded

if os.path.exists(HISTORY_PATH) and not st.session_state.history:
    try:
        dfh = pd.read_csv(HISTORY_PATH)
        if "Kết quả" in dfh.columns:
            st.session_state.history = dfh["Kết quả"].tolist()
            st.session_state.ai_conf = [1.0]*len(st.session_state.history)
    except Exception:
        pass

# ----------------- UI Layout -----------------
sidebar = st.sidebar
sidebar.header("Controls")
with sidebar:
    adj_strength = st.slider("Adjust: Pattern vs Model (0=model only → 1=pattern heavy)", 0.0, 1.0, 0.45, 0.05)
    recent_n = st.number_input("Recent window for rolling ratio (n)", min_value=5, max_value=100, value=20, step=5)
    use_xgb = st.checkbox("Allow XGBoost if available", value=False)
    save_model_on_train = st.checkbox("Save model to disk after training", value=True)
    show_conf_plot = st.checkbox("Show confidence trend", value=True)

# Top area
col1, col2, col3 = st.columns([2,1,1])
with col1:
    st.subheader("📜 Lịch sử (gần nhất)")
    if st.session_state.history:
        st.write(" → ".join(st.session_state.history[-40:]))
    else:
        st.info("Chưa có dữ liệu. Thêm kết quả để bắt đầu.")
with col2:
    if st.button("➕ Thêm Tài"):
        add_result("Tài")
        st.rerun()
    if st.button("➖ Thêm Xỉu"):
        add_result("Xỉu")
        st.rerun()
with col3:
    if st.button("↩️ Undo"):
        undo()
        st.success("Đã undo")
        st.rerun()

st.markdown("---")

# Left: charts & metrics; Right: training/prediction controls
left, right = st.columns([2,1])

with left:
    st.subheader("📊 Charts & Stats")
    buf = plot_history_bar(st.session_state.history)
    if buf:
        st.image(buf, use_column_width=True)
    if show_conf_plot:
        buf2 = plot_confidence_trend(st.session_state.ai_conf)
        if buf2:
            st.image(buf2, use_column_width=True)
    # rolling stats
    if st.session_state.history:
        n = min(len(st.session_state.history), recent_n)
        recent = st.session_state.history[-n:]
        rate_tai = recent.count("Tài")/n
        st.write(f"📈 Tỷ lệ Tài trong {n} ván gần nhất: **{rate_tai:.1%}**")

with right:
    st.subheader("⚙️ Huấn luyện & Dự đoán")
    st.write("Nhấn **Huấn luyện** để train ensemble (chỉ khi bạn muốn).")
    if st.button("🚀 Huấn luyện (Train)"):
        with st.spinner("Đang huấn luyện..."):
            models = train_fusion(tuple(st.session_state.history), tuple(st.session_state.ai_conf), use_xgb=use_xgb, cache_key=str(len(st.session_state.history)))
            if models is None:
                st.error("Không đủ dữ liệu để huấn luyện (cần >=12 entries và cả 2 class).")
            else:
                st.session_state.models = models
                if save_model_on_train:
                    saved = save_state_model(models, MODEL_PATH)
                    if saved:
                        st.success("Huấn luyện xong và đã lưu model.")
                    else:
                        st.success("Huấn luyện xong (không lưu được model).")
                else:
                    st.success("Huấn luyện xong (model chưa được lưu).")
                # show quick metrics if available
                metrics = models.get("metrics", {})
                if metrics:
                    st.write("**Metrics (time-split):**")
                    st.json(metrics)
    st.write("---")
    # Prediction
    if st.button("🤖 Dự đoán (Predict)"):
        if st.session_state.models is None:
            st.warning("Vui lòng huấn luyện model trước.")
        else:
            preds, final = predict_fusion(st.session_state.models, st.session_state.history, adjust_strength=adj_strength, recent_n=recent_n)
            if preds:
                label = "Tài" if final >= 0.5 else "Xỉu"
                st.metric("🎯 Dự đoán chung", f"{label} ({final*100:.2f}%)")
                st.write("Chi tiết:")
                st.write(f"- Voting prob: {preds['VotingProb']:.2%}")
                st.write(f"- Stacking prob: {preds['StackingProb']:.2%}")
                st.write(f"- Model average: {preds['ModelAvg']:.2%}")
                st.write(f"- Pattern prob: {preds['PatternProb']:.2%}")
                st.write(f"- Recent ratio (last {min(len(st.session_state.history), recent_n)}): {preds['RecentRatio']:.2%}")
                # store last pred/conf for ai_conf update when result entered
                st.session_state.ai_last_pred = label
                st.session_state.ai_conf.append(final)
            else:
                st.error("Không thể dự đoán (kiểm tra lịch sử & model).")

    st.write("---")
    # Save/load history & model
    if st.button("💾 Lưu lịch sử hiện tại"):
        try:
            csvb = export_history_csv_bytes()
            with open(HISTORY_PATH, "wb") as f:
                f.write(csvb)
            st.success(f"Đã lưu lịch sử vào {HISTORY_PATH}")
        except Exception as e:
            st.error(f"Lưu lỗi: {e}")

    uploaded = st.file_uploader("📤 Import lịch sử (CSV, cột 'Kết quả')", type=["csv"])
    if uploaded:
        import_history_file(uploaded)
        st.rerun()

    st.download_button("📥 Export hiện tại (CSV)", export_history_csv_bytes(), f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")

st.markdown("---")
st.caption("© Fusion Pro 2025 — Hybrid + Improved. Chỉ dùng cho mục đích học thuật.")
