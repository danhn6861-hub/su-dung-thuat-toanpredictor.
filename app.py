# app.py
import streamlit as st
import numpy as np
import pandas as pd
import os
import io
import base64
import joblib
import random
from datetime import datetime
import matplotlib.pyplot as plt

# Models
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

# Try import xgboost (optional)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ----------------------
# Config & constants
# ----------------------
st.set_page_config(page_title="AI Meta-Hybrid Tài/Xỉu – Cấp Siêu Hợp Nhất", layout="wide")
MODEL_PATH = "/mnt/data/tx_hybrid_model.joblib"
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ----------------------
# Disclaimer
# ----------------------
st.sidebar.markdown("""
### ⚠️ Lưu ý quan trọng
Ứng dụng chỉ mang tính chất _thử nghiệm & nghiên cứu_. Không khuyến khích sử dụng cho **cờ bạc** hoặc mục đích gây tổn thất tài chính. Mọi kết quả đều mang tính tham khảo. Bạn tự chịu trách nhiệm với quyết định của mình.
""")

# ----------------------
# Session state init
# ----------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "ai_confidence" not in st.session_state:
    st.session_state.ai_confidence = []
if "models" not in st.session_state:
    st.session_state.models = None
if "meta_strategies" not in st.session_state:
    # initial strategy scores
    st.session_state.meta_strategies = {
        "pattern_reversal": {"score": 1.0, "desc": "Đảo chiều khi chuỗi dài"},
        "trend_follow": {"score": 1.0, "desc": "Theo xu hướng model"},
        "balanced_model": {"score": 1.0, "desc": "Cân bằng model + pattern"},
        "random_check": {"score": 1.0, "desc": "Ngẫu nhiên đối chứng"},
        "meta_hybrid": {"score": 1.0, "desc": "Tự điều chỉnh cân bằng"}
    }
if "best_strategy" not in st.session_state:
    st.session_state.best_strategy = None
if "ai_last_pred" not in st.session_state:
    st.session_state.ai_last_pred = None
if "undo_stack" not in st.session_state:
    st.session_state.undo_stack = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# ----------------------
# Utilities
# ----------------------
def softmax(arr):
    a = np.array(arr, dtype=float)
    e = np.exp(a - np.max(a))
    return e / e.sum()

def save_models_meta():
    try:
        data = {
            "models": st.session_state.models,
            "meta_strategies": st.session_state.meta_strategies,
            "ai_confidence": st.session_state.ai_confidence,
            "history": st.session_state.history
        }
        joblib.dump(data, MODEL_PATH)
    except Exception as e:
        st.warning(f"Không lưu được model: {e}")

def load_models_meta():
    if os.path.exists(MODEL_PATH):
        try:
            data = joblib.load(MODEL_PATH)
            st.session_state.models = data.get("models", None)
            st.session_state.meta_strategies = data.get("meta_strategies", st.session_state.meta_strategies)
            st.session_state.ai_confidence = data.get("ai_confidence", st.session_state.ai_confidence)
            st.session_state.history = data.get("history", st.session_state.history)
            st.success("Đã load model & trạng thái cũ (nếu có).")
        except Exception as e:
            st.warning(f"Load model thất bại: {e}")

# Try load at start (best-effort)
load_models_meta()

# ----------------------
# Feature engineering (improved)
# ----------------------
def create_features(history, window=6):
    """
    Return X, y arrays. Features include:
    - last window binary indicators
    - tai_ratio
    - change_ratio
    - current_streak
    - last3_tai_count
    """
    if len(history) < window + 1:
        return np.empty((0, window + 4)), np.empty((0,))
    X, y = [], []
    for i in range(window, len(history)):
        window_slice = history[i-window:i]
        base = [1 if x == "Tài" else 0 for x in window_slice]
        tai_ratio = sum(base) / window
        changes = sum(base[j] != base[j-1] for j in range(1, len(base)))
        change_ratio = changes / (window - 1) if window > 1 else 0.0
        # current streak length (count backwards)
        last = base[-1]
        streak = 1
        for j in range(len(base)-2, -1, -1):
            if base[j] == last:
                streak += 1
            else:
                break
        last3 = sum(base[-3:]) if len(base) >= 3 else sum(base)
        features = base + [tai_ratio, change_ratio, streak, last3]
        X.append(features)
        y.append(1 if history[i] == "Tài" else 0)
    return np.array(X, dtype=float), np.array(y, dtype=int)

# ----------------------
# Pattern detector improved
# ----------------------
def pattern_detector(history, lookback=8):
    if len(history) < 3:
        return 0.5
    recent = history[-lookback:] if len(history) >= lookback else history[:]
    tai_recent = sum(1 for x in recent if x == "Tài")
    base_prob = tai_recent / len(recent)
    # detect streaks
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
        adjusted = 1.0 - base_prob  # mean reversion
    else:
        adjusted = 0.5
    return float(np.clip(adjusted, 0.1, 0.9))

# ----------------------
# Train hybrid models (balanced for generalization)
# ----------------------
@st.cache_resource
def train_models_hybrid(history_tuple, ai_conf_tuple, use_xgb=True, cache_key=None):
    history = list(history_tuple)
    ai_conf = list(ai_conf_tuple)
    if len(history) < 8:
        # require at least 8+ to do anything
        return None
    X, y = create_features(history, window=6)
    if X.shape[0] < 6:
        return None
    # ensure target variety
    if len(np.unique(y)) < 2:
        return None

    # sample weights: recent samples slightly heavier, clip to avoid zeros
    recent_weight = np.linspace(0.4, 1.0, len(y))
    if len(ai_conf) >= len(y):
        combined = recent_weight * np.array(ai_conf[:len(y)])
    else:
        combined = recent_weight
    combined = np.clip(combined, 0.2, 2.0)

    # TimeSeriesSplit for CV
    n_splits = min(4, max(2, X.shape[0] // 6))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Base learners
    lr = LogisticRegressionCV(cv=tscv, max_iter=1000, class_weight='balanced', scoring='accuracy', random_state=RANDOM_SEED)
    rf = RandomForestClassifier(n_estimators=60, max_depth=6, min_samples_split=8, class_weight='balanced', random_state=RANDOM_SEED)

    learners = [('lr', lr), ('rf', rf)]

    if use_xgb and HAS_XGB:
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50, random_state=RANDOM_SEED)
        learners.append(('xgb', xgb))

    # Fit base learners with sample weights where possible
    # LogisticRegressionCV accepts sample_weight in fit
    try:
        lr.fit(X, y, sample_weight=combined)
    except TypeError:
        lr.fit(X, y)
    rf.fit(X, y, sample_weight=combined)
    if use_xgb and HAS_XGB:
        try:
            xgb.fit(X, y, sample_weight=combined)
        except Exception:
            xgb.fit(X, y)

    # Calibrate RF/XGB probabilities if small data
    calibrated_rf = CalibratedClassifierCV(base_estimator=rf, cv='prefit')
    try:
        calibrated_rf = calibrated_rf.fit(X, y)
    except Exception:
        calibrated_rf = rf  # fallback

    estimators_for_voting = [('lr', lr), ('rf', calibrated_rf)]
    if use_xgb and HAS_XGB:
        # if xgb exists, calibrate or use directly
        estimators_for_voting.append(('xgb', xgb))

    # Voting soft ensemble
    voting = VotingClassifier(estimators=estimators_for_voting, voting='soft')
    voting.fit(X, y)

    # Stacking as an alternative (meta-model)
    try:
        stacking_estimators = [('lr', lr), ('rf', rf)]
        if use_xgb and HAS_XGB:
            stacking_estimators.append(('xgb', xgb))
        stack = StackingClassifier(estimators=stacking_estimators, final_estimator=LogisticRegression(max_iter=500), passthrough=True)
        stack.fit(X, y)
    except Exception:
        stack = voting  # fallback

    # Evaluate quick metrics (train/test split time-series)
    metrics = {}
    if X.shape[0] > 12:
        split = int(0.8 * X.shape[0])
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

    # Return dict
    return {
        "voting": voting,
        "stacking": stack,
        "lr": lr,
        "rf": rf,
        "xgb": xgb if (use_xgb and HAS_XGB) else None,
        "metrics": metrics
    }

# ----------------------
# Evolutionary meta-strategy routines
# ----------------------
def update_strategy_performance(strategy_key, was_correct, reward_win=1.08, reward_loss=0.92):
    if strategy_key not in st.session_state.meta_strategies:
        return
    old = st.session_state.meta_strategies[strategy_key]["score"]
    new = old * (reward_win if was_correct else reward_loss)
    # clip to avoid extremes
    st.session_state.meta_strategies[strategy_key]["score"] = float(np.clip(new, 0.01, 1000.0))

def choose_strategy_softmax():
    keys = list(st.session_state.meta_strategies.keys())
    scores = [st.session_state.meta_strategies[k]["score"] for k in keys]
    probs = softmax(scores)
    return np.random.choice(keys, p=probs)

def evolve_strategies_return_best():
    scores = {k: v["score"] for k, v in st.session_state.meta_strategies.items()}
    best = max(scores, key=scores.get)
    st.session_state.best_strategy = best
    return best, scores

# ----------------------
# Combined predict (meta-layer)
# ----------------------
def predict_next_hybrid(models_dict, history):
    # Requirements
    if models_dict is None or len(history) < 6:
        return None, None
    X, _ = create_features(history, window=6)
    if X.shape[0] == 0:
        return None, None
    latest = X[-1:].astype(float)
    # Get model probs
    try:
        prob_voting = models_dict['voting'].predict_proba(latest)[0][1]
    except Exception:
        prob_voting = 0.5
    try:
        prob_stack = models_dict['stacking'].predict_proba(latest)[0][1]
    except Exception:
        prob_stack = prob_voting
    # average model probability (stronger ensemble)
    model_prob = float(np.mean([prob_voting, prob_stack]))
    pattern_prob = pattern_detector(history)
    recent_ratio = sum(1 for x in history[-5:] if x == "Tài") / 5

    # Choose strategy (softmax of meta scores) but also allow user override
    strategy = st.session_state.best_strategy or choose_strategy_softmax()

    if strategy == "pattern_reversal":
        final = 0.75 * pattern_prob + 0.25 * (1 - model_prob)
    elif strategy == "trend_follow":
        final = 0.75 * model_prob + 0.25 * recent_ratio
    elif strategy == "balanced_model":
        final = 0.5 * model_prob + 0.5 * pattern_prob
    elif strategy == "random_check":
        final = random.uniform(0.35, 0.65)
    elif strategy == "meta_hybrid":
        adapt = 0.6 if abs(recent_ratio - 0.5) > 0.25 else 0.45
        final = adapt * model_prob + (1 - adapt) * pattern_prob
    else:
        final = model_prob

    # Keep some smoothing to avoid flip-flop
    final = float(np.clip(0.9 * final + 0.1 * 0.5, 0.01, 0.99))

    preds = {
        "VotingProb": prob_voting,
        "StackingProb": prob_stack,
        "ModelAvg": model_prob,
        "PatternProb": pattern_prob,
        "RecentRatio": recent_ratio,
        "Strategy": strategy
    }
    return preds, final

# ----------------------
# Add result / undo / import / export
# ----------------------
def add_result(result):
    if st.session_state.is_processing:
        return
    st.session_state.is_processing = True
    try:
        if result not in ["Tài", "Xỉu"]:
            return
        st.session_state.undo_stack.append((st.session_state.history.copy(), st.session_state.ai_confidence.copy(), st.session_state.meta_strategies.copy()))
        st.session_state.history.append(result)
        # keep sizes bounded
        if len(st.session_state.history) > 500:
            st.session_state.history = st.session_state.history[-500:]
            st.session_state.ai_confidence = st.session_state.ai_confidence[-500:]
        # update ai_confidence based on last prediction correctness
        if st.session_state.ai_last_pred is not None:
            was_correct = (st.session_state.ai_last_pred == result)
            # append confidence reward/penalty
            st.session_state.ai_confidence.append(1.1 if was_correct else 0.9)
        # update strategy performance for the strategy used last time
        last_strategy = st.session_state.best_strategy or None
        if last_strategy:
            was_correct = (st.session_state.ai_last_pred == result)
            update_strategy_performance(last_strategy, was_correct)
    finally:
        st.session_state.is_processing = False

def undo_last():
    if st.session_state.is_processing:
        return
    st.session_state.is_processing = True
    try:
        if st.session_state.undo_stack:
            history, conf, strategies = st.session_state.undo_stack.pop()
            st.session_state.history = history
            st.session_state.ai_confidence = conf
            st.session_state.meta_strategies = strategies
            st.success("Đã undo.")
        else:
            st.info("Không có thao tác để undo.")
    finally:
        st.session_state.is_processing = False

def export_history_csv():
    df = pd.DataFrame({"Kết quả": st.session_state.history})
    return df.to_csv(index=False).encode('utf-8')

def import_history_file(uploaded):
    if uploaded is None:
        return
    try:
        df = pd.read_csv(uploaded)
        if "Kết quả" not in df.columns:
            st.error("CSV phải có cột 'Kết quả'.")
            return
        vals = df["Kết quả"].tolist()
        if not all(v in ["Tài", "Xỉu"] for v in vals):
            st.error("CSV chứa giá trị ngoài 'Tài'/'Xỉu'.")
            return
        st.session_state.undo_stack.append((st.session_state.history.copy(), st.session_state.ai_confidence.copy(), st.session_state.meta_strategies.copy()))
        st.session_state.history = vals
        st.session_state.ai_confidence = [1.0] * len(vals)
        st.success("Import thành công.")
    except Exception as e:
        st.error(f"Lỗi khi import: {e}")

# ----------------------
# Plotting helper
# ----------------------
def plot_history_pie(history):
    if not history:
        return None
    df = pd.Series(history).value_counts(normalize=True) * 100
    fig, ax = plt.subplots()
    labels = df.index.tolist()
    vals = df.values
    ax.bar(labels, vals)
    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Tỷ lệ Tài / Xỉu trong lịch sử")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf

# ----------------------
# UI layout
# ----------------------
st.title("🤖 AI Meta-Hybrid Tài/Xỉu – Mạnh + Ổn định + Nhanh")

col1, col2, col3 = st.columns([2,1,1])
with col1:
    st.markdown("#### 📜 Lịch sử gần đây")
    if st.session_state.history:
        st.write(" → ".join(st.session_state.history[-40:]))
    else:
        st.info("Chưa có dữ liệu. Nhập kết quả để bắt đầu.")

with col2:
    if st.button("🧹 Xóa toàn bộ (confirm)"):
        confirm = st.checkbox("Xác nhận xóa lịch sử (và model đã train)?", key="confirm_clear_all")
        if confirm:
            st.session_state.history = []
            st.session_state.ai_confidence = []
            st.session_state.models = None
            st.session_state.undo_stack = []
            st.session_state.meta_strategies = {k: {"score":1.0, "desc":v["desc"]} for k,v in st.session_state.meta_strategies.items()}
            st.success("Đã xóa mọi thứ!")
            # attempt to remove saved model
            try:
                if os.path.exists(MODEL_PATH):
                    os.remove(MODEL_PATH)
            except Exception:
                pass
            st.experimental_rerun()
with col3:
    if st.button("↩️ Undo"):
        undo_last()
        st.experimental_rerun()

st.divider()

col_t, col_x = st.columns(2)
with col_t:
    if st.button("Nhập Tài"):
        add_result("Tài")
        st.experimental_rerun()
with col_x:
    if st.button("Nhập Xỉu"):
        add_result("Xỉu")
        st.experimental_rerun()

st.divider()

# Train controls
with st.expander("⚙️ Huấn luyện & cấu hình (mở ra nếu muốn)"):
    use_xgb_checkbox = st.checkbox("Cho phép XGBoost nếu có", value=HAS_XGB)
    auto_train_btn = st.button("Huấn luyện lại (hybrid)")
    if auto_train_btn:
        with st.spinner("Đang huấn luyện..."):
            st.session_state.models = train_models_hybrid(tuple(st.session_state.history), tuple(st.session_state.ai_confidence), use_xgb=use_xgb_checkbox)
            if st.session_state.models:
                st.success("Huấn luyện xong.")
                save_models_meta()
            else:
                st.warning("Không đủ dữ liệu để huấn luyện hoặc dữ liệu không đa dạng (cần cả Tài & Xỉu).")

# Prediction & meta-evolution
if len(st.session_state.history) >= 6 and st.session_state.models is not None:
    best, scores = evolve_strategies_return_best()
    preds, final = predict_next_hybrid(st.session_state.models, st.session_state.history)
    if preds:
        st.session_state.ai_last_pred = "Tài" if final >= 0.5 else "Xỉu"
        st.subheader(f"🎯 Dự đoán: **{st.session_state.ai_last_pred}** ({final:.2%})")
        st.caption(f"Chiến lược hiện tại: **{best}** – điểm {scores[best]:.2f}")
        st.write("Chi tiết:")
        st.write(f"- Voting prob: {preds['VotingProb']:.2%}")
        st.write(f"- Stacking prob: {preds['StackingProb']:.2%}")
        st.write(f"- Model average: {preds['ModelAvg']:.2%}")
        st.write(f"- Pattern prob: {preds['PatternProb']:.2%}")
        st.write(f"- Recent 5 ván Tài: {preds['RecentRatio']:.2%}")
        st.progress(final)
        # Soft update: choose strategy randomly weighted by scores for next round
        chosen = choose_strategy_softmax()
        st.write(f"- Strategy chosen (softmax sample): **{chosen}**")
else:
    st.info("Cần ít nhất 6 ván và huấn luyện model để dự đoán.")

st.divider()

# Diagnostics & metrics
with st.expander("📈 Thông tin chẩn đoán & metrics"):
    if st.session_state.models:
        metrics = st.session_state.models.get("metrics", {})
        st.write("Quick metrics (train/test time-split):")
        st.write(metrics)
    st.write("Meta-strategy scores:")
    s_df = pd.DataFrame(
        [(k, v["score"], v["desc"]) for k, v in st.session_state.meta_strategies.items()],
        columns=["Strategy", "Score", "Description"]
    ).sort_values("Score", ascending=False)
    st.dataframe(s_df)
    st.write("AI confidence history (last 50):")
    st.write(st.session_state.ai_confidence[-50:])

st.divider()

# Chart
if st.session_state.history:
    buf = plot_history_pie(st.session_state.history)
    if buf:
        st.image(buf, use_column_width=True)

st.divider()

# Import/export
col_e, col_i = st.columns(2)
with col_e:
    csv = export_history_csv()
    st.download_button("📥 Export lịch sử CSV", csv, f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
with col_i:
    up = st.file_uploader("📤 Import lịch sử CSV", type="csv")
    if up:
        import_history_file(up)
        st.experimental_rerun()

# Save model button
if st.button("💾 Lưu model & trạng thái hiện tại"):
    save_models_meta()
    st.success("Đã lưu (nếu có quyền ghi).")

st.sidebar.markdown("""
### 🧠 Thiết kế chung - Meta Hybrid
- **Mạnh nhất:** Kết hợp nhiều base learners (LogisticCV, RF, XGBoost) + stacking.
- **Ổn định nhất:** Regularization, TimeSeriesSplit, calibrated probabilities, kiểm tra overfitting.
- **Nhanh nhẹ:** Fallback khi dữ liệu nhỏ, caching model, cấu hình giảm nặng khi cần.
""")
