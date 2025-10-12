import streamlit as st
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from scipy.stats import entropy, zscore, norm, binomtest
from collections import deque
import math
import warnings
import pickle
import os
from functools import lru_cache
warnings.filterwarnings("ignore")

# ------------------------------
# Utility helpers
# ------------------------------
def safe_array(arr):
    return np.array(arr, dtype=float)

def handle_outliers(arr):
    arr = safe_array(arr)
    if arr.size < 2:
        return arr.tolist()
    z = np.abs(zscore(arr, ddof=1, nan_policy='omit'))
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    median_val = np.nanmedian(arr)
    arr[z > 3] = median_val
    lower, upper = np.percentile(arr, [5, 95])
    arr = np.clip(arr, lower, upper)
    return arr.tolist()

def ema_smoothing(arr, alpha=0.3):
    smoothed = []
    for i, v in enumerate(arr):
        if i == 0:
            smoothed.append(float(v))
        else:
            smoothed.append(alpha * float(v) + (1 - alpha) * smoothed[-1])
    return smoothed

def weighted_moving_average(arr):
    arr = safe_array(arr)
    if arr.size == 0:
        return 0.5
    weights = np.arange(1, arr.size + 1, dtype=float)
    return np.dot(arr, weights) / weights.sum()

def autocorr(arr, lag=1):
    arr = safe_array(arr)
    if arr.size <= lag:
        return 0.0
    arr_mean = arr.mean()
    num = np.sum((arr[:-lag] - arr_mean) * (arr[lag:] - arr_mean))
    den = np.sum((arr - arr_mean) ** 2)
    return num / den if den != 0 else 0.0

def alternation_score(arr):
    if len(arr) < 2:
        return 0.0
    alt = sum(1 for i in range(1, len(arr)) if int(round(arr[i])) != int(round(arr[i-1])))
    return alt / (len(arr) - 1)

@lru_cache(maxsize=128)
def calc_bias_stats(arr_tuple):
    a = np.array(arr_tuple)
    if a.size == 0:
        return {'var': 0.0}
    return {'var': np.var(a)}

def runs_test(arr):
    arr_int = [int(round(x)) for x in arr]
    if len(arr_int) < 2:
        return 0.0, 1.0
    runs = 1 + sum(1 for i in range(1, len(arr_int)) if arr_int[i] != arr_int[i-1])
    n1 = sum(arr_int)
    n0 = len(arr_int) - n1
    n = len(arr_int)
    mu = 2 * n0 * n1 / n + 1
    sigma_sq = 2 * n0 * n1 * (2 * n0 * n1 - n) / (n**2 * (n - 1))
    sigma = np.sqrt(sigma_sq) if sigma_sq > 0 else 0
    z = (runs - mu) / sigma if sigma > 0 else 0.0
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p

def anomaly_score(history, window=7):
    if len(history) < window:
        return 0.0
    hist_num = [1 if h == "Tài" else 0 for h in history[-window:]]
    iso = IsolationForest(contamination=0.1, random_state=42)
    scores = iso.fit(np.array(hist_num).reshape(-1, 1)).decision_function(np.array(hist_num).reshape(-1, 1))
    return -np.mean(scores)

def binomial_bias_test(arr, p=0.5):
    n = len(arr)
    if n < 10:
        return 1.0, 0.0
    k = sum(1 for x in arr if int(round(x)) == 1)
    result = binomtest(k, n, p=p)
    deviation = abs(k / n - p)
    return result.pvalue, deviation

# ------------------------------
# Feature engineering
# ------------------------------
@lru_cache(maxsize=32)
def create_features_cached(history_tuple, window):
    history = list(history_tuple)
    return create_features(history, window)

def create_features(history, window=7):
    encode = {"Tài": 1, "Xỉu": 0}
    hist_num = [encode.get(h, 0) for h in history]
    hist_smooth = ema_smoothing(hist_num)
    X, y = [], []
    for i in range(window, len(hist_num)):
        ws = hist_smooth[i - window:i]
        ws = handle_outliers(ws)
        ws_int = [int(round(v)) for v in ws]
        counts = np.bincount(ws_int, minlength=2)
        probs = counts / counts.sum() if counts.sum() > 0 else np.array([0.5, 0.5])
        ent = entropy(probs, base=2) if counts.sum() > 0 else 1.0
        streak = 1
        for j in range(2, len(ws) + 1):
            if int(round(ws[-j])) == int(round(ws[-1])):
                streak += 1
            else:
                break
        ratio_tai = counts[1] / counts.sum() if counts.sum() > 0 else 0.5
        wma = weighted_moving_average(ws)
        ac1 = autocorr(ws, lag=1)
        runs_z, runs_p = runs_test(ws)
        anom_score = anomaly_score(history[:i], window)
        binom_p, binom_dev = binomial_bias_test(ws)
        features = ws + [ent, streak, ratio_tai, wma, ac1, runs_z, anom_score, binom_p, binom_dev]
        X.append(features)
        y.append(hist_num[i])
    if not X:
        return np.empty((0, window + 9)), np.empty((0,))
    return np.array(X), np.array(y)

# ------------------------------
# Experts
# ------------------------------
def expert_markov_prob(history):
    if not history:
        return 0.5
    encode = {"Tài": 1, "Xỉu": 0}
    h = [encode.get(x, 0) for x in history]
    last = h[-1]
    next_counts = [1.0, 1.0]
    for i in range(len(h) - 1):
        if h[i] == last:
            next_counts[h[i + 1]] += 1.0
    return next_counts[1] / sum(next_counts)

def expert_freq_prob(history):
    if not history:
        return 0.5
    return sum(1 for x in history if x == "Tài") / len(history)

def expert_wma_prob(history, window=7):
    arr = [1 if x == "Tài" else 0 for x in history[-window:]]
    return weighted_moving_average(arr) if arr else 0.5

def expert_sgd_prob(sgd_model, history, window=7):
    if sgd_model is None or len(history) < window:
        return expert_freq_prob(history)
    X_all, _ = create_features(history, window)
    if X_all.size == 0:
        return expert_freq_prob(history)
    return sgd_model.predict_proba([X_all[-1]])[0][1]

def expert_lgbm_prob(lgbm_model, history, window=7):
    if lgbm_model is None or len(history) < window:
        return 0.5
    X_all, _ = create_features(history, window)
    if X_all.size == 0:
        return 0.5
    return lgbm_model.predict_proba([X_all[-1]])[0][1]

def expert_bayesian_prob(history, alpha=1.0, beta=1.0):
    if not history:
        return 0.5
    successes = sum(1 for x in history if x == "Tài")
    n = len(history)
    return (alpha + successes) / (alpha + beta + n)

def expert_logistic_prob(logistic_model, history, window=7):
    if logistic_model is None or len(history) < window:
        return 0.5
    X_all, _ = create_features(history, window)
    if X_all.size == 0:
        return 0.5
    return logistic_model.predict_proba([X_all[-1]])[0][1]

def expert_nb_prob(nb_model, history, window=7):
    if nb_model is None or len(history) < window:
        return 0.5
    X_all, _ = create_features(history, window)
    if X_all.size == 0:
        return 0.5
    return nb_model.predict_proba([X_all[-1]])[0][1]

# ------------------------------
# Meta-ensemble
# ------------------------------
def init_meta_state():
    return {
        "names": ["markov", "freq", "wma", "sgd", "lgbm", "bayesian", "logistic", "nb"],
        "weights": np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1]),
        "loss_history": deque(maxlen=200),
        "eta": 0.5,
        "decay": 0.999,
        "experience_log": [],
        "historical_accuracy": deque(maxlen=50)
    }

class RLPolicy:
    def __init__(self, state_size=8, n_experts=8):
        self.weights = np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1])
        self.lr = 0.01

    def predict(self, state):
        return self.weights / self.weights.sum()

    def update(self, state, reward, weights):
        self.weights = weights * (1 + self.lr * reward)
        return self.weights / self.weights.sum()

def rl_adjust_weights(rl_policy, state, reward, weights):
    if rl_policy is None:
        return weights
    new_weights = rl_policy.update(state, reward, weights)
    return new_weights

def adaptive_eta(base_eta, entropy_val, streak, t=1):
    ent_term = np.clip(entropy_val, 0.0, 1.0)
    streak_term = np.clip(streak / 10.0, 0.0, 1.0)
    eta = base_eta * (1.0 + 0.5 * streak_term) * (1.0 - 0.7 * ent_term)
    eta = np.clip(eta, 0.01, 1.5)
    eta *= (1.0 - 1e-4 * t)
    return eta

def hedge_update(weights, losses, eta):
    losses = np.array(losses)
    exp_term = np.exp(-eta * losses)
    w = weights * exp_term
    if np.isnan(w).any() or w.sum() <= 0:
        return np.ones_like(weights) / len(weights)
    return w / w.sum()

def log_loss(true_label, prob):
    eps = 1e-12
    p = np.clip(prob, eps, 1 - eps)
    return - (true_label * np.log(p) + (1 - true_label) * np.log(1 - p))

def route_expert(probs, losses, risk_score):
    if risk_score > 0.7:
        return probs[np.argmin(losses)]
    inv_losses = 1 / (np.array(losses) + 1e-8)
    return np.dot(probs, inv_losses / inv_losses.sum())

# ------------------------------
# Combined predict
# ------------------------------
def combined_predict(session_state, history, window=7, label_smoothing_alpha=0.1, risk_threshold=0.55, skip_on_high_risk=True):
    s = session_state
    default_result = {
        "prob": 0.5,
        "raw_prob": 0.5,
        "skip": True,
        "risk_score": 1.0,
        "entropy": 1.0,
        "streak": 0,
        "bias_level": 0.0,
        "runs_z": 0.0,
        "runs_p": 1.0,
        "anom_score": 0.0,
        "binom_p": 1.0,
        "binom_dev": 0.0,
        "dynamic_threshold": risk_threshold,
        "expert_probs": [0.5] * 8,
        "weights": s["meta"].get("weights", np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1])),
        "eta": s["meta"].get("eta", 0.5)
    }
    if not history:
        return default_result
    recent = [1 if x == "Tài" else 0 for x in history[-window:]] if len(history) >= window else [1 if x == "Tài" else 0 for x in history]
    if not recent:
        return default_result
    recent = handle_outliers(recent)
    counts = np.bincount([int(round(x)) for x in recent], minlength=2)
    probs_counts = counts / counts.sum() if counts.sum() > 0 else np.array([0.5, 0.5])
    ent_val = entropy(probs_counts, base=2) if counts.sum() > 0 else 1.0
    ent_norm = min(1.0, ent_val)
    recent_int = np.round(recent).astype(int)
    streak = 1
    for j in range(2, len(recent_int) + 1):
        if recent_int[-j] == recent_int[-1]:
            streak += 1
        else:
            break
    streak_norm = streak / window
    runs_z, runs_p = runs_test(recent)
    anom_score = anomaly_score(history, window)
    binom_p, binom_dev = binomial_bias_test(recent)
    bias_level = (abs(runs_z) + anom_score + (1 - binom_p)) / 3.0
    adjustment = binom_dev if counts[1] > counts[0] else -binom_dev if binom_dev > 0.1 or bias_level > 0.5 else 0.0
    X_all, _ = create_features(history, window)
    use_default = X_all.size == 0
    probs = [
        expert_markov_prob(history),
        expert_freq_prob(history),
        expert_wma_prob(history, window),
        expert_sgd_prob(s.get("sgd_model"), history, window) if not use_default else 0.5,
        expert_lgbm_prob(s.get("lgbm_model"), history, window) if not use_default else 0.5,
        expert_bayesian_prob(history),
        expert_logistic_prob(s.get("logistic_model"), history, window) if not use_default else 0.5,
        expert_nb_prob(s.get("nb_model"), history, window) if not use_default else 0.5
    ]
    base_eta = s["meta"].get("eta", 0.5)
    t = len(s.get("meta_steps", [])) or 1
    eta = adaptive_eta(base_eta, ent_norm, streak, t)
    weights = s["meta"].get("weights", np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1]))
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(8) / 8.0
    losses = [log_loss(1, p) for p in probs]
    final_prob_raw = route_expert(probs, losses, bias_level + ent_norm) if bias_level > 0.3 else np.dot(weights, probs)
    final_prob_raw = np.clip(final_prob_raw + adjustment, 0.0, 1.0)
    prob_smoothed = label_smoothing_alpha + (1 - 2 * label_smoothing_alpha) * final_prob_raw
    prob_smoothed = np.clip(prob_smoothed, 0.0, 1.0)
    alt = alternation_score(recent)
    ac1 = autocorr(recent, lag=1)
    risk_score = ent_norm * (1 - abs(ac1)) * (0.5 + alt) * (1 + bias_level)
    hist_acc = np.mean(s["meta"]["historical_accuracy"]) if s["meta"]["historical_accuracy"] else 0.5
    dynamic_threshold = risk_threshold * (1.0 - 0.2 * (1 - hist_acc))
    skip = skip_on_high_risk and (risk_score > 0.7 or max(prob_smoothed, 1 - prob_smoothed) < dynamic_threshold)
    result = {
        "prob": prob_smoothed,
        "raw_prob": final_prob_raw,
        "expert_probs": probs,
        "weights": weights,
        "entropy": ent_val,
        "streak": streak,
        "risk_score": risk_score,
        "skip": skip,
        "eta": eta,
        "bias_level": bias_level,
        "runs_z": runs_z,
        "runs_p": runs_p,
        "anom_score": anom_score,
        "binom_p": binom_p,
        "binom_dev": binom_dev,
        "dynamic_threshold": dynamic_threshold
    }
    return result

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="AI Meta-Ensemble v5 — T/X Predictor", layout="wide")
st.title("🧠 AI Meta-Ensemble v5 — Real-time T/X Predictor")

if "history" not in st.session_state: st.session_state.history = []
if "window" not in st.session_state: st.session_state.window = 7
if "meta" not in st.session_state: st.session_state.meta = init_meta_state()
if "sgd_model" not in st.session_state: st.session_state.sgd_model = None
if "lgbm_model" not in st.session_state: st.session_state.lgbm_model = None
if "rl_policy" not in st.session_state: st.session_state.rl_policy = RLPolicy()
if "metrics" not in st.session_state: st.session_state.metrics = {"rounds": [], "pred_prob": [], "real": [], "loss": []}
if "meta_steps" not in st.session_state: st.session_state.meta_steps = []
if "logistic_model" not in st.session_state: st.session_state.logistic_model = None
if "nb_model" not in st.session_state: st.session_state.nb_model = None

st.sidebar.header("Settings")
window = st.sidebar.number_input("Window size (features)", min_value=3, max_value=20, value=st.session_state.window)
st.session_state.window = window
label_smoothing_alpha = st.sidebar.slider("Label smoothing α", 0.0, 0.3, 0.1, 0.01)
confidence_threshold = st.sidebar.slider("Base Confidence threshold", 0.5, 0.9, 0.55, 0.01)
risk_skip_enabled = st.sidebar.checkbox("Skip predictions when risk high", value=True)
base_eta = st.sidebar.slider("Base eta (Hedge)", 0.01, 1.0, st.session_state.meta.get("eta", 0.5), 0.01)
st.session_state.meta["eta"] = base_eta

st.sidebar.header("State Management")
save_file = "app_state.pkl"
if st.sidebar.button("Save State"):
    state = {k: v for k, v in st.session_state.items()}
    with open(save_file, 'wb') as f:
        pickle.dump(state, f)
    st.sidebar.success("State saved.")

if st.sidebar.button("Load State"):
    if os.path.exists(save_file):
        with open(save_file, 'rb') as f:
            state = pickle.load(f)
        for k, v in state.items():
            st.session_state[k] = v
        st.sidebar.success("State loaded.")
    else:
        st.sidebar.error("No saved state found.")

if st.sidebar.button("Reset all"):
    st.session_state.history = []
    st.session_state.meta = init_meta_state()
    st.session_state.sgd_model = None
    st.session_state.lgbm_model = None
    st.session_state.rl_policy = RLPolicy()
    st.session_state.logistic_model = None
    st.session_state.nb_model = None
    st.session_state.metrics = {"rounds": [], "pred_prob": [], "real": [], "loss": []}
    st.session_state.meta_steps = []
    st.success("Reset xong.")

st.subheader("1 — Nhập kết quả (mới nhất cuối)")
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🎯 Tài"):
        st.session_state.history.append("Tài")
        if len(st.session_state.history) > 500:
            st.session_state.history = st.session_state.history[-500:]
with c2:
    if st.button("🎯 Xỉu"):
        st.session_state.history.append("Xỉu")
        if len(st.session_state.history) > 500:
            st.session_state.history = st.session_state.history[-500:]
with c3:
    if st.button("Hoàn tác 1 ván"):
        if st.session_state.history:
            st.session_state.history.pop()

st.write("Số ván hiện có:", len(st.session_state.history))
st.write("Lịch sử (mới nhất cuối):", st.session_state.history)

st.subheader("2 — Dự đoán ván TIẾP THEO")
with st.spinner("Đang tính toán dự đoán..."):
    pred_info = combined_predict(st.session_state, st.session_state.history, window=window,
                                 label_smoothing_alpha=label_smoothing_alpha,
                                 risk_threshold=confidence_threshold,
                                 skip_on_high_risk=risk_skip_enabled)
prob = pred_info["prob"]
skip = pred_info["skip"]
pred_label = "Tài" if prob > 0.5 else "Xỉu"
conf = max(prob, 1 - prob)

if skip:
    st.warning(f"⚠️ Không dự đoán (skip) — Risk score {pred_info['risk_score']:.3f}, Entropy {pred_info['entropy']:.3f}, Confidence {conf:.2%}, Bias level {pred_info['bias_level']:.3f}, Dynamic threshold {pred_info['dynamic_threshold']:.3f}")
else:
    status = "Đáng tin cậy ✅" if conf >= pred_info['dynamic_threshold'] else "Xác suất thấp ⚠️"
    st.success(f"Dự đoán: **{pred_label}** — Xác suất Tài (smoothed): {prob:.2%} — {status} (Dynamic threshold: {pred_info['dynamic_threshold']:.3f})")

st.subheader("3 — Experts & Weights")
names = st.session_state.meta["names"]
cols = st.columns(min(len(names), 8))
for i, name in enumerate(names):
    cols[i % 8].metric(name, f"{pred_info['expert_probs'][i]:.2%}" if i < len(pred_info['expert_probs']) else "N/A")
wcols = st.columns(min(len(names), 8))
for i, name in enumerate(names):
    wcols[i % 8].metric(name, f"{pred_info['weights'][i]:.3f}" if i < len(pred_info['weights']) else "N/A")

st.subheader("4 — Micro-patterns & Bias (recent window)")
recent = [1 if x == "Tài" else 0 for x in st.session_state.history[-window:]] if st.session_state.history else []
if recent:
    ac1 = autocorr(recent, lag=1)
    alt = alternation_score(recent)
    st.write(f"Entropy (base2): {pred_info['entropy']:.3f} | Autocorr1: {ac1:.3f} | Alternation: {alt:.3f}")
    st.write(f"Runs test Z: {pred_info.get('runs_z', 0.0):.3f} (p: {pred_info['runs_p']:.3f}) | Anomaly score: {pred_info['anom_score']:.3f}")
    st.write(f"Binomial bias test p: {pred_info['binom_p']:.3f} | Deviation from 50%: {pred_info['binom_dev']:.3f}")
else:
    st.write("Chưa đủ dữ liệu để tính micro-patterns.")

if len(st.session_state.history) >= 2:
    idx = len(st.session_state.history) - 1
    history_before = st.session_state.history[:idx]
    true_label = 1 if st.session_state.history[idx] == "Tài" else 0
    if len(history_before) >= window:
        probs_before = [
            expert_markov_prob(history_before),
            expert_freq_prob(history_before),
            expert_wma_prob(history_before, window),
            expert_sgd_prob(st.session_state.get("sgd_model"), history_before, window),
            expert_lgbm_prob(st.session_state.get("lgbm_model"), history_before, window),
            expert_bayesian_prob(history_before),
            expert_logistic_prob(st.session_state.get("logistic_model"), history_before, window),
            expert_nb_prob(st.session_state.get("nb_model"), history_before, window)
        ]
        losses = [log_loss(true_label, p) for p in probs_before]
        recent_hist = [1 if x == "Tài" else 0 for x in history_before[-window:]] if history_before else []
        counts = np.bincount(recent_hist, minlength=2) if recent_hist else np.array([1, 1])
        probs_counts = counts / counts.sum() if counts.sum() > 0 else np.array([0.5, 0.5])
        ent_val = entropy(probs_counts, base=2) if counts.sum() > 0 else 1.0
        recent_int = np.round(recent_hist).astype(int)
        streak = 1
        for j in range(2, len(recent_int) + 1):
            if recent_int[-j] == recent_int[-1]:
                streak += 1
            else:
                break
        eta = adaptive_eta(st.session_state.meta["eta"], ent_val, streak, len(st.session_state.meta_steps) + 1)
        old_w = st.session_state.meta["weights"]
        new_w = hedge_update(old_w, losses, eta)
        ac1 = autocorr(recent_hist, lag=1)
        alt = alternation_score(recent_hist)
        binom_p, binom_dev = binomial_bias_test(recent_hist)
        state = [ent_val, streak, pred_info['risk_score'], pred_info['bias_level'], np.mean(losses), ac1, alt]
        ensemble_cb = combined_predict(st.session_state, history_before, window=window,
                                       label_smoothing_alpha=label_smoothing_alpha,
                                       risk_threshold=confidence_threshold,
                                       skip_on_high_risk=risk_skip_enabled)
        ensemble_prob_before = ensemble_cb.get("prob", 0.5)
        ens_loss = log_loss(true_label, ensemble_prob_before)
        reward = 1.0 if (ensemble_prob_before > 0.5) == true_label else -ens_loss
        new_w = rl_adjust_weights(st.session_state.rl_policy, state, reward, new_w)
        st.session_state.meta["weights"] = new_w
        st.session_state.meta_steps.append({"losses": losses, "eta": eta, "old_w": old_w.tolist(), "new_w": new_w.tolist(), "reward": reward})
        correct = 1 if (ensemble_prob_before > 0.5) == true_label else 0
        st.session_state.meta["historical_accuracy"].append(correct)
        if reward < 0:
            reason = "High entropy" if ent_val > 0.8 else "Streak mismatch" if streak > 3 else "Bias undetected" if binom_dev > 0.1 else "Pattern luân phiên thất bại"
            st.session_state.meta["experience_log"].append(f"Thua ván {idx + 1}: {reason}. Adjust weight cho LGBM/Bayesian.")
        else:
            st.session_state.meta["experience_log"].append(f"Thắng ván {idx + 1}: Good pattern match.")
        if len(history_before) > window and len(st.session_state.history) % 5 == 0:  # Huấn luyện mỗi 5 ván
            Xb, yb = create_features(history_before + [st.session_state.history[idx]], window)
            if Xb.size > 0:
                batch_size = min(32, Xb.shape[0])
                new_X = Xb[-batch_size:]
                new_y = yb[-batch_size:]
                if st.session_state.sgd_model is None:
                    st.session_state.sgd_model = SGDClassifier(loss="log_loss", max_iter=500, tol=1e-3, random_state=42)
                st.session_state.sgd_model.partial_fit(new_X, new_y, classes=[0, 1])
                if st.session_state.lgbm_model is None:
                    st.session_state.lgbm_model = LGBMClassifier(n_estimators=20, random_state=42)
                st.session_state.lgbm_model.fit(new_X, new_y)
                if st.session_state.logistic_model is None:
                    st.session_state.logistic_model = LogisticRegression(max_iter=100, random_state=42)
                st.session_state.logistic_model.fit(new_X, new_y)
                if st.session_state.nb_model is None:
                    st.session_state.nb_model = GaussianNB()
                st.session_state.nb_model.fit(new_X, new_y)

st.subheader("5 — Thống Kê Hiệu Suất")
if st.session_state.meta["historical_accuracy"]:
    accuracy = np.mean(st.session_state.meta["historical_accuracy"])
    st.write(f"Độ chính xác (trên {len(st.session_state.meta['historical_accuracy'])} ván gần nhất): {accuracy:.2%}")
    recent_losses = [x['reward'] for x in st.session_state.meta_steps[-10:]]
    if recent_losses:
        avg_loss = np.mean([x for x in recent_losses if x < 0])
        st.write(f"Loss trung bình (10 ván gần nhất): {avg_loss:.3f}")
else:
    st.write("Chưa có dữ liệu hiệu suất.")

st.subheader("6 — Kinh Nghiệm Thắng/Thua")
if st.session_state.meta["experience_log"]:
    for log in st.session_state.meta["experience_log"][-5:]:
        st.write(log)
else:
    st.write("Chưa có kinh nghiệm được ghi lại.")
