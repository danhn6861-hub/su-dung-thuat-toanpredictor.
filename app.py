import streamlit as st
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.stats import entropy, zscore, norm, binomtest
from scipy.fft import fft
from collections import deque
import warnings
import pickle
import os
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    weights = np.arange(1, arr.size + 1, dtype=float) ** 1.5
    return np.dot(arr, weights) / weights.sum()

def autocorr(arr, lag=1):
    arr = safe_array(arr)
    if arr.size <= lag:
        return 0.0
    arr_mean = arr.mean()
    num = np.sum((arr[:-lag] - arr_mean) * (arr[lag:] - arr_mean))
    den = np.sum((arr - arr_mean) ** 2)
    return num / den if den != 0 else 0.0

def switch_rate(arr):
    if len(arr) < 2:
        return 0.0
    switches = sum(1 for i in range(1, len(arr)) if arr[i] != arr[i-1])
    return switches / (len(arr) - 1)

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

def spectral_bias(arr):
    if len(arr) < 4:
        return 0.0, 0.0
    arr = safe_array(arr)
    f = fft(arr)
    power = np.abs(f) ** 2
    power_nonzero = power[1:len(arr)//2]
    if np.sum(power_nonzero) == 0:
        return 0.0, 0.0
    dominant_freq = np.argmax(power_nonzero) + 1
    cycle_length = len(arr) / dominant_freq if dominant_freq > 0 else len(arr)
    cycle_strength = np.max(power_nonzero) / np.sum(power_nonzero)
    return cycle_length / len(arr), cycle_strength

def repetitive_score(arr):
    if len(arr) < 5:
        return 0.0
    recent = arr[-5:]
    if all(x == recent[0] for x in recent):
        return 1.0
    return 0.0

def binomial_bias_test(arr, p=0.5):
    n = len(arr)
    if n < 10:
        return 1.0, 0.0
    k = sum(1 for x in arr if int(round(x)) == 1)
    result = binomtest(k, n, p=p)
    deviation = abs(k / n - p)
    return result.pvalue, deviation

def check_data_bias(history, threshold=0.75):
    if not history:
        return False, 0.5
    tai_count = sum(1 for x in history if x == "Tài")
    ratio = tai_count / len(history)
    return ratio > threshold or ratio < (1 - threshold), ratio

# ------------------------------
# Feature engineering
# ------------------------------
def create_features(history, window=7):
    logger.info("Bắt đầu tạo đặc trưng")
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
        ac2 = autocorr(ws, lag=2)
        switch = switch_rate(ws_int)
        runs_z, runs_p = runs_test(ws)
        cycle_length, cycle_strength = spectral_bias(ws)
        binom_p, binom_dev = binomial_bias_test(ws)
        rep_score = repetitive_score(ws_int)
        features = ws + [ent, streak, ratio_tai, wma, ac1, ac2, switch, runs_z, runs_p, cycle_length, cycle_strength, binom_p, binom_dev, rep_score]
        X.append(features)
        y.append(hist_num[i])
    logger.info("Hoàn thành tạo đặc trưng")
    if not X:
        return np.empty((0, window + 14)), np.empty((0,))
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

@st.cache_resource
def expert_sgd_prob(_sgd_model, history, window=7):
    if _sgd_model is None or len(history) < window:
        return expert_freq_prob(history)
    X_all, _ = create_features(history, window)
    if X_all.size == 0:
        return expert_freq_prob(history)
    return _sgd_model.predict_proba([X_all[-1]])[0][1]

@st.cache_resource
def expert_lgbm_prob(_lgbm_model, history, window=7):
    if _lgbm_model is None or len(history) < window:
        return 0.5
    X_all, _ = create_features(history, window)
    if X_all.size == 0:
        return 0.5
    return _lgbm_model.predict_proba([X_all[-1]])[0][1]

def expert_bayesian_prob(history, alpha=1.0, beta=1.0):
    if not history:
        return 0.5
    successes = sum(1 for x in history if x == "Tài")
    n = len(history)
    return (alpha + successes) / (alpha + beta + n)

@st.cache_resource
def expert_logistic_prob(_logistic_model, history, window=7):
    if _logistic_model is None or len(history) < window:
        return 0.5
    X_all, _ = create_features(history, window)
    if X_all.size == 0:
        return 0.5
    return _logistic_model.predict_proba([X_all[-1]])[0][1]

@st.cache_resource
def expert_nb_prob(_nb_model, history, window=7):
    if _nb_model is None or len(history) < window:
        return 0.5
    X_all, _ = create_features(history, window)
    if X_all.size == 0:
        return 0.5
    return _nb_model.predict_proba([X_all[-1]])[0][1]

@st.cache_resource
def expert_catboost_prob(_catboost_model, history, window=7):
    if _catboost_model is None or len(history) < window:
        return 0.5
    X_all, _ = create_features(history, window)
    if X_all.size == 0:
        return 0.5
    return _catboost_model.predict_proba([X_all[-1]])[0][1]

# ------------------------------
# Meta-ensemble
# ----------------------
def init_meta_state():
    return {
        "names": ["markov", "freq", "wma", "sgd", "lgbm", "bayesian", "logistic", "nb", "catboost"],
        "weights": np.array([0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1, 0.2]),
        "loss_history": deque(maxlen=200),
        "eta": 0.5,
        "decay": 0.999,
        "experience_log": [],
        "historical_accuracy": deque(maxlen=50)
    }

@st.cache_resource
def init_rl_policy(_state_size=9, _n_experts=9):
    return RLPolicy(_state_size, _n_experts)

class RLPolicy:
    def __init__(self, state_size=9, n_experts=9):
        self.weights = np.array([0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1, 0.2])
        self.lr = 0.05

    def predict(self, state):
        weights = self.weights + np.random.normal(0, 0.02, len(self.weights))
        return weights / weights.sum()

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
    min_weight = 0.05
    w = np.clip(w, min_weight, None)
    if np.isnan(w).any() or w.sum() <= 0:
        return np.ones_like(weights) / len(weights)
    return w / w.sum()

def log_loss(true_label, prob):
    eps = 1e-12
    p = np.clip(prob, eps, 1 - eps)
    return - (true_label * np.log(p) + (1 - true_label) * np.log(1 - p))

def route_expert(probs, losses, risk_score, rep_score):
    if risk_score > 0.7 or rep_score > 0.8:
        return probs[np.argmin(losses)]
    inv_losses = 1 / (np.array(losses) + 1e-8)
    return np.dot(probs, inv_losses / inv_losses.sum())

# ------------------------------
# Combined predict
# ----------------------
def combined_predict(_session_state, history_tuple, window=7, label_smoothing_alpha=0.1, risk_threshold=0.55, skip_on_high_risk=True):
    logger.info("Bắt đầu dự đoán")
    s = _session_state
    history = list(history_tuple)
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
        "cycle_length": 0.0,
        "cycle_strength": 0.0,
        "rep_score": 0.0,
        "binom_p": 1.0,
        "binom_dev": 0.0,
        "dynamic_threshold": risk_threshold,
        "expert_probs": [0.5] * 9,
        "weights": [1/9] * 9,
        "eta": 0.5
    }
    if not history:
        return default_result
    is_bias, tai_ratio = check_data_bias(history)
    if is_bias:
        logger.warning(f"Dữ liệu bias: Tỷ lệ Tài = {tai_ratio:.2f}")
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
    cycle_length, cycle_strength = spectral_bias(recent)
    rep_score = repetitive_score(recent_int)
    binom_p, binom_dev = binomial_bias_test(recent)
    bias_level = (abs(runs_z) + (1 - binom_p) + cycle_strength) / 3.0
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
        expert_nb_prob(s.get("nb_model"), history, window) if not use_default else 0.5,
        expert_catboost_prob(s.get("catboost_model"), history, window) if not use_default else 0.5
    ]
    base_eta = s["meta"].get("eta", 0.5)
    t = len(s.get("meta_steps", [])) or 1
    eta = adaptive_eta(base_eta, ent_norm, streak, t)
    weights = s["meta"].get("weights", np.array([0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1, 0.2]))
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(9) / 9.0
    losses = [log_loss(1, p) for p in probs]
    final_prob_raw = route_expert(probs, losses, bias_level + ent_norm, rep_score) if bias_level > 0.3 or rep_score > 0.8 else np.dot(weights, probs)
    final_prob_raw = np.clip(final_prob_raw + adjustment, 0.0, 1.0)
    prob_smoothed = label_smoothing_alpha + (1 - 2 * label_smoothing_alpha) * final_prob_raw
    prob_smoothed = np.clip(prob_smoothed, 0.0, 1.0)
    alt = switch_rate(recent)
    ac1 = autocorr(recent, lag=1)
    risk_score = ent_norm * (1 - abs(ac1)) * (0.5 + alt) * (1 + bias_level)
    hist_acc = np.mean(s["meta"]["historical_accuracy"]) if s["meta"]["historical_accuracy"] else 0.5
    dynamic_threshold = risk_threshold * (1.0 - 0.2 * (1 - hist_acc))
    skip = skip_on_high_risk and (risk_score > 0.7 or max(prob_smoothed, 1 - prob_smoothed) < dynamic_threshold or rep_score > 0.8)
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
        "cycle_length": cycle_length,
        "cycle_strength": cycle_strength,
        "rep_score": rep_score,
        "binom_p": binom_p,
        "binom_dev": binom_dev,
        "dynamic_threshold": dynamic_threshold
    }
    logger.info("Hoàn thành dự đoán")
    return result

# ------------------------------
# Training function
# ----------------------
@st.cache_resource
def train_models(_sgd_model, _lgbm_model, _logistic_model, _nb_model, _catboost_model, history, window):
    logger.info("Bắt đầu huấn luyện mô hình")
    Xb, yb = create_features(history, window)
    if Xb.size > 0 and len(np.unique(yb)) > 1 and len(Xb) >= 40:
        batch_size = min(64, Xb.shape[0])
        new_X = Xb[-batch_size:]
        new_y = yb[-batch_size:]
        if not np.all(np.isfinite(new_X)) or not np.all(np.isfinite(new_y)):
            logger.warning("Dữ liệu huấn luyện chứa giá trị không hợp lệ")
            return _sgd_model, _lgbm_model, _logistic_model, _nb_model, _catboost_model
        if _sgd_model is None:
            _sgd_model = SGDClassifier(loss="log_loss", max_iter=500, tol=1e-3, random_state=42)
        _sgd_model.partial_fit(new_X, new_y, classes=[0, 1])
        if _lgbm_model is None:
            _lgbm_model = LGBMClassifier(n_estimators=20, reg_alpha=0.1, reg_lambda=0.1, random_state=42)
        _lgbm_model.fit(new_X, new_y)
        if _logistic_model is None:
            _logistic_model = LogisticRegression(max_iter=100, random_state=42)
        _logistic_model.fit(new_X, new_y)
        if _nb_model is None:
            _nb_model = GaussianNB()
        _nb_model.fit(new_X, new_y)
        if _catboost_model is None:
            _catboost_model = CatBoostClassifier(iterations=30, depth=3, learning_rate=0.15, l2_leaf_reg=3, random_state=42, verbose=False)
        _catboost_model.fit(new_X, new_y)
    logger.info("Hoàn thành huấn luyện mô hình")
    return _sgd_model, _lgbm_model, _logistic_model, _nb_model, _catboost_model

# ------------------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="AI Meta-Ensemble v6 — T/X Predictor", layout="wide")
st.title("🧠 AI Meta-Ensemble v6 — Real-time T/X Predictor")

# Khởi tạo session_state
if "history" not in st.session_state:
    st.session_state.history = []
if "window" not in st.session_state:
    st.session_state.window = 7
if "meta" not in st.session_state:
    st.session_state.meta = init_meta_state()
if "sgd_model" not in st.session_state:
    st.session_state.sgd_model = None
if "lgbm_model" not in st.session_state:
    st.session_state.lgbm_model = None
if "rl_policy" not in st.session_state:
    st.session_state.rl_policy = init_rl_policy()
if "metrics" not in st.session_state:
    st.session_state.metrics = {"rounds": [], "pred_prob": [], "real": [], "loss": []}
if "meta_steps" not in st.session_state:
    st.session_state.meta_steps = []
if "logistic_model" not in st.session_state:
    st.session_state.logistic_model = None
if "nb_model" not in st.session_state:
    st.session_state.nb_model = None
if "catboost_model" not in st.session_state:
    st.session_state.catboost_model = None
if "last_trained" not in st.session_state:
    st.session_state.last_trained = 0

# Sidebar
st.sidebar.header("Settings")
window = st.sidebar.number_input("Window size (features)", min_value=3, max_value=10, value=st.session_state.window)
st.session_state.window = window
label_smoothing_alpha = st.sidebar.slider("Label smoothing α", 0.0, 0.3, 0.1, 0.01)
confidence_threshold = st.sidebar.slider("Base Confidence threshold", 0.5, 0.9, 0.55, 0.01)
risk_skip_enabled = st.sidebar.checkbox("Skip predictions when risk high", value=True)
base_eta = st.sidebar.slider("Base eta (Hedge)", 0.01, 1.0, st.session_state.meta.get("eta", 0.5), 0.01)
st.session_state.meta["eta"] = base_eta

st.sidebar.header("State Management")
save_file = "app_state.pkl"
if st.sidebar.button("Save State"):
    try:
        state = {k: v for k, v in st.session_state.items() if k != 'rl_policy'}
        with open(save_file, 'wb') as f:
            pickle.dump(state, f)
        st.sidebar.success("State saved.")
    except Exception as e:
        logger.error(f"Lỗi khi lưu state: {e}")
        st.sidebar.error("Không thể lưu state.")

if st.sidebar.button("Load State"):
    if os.path.exists(save_file):
        try:
            with open(save_file, 'rb') as f:
                state = pickle.load(f)
            for k, v in state.items():
                st.session_state[k] = v
            st.sidebar.success("State loaded.")
        except Exception as e:
            logger.error(f"Lỗi khi load state: {e}")
            st.sidebar.error("Không thể load state.")
    else:
        st.sidebar.error("No saved state found.")

if st.sidebar.button("Reset all"):
    st.session_state.history = []
    st.session_state.meta = init_meta_state()
    st.session_state.sgd_model = None
    st.session_state.lgbm_model = None
    st.session_state.rl_policy = init_rl_policy()
    st.session_state.logistic_model = None
    st.session_state.nb_model = None
    st.session_state.catboost_model = None
    st.session_state.metrics = {"rounds": [], "pred_prob": [], "real": [], "loss": []}
    st.session_state.meta_steps = []
    st.session_state.last_trained = 0
    st.success("Reset xong.")

# Nhập kết quả
st.subheader("1 — Nhập kết quả (mới nhất cuối)")
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("🎯 Tài"):
        st.session_state.history.append("Tài")
        if len(st.session_state.history) > 200:
            st.session_state.history = st.session_state.history[-200:]
with c2:
    if st.button("🎯 Xỉu"):
        st.session_state.history.append("Xỉu")
        if len(st.session_state.history) > 200:
            st.session_state.history = st.session_state.history[-200:]
with c3:
    if st.button("Hoàn tác 1 ván"):
        if st.session_state.history:
            st.session_state.history.pop()

st.write("Số ván hiện có:", len(st.session_state.history))
is_bias, tai_ratio = check_data_bias(st.session_state.history)
if is_bias:
    st.warning(f"Dữ liệu bias! Tỷ lệ Tài: {tai_ratio:.2%}. Vui lòng nhập thêm dữ liệu đa dạng.")
st.write("Lịch sử (mới nhất cuối):", st.session_state.history[-20:])

# Dự đoán
st.subheader("2 — Dự đoán ván TIẾP THEO")
pred_placeholder = st.empty()
pred_info = None
if len(st.session_state.history) < window:
    pred_placeholder.warning(f"Chưa đủ {window} ván để dự đoán. Vui lòng nhập thêm dữ liệu.")
else:
    try:
        with st.spinner("Đang tính toán dự đoán..."):
            pred_info = combined_predict(st.session_state, tuple(st.session_state.history), window=window,
                                        label_smoothing_alpha=label_smoothing_alpha,
                                        risk_threshold=confidence_threshold,
                                        skip_on_high_risk=risk_skip_enabled)
    except Exception as e:
        logger.error(f"Lỗi trong combined_predict: {e}")
        pred_placeholder.error("Lỗi khi tính toán dự đoán. Vui lòng kiểm tra logs.")
        pred_info = {
            "prob": 0.5,
            "raw_prob": 0.5,
            "skip": True,
            "risk_score": 1.0,
            "entropy": 1.0,
            "streak": 0,
            "bias_level": 0.0,
            "runs_z": 0.0,
            "runs_p": 1.0,
            "cycle_length": 0.0,
            "cycle_strength": 0.0,
            "rep_score": 0.0,
            "binom_p": 1.0,
            "binom_dev": 0.0,
            "dynamic_threshold": confidence_threshold,
            "expert_probs": [0.5] * 9,
            "weights": [1/9] * 9,
            "eta": 0.5
        }
if pred_info:
    prob = pred_info["prob"]
    skip = pred_info["skip"]
    pred_label = "Tài" if prob > 0.5 else "Xỉu"
    conf = max(prob, 1 - prob)
    if skip:
        pred_placeholder.warning(f"⚠️ Không dự đoán (skip) — Risk score {pred_info['risk_score']:.3f}, Entropy {pred_info['entropy']:.3f}, Confidence {conf:.2%}, Bias level {pred_info['bias_level']:.3f}, Repetitive score {pred_info['rep_score']:.3f}, Dynamic threshold {pred_info['dynamic_threshold']:.3f}")
    else:
        status = "Đáng tin cậy ✅" if conf >= pred_info['dynamic_threshold'] else "Xác suất thấp ⚠️"
        pred_placeholder.success(f"Dự đoán: **{pred_label}** — Xác suất Tài (smoothed): {prob:.2%} — {status} (Dynamic threshold: {pred_info['dynamic_threshold']:.3f})")

# Experts & Weights
with st.expander("3 — Experts & Weights"):
    names = st.session_state.meta.get("names", ["markov", "freq", "wma", "sgd", "lgbm", "bayesian", "logistic", "nb", "catboost"])
    expert_probs = pred_info.get("expert_probs", [0.5] * len(names)) if pred_info else [0.5] * len(names)
    weights = pred_info.get("weights", [1/len(names)] * len(names)) if pred_info else [1/len(names)] * len(names)
    num_cols = min(len(names), 9)
    cols = st.columns(num_cols)
    for i, name in enumerate(names):
        if i < len(expert_probs):
            cols[i % num_cols].metric(name, f"{expert_probs[i]:.2%}" if expert_probs[i] is not None else "N/A")
        else:
            cols[i % num_cols].metric(name, "N/A")
    wcols = st.columns(num_cols)
    for i, name in enumerate(names):
        if i < len(weights):
            wcols[i % num_cols].metric(name, f"{weights[i]:.3f}" if weights[i] is not None else "N/A")
        else:
            wcols[i % num_cols].metric(name, "N/A")

# Micro-patterns & Bias
with st.expander("4 — Micro-patterns & Bias (recent window)"):
    recent = [1 if x == "Tài" else 0 for x in st.session_state.history[-window:]] if st.session_state.history else []
    if recent and pred_info:
        ac1 = autocorr(recent, lag=1)
        ac2 = autocorr(recent, lag=2)
        alt = switch_rate(recent)
        st.write(f"Entropy (base2): {pred_info.get('entropy', 1.0):.3f} | Autocorr1: {ac1:.3f} | Autocorr2: {ac2:.3f} | Switch rate: {alt:.3f}")
        st.write(f"Runs test Z: {pred_info.get('runs_z', 0.0):.3f} (p: {pred_info.get('runs_p', 1.0):.3f})")
        st.write(f"Cycle length: {pred_info.get('cycle_length', 0.0):.3f} | Cycle strength: {pred_info.get('cycle_strength', 0.0):.3f}")
        st.write(f"Repetitive score: {pred_info.get('rep_score', 0.0):.3f}")
        st.write(f"Binomial bias test p: {pred_info.get('binom_p', 1.0):.3f} | Deviation from 50%: {pred_info.get('binom_dev', 0.0):.3f}")
    else:
        st.write("Chưa đủ dữ liệu hoặc không thể tính micro-patterns. Vui lòng nhập thêm ván.")

# Huấn luyện mô hình
st.sidebar.header("Training")
if st.sidebar.button("Train Models Now"):
    if len(st.session_state.history) < 40:
        st.sidebar.warning("Cần ít nhất 40 ván để huấn luyện. Vui lòng nhập thêm dữ liệu.")
    else:
        with st.spinner("Đang huấn luyện mô hình..."):
            st.session_state.sgd_model, st.session_state.lgbm_model, st.session_state.logistic_model, st.session_state.nb_model, st.session_state.catboost_model = train_models(
                st.session_state.get("sgd_model"), st.session_state.get("lgbm_model"),
                st.session_state.get("logistic_model"), st.session_state.get("nb_model"),
                st.session_state.get("catboost_model"), st.session_state.history, window
            )
            st.session_state.last_trained = len(st.session_state.history)
            st.success("Huấn luyện hoàn tất!")

# Cập nhật trọng số và kinh nghiệm
if len(st.session_state.history) >= 2 and len(st.session_state.history) % 5 == 0 and len(st.session_state.history) > st.session_state.last_trained:
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
            expert_nb_prob(st.session_state.get("nb_model"), history_before, window),
            expert_catboost_prob(st.session_state.get("catboost_model"), history_before, window)
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
        ac2 = autocorr(recent_hist, lag=2)
        alt = switch_rate(recent_hist)
        binom_p, binom_dev = binomial_bias_test(recent_hist)
        rep_score = repetitive_score(recent_hist)
        state = [ent_val, streak, pred_info.get('risk_score', 1.0), pred_info.get('bias_level', 0.0), np.mean(losses), ac1, ac2, alt, rep_score]
        ensemble_cb = combined_predict(st.session_state, tuple(history_before), window=window,
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
            reason = "High entropy" if ent_val > 0.8 else "Streak mismatch" if streak > 3 else "Bias undetected" if binom_dev > 0.1 else "Repetitive pattern" if rep_score > 0.8 else "Pattern luân phiên thất bại"
            st.session_state.meta["experience_log"].append(f"Thua ván {idx + 1}: {reason}. Adjust weight cho LGBM/Bayesian/CatBoost.")
        else:
            st.session_state.meta["experience_log"].append(f"Thắng ván {idx + 1}: Good pattern match.")

# Thống kê hiệu suất
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

# Kinh nghiệm thắng/thua
with st.expander("6 — Kinh Nghiệm Thắng/Thua"):
    if st.session_state.meta["experience_log"]:
        for log in st.session_state.meta["experience_log"][-5:]:
            st.write(log)
    else:
        st.write("Chưa có kinh nghiệm được ghi lại.")
