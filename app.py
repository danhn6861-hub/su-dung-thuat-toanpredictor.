import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import hashlib
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy, zscore, skew, kurtosis, norm
from scipy.fft import fft
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings("ignore")

# CONFIG (Giảm max samples để nhanh hơn)
MIN_GAMES_TO_PREDICT = 60
WINDOW = 10
MAX_TRAIN_SAMPLES = 2000  # Giảm để train nhanh
SEED = 42
HISTORY_FILE = "history.csv"
MODELS_DIR = "models_store"
os.makedirs(MODELS_DIR, exist_ok=True)

# UTILITY FUNCTIONS
def safe_float_array(lst, length=None, fill=0.0):
    try:
        arr = np.array(lst, dtype=float)
    except:
        arr = np.array([float(x) if _is_num(x) else fill for x in lst], dtype=float)
    if length is not None:
        if arr.size < length:
            arr = np.concatenate([arr, np.full(length - arr.size, fill)])
        elif arr.size > length:
            arr = arr[-length:]
    return arr

def _is_num(x):
    try:
        float(x)
        return True
    except:
        return False

def save_obj(obj, path):
    try:
        joblib.dump(obj, path)
    except:
        pass

def load_obj(path):
    try:
        if os.path.exists(path):
            return joblib.load(path)
    except:
        pass
    return None

# FEATURE ENGINEERING
def handle_outliers(window_data):
    try:
        arr = safe_float_array(window_data)
        if arr.size < 2:
            return arr.tolist()
        z_scores = np.abs(zscore(arr, ddof=0))
        median_val = float(np.median(arr))
        arr[z_scores > 3] = median_val
        return arr.tolist()
    except:
        return [float(x) if _is_num(x) else 0.0 for x in window_data]

def calculate_streaks(binary_seq):
    try:
        if len(binary_seq) == 0:
            return 0
        cur = mx = 1
        for i in range(1, len(binary_seq)):
            if binary_seq[i] == binary_seq[i-1]:
                cur += 1
                mx = max(mx, cur)
            else:
                cur = 1
        return mx
    except:
        return 0

def calculate_alternations(binary_seq):
    try:
        if len(binary_seq) < 2:
            return 0.0
        alt = sum(binary_seq[i] != binary_seq[i-1] for i in range(1, len(binary_seq)))
        return alt / (len(binary_seq) - 1)
    except:
        return 0.0

def calculate_autocorrelation(binary_seq, lag=1):
    try:
        arr = np.array(binary_seq, dtype=float)
        if arr.size < lag + 1:
            return 0.0
        m, v = arr.mean(), arr.var()
        if v == 0:
            return 0.0
        ac = ((arr[:-lag] - m) * (arr[lag:] - m)).sum() / (v * arr.size)
        return float(ac)
    except:
        return 0.0

def calculate_bias_metrics(binary_seq):
    try:
        arr = np.array(binary_seq, dtype=float)
        if arr.size < 2:
            return 0.0, 0.0, 0.0
        return float(arr.var()), float(skew(arr)), float(kurtosis(arr))
    except:
        return 0.0, 0.0, 0.0

def runs_test_p(binary_seq):
    try:
        arr = [int(round(x)) for x in binary_seq]
        n1 = sum(x == 1 for x in arr)
        n0 = len(arr) - n1
        n = len(arr)
        if n0 == 0 or n1 == 0 or n < 2:
            return 1.0
        runs = 1 + sum(arr[i] != arr[i-1] for i in range(1, n))
        expected = 1 + (2.0 * n1 * n0) / n
        var_runs = (2.0 * n1 * n0 * (2.0 * n1 * n0 - n)) / ((n**2) * (n - 1)) if n > 1 else 0
        if var_runs <= 0:
            return 1.0
        z = (runs - expected) / np.sqrt(var_runs)
        p = 2.0 * (1.0 - norm.cdf(abs(z)))
        return float(np.clip(p, 0.0, 1.0))
    except:
        return 1.0

@st.cache_data(ttl=3600, max_entries=10)
def create_features(history, window=WINDOW):
    enc = {"Tài": 1, "Xỉu": 0}
    hist_num = [enc.get(x, 0) for x in history]
    X, y = [], []
    for i in range(window, len(hist_num)):
        w = hist_num[i - window:i]
        w_clean = handle_outliers(w)
        w_clean = safe_float_array(w_clean, length=window)
        counts = np.bincount(np.round(w_clean).astype(int), minlength=2)
        probs = counts / counts.sum() if counts.sum() > 0 else np.array([0.5, 0.5])
        ent = float(entropy(probs, base=2))
        momentum = float(np.mean(np.diff(w_clean[-3:])) if len(w_clean) >= 2 else 0.0)
        streaks = calculate_streaks(w_clean)
        altern = calculate_alternations(w_clean)
        autoc = calculate_autocorrelation(w_clean)
        var, sk, kur = calculate_bias_metrics(w_clean)
        p_runs = runs_test_p(w_clean)
        df_w = pd.Series(w_clean)
        roll_mean = df_w.rolling(3).mean().iloc[-1] if len(df_w) >= 3 else 0.0
        roll_std = df_w.rolling(3).std().iloc[-1] if len(df_w) >= 3 else 0.0
        exp_min = df_w.expanding().min().iloc[-1]
        exp_max = df_w.expanding().max().iloc[-1]
        cep_prob = np.mean(w_clean[-5:]) if len(w_clean) >= 5 else 0.5
        fft_vals = np.abs(fft(w_clean))[:window // 2]
        fft_norm = MinMaxScaler().fit_transform(fft_vals.reshape(-1, 1)).flatten().tolist()
        trans_00 = sum(w_clean[j-1] == 0 and w_clean[j] == 0 for j in range(1, len(w_clean))) / max(1, sum(x == 0 for x in w_clean[:-1]))
        trans_11 = sum(w_clean[j-1] == 1 and w_clean[j] == 1 for j in range(1, len(w_clean))) / max(1, sum(x == 1 for x in w_clean[:-1]))
        trans_01 = sum(w_clean[j-1] == 0 and w_clean[j] == 1 for j in range(1, len(w_clean))) / max(1, sum(x == 0 for x in w_clean[:-1]))
        trans_10 = sum(w_clean[j-1] == 1 and w_clean[j] == 0 for j in range(1, len(w_clean))) / max(1, sum(x == 1 for x in w_clean[:-1]))
        autoc_lag2 = calculate_autocorrelation(w_clean, lag=2)
        autoc_lag3 = calculate_autocorrelation(w_clean, lag=3)
        roll_mean5 = df_w.rolling(5).mean().iloc[-1] if len(df_w) >= 5 else 0.0
        roll_std5 = df_w.rolling(5).std().iloc[-1] if len(df_w) >= 5 else 0.0
        sub_ent = entropy(np.bincount(np.round(w_clean[-5:]).astype(int), minlength=2) / 5, base=2) if len(w_clean) >= 5 else 1.0
        lag1 = w_clean[-1] if len(w_clean) > 0 else 0.0
        lag3 = w_clean[-3] if len(w_clean) > 2 else 0.0
        new_feats = [lag1, lag3, roll_mean, roll_std, roll_mean5, roll_std5, exp_min, exp_max, cep_prob, trans_00, trans_11, trans_01, trans_10, autoc_lag2, autoc_lag3, sub_ent] + fft_norm
        feats = list(w_clean) + [ent, momentum, streaks, altern, autoc, var, sk, kur, p_runs] + new_feats
        X.append(feats)
        y.append(hist_num[i])
    X = np.array(X, dtype=float) if X else np.empty((0, window + 9 + len(new_feats)), dtype=float)
    y = np.array(y, dtype=int) if y else np.empty((0,), dtype=int)
    selector = None
    if X.shape[0] > 0:
        try:
            k = min(15, X.shape[1])  # Giảm k để nhanh
            selector = SelectKBest(f_classif, k=k)
            Xt = selector.fit_transform(X, y)
            return Xt, y, selector
        except:
            return X, y, None
    return X, y, None

# DATA AUGMENTATION (Giảm factor để nhanh)
@st.cache_data(ttl=3600)
def augment_data(X, y, factor=2):  # Giảm factor
    try:
        augmented_X = list(X)
        augmented_y = list(y)
        for i in range(X.shape[0]):
            for _ in range(factor - 1):
                sample = X[i].copy()
                noise = np.random.normal(0, 0.03, sample.shape)
                sample += noise
                sample = np.clip(sample, 0, 1)
                scale = np.random.normal(1, 0.05)
                sample *= scale
                sample = np.clip(sample, 0, 1)
                if np.random.rand() > 0.5:
                    flip_start = np.random.randint(0, len(sample) - 3)
                    sample[flip_start:flip_start+3] = 1 - sample[flip_start:flip_start+3]
                augmented_X.append(sample)
                augmented_y.append(y[i])
        return np.array(augmented_X), np.array(augmented_y)
    except:
        return X, y

# SESSION STATE
if "history" not in st.session_state:
    st.session_state.history = []
if "models" not in st.session_state:
    st.session_state.models = None
if "weights" not in st.session_state:
    st.session_state.weights = None
if "selector" not in st.session_state:
    st.session_state.selector = None
if "trained_hash" not in st.session_state:
    st.session_state.trained_hash = ""

def save_history_csv(history, path=HISTORY_FILE):
    try:
        pd.DataFrame({"result": history}).to_csv(path, index=False)
    except:
        pass

def load_history_csv(path=HISTORY_FILE):
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df["result"].astype(str).tolist()
    except:
        pass
    return []

st.session_state.history = st.session_state.history or load_history_csv()

# USER INTERFACE
st.title("🎲 AI Tài Xỉu — Tối ưu Hóa")
st.markdown("Nhấn Tài/Xỉu để lưu. Huấn luyện khi bấm nút.")

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("🎯 Tài"):
        st.session_state.history.append("Tài")
        save_history_csv(st.session_state.history)
        st.success("Lưu: Tài")
with col2:
    if st.button("🎯 Xỉu"):
        st.session_state.history.append("Xỉu")
        save_history_csv(st.session_state.history)
        st.success("Lưu: Xỉu")
with col3:
    if st.button("🗑️ Xóa"):
        st.session_state.history = []
        save_history_csv([])
        st.success("Xóa lịch sử")

st.markdown("**Lịch sử (tối đa 200):**")
st.write(st.session_state.history[-200:])

if st.session_state.history:
    csv = pd.DataFrame({"result": st.session_state.history}).to_csv(index=False).encode("utf-8")
    st.download_button("📥 Tải lịch sử", csv, "history.csv", "text/csv")

# MODEL INFRASTRUCTURE (Giảm params để nhanh)
class LSTMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, units=64, epochs=30, dropout=0.2):  # Giảm units, epochs
        self.units = units
        self.epochs = epochs
        self.dropout = dropout
    def fit(self, X, y):
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        self.model = Sequential([
            LSTM(self.units, input_shape=(X.shape[1], 1), return_sequences=True),
            Dropout(self.dropout),
            LSTM(self.units // 2),
            Dropout(self.dropout),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(Adam(learning_rate=0.001), 'binary_crossentropy', metrics=['accuracy'])
        self.model.fit(X_reshaped, y, epochs=self.epochs, batch_size=64, verbose=0)  # Tăng batch_size để nhanh
        return self
    def predict_proba(self, X):
        return self.model.predict(X.reshape((X.shape[0], X.shape[1], 1)), verbose=0)
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

@st.cache_resource(ttl=3600)
def base_model_defs():
    return {
        "xgb": XGBClassifier(n_estimators=30, max_depth=2, learning_rate=0.05, n_jobs=1, verbosity=0, random_state=SEED),  # Giảm params
        "cat": CatBoostClassifier(iterations=30, depth=2, learning_rate=0.05, verbose=0, random_state=SEED),
        "rf": RandomForestClassifier(n_estimators=30, max_depth=4, n_jobs=1, random_state=SEED),
        "lr": LogisticRegression(max_iter=200, solver='lbfgs', random_state=SEED),
        "lstm": LSTMWrapper(units=64, epochs=30, dropout=0.2),
    }

def fit_single(key, model, X, y):
    try:
        model.fit(X, y)  # Bỏ tune để nhanh
        return key, model, True
    except:
        fallback = {
            "xgb": XGBClassifier(n_estimators=10, max_depth=1, n_jobs=1, verbosity=0, random_state=SEED),
            "cat": CatBoostClassifier(iterations=10, depth=1, verbose=0, random_state=SEED),
            "rf": RandomForestClassifier(n_estimators=10, max_depth=2, n_jobs=1, random_state=SEED),
            "lr": LogisticRegression(max_iter=100, solver='liblinear', random_state=SEED),
            "lstm": LSTMWrapper(units=32, epochs=10),
        }.get(key)
        if fallback:
            fallback.fit(X, y)
            return key, fallback, True
    return key, None, False

def train_models_parallel(X, y):
    if X.shape[0] > MAX_TRAIN_SAMPLES:
        X, y = X[-MAX_TRAIN_SAMPLES:], y[-MAX_TRAIN_SAMPLES:]
    X_aug, y_aug = augment_data(X, y)
    defs = base_model_defs()
    results = Parallel(n_jobs=2)(delayed(fit_single)(k, m, X_aug, y_aug) for k, m in defs.items())
    return {k: m for k, m, ok in results if ok}

def compute_adaptive_weights(models, X_val, y_val):
    try:
        scores = [accuracy_score(y_val, models[k].predict(X_val)) for k in models]
        scores = [max(s, 1e-6) for s in scores]
        total = sum(scores)
        return {k: s / total for k, s in zip(models, scores)}
    except:
        return {k: 1.0 / len(models) for k in models}

# TRAINING INTERFACE
st.header("Huấn luyện")
colA, colB = st.columns(2)
with colA:
    if st.button("🛠️ Huấn luyện"):
        if len(st.session_state.history) < MIN_GAMES_TO_PREDICT:
            st.warning(f"Cần ≥ {MIN_GAMES_TO_PREDICT} ván (hiện {len(st.session_state.history)}).")
        else:
            with st.spinner("Huấn luyện..."):
                try:
                    X_all, y_all, selector = create_features(st.session_state.history, WINDOW)
                    st.session_state.selector = selector
                    if X_all.shape[0] < 10 or len(np.unique(y_all)) < 2:
                        st.error("Dữ liệu không đủ.")
                    else:
                        X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=SEED) if X_all.shape[0] > 10 else (X_all, X_all, y_all, y_all)
                        trained = train_models_parallel(X_tr, y_tr)
                        if not trained:
                            st.error("Huấn luyện thất bại.")
                        else:
                            st.session_state.models = trained
                            st.session_state.weights = compute_adaptive_weights(trained, X_val, y_val)
                            st.success("Huấn luyện xong!")
                            for k, m in trained.items():
                                save_obj(m, os.path.join(MODELS_DIR, f"{k}.joblib"))
                            save_obj(selector, os.path.join(MODELS_DIR, "selector.joblib"))
                            save_obj(st.session_state.weights, os.path.join(MODELS_DIR, "weights.joblib"))
                            st.session_state.trained_hash = hashlib.sha256(str(st.session_state.history).encode()).hexdigest()
                            overall_acc = np.mean([accuracy_score(y_val, m.predict(X_val)) for m in trained.values()])
                            st.write(f"Acc validation trung bình: {overall_acc:.2%}")
                            if overall_acc >= 1.0:
                                st.success("Đạt 100% trên validation!")
                            else:
                                st.info("Chưa 100%, thử thêm dữ liệu.")
                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")

with colB:
    if st.button("🔁 Xóa models"):
        st.session_state.models = st.session_state.weights = st.session_state.selector = None
        try:
            for f in os.listdir(MODELS_DIR):
                os.remove(os.path.join(MODELS_DIR, f))
        except:
            pass
        st.success("Xóa models.")

# LOAD MODELS
if st.session_state.models is None:
    try:
        loaded = {k: load_obj(os.path.join(MODELS_DIR, f"{k}.joblib")) for k in ["xgb", "cat", "rf", "lr", "lstm"] if load_obj(os.path.join(MODELS_DIR, f"{k}.joblib"))}
        if loaded:
            st.session_state.models = loaded
            st.session_state.weights = load_obj(os.path.join(MODELS_DIR, "weights.joblib"))
            st.session_state.selector = load_obj(os.path.join(MODELS_DIR, "selector.joblib"))
            st.info("Tải models từ lưu trữ.")
    except:
        pass

# PREDICTION INTERFACE
st.header("Dự đoán")
if st.session_state.models is None:
    st.info("Huấn luyện trước để dự đoán.")
else:
    try:
        if len(st.session_state.history) < WINDOW:
            st.warning(f"Cần ≥ {WINDOW} ván (hiện {len(st.session_state.history)}).")
        else:
            X_feats, y_feats, _ = create_features(st.session_state.history, WINDOW)
            if X_feats.shape[0] < 1:
                st.error("Không tạo đặc trưng.")
            else:
                feat = X_feats[-1].reshape(1, -1)
                base_probs = {}
                for k, m in st.session_state.models.items():
                    try:
                        p = m.predict_proba(feat)[0][1]
                    except:
                        try:
                            df = m.decision_function(feat)[0]
                            p = 1.0 / (1.0 + np.exp(-df))
                        except:
                            p = 0.5
                    base_probs[k] = np.clip(p, 0, 1)
                st.write("Xác suất Tài từ models:", base_probs)
                weights = st.session_state.weights or {k: 1.0 / len(base_probs) for k in base_probs}
                probs_arr = np.array([base_probs[k] for k in weights
