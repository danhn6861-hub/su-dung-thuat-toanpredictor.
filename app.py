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
WINDOW = 7  # Giảm window để tạo features nhanh hơn
MAX_TRAIN_SAMPLES = 1000  # Giảm để train nhanh
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

# FEATURE ENGINEERING (Giảm features để nhanh)
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
        cep_prob = np.mean(w_clean[-5:]) if len(w_clean) >= 5 else 0.5
        fft_vals = np.abs(fft(w_clean))[:window // 2]
        fft_norm = MinMaxScaler().fit_transform(fft_vals.reshape(-1, 1)).flatten().tolist()
        trans_00 = sum(w_clean[j-1] == 0 and w_clean[j] == 0 for j in range(1, len(w_clean))) / max(1, sum(x == 0 for x in w_clean[:-1]))
        trans_11 = sum(w_clean[j-1] == 1 and w_clean[j] == 1 for j in range(1, len(w_clean))) / max(1, sum(x == 1 for x in w_clean[:-1]))
        lag1 = w_clean[-1] if len(w_clean) > 0 else 0.0
        lag3 = w_clean[-3] if len(w_clean) > 2 else 0.0
        new_feats = [lag1, lag3, roll_mean, roll_std, cep_prob, trans_00, trans_11] + fft_norm  # Giảm new_feats
        feats = list(w_clean) + [ent, momentum, streaks, altern, autoc, var, sk, kur, p_runs] + new_feats
        X.append(feats)
        y.append(hist_num[i])
    X = np.array(X, dtype=float) if X else np.empty((0, window + 9 + len(new_feats)), dtype=float)
    y = np.array(y, dtype=int) if y else np.empty((0,), dtype=int)
    selector = None
    if X.shape[0] > 0:
        try:
            k = min(10, X.shape[1])  # Giảm k
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
    def __init__(self, units=32, epochs=20, dropout=0.2):  # Giảm units, epochs
        self.units = units
        self.epochs = epochs
        self.dropout = dropout
    def fit(self, X, y):
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        self.model = Sequential([
            LSTM(self.units, input_shape=(X.shape[1], 1)),
            Dropout(self.dropout),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(Adam(learning_rate=0.001), 'binary_crossentropy', metrics=['accuracy'])
        self.model.fit(X_reshaped, y, epochs=self.epochs, batch_size=64, verbose=0)  # Tăng batch_size
        return self
    def predict_proba(self, X):
        return self.model.predict(X.reshape((X.shape[0], X.shape[1], 1)), verbose=0)
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

@st.cache_resource(ttl=3600)
def base_model_defs():
    return {
        "xgb": XGBClassifier(n_estimators=20, max_depth=2, learning_rate=0.1, n_jobs=1, verbosity=0, random_state=SEED),  # Giảm n_estimators
        "cat": CatBoostClassifier(iterations=20, depth=2, learning_rate=0.1, verbose=0, random_state=SEED),
        "rf": RandomForestClassifier(n_estimators=20, max_depth=3, n_jobs=1, random_state=SEED),
        "lr": LogisticRegression(max_iter=100, solver='lbfgs', random_state=SEED),
        "lstm": LSTMWrapper(units=32, epochs=20, dropout=0.2),
    }

def fit_single(key, model, X, y):
    try:
        model.fit(X, y)
        return key, model, True
    except:
        fallback = {
            "xgb": XGBClassifier(n_estimators=10, max_depth=1, n_jobs=1, verbosity=0, random_state=SEED),
            "cat": CatBoostClassifier(iterations=10, depth=1, verbose=0, random_state=SEED),
            "rf": RandomForestClassifier(n_estimators=10, max_depth=2, n_jobs=1, random_state=SEED),
            "lr": LogisticRegression(max_iter=50, solver='liblinear', random_state=SEED),
            "lstm": LSTMWrapper(units=16, epochs=10),
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
                probs_arr = np.array([base_probs[k] for k in weights])
                w_arr = np.array([weights[k] for k in weights])
                final_prob_tai = np.dot(w_arr, probs_arr)
                pred_vote = "Tài" if final_prob_tai > 0.5 else "Xỉu"
                st.markdown(f"### Bỏ phiếu: **{pred_vote}** (Tài {final_prob_tai:.2%})")
                try:
                    X_meta, y_meta, _ = create_features(st.session_state.history, WINDOW)
                    if X_meta.shape[0] >= 10:
                        model_keys = list(st.session_state.models.keys())
                        meta_train = np.zeros((X_meta.shape[0], len(model_keys)))
                        kf = KFold(n_splits=min(5, X_meta.shape[0] // 10 or 2), shuffle=True, random_state=SEED)
                        for i, key in enumerate(model_keys):
                            m = st.session_state.models[key]
                            oof = np.zeros(X_meta.shape[0])
                            for train_idx, val_idx in kf.split(X_meta):
                                try:
                                    clone = m.__class__(**m.get_params()) if hasattr(m, 'get_params') else m
                                    clone.fit(X_meta[train_idx], y_meta[train_idx])
                                    if hasattr(clone, 'predict_proba'):
                                        oof[val_idx] = clone.predict_proba(X_meta[val_idx])[:, 1]
                                    elif hasattr(clone, 'decision_function'):
                                        oof[val_idx] = 1.0 / (1.0 + np.exp(-clone.decision_function(X_meta[val_idx])))
                                    else:
                                        oof[val_idx] = clone.predict(X_meta[val_idx])
                                except:
                                    oof[val_idx] = 0.5
                            meta_train[:, i] = oof
                        meta_clf = LogisticRegression(max_iter=400, solver='lbfgs', random_state=SEED)
                        meta_clf.fit(meta_train, y_meta)
                        meta_input = np.array([base_probs.get(k, 0.5) for k in model_keys]).reshape(1, -1)
                        p_meta = meta_clf.predict_proba(meta_input)[0, 1]
                        pred_meta = "Tài" if p_meta > 0.5 else "Xỉu"
                        st.markdown(f"### Xếp chồng: **{pred_meta}** (Tài {p_meta:.2%})")
                    else:
                        st.info("Không đủ cho meta (cần ≥10).")
                except Exception as e:
                    st.warning(f"Meta lỗi: {str(e)}; dùng bỏ phiếu.")
                st.write("---")
                st.write("Gợi ý: Nếu đồng ý, tin cậy cao; không thì bỏ ván.")
                if X_feats.shape[0] > 0:
                    hist_preds = np.mean([m.predict(X_feats) for m in st.session_state.models.values()], axis=0) > 0.5
                    hist_acc = accuracy_score(y_feats, hist_preds.astype(int))
                    st.write(f"Acc lịch sử: {hist_acc:.2%}")
                    if hist_acc >= 1.0:
                        st.success("100% trên lịch sử!")
                    else:
                        st.info("Chưa 100%, cần thêm dữ liệu.")
    except Exception as e:
        st.error(f"Lỗi dự đoán: {str(e)}")

st.markdown("---")
st.info("Lưu ý: Lưu tạm thời. Tải history.csv và models_store để lưu lâu.")
