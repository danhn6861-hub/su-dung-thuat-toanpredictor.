import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import hashlib
import traceback
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import entropy, zscore, skew, kurtosis, norm
from scipy.fft import fft
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings("ignore")

# CONFIGURATION
MIN_GAMES_TO_PREDICT = 60
WINDOW = 7
MAX_TRAIN_SAMPLES = 3000
SEED = 42
HISTORY_FILE = "history.csv"
MODELS_DIR = "models_store"
os.makedirs(MODELS_DIR, exist_ok=True)

# UTILITY FUNCTIONS
def safe_float_array(lst, length=None, fill=0.0):
    try:
        arr = np.array(lst, dtype=float)
    except Exception:
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
    except Exception:
        return False

def save_obj(obj, path):
    try:
        joblib.dump(obj, path)
    except Exception:
        pass

def load_obj(path):
    try:
        if os.path.exists(path):
            return joblib.load(path)
    except Exception:
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
    except Exception:
        return [float(x) if _is_num(x) else 0.0 for x in window_data]

def calculate_streaks(binary_seq):
    try:
        if len(binary_seq) == 0:
            return 0
        cur = 1
        mx = 1
        for i in range(1, len(binary_seq)):
            if binary_seq[i] == binary_seq[i-1]:
                cur += 1
                if cur > mx:
                    mx = cur
            else:
                cur = 1
        return mx
    except Exception:
        return 0

def calculate_alternations(binary_seq):
    try:
        if len(binary_seq) < 2:
            return 0.0
        alt = sum(1 for i in range(1, len(binary_seq)) if binary_seq[i] != binary_seq[i-1])
        return alt / (len(binary_seq) - 1)
    except Exception:
        return 0.0

def calculate_autocorrelation(binary_seq, lag=1):
    try:
        arr = np.array(binary_seq, dtype=float)
        if arr.size < lag + 1:
            return 0.0
        m = arr.mean()
        v = arr.var()
        if v == 0:
            return 0.0
        ac = ((arr[:-lag] - m) * (arr[lag:] - m)).sum() / (v * arr.size)
        return float(ac)
    except Exception:
        return 0.0

def calculate_bias_metrics(binary_seq):
    try:
        arr = np.array(binary_seq, dtype=float)
        if arr.size < 2:
            return 0.0, 0.0, 0.0
        return float(arr.var()), float(skew(arr)), float(kurtosis(arr))
    except Exception:
        return 0.0, 0.0, 0.0

def runs_test_p(binary_seq):
    try:
        arr = [int(round(x)) for x in binary_seq]
        n1 = sum(1 for x in arr if x == 1)
        n0 = sum(1 for x in arr if x == 0)
        n = n0 + n1
        if n0 == 0 or n1 == 0 or n < 2:
            return 1.0
        runs = 1
        for i in range(1, len(arr)):
            if arr[i] != arr[i-1]:
                runs += 1
        expected = 1 + (2.0 * n1 * n0) / n
        num = 2.0 * n1 * n0 * (2.0 * n1 * n0 - n)
        den = (n**2) * (n - 1)
        if den == 0 or num <= 0:
            return 1.0
        var_runs = num / den
        z = (runs - expected) / np.sqrt(var_runs)
        p = 2.0 * (1.0 - norm.cdf(abs(z)))
        return float(np.clip(p, 0.0, 1.0))
    except Exception:
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
        lag1 = w_clean[-1] if len(w_clean) > 0 else 0.0
        lag3 = w_clean[-3] if len(w_clean) > 2 else 0.0
        cep_prob = np.mean(w_clean[-5:]) if len(w_clean) >= 5 else 0.5
        fft_vals = np.abs(fft(w_clean))[:window // 2]
        fft_norm = MinMaxScaler().fit_transform(fft_vals.reshape(-1, 1)).flatten().tolist()

        feats = list(w_clean) + [ent, momentum, streaks, altern, autoc, var, sk, kur, p_runs, lag1, lag3, roll_mean, roll_std, cep_prob] + fft_norm
        X.append(feats)
        y.append(hist_num[i])
    X = np.array(X, dtype=float) if X else np.empty((0, window + 14 + len(fft_norm)), dtype=float)
    y = np.array(y, dtype=int) if y else np.empty((0,), dtype=int)
    selector = None
    if X.shape[0] > 0:
        try:
            k = min(12, X.shape[1])
            selector = SelectKBest(f_classif, k=k)
            Xt = selector.fit_transform(X, y)
            return Xt, y, selector
        except Exception:
            return X, y, None
    return X, y, None

# DATA AUGMENTATION
@st.cache_data(ttl=3600)
def augment_data(X, y, factor=3):
    try:
        augmented_X, augmented_y = list(X), list(y)
        for i in range(X.shape[0]):
            for _ in range(factor - 1):
                sample = X[i].copy()
                noise = np.random.normal(0, 0.05, sample.shape)
                sample += noise
                sample = np.clip(sample, 0, 1)

                scale = np.random.normal(1, 0.1)
                sample *= scale
                sample = np.clip(sample, 0, 1)

                warp_factor = np.random.choice([0.5, 2.0])
                warp_start = np.random.randint(0, len(sample) // 2)
                warp_len = np.random.randint(3, len(sample) // 4)
                warped_slice = np.interp(np.linspace(0, warp_len, int(warp_len * warp_factor)), np.arange(warp_len), sample[warp_start:warp_start + warp_len])
                sample = np.concatenate([sample[:warp_start], warped_slice, sample[warp_start + warp_len:]])[:len(sample)]
                
                augmented_X.append(sample)
                augmented_y.append(y[i])
        return np.array(augmented_X), np.array(augmented_y)
    except Exception:
        return X, y

# SESSION AND HISTORY MANAGEMENT
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
if "force_train" not in st.session_state:
    st.session_state.force_train = False

def save_history_csv(history, path=HISTORY_FILE):
    try:
        pd.DataFrame({"result": history}).to_csv(path, index=False)
    except Exception:
        pass

def load_history_csv(path=HISTORY_FILE):
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df["result"].astype(str).tolist()
    except Exception:
        pass
    return []

if not st.session_state.history:
    st.session_state.history = load_history_csv()

# USER INTERFACE: INPUT BUTTONS
st.title("🎲 AI Tài Xỉu — Optimized Speed Version")
st.markdown("Press **Tài** / **Xỉu** to log a game. Training runs only when you click **Train Model**.")

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("🎯 Tài"):
        st.session_state.history.append("Tài")
        save_history_csv(st.session_state.history)
        st.success("Logged: Tài")
with col2:
    if st.button("🎯 Xỉu"):
        st.session_state.history.append("Xỉu")
        save_history_csv(st.session_state.history)
        st.success("Logged: Xỉu")
with col3:
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        save_history_csv([])
        st.success("History cleared")

st.markdown("**History (latest at end, max 200 displayed):**")
st.write(st.session_state.history[-200:])

if st.session_state.history:
    csv = pd.DataFrame({"result": st.session_state.history}).to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download History", data=csv, file_name="history.csv", mime="text/csv")

# MODEL INFRASTRUCTURE
class LSTMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, units=50, epochs=30):
        self.units = units
        self.epochs = epochs
    def fit(self, X, y):
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        self.model = Sequential([LSTM(self.units, input_shape=(X.shape[1], 1)), Dense(1, activation='sigmoid')])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(X_reshaped, y, epochs=self.epochs, verbose=0)
        return self
    def predict_proba(self, X):
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        return self.model.predict(X_reshaped, verbose=0)
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

@st.cache_resource(ttl=3600)
def base_model_defs():
    return {
        "xgb": XGBClassifier(n_estimators=30, max_depth=2, learning_rate=0.05, n_jobs=1, verbosity=0, random_state=SEED),
        "cat": CatBoostClassifier(iterations=40, depth=2, learning_rate=0.05, verbose=0, random_state=SEED),
        "rf": RandomForestClassifier(n_estimators=40, max_depth=4, n_jobs=1, random_state=SEED),
        "lr": LogisticRegression(max_iter=200, solver='lbfgs', random_state=SEED),
        "lstm": LSTMWrapper(units=50, epochs=30)
    }

def fit_single(key, model, X, y):
    try:
        model.fit(X, y)
        return key, model, True
    except Exception:
        try:
            if key == "xgb":
                m = XGBClassifier(n_estimators=20, max_depth=1, n_jobs=1, verbosity=0, random_state=SEED)
                m.fit(X, y)
                return key, m, True
            if key == "cat":
                m = CatBoostClassifier(iterations=20, depth=1, verbose=0, random_state=SEED)
                m.fit(X, y)
                return key, m, True
            if key == "rf":
                m = RandomForestClassifier(n_estimators=20, max_depth=2, n_jobs=1, random_state=SEED)
                m.fit(X, y)
                return key, m, True
            if key == "lr":
                m = LogisticRegression(max_iter=100, solver='liblinear', random_state=SEED)
                m.fit(X, y)
                return key, m, True
            if key == "lstm":
                m = LSTMWrapper(units=20, epochs=10)
                m.fit(X, y)
                return key, m, True
        except Exception:
            return key, None, False
    return key, None, False

def train_models_parallel(X, y):
    if X.shape[0] > MAX_TRAIN_SAMPLES:
        X = X[-MAX_TRAIN_SAMPLES:]
        y = y[-MAX_TRAIN_SAMPLES:]
    X_aug, y_aug = augment_data(X, y)
    defs = base_model_defs()
    results = Parallel(n_jobs=-1)(delayed(fit_single)(k, m, X_aug, y_aug) for k, m in defs.items())
    trained = {k: m for k, m, ok in results if ok and m is not None}
    return trained

def compute_adaptive_weights(models, X_val, y_val):
    weights = {}
    try:
        scores = []
        keys = list(models.keys())
        for k in keys:
            try:
                p = models[k].predict(X_val)
                acc = accuracy_score(y_val, p)
            except Exception:
                acc = 0.0
            scores.append(max(acc, 1e-6))
        arr = np.array(scores, dtype=float)
        weights = {k: float(v / arr.sum()) for k, v in zip(keys, arr)}
    except Exception:
        n = len(models)
        weights = {k: 1.0 / n for k in models}
    return weights

# TRAINING INTERFACE
st.header("Training (runs only on button press)")
colA, colB = st.columns(2)
with colA:
    if st.button("🛠️ Train Model"):
        if len(st.session_state.history) < MIN_GAMES_TO_PREDICT:
            st.warning(f"Need at least {MIN_GAMES_TO_PREDICT} games to train (currently {len(st.session_state.history)}).")
        else:
            with st.spinner("Generating features and training..."):
                try:
                    X_all, y_all, selector = create_features(st.session_state.history, WINDOW)
                    st.session_state.selector = selector
                    if X_all.shape[0] < 10 or len(np.unique(y_all)) < 2:
                        st.error("Insufficient data for training.")
                    else:
                        X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=SEED) if X_all.shape[0] > 10 else (X_all, X_all, y_all, y_all)
                        trained = train_models_parallel(X_tr, y_tr)
                        if not trained:
                            st.error("No models trained successfully.")
                        else:
                            st.session_state.models = trained
                            st.session_state.weights = compute_adaptive_weights(trained, X_val, y_val)
                            st.success("Training complete! Models saved to session with augmented data.")
                            for k, m in trained.items():
                                save_obj(m, os.path.join(MODELS_DIR, f"{k}.joblib"))
                            if selector is not None:
                                save_obj(selector, os.path.join(MODELS_DIR, "selector.joblib"))
                            save_obj(st.session_state.weights, os.path.join(MODELS_DIR, "weights.joblib"))
                            st.session_state.trained_hash = hashlib.sha256(str(st.session_state.history).encode()).hexdigest()
                            st.write("Adaptive weights:", st.session_state.weights)
                except Exception:
                    st.error("Error during training:")
                    st.error(traceback.format_exc())

with colB:
    if st.button("🔁 Clear Models"):
        st.session_state.models = None
        st.session_state.weights = None
        st.session_state.selector = None
        try:
            for fname in os.listdir(MODELS_DIR):
                os.remove(os.path.join(MODELS_DIR, fname))
        except Exception:
            pass
        st.success("Models cleared from memory.")

# LOAD MODELS FROM DISK
if st.session_state.models is None:
    try:
        loaded = {}
        for fname in os.listdir(MODELS_DIR):
            if fname.endswith(".joblib"):
                key = fname.replace(".joblib", "")
                if key in ("xgb", "cat", "rf", "lr", "lstm"):
                    loaded[key] = load_obj(os.path.join(MODELS_DIR, fname))
        if loaded:
            st.session_state.models = loaded
            w = load_obj(os.path.join(MODELS_DIR, "weights.joblib"))
            st.session_state.weights = w if w is not None else st.session_state.weights
            sel = load_obj(os.path.join(MODELS_DIR, "selector.joblib"))
            st.session_state.selector = sel if sel is not None else st.session_state.selector
            if st.session_state.models:
                st.info("Loaded models from disk.")
    except Exception:
        pass

# PREDICTION INTERFACE
st.header("Predict Next Game (using trained models)")
if st.session_state.models is None:
    st.info("No models available. Train models after logging at least 60 games.")
else:
    try:
        if len(st.session_state.history) < WINDOW:
            st.warning(f"Need at least {WINDOW} games for feature generation (currently {len(st.session_state.history)}).")
        else:
            X_feats, y_feats, _ = create_features(st.session_state.history, WINDOW)
            if X_feats.shape[0] < 1:
                st.error("Cannot generate features for the latest game.")
            else:
                feat = X_feats[-1].reshape(1, -1)
                base_probs = {}
                for k, m in st.session_state.models.items():
                    try:
                        p = m.predict_proba(feat)[0][1]
                    except Exception:
                        try:
                            df = m.decision_function(feat)
                            if np.isscalar(df):
                                df = np.array([df])
                            p = 1.0 / (1.0 + np.exp(-float(df[0])))
                        except Exception:
                            p = 0.5
                    base_probs[k] = float(np.clip(p, 0.0, 1.0))
                st.write("Probabilities (Tài) from each model:", base_probs)

                weights = st.session_state.weights if st.session_state.weights is not None else {k: 1.0 / len(base_probs) for k in base_probs.keys()}
                keys = [k for k in base_probs.keys() if k in weights]
                if not keys:
                    keys = list(base_probs.keys())
                    weights = {k: 1.0 / len(keys) for k in keys}
                probs_arr = np.array([base_probs[k] for k in keys])
                w_arr = np.array([weights[k] for k in keys])
                final_prob_tai = float(np.dot(w_arr, probs_arr))
                pred_vote = "Tài" if final_prob_tai > 0.5 else "Xỉu"
                st.markdown(f"### Voting (Adaptive Weights): **{pred_vote}** — Probability Tài = {final_prob_tai:.2%}")

                try:
                    X_meta, y_meta, _ = create_features(st.session_state.history, WINDOW)
                    if X_meta.shape[0] >= 10:
                        model_keys = list(st.session_state.models.keys())
                        meta_train = np.zeros((X_meta.shape[0], len(model_keys)))
                        kf = KFold(n_splits=min(3, max(2, X_meta.shape[0] // 10)), shuffle=True, random_state=SEED)
                        for i, key in enumerate(model_keys):
                            m = st.session_state.models[key]
                            oof = np.zeros(X_meta.shape[0])
                            for train_idx, val_idx in kf.split(X_meta):
                                try:
                                    clone = m.__class__(**{k: v for k, v in getattr(m, 'get_params', lambda: {})().items()}) if hasattr(m, 'get_params') else m
                                    clone.fit(X_meta[train_idx], y_meta[train_idx])
                                    if hasattr(clone, 'predict_proba'):
                                        oof[val_idx] = clone.predict_proba(X_meta[val_idx])[:, 1]
                                    elif hasattr(clone, 'decision_function'):
                                        df = clone.decision_function(X_meta[val_idx])
                                        oof[val_idx] = 1.0 / (1.0 + np.exp(-df))
                                    else:
                                        oof[val_idx] = clone.predict(X_meta[val_idx])
                                except Exception:
                                    oof[val_idx] = 0.5
                            meta_train[:, i] = oof
                        meta_clf = LogisticRegression(max_iter=400, solver='lbfgs', random_state=SEED)
                        meta_clf.fit(meta_train, y_meta)
                        meta_input = np.array([base_probs.get(k, 0.5) for k in model_keys]).reshape(1, -1)
                        p_meta = meta_clf.predict_proba(meta_input)[0, 1]
                        pred_meta = "Tài" if p_meta > 0.5 else "Xỉu"
                        st.markdown(f"### Stacking (Meta Logistic): **{pred_meta}** — Probability Tài = {p_meta:.2%}")
                    else:
                        st.info("Insufficient samples for reliable meta-stacking (need >=10 samples after window).")
                except Exception:
                    st.warning("Meta-stacking failed; continuing with voting.")

                st.write("---")
                st.write("Tip: If Voting and Stacking agree, confidence is higher. If they disagree, consider skipping the game.")
    except Exception:
        st.error("Error during prediction:")
        st.error(traceback.format_exc())

# FINAL NOTES AND DOWNLOAD
st.markdown("---")
st.info("Note: This app stores history and models temporarily (ephemeral). For persistent storage, download history.csv and models from the 'models_store' directory.")
```
