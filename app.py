import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import hashlib
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, train as xgb_train, DMatrix
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
UPDATE_INTERVAL = 10  # Tự update sau mỗi 10 ván mới

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
            k = min(20, X.shape[1])
            selector = SelectKBest(f_classif, k=k)
            Xt = selector.fit_transform(X, y)
            return Xt, y, selector
        except:
            return X, y, None
    return X, y, None

# DATA AUGMENTATION (Giảm factor để nhanh)
@st.cache_data(ttl=3600)
def augment_data(X, y, factor=3):
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
if "last_update_len" not in st.session_state:
    st.session_state.last_update_len = 0

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
st.title("🎲 AI Tài Xỉu — Tối ưu Hóa Cao Cấp")
st.markdown("Nhấn **Tài** / **Xỉu** để lưu ván. AI tự huấn luyện và cập nhật sau mỗi ván mới (không cần bấm).")

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if st.button("🎯 Tài"):
        st.session_state.history.append("Tài")
        save_history_csv(st.session_state.history)
        st.success("Lưu: Tài")
        auto_update_models()
with col2:
    if st.button("🎯 Xỉu"):
        st.session_state.history.append("Xỉu")
        save_history_csv(st.session_state.history)
        st.success("Lưu: Xỉu")
        auto_update_models()
with col3:
    if st.button("🗑️ Xóa lịch sử"):
        st.session_state.history = []
        save_history_csv([])
        st.success("Đã xóa lịch sử")

st.markdown("**Lịch sử (mới nhất cuối, hiển thị tối đa 200):**")
st.write(st.session_state.history[-200:])

if st.session_state.history:
    csv = pd.DataFrame({"result": st.session_state.history}).to_csv(index=False).encode("utf-8")
    st.download_button("📥 Tải lịch sử", data=csv, file_name="history.csv", mime="text/csv")

# CƠ SỞ HẠ TẦNG MÔ HÌNH (thêm AutoML đơn giản với grid search nhỏ, và incremental update)

class LSTMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, units=50, epochs=30, dropout=0.2):
        self.units = units
        self.epochs = epochs
        self.dropout = dropout
    def fit(self, X, y, initial_epoch=0):
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        self.model = Sequential([LSTM(self.units, input_shape=(X.shape[1], 1), return_sequences=True),
                                 Dropout(self.dropout),
                                 LSTM(self.units // 2),
                                 Dropout(self.dropout),
                                 Dense(1, activation='sigmoid')])
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(X_reshaped, y, epochs=self.epochs, initial_epoch=initial_epoch, batch_size=32, verbose=0)
        return self
    def predict_proba(self, X):
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        return self.model.predict(X_reshaped, verbose=0)
    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)
    def incremental_fit(self, X_new, y_new):
        self.model.fit(X_new.reshape((X_new.shape[0], X_new.shape[1], 1)), y_new, epochs=5, batch_size=32, verbose=0)  # Fine-tune with few epochs

@st.cache_resource(ttl=3600)
def base_model_defs():
    return {
        "xgb": XGBClassifier(n_estimators=30, max_depth=2, learning_rate=0.05, n_jobs=1, verbosity=0, random_state=SEED),
        "cat": CatBoostClassifier(iterations=40, depth=2, learning_rate=0.05, verbose=0, random_state=SEED),
        "rf": RandomForestClassifier(n_estimators=40, max_depth=4, n_jobs=1, random_state=SEED),
        "lr": LogisticRegression(max_iter=200, solver='lbfgs', random_state=SEED),
        "lstm": LSTMWrapper(units=50, epochs=30, dropout=0.2)
    }

def fit_single(key, model, X, y):
    try:
        model.fit(X, y)
        return key, model, True
    except:
        return key, None, False

def train_models_parallel(X, y):
    if X.shape[0] > MAX_TRAIN_SAMPLES:
        X = X[-MAX_TRAIN_SAMPLES:]
        y = y[-MAX_TRAIN_SAMPLES:]
    X_aug, y_aug = augment_data(X, y)
    defs = base_model_defs()
    results = Parallel(n_jobs=2)(delayed(fit_single)(k, m, X_aug, y_aug) for k, m in defs.items())
    trained = {k: m for k, m, ok in results if ok and m is not None}
    return trained

def incremental_update_models(models, X_new, y_new):
    updated = {}
    for k, m in models.items():
        try:
            if k == "xgb":
                params = m.get_params()
                dtrain = DMatrix(X_new, label=y_new)
                updated_model = xgb_train(params, dtrain, num_boost_round=10, xgb_model=m)
                updated[k] = updated_model
            elif k == "rf":
                m.set_params(n_estimators=m.n_estimators + 10, warm_start=True)
                m.fit(X_new, y_new)
                updated[k] = m
            elif k == "lstm":
                m.incremental_fit(X_new, y_new)
                updated[k] = m
            elif k == "cat":
                m.fit(X_new, y_new, init_model=m)  # CatBoost hỗ trợ init_model cho incremental
                updated[k] = m
            elif k == "lr":
                m.partial_fit(X_new, y_new, classes=[0, 1])
                updated[k] = m
        except:
            pass
    return updated or models

def auto_update_models():
    history_len = len(st.session_state.history)
    if history_len < MIN_GAMES_TO_PREDICT:
        return
    if (history_len - st.session_state.last_update_len) >= UPDATE_INTERVAL or st.session_state.models is None:
        with st.spinner("AI đang tự cập nhật mô hình..."):
            try:
                X_all, y_all, selector = create_features(st.session_state.history, WINDOW)
                if X_all.shape[0] < 10:
                    return
                if st.session_state.models is None:
                    X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=SEED)
                    trained = train_models_parallel(X_tr, y_tr)
                    st.session_state.models = trained
                    st.session_state.weights = compute_adaptive_weights(trained, X_val, y_val)
                    st.session_state.selector = selector
                else:
                    # Incremental update
                    new_start = st.session_state.last_update_len - WINDOW if st.session_state.last_update_len > WINDOW else 0
                    X_new, y_new, _ = create_features(st.session_state.history[new_start:], WINDOW)
                    if X_new.shape[0] > 0:
                        st.session_state.models = incremental_update_models(st.session_state.models, X_new, y_new)
                        st.session_state.weights = compute_adaptive_weights(st.session_state.models, X_new, y_new)
                st.session_state.last_update_len = history_len
                st.success("AI đã tự cập nhật mô hình!")
            except Exception as e:
                st.warning(f"Lỗi tự cập nhật: {str(e)}")

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

# GIAO DIỆN HUẤN LUYỆN (giữ để manual nếu cần)
st.header("Huấn luyện thủ công (nếu cần)")
colA, colB = st.columns(2)
with colA:
    if st.button("🛠️ Huấn luyện Mô Hình"):
        auto_update_models()  # Gọi tự update
with colB:
    if st.button("🔁 Gỡ models (clear)"):
        st.session_state.models = None
        st.session_state.weights = None
        st.session_state.selector = None
        st.session_state.last_update_len = 0
        try:
            for fname in os.listdir(MODELS_DIR):
                os.remove(os.path.join(MODELS_DIR, fname))
        except Exception:
            pass
        st.success("Đã gỡ models khỏi bộ nhớ.")

# TẢI MÔ HÌNH TỪ Ổ
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
                st.info("Đã tải models từ ổ lưu tạm.")
    except Exception:
        pass

# GIAO DIỆN DỰ ĐOÁN
st.header("Dự đoán ván tiếp theo (dùng models đã huấn luyện)")
if st.session_state.models is None:
    st.info("Chưa có model. Sau khi huấn luyện (ít nhất 60 ván), bạn có thể dự đoán.")
else:
    try:
        if len(st.session_state.history) < WINDOW:
            st.warning(f"Cần tối thiểu {WINDOW} ván để tạo đặc trưng (hiện {len(st.session_state.history)}).")
        else:
            X_feats, y_feats, _ = create_features(st.session_state.history, WINDOW)
            if X_feats.shape[0] < 1:
                st.error("Không thể tạo đặc trưng cho ván cuối.")
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
                st.write("Xác suất (Tài) từ từng model:", base_probs)

                weights = st.session_state.weights if st.session_state.weights is not None else {k: 1.0 / len(base_probs) for k in base_probs.keys()}
                keys = [k for k in base_probs.keys() if k in weights]
                if not keys:
                    keys = list(base_probs.keys())
                    weights = {k: 1.0 / len(keys) for k in keys}
                probs_arr = np.array([base_probs[k] for k in keys])
                w_arr = np.array([weights[k] for k in keys])
                final_prob_tai = float(np.dot(w_arr, probs_arr))
                pred_vote = "Tài" if final_prob_tai > 0.5 else "Xỉu"
                st.markdown(f"### Bỏ phiếu (Trọng số Thích nghi): **{pred_vote}** — Xác suất Tài = {final_prob_tai:.2%}")

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
                        meta_clf = LogisticRegression(max_iter=200, solver='lbfgs', random_state=SEED)
                        meta_clf.fit(meta_train, y_meta)
                        meta_input = np.array([base_probs.get(k, 0.5) for k in model_keys]).reshape(1, -1)
                        p_meta = meta_clf.predict_proba(meta_input)[0, 1]
                        pred_meta = "Tài" if p_meta > 0.5 else "Xỉu"
                        st.markdown(f"### Xếp chồng (Meta Logistic): **{pred_meta}** — Xác suất Tài = {p_meta:.2%}")
                    else:
                        st.info("Không đủ mẫu để chạy xếp chồng meta đáng tin cậy (cần >=10 mẫu sau cửa sổ).")
                except Exception:
                    st.warning("Xếp chồng meta gặp lỗi; tiếp tục với bỏ phiếu.")

                st.write("---")
                st.write("Gợi ý: Nếu Bỏ phiếu và Xếp chồng đồng ý, độ tin cậy cao hơn. Nếu không, cân nhắc bỏ qua ván.")

                if X_feats.shape[0] > 0:
                    hist_preds = np.mean([m.predict(X_feats) for m in st.session_state.models.values()], axis=0) > 0.5
                    hist_acc = accuracy_score(y_feats, hist_preds.astype(int))
                    st.write(f"Độ chính xác trên lịch sử (sau window): {hist_acc:.2%}")
                    if hist_acc >= 1.0:
                        st.success("Đạt 100% độ chính xác trên dữ liệu lịch sử! Mô hình đã học hoàn hảo pattern.")
                    else:
                        st.info("Chưa đạt 100%, có thể cần thêm dữ liệu hoặc huấn luyện lại để capture pattern tốt hơn.")
    except Exception:
        st.error("Lỗi khi dự đoán:")
        st.error(traceback.format_exc())

# GHI CHÚ CUỐI VÀ TẢI XUỐNG
st.markdown("---")
st.info("Lưu ý: Ứng dụng này lưu lịch sử và mô hình tạm thời (ephemeral). Nếu muốn lưu lâu dài, tải file history.csv và mô hình từ thư mục 'models_store' về máy.")
