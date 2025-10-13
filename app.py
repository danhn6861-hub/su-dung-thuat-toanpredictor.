# app.py
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
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy, zscore, skew, kurtosis, norm
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# CONFIG
# -----------------------
MIN_GAMES_TO_PREDICT = 60
WINDOW = 7
MAX_TRAIN_SAMPLES = 3000
SEED = 42
HISTORY_FILE = "history.csv"
MODELS_DIR = "models_store"
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------
# UTILITIES
# -----------------------
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
        float(x); return True
    except Exception: return False

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

# -----------------------
# FEATURE ENGINEERING (robust)
# -----------------------
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
        cur = 1; mx = 1
        for i in range(1, len(binary_seq)):
            if binary_seq[i] == binary_seq[i-1]:
                cur += 1
                if cur > mx: mx = cur
            else:
                cur = 1
        return mx
    except Exception:
        return 0

def calculate_alternations(binary_seq):
    try:
        if len(binary_seq) < 2: return 0.0
        alt = sum(1 for i in range(1, len(binary_seq)) if binary_seq[i] != binary_seq[i-1])
        return alt / (len(binary_seq) - 1)
    except Exception:
        return 0.0

def calculate_autocorrelation(binary_seq, lag=1):
    try:
        arr = np.array(binary_seq, dtype=float)
        if arr.size < lag + 1: return 0.0
        m = arr.mean(); v = arr.var()
        if v == 0: return 0.0
        ac = ((arr[:-lag] - m) * (arr[lag:] - m)).sum() / (v * arr.size)
        return float(ac)
    except Exception:
        return 0.0

def calculate_bias_metrics(binary_seq):
    try:
        arr = np.array(binary_seq, dtype=float)
        if arr.size < 2: return 0.0, 0.0, 0.0
        return float(arr.var()), float(skew(arr)), float(kurtosis(arr))
    except Exception:
        return 0.0, 0.0, 0.0

def runs_test_p(binary_seq):
    try:
        arr = [int(round(x)) for x in binary_seq]
        n1 = sum(1 for x in arr if x == 1); n0 = sum(1 for x in arr if x == 0); n = n0 + n1
        if n0 == 0 or n1 == 0 or n < 2: return 1.0
        runs = 1
        for i in range(1, len(arr)):
            if arr[i] != arr[i-1]: runs += 1
        expected = 1 + (2.0 * n1 * n0) / n
        num = 2.0 * n1 * n0 * (2.0 * n1 * n0 - n)
        den = (n**2) * (n - 1)
        if den == 0 or num <= 0: return 1.0
        var_runs = num / den
        z = (runs - expected) / np.sqrt(var_runs)
        p = 2.0 * (1.0 - norm.cdf(abs(z)))
        return float(np.clip(p, 0.0, 1.0))
    except Exception:
        return 1.0

@st.cache_data
def create_features(history, window=WINDOW):
    enc = {"Tài":1, "Xỉu":0}
    hist_num = [enc.get(x, 0) for x in history]
    X, y = [], []
    for i in range(window, len(hist_num)):
        w = hist_num[i-window:i]
        w_clean = handle_outliers(w)
        w_clean = safe_float_array(w_clean, length=window)
        counts = np.bincount(np.round(w_clean).astype(int), minlength=2)
        probs = counts / counts.sum() if counts.sum()>0 else np.array([0.5,0.5])
        ent = float(entropy(probs, base=2))
        momentum = float(np.mean(np.diff(w_clean[-3:])) if len(w_clean)>=2 else 0.0)
        streaks = calculate_streaks(w_clean)
        altern = calculate_alternations(w_clean)
        autoc = calculate_autocorrelation(w_clean)
        var, sk, kur = calculate_bias_metrics(w_clean)
        p_runs = runs_test_p(w_clean)
        feats = list(w_clean) + [ent, momentum, streaks, altern, autoc, var, sk, kur, p_runs]
        X.append(feats); y.append(hist_num[i])
    X = np.array(X, dtype=float) if X else np.empty((0, window+9), dtype=float)
    y = np.array(y, dtype=int) if y else np.empty((0,), dtype=int)
    selector = None
    if X.shape[0] > 0:
        try:
            k = min(10, X.shape[1])
            selector = SelectKBest(f_classif, k=k)
            Xt = selector.fit_transform(X, y)
            return Xt, y, selector
        except Exception:
            return X, y, None
    return X, y, None

# -----------------------
# SESSION & HISTORY
# -----------------------
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

# recover
if not st.session_state.history:
    st.session_state.history = load_history_csv()

# -----------------------
# UI: input buttons
# -----------------------
st.title("🎲 AI Tài Xỉu — Train on Demand")
st.markdown("Nhấn **Tài** / **Xỉu** để lưu ván. Huấn luyện chỉ chạy khi bạn ấn **Huấn luyện Mô Hình**.")

col1, col2, col3 = st.columns([1,1,2])
with col1:
    if st.button("🎯 Tài"):
        st.session_state.history.append("Tài"); save_history_csv(st.session_state.history)
        st.success("Lưu: Tài")
with col2:
    if st.button("🎯 Xỉu"):
        st.session_state.history.append("Xỉu"); save_history_csv(st.session_state.history)
        st.success("Lưu: Xỉu")
with col3:
    if st.button("🗑️ Xóa lịch sử"):
        st.session_state.history = []; save_history_csv([]); st.success("Đã xóa lịch sử")

st.markdown("**Lịch sử (mới nhất cuối, hiển thị tối đa 200):**")
st.write(st.session_state.history[-200:])

if st.session_state.history:
    csv = pd.DataFrame({"result": st.session_state.history}).to_csv(index=False).encode("utf-8")
    st.download_button("📥 Tải lịch sử", data=csv, file_name="history.csv", mime="text/csv")

# -----------------------
# MODEL INFRASTRUCTURE
# -----------------------
def base_model_defs():
    # conservative params for speed and stability
    return {
        "xgb": XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, n_jobs=1, verbosity=0, random_state=SEED),
        "cat": CatBoostClassifier(iterations=60, depth=3, learning_rate=0.1, verbose=0, random_state=SEED),
        "rf": RandomForestClassifier(n_estimators=60, max_depth=5, n_jobs=1, random_state=SEED),
        "lr": LogisticRegression(max_iter=400, solver='lbfgs', random_state=SEED)
    }

def fit_single(key, model, X, y):
    try:
        model.fit(X, y)
        return key, model, True
    except Exception:
        # fallback light params
        try:
            if key == "xgb":
                m = XGBClassifier(n_estimators=20, max_depth=2, n_jobs=1, verbosity=0, random_state=SEED); m.fit(X, y); return key, m, True
            if key == "cat":
                m = CatBoostClassifier(iterations=30, depth=2, verbose=0, random_state=SEED); m.fit(X, y); return key, m, True
            if key == "rf":
                m = RandomForestClassifier(n_estimators=20, max_depth=3, n_jobs=1, random_state=SEED); m.fit(X, y); return key, m, True
            if key == "lr":
                m = LogisticRegression(max_iter=200, solver='liblinear', random_state=SEED); m.fit(X, y); return key, m, True
        except Exception:
            return key, None, False

def train_models_parallel(X, y):
    # cap
    if X.shape[0] > MAX_TRAIN_SAMPLES:
        X = X[-MAX_TRAIN_SAMPLES:]; y = y[-MAX_TRAIN_SAMPLES:]
    defs = base_model_defs()
    results = Parallel(n_jobs=2)(delayed(fit_single)(k, m, X, y) for k, m in defs.items())
    trained = {k:m for k,m,ok in results if ok and m is not None}
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
        weights = {k: float(v/arr.sum()) for k,v in zip(keys, arr)}
    except Exception:
        # fallback equal
        n = len(models)
        weights = {k: 1.0/n for k in models}
    return weights

# -----------------------
# TRAIN UI (on demand)
# -----------------------
st.header("Huấn luyện (chỉ khi bấm)")
colA, colB = st.columns(2)
with colA:
    if st.button("🛠️ Huấn luyện Mô Hình"):
        if len(st.session_state.history) < MIN_GAMES_TO_PREDICT:
            st.warning(f"Cần ít nhất {MIN_GAMES_TO_PREDICT} ván để huấn luyện (hiện {len(st.session_state.history)}).")
        else:
            with st.spinner("Đang tạo đặc trưng và huấn luyện..."):
                try:
                    X_all, y_all, selector = create_features(st.session_state.history, WINDOW)
                    st.session_state.selector = selector
                    if X_all.shape[0] < 10 or len(np.unique(y_all)) < 2:
                        st.error("Dữ liệu không đủ để huấn luyện.")
                    else:
                        # split quick val for weights
                        X_tr, X_val, y_tr, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=SEED) if X_all.shape[0] > 10 else (X_all, X_all, y_all, y_all)
                        trained = train_models_parallel(X_tr, y_tr)
                        if not trained:
                            st.error("Không huấn luyện được model nào.")
                        else:
                            st.session_state.models = trained
                            # compute adaptive weights
                            st.session_state.weights = compute_adaptive_weights(trained, X_val, y_val)
                            st.success("Huấn luyện xong! Models đã lưu vào session.")
                            # persist models to disk for session persist (ephemeral)
                            for k, m in trained.items():
                                save_obj(m, os.path.join(MODELS_DIR, f"{k}.joblib"))
                            if selector is not None:
                                save_obj(selector, os.path.join(MODELS_DIR, "selector.joblib"))
                            save_obj(st.session_state.weights, os.path.join(MODELS_DIR, "weights.joblib"))
                            # store trained hash
                            st.session_state.trained_hash = hashlib.sha256(str(st.session_state.history).encode()).hexdigest()
                            st.write("Trọng số adaptive:", st.session_state.weights)
                except Exception:
                    st.error("Lỗi khi huấn luyện:")
                    st.error(traceback.format_exc())

with colB:
    if st.button("🔁 Gỡ models (clear)"):
        st.session_state.models = None
        st.session_state.weights = None
        st.session_state.selector = None
        # try remove files
        try:
            for fname in os.listdir(MODELS_DIR):
                os.remove(os.path.join(MODELS_DIR, fname))
        except Exception:
            pass
        st.success("Đã gỡ models khỏi bộ nhớ.")

# auto-load models if present on disk
if st.session_state.models is None:
    try:
        loaded = {}
        for fname in os.listdir(MODELS_DIR):
            if fname.endswith(".joblib"):
                key = fname.replace(".joblib", "")
                if key in ("xgb","cat","rf","lr"):
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

# -----------------------
# PREDICTION UI
# -----------------------
st.header("Dự đoán ván tiếp theo (dùng models đã huấn luyện)")
if st.session_state.models is None:
    st.info("Chưa có model. Sau khi huấn luyện (ít nhất 60 ván), bạn có thể dự đoán.")
else:
    try:
        if len(st.session_state.history) < WINDOW:
            st.warning(f"Cần tối thiểu {WINDOW} ván để tạo features (hiện {len(st.session_state.history)}).")
        else:
            X_feats, y_feats, _ = create_features(st.session_state.history, WINDOW)
            if X_feats.shape[0] < 1:
                st.error("Không thể tạo features cho ván cuối.")
            else:
                feat = X_feats[-1].reshape(1, -1)  # last sample
                # get base model probs
                base_probs = {}
                for k, m in st.session_state.models.items():
                    try:
                        p = m.predict_proba(feat)[0][1]
                    except Exception:
                        try:
                            df = m.decision_function(feat)
                            if np.isscalar(df): df = np.array([df])
                            p = 1.0 / (1.0 + np.exp(-float(df[0])))
                        except Exception:
                            p = 0.5
                    base_probs[k] = float(np.clip(p, 0.0, 1.0))
                st.write("Xác suất (Tài) từ từng model:", base_probs)

                # Weighted Voting (adaptive)
                weights = st.session_state.weights if st.session_state.weights is not None else {k: 1.0/len(base_probs) for k in base_probs.keys()}
                keys = [k for k in base_probs.keys() if k in weights]
                if not keys:
                    keys = list(base_probs.keys())
                    weights = {k: 1.0/len(keys) for k in keys}
                probs_arr = np.array([base_probs[k] for k in keys])
                w_arr = np.array([weights[k] for k in keys])
                final_prob_tai = float(np.dot(w_arr, probs_arr))
                pred_vote = "Tài" if final_prob_tai > 0.5 else "Xỉu"
                st.markdown(f"### Voting (Weighted Adaptive): **{pred_vote}** — Xác suất Tài = {final_prob_tai:.2%}")

                # Stacking meta (fast OOF)
                try:
                    # prepare meta features from training history (fast)
                    X_meta, y_meta, _ = create_features(st.session_state.history, WINDOW)
                    if X_meta.shape[0] >= 10:
                        model_keys = list(st.session_state.models.keys())
                        meta_train = np.zeros((X_meta.shape[0], len(model_keys)))
                        kf = KFold(n_splits=min(3, max(2, X_meta.shape[0]//10)), shuffle=True, random_state=SEED)
                        for i, key in enumerate(model_keys):
                            m = st.session_state.models[key]
                            oof = np.zeros(X_meta.shape[0])
                            for train_idx, val_idx in kf.split(X_meta):
                                try:
                                    # fit clone of m on train_idx
                                    clone = m.__class__(**{k:v for k,v in getattr(m,'get_params', lambda: {})().items()}) if hasattr(m,'get_params') else m
                                    clone.fit(X_meta[train_idx], y_meta[train_idx])
                                    if hasattr(clone, 'predict_proba'):
                                        oof[val_idx] = clone.predict_proba(X_meta[val_idx])[:,1]
                                    elif hasattr(clone, 'decision_function'):
                                        df = clone.decision_function(X_meta[val_idx])
                                        oof[val_idx] = 1.0/(1.0+np.exp(-df))
                                    else:
                                        oof[val_idx] = clone.predict(X_meta[val_idx])
                                except Exception:
                                    oof[val_idx] = 0.5
                            meta_train[:, i] = oof
                        meta_clf = LogisticRegression(max_iter=400, solver='lbfgs', random_state=SEED)
                        meta_clf.fit(meta_train, y_meta)
                        # create meta input for current feat
                        meta_input = np.array([base_probs.get(k, 0.5) for k in model_keys]).reshape(1, -1)
                        p_meta = meta_clf.predict_proba(meta_input)[0,1]
                        pred_meta = "Tài" if p_meta > 0.5 else "Xỉu"
                        st.markdown(f"### Stacking (Meta Logistic): **{pred_meta}** — Xác suất Tài = {p_meta:.2%}")
                    else:
                        st.info("Không đủ mẫu để chạy stacking meta reliably (cần >=10 samples after window).")
                except Exception:
                    st.warning("Stacking meta gặp lỗi; tiếp tục với voting.")

                # quick advice
                st.write("---")
                st.write("Gợi ý: nếu Voting và Stacking đồng ý, tin cậy cao hơn. Nếu không, cân nhắc bỏ qua ván.")
    except Exception:
        st.error("Lỗi khi dự đoán:")
        st.error(traceback.format_exc())

# -----------------------
# FINAL NOTES & DOWNLOADS
# -----------------------
st.markdown("---")
st.info("Lưu ý: App này lưu lịch sử và models tạm thời (ephemeral). Nếu muốn persist lâu dài, lưu file history.csv và models từ thư mục 'models_store' ra máy.")
