# app.py
import streamlit as st
import numpy as np
import pandas as pd
import os
import hashlib
import traceback
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy, zscore, skew, kurtosis, norm
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# Config
# -----------------------
MIN_GAMES_TO_PREDICT = 60
DEFAULT_WINDOW = 7
MAX_TRAIN_SAMPLES = 3000
SEED = 42
HISTORY_CSV = "history.csv"  # stored in working dir (ephemeral on cloud)

# -----------------------
# Utility helpers
# -----------------------
def _is_number_like(x):
    try:
        float(x)
        return True
    except Exception:
        return False

def safe_float_array(lst, length=None, fill=0.0):
    try:
        arr = np.array(lst, dtype=float)
    except Exception:
        arr = np.array([float(x) if _is_number_like(x) else fill for x in lst], dtype=float)
    if length is not None:
        if arr.size < length:
            arr = np.concatenate([arr, np.full(length - arr.size, fill)])
        elif arr.size > length:
            arr = arr[-length:]
    return arr

# -----------------------
# Feature engineering
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
        return [float(x) if _is_number_like(x) else 0.0 for x in window_data]

def calculate_streaks(binary_seq):
    try:
        if len(binary_seq) == 0:
            return 0
        current = 1
        mx = 1
        for i in range(1, len(binary_seq)):
            if binary_seq[i] == binary_seq[i-1]:
                current += 1
                if current > mx:
                    mx = current
            else:
                current = 1
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
        mean = arr.mean()
        var = arr.var()
        if var == 0:
            return 0.0
        ac = ((arr[:-lag] - mean) * (arr[lag:] - mean)).sum() / (var * arr.size)
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

@st.cache_data
def create_advanced_features(history, window=DEFAULT_WINDOW):
    encode = {"Tài": 1, "Xỉu": 0}
    history_num = [encode.get(r, 0) for r in history]
    X, y = [], []
    for i in range(window, len(history_num)):
        basic_feats = history_num[i-window:i]
        basic_feats_clean = handle_outliers(basic_feats)
        basic_feats_clean = safe_float_array(basic_feats_clean, length=window)
        counts = np.bincount(np.round(basic_feats_clean).astype(int), minlength=2)
        probs = counts / counts.sum() if counts.sum() > 0 else np.array([0.5, 0.5])
        entropy_val = float(entropy(probs, base=2))
        momentum = float(np.mean(np.diff(basic_feats_clean[-3:])) if len(basic_feats_clean) >= 2 else 0.0)
        streaks = calculate_streaks(basic_feats_clean)
        alternations = calculate_alternations(basic_feats_clean)
        autocorr = calculate_autocorrelation(basic_feats_clean)
        var, sk, kur = calculate_bias_metrics(basic_feats_clean)
        p_runs = runs_test_p(basic_feats_clean)
        features = list(basic_feats_clean) + [entropy_val, momentum, streaks, alternations, autocorr, var, sk, kur, p_runs]
        X.append(features)
        y.append(history_num[i])
    X = np.array(X, dtype=float) if X else np.empty((0, window + 9), dtype=float)
    y = np.array(y, dtype=int) if y else np.empty((0,), dtype=int)
    selector = None
    if X.shape[0] > 0:
        try:
            k = min(10, X.shape[1])
            selector = SelectKBest(f_classif, k=k)
            X_trans = selector.fit_transform(X, y)
            return X_trans, y, selector
        except Exception:
            return X, y, None
    return X, y, None

# -----------------------
# Session / history helpers
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

def save_history_to_csv(hist, path=HISTORY_CSV):
    try:
        df = pd.DataFrame({"result": hist})
        df.to_csv(path, index=False)
    except Exception:
        pass

def load_history_from_csv(path=HISTORY_CSV):
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df["result"].astype(str).tolist()
    except Exception:
        pass
    return []

# Try to recover persisted history on startup
if not st.session_state.history:
    st.session_state.history = load_history_from_csv()

# -----------------------
# UI: Buttons to add results
# -----------------------
st.header("1) Ghi nhận kết quả (thủ công)")
col_a, col_b, col_c = st.columns([1,1,2])
with col_a:
    if st.button("🎯 Tài"):
        st.session_state.history.append("Tài")
        save_history_to_csv(st.session_state.history)
        st.success("Đã lưu: Tài")
with col_b:
    if st.button("🎯 Xỉu"):
        st.session_state.history.append("Xỉu")
        save_history_to_csv(st.session_state.history)
        st.success("Đã lưu: Xỉu")
with col_c:
    if st.button("🗑️ Xóa lịch sử"):
        st.session_state.history = []
        save_history_to_csv(st.session_state.history)
        st.success("Đã xóa lịch sử.")

st.markdown("Lịch sử (mới nhất cuối):")
st.write(st.session_state.history[-200:])  # show last 200 for UX

# Download history CSV
if st.session_state.history:
    csv = pd.DataFrame({"result": st.session_state.history}).to_csv(index=False).encode("utf-8")
    st.download_button("📥 Tải lịch sử (CSV)", data=csv, file_name="history.csv", mime="text/csv")

# -----------------------
# Training helpers
# -----------------------
def get_base_models():
    return {
        'xgb': XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.1, n_jobs=1, verbosity=0, random_state=SEED),
        'cat': CatBoostClassifier(iterations=80, depth=4, learning_rate=0.1, verbose=0, random_state=SEED),
        'rf': RandomForestClassifier(n_estimators=80, max_depth=6, n_jobs=1, random_state=SEED),
        'lr': LogisticRegression(max_iter=500, solver='lbfgs', random_state=SEED)
    }

def fit_one(key, model, X, y):
    try:
        model.fit(X, y)
        return key, model, True
    except Exception:
        # fallback: try lighter params
        try:
            if key == 'xgb':
                m = XGBClassifier(n_estimators=30, max_depth=2, n_jobs=1, verbosity=0, random_state=SEED)
            elif key == 'cat':
                m = CatBoostClassifier(iterations=40, depth=3, verbose=0, random_state=SEED)
            elif key == 'rf':
                m = RandomForestClassifier(n_estimators=30, max_depth=3, n_jobs=1, random_state=SEED)
            else:
                m = LogisticRegression(max_iter=200, solver='liblinear', random_state=SEED)
            m.fit(X, y)
            return key, m, True
        except Exception:
            return key, None, False

@st.cache_resource
def train_models_cached(X, y):
    # cap samples
    if X.shape[0] > MAX_TRAIN_SAMPLES:
        X = X[-MAX_TRAIN_SAMPLES:]
        y = y[-MAX_TRAIN_SAMPLES:]
    models = get_base_models()
    results = Parallel(n_jobs=4)(delayed(fit_one)(k, m, X, y) for k, m in models.items())
    trained = {k: m for k, m, ok in results if ok and m is not None}
    return trained

# -----------------------
# UI: Train model button (only when user clicks)
# -----------------------
st.header("2) Huấn luyện mô hình (chỉ khi bạn bấm)")
col1, col2 = st.columns(2)
with col1:
    if st.button("🛠️ Huấn luyện Mô Hình"):
        if len(st.session_state.history) < MIN_GAMES_TO_PREDICT:
            st.warning(f"Bạn cần ít nhất {MIN_GAMES_TO_PREDICT} ván để huấn luyện. Hiện có {len(st.session_state.history)} ván.")
        else:
            with st.spinner("Đang tạo đặc trưng & huấn luyện..."):
                try:
                    X, y, selector = create_advanced_features(st.session_state.history, DEFAULT_WINDOW)
                    st.session_state.selector = selector
                    if X.shape[0] < 10 or len(np.unique(y)) < 2:
                        st.error("Dữ liệu không đủ để huấn luyện.")
                    else:
                        # optional SMOTE: skipped here for safety unless implemented carefully
                        # train
                        trained = train_models_cached(X, y)
                        if not trained:
                            st.error("Không huấn luyện được model nào.")
                        else:
                            st.session_state.models = trained
                            # compute internal accuracies for weights
                            scores = {}
                            for name, m in trained.items():
                                try:
                                    preds = m.predict(X)
                                    scores[name] = float(accuracy_score(y, preds))
                                except Exception:
                                    scores[name] = 0.0
                            # normalize weights
                            vals = np.array(list(scores.values()), dtype=float)
                            if vals.sum() <= 0:
                                weights = {k: 1.0/len(scores) for k in scores}
                            else:
                                weights = {k: float(v/vals.sum()) for k, v in zip(scores.keys(), vals)}
                            st.session_state.weights = weights
                            st.success("Huấn luyện hoàn tất! Mô hình đã được lưu.")
                            st.write("Độ chính xác nội bộ từng model (train):", scores)
                            # save trained hash
                            st.session_state.trained_hash = hashlib.sha256(str(st.session_state.history).encode()).hexdigest()
                except Exception as e:
                    st.error("Lỗi huấn luyện:")
                    st.error(traceback.format_exc())

with col2:
    if st.button("🔁 Huỷ huấn luyện (gỡ models)"):
        st.session_state.models = None
        st.session_state.weights = None
        st.session_state.selector = None
        st.success("Đã gỡ mô hình khỏi bộ nhớ.")

# -----------------------
# Prediction area (use trained models)
# -----------------------
st.header("3) Dự đoán ván tiếp theo (dùng models đã huấn luyện)")
if st.session_state.models is None:
    st.info("Chưa có mô hình. Huấn luyện trước khi dự đoán.")
else:
    try:
        # construct last-window features for prediction
        if len(st.session_state.history) < DEFAULT_WINDOW:
            st.warning(f"Cần ít nhất {DEFAULT_WINDOW} ván để tạo features dự đoán (hiện {len(st.session_state.history)}).")
        else:
            X_full, y_full, _ = create_advanced_features(st.session_state.history, DEFAULT_WINDOW)
            if X_full.shape[0] < 1:
                st.error("Không thể tạo features cho ván cuối.")
            else:
                feat = X_full[-1].reshape(1, -1)
                # if selector exists, ensure transform compatibility
                X_input = feat
                # safe model proba
                base_probs = {}
                for k, m in st.session_state.models.items():
                    try:
                        p = m.predict_proba(X_input)[0][1]
                    except Exception:
                        try:
                            # decision_function fallback
                            df = m.decision_function(X_input)
                            if np.isscalar(df):
                                df = np.array([df])
                            p = 1.0 / (1.0 + np.exp(-float(df[0])))
                        except Exception:
                            p = 0.5
                    base_probs[k] = float(np.clip(p, 0.0, 1.0))

                st.write("Xác suất từng model (Tài):")
                st.write(base_probs)

                # Weighted Voting
                weights = st.session_state.weights if st.session_state.weights is not None else {k: 1.0/len(base_probs) for k in base_probs}
                common_keys = [k for k in base_probs.keys() if k in weights]
                if not common_keys:
                    common_keys = list(base_probs.keys())
                    weights = {k: 1.0/len(common_keys) for k in common_keys}
                probs_arr = np.array([base_probs[k] for k in common_keys])
                w_arr = np.array([weights[k] for k in common_keys])
                final_prob_tai = float(np.dot(w_arr, probs_arr))
                pred_weighted = "Tài" if final_prob_tai > 0.5 else "Xỉu"
                st.markdown(f"### Voting (Weighted): **{pred_weighted}** — Xác suất Tài = {final_prob_tai:.2%}")

                # Stacking meta (fast OOF on train)
                try:
                    # build meta input from base models on training set
                    X_train_meta, y_train_meta, _ = create_advanced_features(st.session_state.history, DEFAULT_WINDOW)
                    # ensure trained models exist for keys
                    model_keys = list(st.session_state.models.keys())
                    meta_train = np.zeros((X_train_meta.shape[0], len(model_keys)))
                    kf = KFold(n_splits=min(3, max(2, X_train_meta.shape[0]//10)), shuffle=True, random_state=SEED)
                    for i, key in enumerate(model_keys):
                        m = st.session_state.models[key]
                        oof = np.zeros(X_train_meta.shape[0])
                        for train_idx, val_idx in kf.split(X_train_meta):
                            try:
                                m_clone = m.__class__(**{k:v for k,v in getattr(m, '__dict__', {}).items() if not k.startswith('_')}) if hasattr(m, '__class__') else m
                                # attempt to fit clone on train_idx (best-effort) — wrap in try/except
                                m_clone.fit(X_train_meta[train_idx], y_train_meta[train_idx])
                                if hasattr(m_clone, 'predict_proba'):
                                    oof[val_idx] = m_clone.predict_proba(X_train_meta[val_idx])[:, 1]
                                elif hasattr(m_clone, 'decision_function'):
                                    df = m_clone.decision_function(X_train_meta[val_idx])
                                    oof[val_idx] = 1.0 / (1.0 + np.exp(-df))
                                else:
                                    oof[val_idx] = m_clone.predict(X_train_meta[val_idx])
                            except Exception:
                                oof[val_idx] = 0.5
                        meta_train[:, i] = oof
                    # meta model
                    meta = LogisticRegression(max_iter=500)
                    meta.fit(meta_train, y_train_meta)
                    # prepare single-row meta input from base_probs
                    meta_input = np.array([base_probs.get(k, 0.5) for k in model_keys]).reshape(1, -1)
                    p_meta = meta.predict_proba(meta_input)[0,1]
                    pred_meta = "Tài" if p_meta > 0.5 else "Xỉu"
                    st.markdown(f"### Stacking (Meta): **{pred_meta}** — Xác suất Tài = {p_meta:.2%}")
                except Exception:
                    st.warning("Không thể chạy stacking meta (bỏ qua).")

    except Exception:
        st.error("Lỗi khi dự đoán:")
        st.error(traceback.format_exc())

# -----------------------
# Footer / Notes
# -----------------------
st.markdown("---")
st.info("Ghi chú: App lưu lịch sử vào file history.csv trên môi trường hiện tại (ephemeral).\
 Bấm 'Tải lịch sử' để lưu về máy. Huấn luyện chỉ chạy khi bấm 'Huấn luyện Mô Hình'.")

