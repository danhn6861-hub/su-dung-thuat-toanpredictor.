import streamlit as st
import numpy as np
import hashlib
import traceback
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from scipy.stats import entropy, zscore, skew, kurtosis, norm
import matplotlib.pyplot as plt

# ------------------------------
# CONFIG
# ------------------------------
MIN_GAMES_TO_PREDICT = 60
DEFAULT_WINDOW = 7
RANDOM_STATE = 42
MAX_TRAIN_SAMPLES = 3000  # cap to avoid long training

# ------------------------------
# HELPERS (robust conversion/ops)
# ------------------------------
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

# ------------------------------
# Feature engineering (robust)
# ------------------------------

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
                mx = max(mx, current)
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

# ------------------------------
# MODEL TRAINING (explicit: only train when user presses button)
# Use joblib to parallelize per-model training
# ------------------------------
@st.cache_resource
def train_models_cached(X, y):
    # Cap samples to prevent long runs
    if X.shape[0] > MAX_TRAIN_SAMPLES:
        X = X[-MAX_TRAIN_SAMPLES:]
        y = y[-MAX_TRAIN_SAMPLES:]

    # Safe model definitions (conservative params)
    models = {
        'xgb': XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.1, n_jobs=1, verbosity=0, random_state=RANDOM_STATE),
        'cat': CatBoostClassifier(iterations=80, depth=4, learning_rate=0.1, verbose=0, random_state=RANDOM_STATE),
        'rf': RandomForestClassifier(n_estimators=80, max_depth=6, n_jobs=1, random_state=RANDOM_STATE),
        'lr': LogisticRegression(max_iter=500, solver='lbfgs', random_state=RANDOM_STATE)
    }

    def fit_one(key, model, X_local, y_local):
        try:
            model.fit(X_local, y_local)
            return key, model, True
        except Exception:
            # Try simple fallback params
            try:
                if key == 'xgb':
                    m = XGBClassifier(n_estimators=30, max_depth=2, n_jobs=1, verbosity=0, random_state=RANDOM_STATE)
                    m.fit(X_local, y_local)
                    return key, m, True
                if key == 'cat':
                    m = CatBoostClassifier(iterations=40, depth=3, verbose=0, random_state=RANDOM_STATE)
                    m.fit(X_local, y_local)
                    return key, m, True
                if key == 'rf':
                    m = RandomForestClassifier(n_estimators=30, max_depth=3, n_jobs=1, random_state=RANDOM_STATE)
                    m.fit(X_local, y_local)
                    return key, m, True
                if key == 'lr':
                    m = LogisticRegression(max_iter=200, solver='liblinear', random_state=RANDOM_STATE)
                    m.fit(X_local, y_local)
                    return key, m, True
            except Exception:
                return key, None, False

    # Parallel training (joblib Parallel)
    results = Parallel(n_jobs=4)(delayed(fit_one)(k, m, X, y) for k, m in models.items())
    trained = {k: m for k, m, ok in results if ok and m is not None}
    return trained

# ------------------------------
# Stacking meta-learner training (fast OOF logistic)
# ------------------------------

def train_stacking_meta(base_models, X, y, folds=3):
    try:
        # produce out-of-fold predictions for each base model
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
        meta_features = np.zeros((X.shape[0], len(base_models)))
        for i, (name, model) in enumerate(base_models.items()):
            preds_oof = np.zeros(X.shape[0])
            for train_idx, val_idx in skf.split(X, y):
                try:
                    m_clone = clone_model(model)
                    m_clone.fit(X[train_idx], y[train_idx])
                    preds_oof[val_idx] = m_clone.predict_proba(X[val_idx])[:, 1] if hasattr(m_clone, 'predict_proba') else (m_clone.decision_function(X[val_idx]) if hasattr(m_clone, 'decision_function') else m_clone.predict(X[val_idx]))
                except Exception:
                    preds_oof[val_idx] = 0.5
            meta_features[:, i] = preds_oof
        # Train logistic on meta_features
        meta = LogisticRegression(max_iter=500, solver='lbfgs', random_state=RANDOM_STATE)
        meta.fit(meta_features, y)
        return meta
    except Exception:
        return None


def clone_model(model):
    # quick clone: try sklearn clone, else instantiate same class with safe params
    from sklearn.base import clone
    try:
        return clone(model)
    except Exception:
        try:
            cls = model.__class__
            return cls()
        except Exception:
            return model

# ------------------------------
# Predict helpers
# ------------------------------

def _safe_proba(model, X):
    try:
        if hasattr(model, 'predict_proba'):
            p = model.predict_proba(X)
            if p.ndim == 2 and p.shape[1] >= 2:
                return float(p[0, 1])
            if p.ndim == 2 and p.shape[1] == 1:
                return float(p[0, 0])
        if hasattr(model, 'decision_function'):
            df = model.decision_function(X)
            if np.isscalar(df): df = np.array([df])
            val = float(df[0])
            return float(1.0 / (1.0 + np.exp(-val)))
        pred = model.predict(X)[0]
        return 1.0 if int(pred) == 1 else 0.0
    except Exception:
        return 0.5

# ------------------------------
# STREAMLIT APP
# ------------------------------
st.set_page_config(page_title='🎲 Taixiu Optimized', layout='wide')
st.title('🎲 AI Tài Xỉu - Full Optimized')
st.markdown('Giữ XGBoost, CatBoost, RandomForest, Logistic. Có Weighted Voting & Stacking. Huấn luyện **chỉ** khi bạn ấn nút.')

# session init
if 'history' not in st.session_state:
    st.session_state.history = []
if 'models' not in st.session_state:
    st.session_state.models = None
if 'meta' not in st.session_state:
    st.session_state.meta = None
if 'weights' not in st.session_state:
    st.session_state.weights = None
if 'selector' not in st.session_state:
    st.session_state.selector = None
if 'trained_hash' not in st.session_state:
    st.session_state.trained_hash = ''
if 'force_train' not in st.session_state:
    st.session_state.force_train = False

# Input controls
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button('🎯 Tài'):
        st.session_state.history.append('Tài')
with col2:
    if st.button('🎯 Xỉu'):
        st.session_state.history.append('Xỉu')
with col3:
    train_click = st.button('🛠️ Huấn luyện Mô Hình (Bấm khi sẵn sàng)')
    if train_click:
        st.session_state.force_train = True
with col4:
    if st.button('🗑️ Xóa toàn bộ dữ liệu'):
        st.session_state.history = []
        st.session_state.models = None
        st.session_state.meta = None
        st.session_state.weights = None
        st.session_state.selector = None
        st.session_state.trained_hash = ''
        st.session_state.force_train = False
        st.success('Đã xóa.')

st.write(f'Hiện có {len(st.session_state.history)} ván lưu trữ.')

# Analysis area
st.subheader('Phân tích ngắn')
if st.session_state.history:
    st.write('Lịch sử (mới nhất cuối):', st.session_state.history[-50:])
    try:
        X_temp, y_temp, _ = create_advanced_features(st.session_state.history, DEFAULT_WINDOW)
        st.write('Đặc trưng hiện có:', X_temp.shape)
    except Exception:
        st.write('Không thể tạo đặc trưng preview')
else:
    st.info('Chưa có dữ liệu. Nhập ít nhất vài ván.')

# Training logic (only when user presses and enough games)
should_train = st.session_state.force_train and len(st.session_state.history) >= MIN_GAMES_TO_PREDICT
if st.session_state.force_train and len(st.session_state.history) < MIN_GAMES_TO_PREDICT:
    st.warning(f'Bạn cần ít nhất {MIN_GAMES_TO_PREDICT} ván để huấn luyện. Hiện: {len(st.session_state.history)}')

if should_train:
    st.info('🔄 Đang tạo đặc trưng và huấn luyện — không đóng trang này cho đến khi hoàn tất.')
    try:
        X, y, selector = create_advanced_features(st.session_state.history, DEFAULT_WINDOW)
        st.session_state.selector = selector
        if X.shape[0] < 10 or len(np.unique(y)) < 2:
            st.error('Dữ liệu không đủ đa dạng để huấn luyện.')
        else:
            # optional SMOTE if highly imbalanced
            try:
                imbalance = abs(st.session_state.history.count('Tài')/len(st.session_state.history) - 0.5)
                if imbalance > 0.12 and X.shape[0] > 20:
                    sm = SMOTE(random_state=RANDOM_STATE, k_neighbors= min(3, max(1, int(np.min(np.bincount(y)) - 1))))
                    X, y = sm.fit_resample(X, y)
                    st.info('Áp dụng SMOTE (an toàn).')
            except Exception:
                pass

            # limit samples
            if X.shape[0] > MAX_TRAIN_SAMPLES:
                X = X[-MAX_TRAIN_SAMPLES:]
                y = y[-MAX_TRAIN_SAMPLES:]

            # train base models
            trained = train_models_cached(X, y)
            if not trained:
                st.error('Không huấn luyện được model nào. Kiểm tra logs.')
            else:
                st.session_state.models = trained
                # compute simple validation scores for weights
                scores = {}
                for name, m in trained.items():
                    try:
                        preds = m.predict(X)
                        scores[name] = float(accuracy_score(y, preds))
                    except Exception:
                        scores[name] = 0.0
                # normalize weights
                w_vals = np.array([scores[k] for k in scores], dtype=float)
                if w_vals.sum() <= 0:
                    w_norm = {k: 1.0/len(w_vals) for k in scores}
                else:
                    w_norm = {k: float(v / w_vals.sum()) for k, v in zip(scores.keys(), w_vals)}
                st.session_state.weights = w_norm

                # train stacking meta (fast; 3-fold)
                try:
                    # build meta features quickly (use small folds to save time)
                    from sklearn.model_selection import StratifiedKFold
                    skf = StratifiedKFold(n_splits=min(3, max(2, int(min(5, X.shape[0]//10)))), shuffle=True, random_state=RANDOM_STATE)
                    meta_X = np.zeros((X.shape[0], len(trained)))
                    model_keys = list(trained.keys())
                    for i, k in enumerate(model_keys):
                        preds_oof = np.zeros(X.shape[0])
                        for train_idx, val_idx in skf.split(X, y):
                            try:
                                m_clone = clone_model(trained[k])
                                m_clone.fit(X[train_idx], y[train_idx])
                                if hasattr(m_clone, 'predict_proba'):
                                    preds_oof[val_idx] = m_clone.predict_proba(X[val_idx])[:, 1]
                                elif hasattr(m_clone, 'decision_function'):
                                    preds_oof[val_idx] = 1.0 / (1.0 + np.exp(-m_clone.decision_function(X[val_idx])))
                                else:
                                    preds_oof[val_idx] = m_clone.predict(X[val_idx])
                            except Exception:
                                preds_oof[val_idx] = 0.5
                        meta_X[:, i] = preds_oof
                    meta_clf = LogisticRegression(max_iter=500, solver='lbfgs', random_state=RANDOM_STATE)
                    meta_clf.fit(meta_X, y)
                    st.session_state.meta = {
                        'model': meta_clf,
                        'base_keys': model_keys
                    }
                except Exception:
                    st.warning('Không huấn luyện được stacking meta (bỏ qua).')

                st.success('Huấn luyện hoàn tất!')
                st.session_state.trained_hash = hashlib.sha256(str(st.session_state.history).encode()).hexdigest()
                st.session_state.force_train = False
    except Exception as e:
        st.error('Lỗi khi huấn luyện:')
        st.error(traceback.format_exc())
        st.session_state.force_train = False

# PREDICTION (only use trained models; never auto-train)
st.subheader('Dự đoán (chỉ dùng models đã train)')
if st.session_state.models is None:
    st.info('Chưa có model đã huấn luyện. Bấm "Huấn luyện Mô Hình" để train khi đủ dữ liệu.')
else:
    # prepare last window
    if len(st.session_state.history) < DEFAULT_WINDOW:
        st.warning(f'Cần tối thiểu {DEFAULT_WINDOW} ván để tạo features dự đoán. Hiện: {len(st.session_state.history)}')
    else:
        try:
            last_window = st.session_state.history[-DEFAULT_WINDOW:]
            X_last, _, _ = create_advanced_features(st.session_state.history[-(DEFAULT_WINDOW+1):], DEFAULT_WINDOW)
            # take last sample features
            if X_last.shape[0] > 0:
                feat = X_last[-1].reshape(1, -1)
            else:
                st.error('Không tạo được features cho ván cuối.'); feat = None

            # apply selector if exists
            if feat is not None:
                if st.session_state.selector is not None:
                    try:
                        feat_full = st.session_state.selector.inverse_transform(feat) if hasattr(st.session_state.selector, 'inverse_transform') else feat
                    except Exception:
                        feat_full = feat
                else:
                    feat_full = feat

                # create X_input for model proba: if models trained on transformed features, pass feat; else try to adapt
                X_input = feat

                # collect base probs
                base_probs = {}
                for k, m in st.session_state.models.items():
                    try:
                        p = _safe_proba(m, X_input)
                    except Exception:
                        p = 0.5
                    base_probs[k] = float(np.clip(p, 0.0, 1.0))

                st.write('✅ Xác suất từ từng model (base):')
                st.write(base_probs)

                # Weighted Voting
                weights = st.session_state.weights if st.session_state.weights is not None else {k: 1.0/len(base_probs) for k in base_probs}
                # align keys
                common_keys = [k for k in base_probs.keys() if k in weights]
                if not common_keys:
                    common_keys = list(base_probs.keys())\                    weights = {k: 1.0/len(common_keys) for k in common_keys}
                probs_arr = np.array([base_probs[k] for k in common_keys])
                w_arr = np.array([weights[k] for k in common_keys])
                final_prob_weighted = float(np.dot(w_arr, probs_arr))
                pred_weighted = 'Tài' if final_prob_weighted > 0.5 else 'Xỉu'

                st.markdown(f'### Voting thông minh (Weighted): **{pred_weighted}** — Xác suất Tài={final_prob_weighted:.2%}')

                # Stacking meta prediction
                if st.session_state.meta is not None:
                    try:
                        meta_model = st.session_state.meta['model']
                        base_keys = st.session_state.meta['base_keys']
                        meta_input = np.array([base_probs[k] if k in base_probs else 0.5 for k in base_keys]).reshape(1, -1)
                        p_meta = _safe_proba(meta_model, meta_input)
                        pred_meta = 'Tài' if p_meta > 0.5 else 'Xỉu'
                        st.markdown(f'### Stacking (Meta Logistic): **{pred_meta}** — Xác suất Tài={p_meta:.2%}')
                    except Exception:
                        st.warning('Không thể dự đoán bằng stacking meta.')
                else:
                    st.info('Chưa có stacking meta (bấm huấn luyện để tạo).')

                # Combined display and suggestion
                st.markdown('---')
                st.write('Gợi ý: nếu cả 2 phương pháp cùng nhau đồng ý — xác suất thường đáng tin cậy hơn.')

                # plot base probs
                try:
                    fig, ax = plt.subplots()
                    keys = list(base_probs.keys())
                    vals = [base_probs[k] for k in keys]
                    ax.bar(keys, vals)
                    ax.set_ylim(0,1)
                    ax.set_ylabel('Xác suất Tài')
                    st.pyplot(fig)
                except Exception:
                    pass
            else:
                st.error('Không có features để dự đoán.')
        except Exception:
            st.error('Lỗi khi dự đoán:')
            st.error(traceback.format_exc())

# Footer notes
st.markdown('---')
st.info('Mô hình **không tự huấn luyện sau mỗi ván**. Bấm **Huấn luyện Mô Hình** khi bạn đã nhập ≥60 ván và muốn cập nhật mô hình.')
