import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from scipy.stats import entropy, zscore, skew, kurtosis, norm
import matplotlib.pyplot as plt
import hashlib
import traceback

# -----------------------------------------------------------------------------
# Robust Streamlit app for Tài/Xỉu predictions
# Goals: prevent crashes, validate shapes, provide safe fallbacks, and self-check
# -----------------------------------------------------------------------------

# ------------------------------
# Config
# ------------------------------
min_games_to_predict = 60  # Minimum games before training/predicting
DEFAULT_WINDOW = 7         # Keep window=7 for lagged features
RANDOM_STATE = 42

# ------------------------------
# Utility helpers
# ------------------------------
def safe_float_array(lst, length=None, fill=0.0):
    """Convert list to float numpy array, ensure length by padding/truncating."""
    try:
        arr = np.array(lst, dtype=float)
    except Exception:
        arr = np.array([float(x) if _is_number_like(x) else 0.0 for x in lst], dtype=float)
    if length is not None:
        if arr.size < length:
            arr = np.concatenate([arr, np.full(length - arr.size, fill)])
        elif arr.size > length:
            arr = arr[-length:]
    return arr


def _is_number_like(x):
    try:
        float(x)
        return True
    except Exception:
        return False


# ------------------------------
# 1. Robust outlier handling
# ------------------------------
@st.cache_data
def handle_outliers(window_data):
    """Replace outliers using z-score; return list of floats same length as input."""
    try:
        arr = safe_float_array(window_data)
        if arr.size < 2:
            return arr.tolist()
        # use ddof=0 for stability when small n
        z_scores = np.abs(zscore(arr, ddof=0))
        median_val = float(np.median(arr)) if arr.size > 0 else 0.0
        arr[z_scores > 3] = median_val
        return arr.tolist()
    except Exception:
        # On any failure, fallback to converting items safely
        try:
            return [float(x) if _is_number_like(x) else 0.0 for x in window_data]
        except Exception:
            return [0.0 for _ in window_data]

# ------------------------------
# Micro-pattern functions
# ------------------------------

def calculate_streaks(binary_seq):
    try:
        if not binary_seq:
            return 0
        current_streak = 1
        max_streak = 1
        for i in range(1, len(binary_seq)):
            if binary_seq[i] == binary_seq[i-1]:
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
            else:
                current_streak = 1
        return max_streak
    except Exception:
        return 0


def calculate_alternations(binary_seq):
    try:
        if len(binary_seq) < 2:
            return 0.0
        alternations = sum(1 for i in range(1, len(binary_seq)) if binary_seq[i] != binary_seq[i-1])
        return alternations / (len(binary_seq) - 1)
    except Exception:
        return 0.0


def calculate_autocorrelation(binary_seq, lag=1):
    try:
        if len(binary_seq) < lag + 1:
            return 0.0
        arr = np.array(binary_seq, dtype=float)
        mean = arr.mean()
        var = arr.var()
        if var == 0:
            return 0.0
        ac = ((arr[:-lag] - mean) * (arr[lag:] - mean)).sum() / (var * len(arr))
        return float(ac)
    except Exception:
        return 0.0

# ------------------------------
# Bias metrics
# ------------------------------

def calculate_bias_metrics(binary_seq):
    try:
        if len(binary_seq) < 2:
            return 0.0, 0.0, 0.0
        arr = np.array(binary_seq, dtype=float)
        var = float(arr.var())
        sk = float(skew(arr))
        kur = float(kurtosis(arr))
        return var, sk, kur
    except Exception:
        return 0.0, 0.0, 0.0

# ------------------------------
# Runs test (returns p-value) - robust
# ------------------------------
def runs_test_p(binary_seq):
    try:
        arr = [int(x) for x in binary_seq]
        n1 = int(sum(1 for x in arr if x == 1))
        n0 = int(sum(1 for x in arr if x == 0))
        n = n0 + n1
        if n0 == 0 or n1 == 0 or n < 2:
            return 1.0
        runs = 1
        for i in range(1, len(arr)):
            if arr[i] != arr[i-1]:
                runs += 1
        expected_runs = 1 + (2.0 * n1 * n0) / n
        numerator = 2.0 * n1 * n0 * (2.0 * n1 * n0 - n)
        denom = (n**2) * (n - 1)
        if denom == 0 or numerator <= 0:
            return 1.0
        variance_runs = numerator / denom
        if variance_runs <= 0:
            return 1.0
        z = (runs - expected_runs) / np.sqrt(variance_runs)
        p_value = 2.0 * (1.0 - norm.cdf(abs(z)))
        return float(np.clip(p_value, 0.0, 1.0))
    except Exception:
        return 1.0

# ------------------------------
# 2. Feature creation (returns selector too)
# ------------------------------
@st.cache_data
def create_advanced_features(history, window=DEFAULT_WINDOW):
    encode = {"Tài": 1, "Xỉu": 0}
    history_num = []
    for r in history:
        try:
            history_num.append(encode.get(r, int(r) if _is_number_like(r) else 0))
        except Exception:
            history_num.append(0)

    X, y = [], []
    for i in range(window, len(history_num)):
        basic_feats = history_num[i-window:i]
        basic_feats_clean = handle_outliers(basic_feats)
        # Ensure fixed length
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
            # If selector fails, return raw X
            return X, y, None
    return X, y, None

# ------------------------------
# 3. Randomness analysis string (robust)
# ------------------------------
@st.cache_data
def analyze_randomness_window(history, window=DEFAULT_WINDOW):
    try:
        if len(history) < window:
            return "🔴 Chưa đủ dữ liệu."
        encode = {"Tài": 1, "Xỉu": 0}
        last_window = [encode.get(r, 0) for r in history[-window:]]
        last_window = handle_outliers(last_window)
        last_window = safe_float_array(last_window, length=window)

        counts = np.bincount(np.round(last_window).astype(int), minlength=2)
        probabilities = counts / counts.sum() if counts.sum() > 0 else np.array([0.5, 0.5])
        ent_val = float(entropy(probabilities, base=2))

        streaks = calculate_streaks(last_window)
        alternations = calculate_alternations(last_window)
        autocorr = calculate_autocorrelation(last_window)
        var, sk, kur = calculate_bias_metrics(last_window)
        p_runs = runs_test_p(last_window)

        base_status = f"🔴 Entropy: {ent_val:.2f}. "
        if ent_val > 0.95:
            base_status += "Cực kỳ ngẫu nhiên! "
        elif ent_val > 0.85:
            base_status += "Khá ngẫu nhiên. "
        elif ent_val > 0.70:
            base_status += "Có một số pattern. "
        else:
            base_status += "Pattern rõ ràng. "

        micro_status = f"🔍 Micro-patterns: Max Streak={streaks}, Alternations={alternations:.2f}, Autocorr={autocorr:.2f}. "
        bias_status = f"📊 Bias: Var={var:.2f}, Skew={sk:.2f}, Kurt={kur:.2f}, Runs_p={p_runs:.3f}."

        return base_status + micro_status + bias_status
    except Exception:
        return "🔴 Không thể phân tích do lỗi nội bộ."

# ------------------------------
# 4. Safe prediction with fallbacks
# ------------------------------

def _safe_model_prob(model, X):
    """Try predict_proba, else try decision_function->sigmoid, else try predict mapping."""
    try:
        proba = model.predict_proba(X)
        # ensure shape [n_samples, 2]
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return float(proba[0][1])
        # If single-column proba, assume it's probability of positive
        if proba.ndim == 2 and proba.shape[1] == 1:
            return float(proba[0][0])
    except Exception:
        pass
    try:
        # decision_function -> sigmoid
        df = model.decision_function(X)
        if np.isscalar(df):
            df = np.array([df])
        val = float(df[0])
        # sigmoid
        prob = 1.0 / (1.0 + np.exp(-val))
        return float(np.clip(prob, 0.0, 1.0))
    except Exception:
        pass
    try:
        pred = model.predict(X)[0]
        # Map prediction to 0/1
        return 1.0 if int(pred) == 1 else 0.0
    except Exception:
        return 0.5


def predict_next_ensemble(models, weights, history, window=DEFAULT_WINDOW, confidence_threshold=0.7):
    try:
        encode = {"Tài": 1, "Xỉu": 0}
        if len(history) < window or not models:
            return "Chưa đủ dữ liệu", 0.5, "Chưa đủ", np.nan

        last_window = [encode.get(r, 0) for r in history[-window:]]
        last_window = handle_outliers(last_window)
        last_window = safe_float_array(last_window, length=window)

        counts = np.bincount(np.round(last_window).astype(int), minlength=2)
        probs = counts / counts.sum() if counts.sum() > 0 else np.array([0.5, 0.5])
        entropy_val = float(entropy(probs, base=2))
        momentum = float(np.mean(np.diff(last_window[-3:])) if len(last_window) >= 2 else 0.0)
        streaks = calculate_streaks(last_window)
        alternations = calculate_alternations(last_window)
        autocorr = calculate_autocorrelation(last_window)
        var, sk, kur = calculate_bias_metrics(last_window)
        p_runs = runs_test_p(last_window)

        final_feats = list(last_window) + [entropy_val, momentum, streaks, alternations, autocorr, var, sk, kur, p_runs]
        final_feats_arr = safe_float_array(final_feats)

        # Apply selector if exists and compatible
        final_feats_trans = final_feats_arr.reshape(1, -1)
        try:
            selector = st.session_state.get('selector', None)
            if selector is not None:
                # check compatibility
                if hasattr(selector, 'n_features_in_'):
                    needed = int(selector.n_features_in_)
                    # If original selector was fitted on full features, ensure we pass correct length
                    if final_feats_trans.shape[1] != selector.n_features_in_:
                        # If mismatch, try to pad or truncate to match expected input
                        final_feats_arr = safe_float_array(final_feats_arr, length=selector.n_features_in_)
                        final_feats_trans = final_feats_arr.reshape(1, -1)
                final_feats_trans = selector.transform(final_feats_trans)
        except Exception:
            # If selector transform fails, revert to raw features
            final_feats_trans = final_feats_arr.reshape(1, -1)

        # Ensuring weights consistent
        try:
            weights = np.array(weights, dtype=float)
            if weights.size != len(models):
                weights = np.ones(len(models), dtype=float) / len(models)
            else:
                s = weights.sum()
                if s <= 0:
                    weights = np.ones(len(models), dtype=float) / len(models)
                else:
                    weights = weights / s
        except Exception:
            weights = np.ones(len(models), dtype=float) / len(models)

        probs_list = []
        for model in models:
            try:
                p = _safe_model_prob(model, final_feats_trans)
            except Exception:
                p = 0.5
            # ensure in [0,1]
            try:
                p = float(np.clip(p, 0.0, 1.0))
            except Exception:
                p = 0.5
            probs_list.append(p)

        probs_arr = np.array(probs_list, dtype=float)
        if probs_arr.size != weights.size:
            # fallback to mean
            final_prob_tai = float(np.clip(probs_arr.mean(), 0.0, 1.0)) if probs_arr.size > 0 else 0.5
        else:
            final_prob_tai = float(np.clip(np.dot(weights, probs_arr), 0.0, 1.0))

        if final_prob_tai > 0.5:
            pred, prob = "Tài", final_prob_tai
        else:
            pred, prob = "Xỉu", 1 - final_prob_tai

        confidence_status = "Đáng tin cậy ✅"
        adaptive_threshold = float(confidence_threshold)
        if entropy_val > 0.85 or kur > 0:
            adaptive_threshold = max(adaptive_threshold, 0.7)
        if prob < adaptive_threshold:
            confidence_status = f"Thấp! (<{adaptive_threshold:.0%}) ⚠️"
            pred = "KHÔNG DỰ ĐOÁN"

        return pred, prob, confidence_status, entropy_val
    except Exception as e:
        # On unexpected error, return safe fallback
        st.error(f"Lỗi dự đoán nội bộ: {str(e)}")
        st.error(traceback.format_exc())
        return "Chưa dự đoán", 0.5, "Lỗi nội bộ", np.nan

# ------------------------------
# probs list safe computation
# ------------------------------
@st.cache_data
def compute_probs_list(history, window, _models, _weights):
    probs_list = []
    try:
        for i in range(window, len(history)):
            history_slice = history[:i]
            try:
                _, prob_tmp, _, _ = predict_next_ensemble(_models, _weights, history_slice, window, confidence_threshold=0.0)
                probs_list.append(prob_tmp)
            except Exception:
                probs_list.append(0.5)
    except Exception:
        return [0.5]
    return probs_list

# ------------------------------
# Streamlit UI and robust training
# ------------------------------

st.set_page_config(page_title="🎲 Robust AI Tài Xỉu", layout="wide")
st.title("🎲 Robust AI Dự đoán Tài Xỉu (Không crash)")
st.markdown("**Mục tiêu:** tránh mọi lỗi làm ảnh hưởng tới dự đoán — fallback an toàn luôn có sẵn.")

# session state init
for k, v in {
    'history': [], 'models': None, 'weights': None, 'window': DEFAULT_WINDOW,
    'prev_hash': '', 'force_train': False, 'selector': None
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

window = int(st.session_state.window)

# truncate history
max_history = 1000
if len(st.session_state.history) > max_history:
    st.session_state.history = st.session_state.history[-max_history:]

# Input area
st.subheader("1) Nhập kết quả")
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🎯 Tài"):
        st.session_state.history.append("Tài")
with col2:
    if st.button("🎯 Xỉu"):
        st.session_state.history.append("Xỉu")
with col3:
    if st.button("🛠️ Huấn Luyện Mô Hình"):
        st.session_state.force_train = True
with col4:
    if st.button("🗑️ Xóa Toàn Bộ Dữ Liệu"):
        st.session_state.history = []
        st.session_state.models = None
        st.session_state.weights = None
        st.session_state.prev_hash = ''
        st.session_state.force_train = False
        st.session_state.selector = None
        st.success("Đã xóa toàn bộ dữ liệu.")

# Hash fn
def hash_history(hist):
    return hashlib.sha256(str(hist).encode()).hexdigest()

current_hash = hash_history(st.session_state.history)

# Analysis
st.subheader("2) Phân tích lịch sử")
if st.session_state.history:
    try:
        st.write("Lịch sử (mới nhất cuối):", st.session_state.history)
        st.markdown(analyze_randomness_window(st.session_state.history, window))
        count_tai = st.session_state.history.count("Tài")
        count_xiu = st.session_state.history.count("Xỉu")
        total = len(st.session_state.history)
        st.write(f"Tài: {count_tai} ({count_tai/total:.2%}) | Xỉu: {count_xiu} ({count_xiu/total:.2%})")
    except Exception:
        st.error("Không thể hiển thị phân tích lịch sử do lỗi nội bộ.")
else:
    st.info("Chưa có dữ liệu lịch sử. Nhập ít nhất một vài ván.")

# Training block (only if >= min_games_to_predict)
if len(st.session_state.history) >= min_games_to_predict and (st.session_state.prev_hash != current_hash or st.session_state.force_train):
    with st.spinner("Đang tạo đặc trưng và huấn luyện (rất an toàn)..."):
        try:
            X, y, selector = create_advanced_features(st.session_state.history, window)
            if X.shape[0] < 10 or len(np.unique(y)) < 2:
                st.error("Dữ liệu không đủ để huấn luyện (ít mẫu hoặc chỉ 1 class).")
            else:
                st.write(f"Chuẩn bị huấn luyện trên {X.shape[0]} mẫu và {X.shape[1]} đặc trưng (trước/sau selector).")

                class_counts = np.bincount(y, minlength=2)
                min_class_count = int(min(class_counts)) if class_counts.size > 0 else 0
                n_splits = min(3, len(X), max(2, min_class_count)) if min_class_count > 0 else 2

                # SMOTE safe
                imbalance_ratio = abs(st.session_state.history.count('Tài')/len(st.session_state.history) - 0.5) if len(st.session_state.history)>0 else 0
                if imbalance_ratio > 0.1 and len(X) > 10 and min_class_count > 5:
                    try:
                        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(3, min_class_count-1))
                        X, y = smote.fit_resample(X, y)
                        st.info("Áp dụng SMOTE thành công.")
                    except Exception:
                        st.warning("Không thể áp dụng SMOTE — bỏ qua.")

                # Augment small dataset
                if X.shape[0] < 100:
                    try:
                        noise = np.random.normal(0, 0.01, X.shape)
                        X = np.vstack([X, X + noise])
                        y = np.hstack([y, y])
                        st.info("Tăng dữ liệu bằng noise nhỏ.")
                    except Exception:
                        pass

                cv = StratifiedKFold(n_splits=max(2, n_splits), shuffle=True, random_state=RANDOM_STATE)

                models = []
                scores = []

                # Logistic Regression (safe)
                try:
                    grid_lr = GridSearchCV(LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, max_iter=500),
                                            {'C':[0.1,0.5,1,5]}, cv=cv)
                    grid_lr.fit(X, y)
                    model_lr = grid_lr.best_estimator_
                    score_lr = float(grid_lr.best_score_)
                except Exception:
                    model_lr = LogisticRegression(solver='liblinear', C=1.0, random_state=RANDOM_STATE, max_iter=500).fit(X, y)
                    score_lr = float(accuracy_score(y, model_lr.predict(X)))
                models.append(model_lr); scores.append(score_lr)

                # Random Forest
                try:
                    grid_rf = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), {'n_estimators':[50,100],'max_depth':[3,5]}, cv=cv)
                    grid_rf.fit(X, y)
                    model_rf = grid_rf.best_estimator_
                    score_rf = float(grid_rf.best_score_)
                except Exception:
                    model_rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=RANDOM_STATE).fit(X, y)
                    score_rf = float(accuracy_score(y, model_rf.predict(X)))
                models.append(model_rf); scores.append(score_rf)

                # XGBoost (very conservative params)
                try:
                    grid_xgb = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE),
                                            {'n_estimators':[50],'max_depth':[3]}, cv=cv)
                    grid_xgb.fit(X, y)
                    model_xgb = grid_xgb.best_estimator_
                    score_xgb = float(grid_xgb.best_score_)
                except Exception:
                    try:
                        model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50, max_depth=3, random_state=RANDOM_STATE).fit(X, y)
                        score_xgb = float(accuracy_score(y, model_xgb.predict(X)))
                    except Exception:
                        model_xgb = None
                        score_xgb = 0.0
                if model_xgb is not None:
                    models.append(model_xgb); scores.append(score_xgb)

                # SVM
                try:
                    model_svm = SVC(probability=True, kernel='rbf', random_state=RANDOM_STATE).fit(X, y)
                    score_svm = float(accuracy_score(y, model_svm.predict(X)))
                except Exception:
                    model_svm = SVC(probability=True, kernel='linear', random_state=RANDOM_STATE).fit(X, y)
                    score_svm = float(accuracy_score(y, model_svm.predict(X)))
                models.append(model_svm); scores.append(score_svm)

                # MLP - keep but with safe params
                try:
                    grid_mlp = GridSearchCV(MLPClassifier(random_state=RANDOM_STATE, max_iter=500), {'hidden_layer_sizes':[(20,),(50,)]}, cv=cv)
                    grid_mlp.fit(X, y)
                    model_mlp = grid_mlp.best_estimator_
                    score_mlp = float(grid_mlp.best_score_)
                except Exception:
                    try:
                        model_mlp = MLPClassifier(hidden_layer_sizes=(20,), max_iter=500, random_state=RANDOM_STATE).fit(X, y)
                        score_mlp = float(accuracy_score(y, model_mlp.predict(X)))
                    except Exception:
                        model_mlp = None
                        score_mlp = 0.0
                if model_mlp is not None:
                    models.append(model_mlp); scores.append(score_mlp)

                # Naive Bayes
                try:
                    model_nb = GaussianNB().fit(X, y)
                    score_nb = float(accuracy_score(y, model_nb.predict(X)))
                except Exception:
                    model_nb = None
                    score_nb = 0.0
                if model_nb is not None:
                    models.append(model_nb); scores.append(score_nb)

                # Finalize session models and weights
                if len(models) == 0:
                    st.error("Không tạo được model nào. Huấn luyện thất bại.")
                else:
                    weights = np.array(scores, dtype=float)
                    if weights.sum() <= 0:
                        weights = np.ones(len(models), dtype=float) / len(models)
                    else:
                        weights = weights / weights.sum()

                    st.session_state.models = models
                    st.session_state.weights = weights
                    st.session_state.selector = selector
                    st.session_state.prev_hash = current_hash
                    st.session_state.force_train = False
                    st.success(f"Huấn luyện xong với {len(models)} models. Weights: {np.round(weights,3).tolist()}")

                    # Try a safe ensemble evaluation (best-effort)
                    try:
                        # generate ensemble predictions safely
                        ensemble_probs = []
                        for m in models:
                            try:
                                p = np.array([_safe_model_prob(m, X[:1])])
                                ensemble_probs.append(p)
                            except Exception:
                                ensemble_probs.append(np.array([0.5]))
                        if len(ensemble_probs) > 0:
                            ensemble_probs = np.hstack(ensemble_probs).mean(axis=1)
                            y_pred = (ensemble_probs > 0.5).astype(int)
                            st.write("Backtest (very small sample) acc:", float(np.mean(y_pred == y[:len(y_pred)])))
                    except Exception:
                        pass
        except Exception as e:
            st.error("Lỗi khi huấn luyện: kiểm tra logs.")
            st.error(traceback.format_exc())

# Prediction & evaluation only if models exist and enough history
if st.session_state.models is not None and len(st.session_state.history) >= min_games_to_predict:
    st.subheader("3) Dự đoán ván tiếp theo")
    try:
        pred, prob, conf_status, entropy_val = predict_next_ensemble(st.session_state.models, st.session_state.weights, st.session_state.history, window)
        st.markdown(f"**Dự đoán (Meta-Ensemble):** **{pred}** | **Độ tin cậy:** {prob:.2%} | Trạng thái: {conf_status}")
        if pred == "KHÔNG DỰ ĐOÁN":
            st.warning("⚠️ Độ tin cậy thấp. Nên cân nhắc bỏ qua ván này.")
    except Exception:
        st.error("Lỗi khi dự đoán. App sẽ dùng fallback an toàn.")

    # Evaluate previous
    if len(st.session_state.history) > window + 1:
        st.subheader("4) Đánh giá ván trước")
        try:
            pred_prev, prob_prev, _, _ = predict_next_ensemble(st.session_state.models, st.session_state.weights, st.session_state.history[:-1], window, confidence_threshold=0.0)
            last_real = st.session_state.history[-1]
            lesson = "Thắng ✅" if last_real == pred_prev else "Thua ❌"
            st.markdown(f"**Kết quả ván trước:** {last_real} | **Dự đoán:** {pred_prev} ({prob_prev:.2%}) | **Bài học:** {lesson}")
        except Exception:
            st.warning("Không thể đánh giá ván trước do lỗi.")

        # Plot safe
        st.subheader("5) Biểu đồ xác suất")
        try:
            probs_list = compute_probs_list(st.session_state.history, window, st.session_state.models, st.session_state.weights)
            rounds = list(range(window+1, window+1+len(probs_list)))
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(rounds, probs_list, marker='o', markersize=3, label="Xác suất Tin cậy")
            ax.axhline(0.7, linestyle='--', label='Ngưỡng 70%')
            ax.axhline(0.5, linestyle='-', label='Ngưỡng 50%')
            ax.set_xlabel('Ván chơi')
            ax.set_ylabel('Độ Tin Cậy')
            ax.set_title('Độ Tin Cậy Dự Đoán qua các ván')
            ax.legend()
            st.pyplot(fig)
        except Exception:
            st.warning("Không thể vẽ biểu đồ do lỗi.")
else:
    st.info(f"Cần ít nhất {min_games_to_predict} kết quả để huấn luyện và dự đoán.")

# ------------------------------
# Self-check utilities: quick diagnostics to ensure no crashes
# ------------------------------
st.markdown("---")
st.header("Tự kiểm tra (Self-check)")
if st.button("Chạy Self-Check"):
    try:
        # Simulate 60 random games
        hist = np.random.choice(['Tài','Xỉu'], size=max(min_games_to_predict, 60)).tolist()
        X, y, selector = create_advanced_features(hist, window)
        ok = True
        if X.shape[0] < 10:
            ok = False
        # try training lightweight model
        if ok:
            m = LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, max_iter=200)
            m.fit(X, y)
            p = _safe_model_prob(m, X[:1])
            st.success('Self-check OK: training+predict worked on synthetic data. Example prob: ' + str(round(p,3)))
        else:
            st.warning('Self-check: không đủ mẫu cho test (số mẫu nhỏ).')
    except Exception:
        st.error('Self-check failed — có lỗi nội bộ.')
        st.error(traceback.format_exc())

st.markdown("**Ghi chú:** App đã được tối ưu để tránh crash. Nếu vẫn gặp lỗi, vui lòng gửi screenshot/traceback — mình sẽ sửa tiếp.")
