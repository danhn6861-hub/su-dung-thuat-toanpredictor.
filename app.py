import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from scipy.stats import entropy, zscore, skew, kurtosis
import matplotlib.pyplot as plt
import hashlib

# ------------------------------
# 1. Hàm loại bỏ hoặc thay thế outliers
# ------------------------------
@st.cache_data
def handle_outliers(window_data):
    """Xử lý outliers bằng Z-score và thay thế bằng trung vị."""
    arr = np.array(window_data, dtype=float)
    if len(arr) < 2:
        return arr.tolist()
    
    z_scores = np.abs(zscore(arr, ddof=1))
    median_val = np.median(arr)
    arr[z_scores > 3] = median_val
    return arr.tolist()

# ------------------------------
# Hàm tính micro-patterns
# ------------------------------
def calculate_streaks(binary_seq):
    if not binary_seq:
        return 0
    current_streak = 1
    max_streak = 1
    for i in range(1, len(binary_seq)):
        if binary_seq[i] == binary_seq[i-1]:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 1
    return max_streak

def calculate_alternations(binary_seq):
    if len(binary_seq) < 2:
        return 0
    alternations = sum(1 for i in range(1, len(binary_seq)) if binary_seq[i] != binary_seq[i-1])
    return alternations / (len(binary_seq) - 1)

def calculate_autocorrelation(binary_seq, lag=1):
    if len(binary_seq) < lag + 1:
        return 0
    mean = np.mean(binary_seq)
    var = np.var(binary_seq)
    if var == 0:
        return 0
    ac = sum((binary_seq[i] - mean) * (binary_seq[i+lag] - mean) for i in range(len(binary_seq)-lag)) / (var * len(binary_seq))
    return ac

# ------------------------------
# Hàm tính bias ẩn
# ------------------------------
def calculate_bias_metrics(binary_seq):
    if len(binary_seq) < 2:
        return 0, 0, 0
    var = np.var(binary_seq)
    sk = skew(binary_seq)
    kur = kurtosis(binary_seq)
    return var, sk, kur

# ------------------------------
# 2. Hàm tạo đặc trưng nâng cao
# ------------------------------
@st.cache_data(hash_funcs={list: lambda x: hashlib.sha256(str(x).encode()).hexdigest()})
def create_advanced_features(history, window=5):
    encode = {"Tài": 1, "Xỉu": 0}
    history_num = [encode[r] for r in history]

    X, y = [], []
    for i in range(window, len(history_num)):
        basic_feats = history_num[i-window:i]
        basic_feats_clean = handle_outliers(basic_feats)

        counts = np.bincount(basic_feats_clean, minlength=2)
        probabilities = counts / counts.sum()
        entropy_val = entropy(probabilities, base=2)

        momentum = np.mean(np.diff(basic_feats_clean[-3:])) if len(basic_feats_clean) >= 2 else 0
        streaks = calculate_streaks(basic_feats_clean)
        alternations = calculate_alternations(basic_feats_clean)
        autocorr = calculate_autocorrelation(basic_feats_clean)
        var, sk, kur = calculate_bias_metrics(basic_feats_clean)

        features = basic_feats_clean + [entropy_val, momentum, streaks, alternations, autocorr, var, sk, kur]
        X.append(features)
        y.append(history_num[i])

    X = np.array(X)
    y = np.array(y)

    # Feature selection
    if len(X) > 0:
        selector = SelectKBest(f_classif, k=min(10, X.shape[1]))
        X = selector.fit_transform(X, y)

    return X, y

# ------------------------------
# 3. Phân tích độ ngẫu nhiên
# ------------------------------
@st.cache_data(hash_funcs={list: lambda x: hashlib.sha256(str(x).encode()).hexdigest()})
def analyze_randomness_window(history, window=5):
    if len(history) < window:
        return "🔴 Chưa đủ dữ liệu."
    encode = {"Tài": 1, "Xỉu": 0}
    last_window = [encode[r] for r in history[-window:]]
    last_window = handle_outliers(last_window)
    counts = np.bincount(last_window, minlength=2)
    probabilities = counts / counts.sum()
    ent_val = entropy(probabilities, base=2)
    
    streaks = calculate_streaks(last_window)
    alternations = calculate_alternations(last_window)
    autocorr = calculate_autocorrelation(last_window)
    var, sk, kur = calculate_bias_metrics(last_window)
    
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
    bias_status = f"📊 Bias: Var={var:.2f}, Skew={sk:.2f}, Kurt={kur:.2f}."
    
    return base_status + micro_status + bias_status

# ------------------------------
# 4. Dự đoán ván tiếp theo
# ------------------------------
def predict_next_ensemble(models, weights, history, window=5, confidence_threshold=0.65):
    encode = {"Tài": 1, "Xỉu": 0}
    if len(history) < window or not models:
        return "Chưa đủ dữ liệu", 0.5, "Chưa đủ", np.nan

    last_window = [encode[r] for r in history[-window:]]
    last_window = handle_outliers(last_window)

    counts = np.bincount(last_window, minlength=2)
    probabilities = counts / counts.sum()
    entropy_val = entropy(probabilities, base=2)
    momentum = np.mean(np.diff(last_window[-3:])) if len(last_window) >= 2 else 0
    streaks = calculate_streaks(last_window)
    alternations = calculate_alternations(last_window)
    autocorr = calculate_autocorrelation(last_window)
    var, sk, kur = calculate_bias_metrics(last_window)

    final_feats = last_window + [entropy_val, momentum, streaks, alternations, autocorr, var, sk, kur]

    # Adaptive strategy
    adaptive_threshold = confidence_threshold
    if entropy_val > 0.85:
        adaptive_threshold = 0.70
    if kur > 0:
        adaptive_threshold = 0.70
    if entropy_val > 0.85 and abs(sk) < 0.1:
        majority = "Tài" if sum(last_window) > window / 2 else "Xỉu"
        return majority, 0.5, "Fallback do nhiễu cao ⚠️", entropy_val

    probs = []
    for model in models:
        try:
            prob = model.predict_proba([final_feats])[0][1]
        except:
            prob = 0.5
        probs.append(prob)
    probs = np.array(probs)
    final_prob_tai = np.clip(np.dot(weights, probs), 0.0, 1.0)

    if final_prob_tai > 0.5:
        pred, prob = "Tài", final_prob_tai
    else:
        pred, prob = "Xỉu", 1 - final_prob_tai

    confidence_status = "Đáng tin cậy ✅"
    if prob < adaptive_threshold:
        confidence_status = f"Thấp! (<{adaptive_threshold:.0%}) ⚠️"
        pred = "KHÔNG DỰ ĐOÁN"

    return pred, prob, confidence_status, entropy_val

# ------------------------------
# Hàm tính probs_list cho biểu đồ
# ------------------------------
@st.cache_data(hash_funcs={list: lambda x: hashlib.sha256(str(x).encode()).hexdigest()})
def compute_probs_list(history, window, _models, _weights):
    probs_list = []
    for i in range(window, len(history)):
        history_slice = history[:i]
        _, prob_tmp, _, _ = predict_next_ensemble(_models, _weights, history_slice, window, confidence_threshold=0.0)
        probs_list.append(prob_tmp)
    return probs_list

# ------------------------------
# 5. Streamlit App
# ------------------------------
st.set_page_config(page_title="🎲 AI Dự đoán Tài Xỉu Nâng Cao", layout="wide")
st.title("🎲 AI Dự đoán Tài Xỉu Nâng Cao")
st.markdown("**Meta-Ensemble (5 models, weights động) | Micro-patterns + Bias + Adaptive Strategy + Tối Ưu cho Small Data**")

# Session state
if "history" not in st.session_state: 
    st.session_state.history = []
if "models" not in st.session_state: 
    st.session_state.models = None
if "weights" not in st.session_state: 
    st.session_state.weights = None
if "window" not in st.session_state: 
    st.session_state.window = 7
if "prev_hash" not in st.session_state:
    st.session_state.prev_hash = ""
if "force_train" not in st.session_state:
    st.session_state.force_train = False
window = st.session_state.window

# Giới hạn history
max_history = 1000
if len(st.session_state.history) > max_history:
    st.session_state.history = st.session_state.history[-max_history:]

# --- Nhập kết quả và quản lý dữ liệu ---
st.subheader("1. Nhập Kết Quả Ván Chơi và Quản Lý Dữ Liệu")
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
        st.session_state.prev_hash = ""
        st.session_state.force_train = False
        st.success("Đã xóa toàn bộ dữ liệu.")

# Tính hash history
def hash_history(hist):
    return hashlib.sha256(str(hist).encode()).hexdigest()

current_hash = hash_history(st.session_state.history)

# --- Phân tích lịch sử ---
st.subheader("2. Phân Tích Lịch Sử")
if st.session_state.history:
    st.write("Lịch sử kết quả (mới nhất cuối):", st.session_state.history)
    st.markdown(analyze_randomness_window(st.session_state.history, window))
    count_tai = st.session_state.history.count("Tài")
    count_xiu = st.session_state.history.count("Xỉu")
    total = len(st.session_state.history)
    st.write(f"Tài: {count_tai} ({count_tai/total:.2%}) | Xỉu: {count_xiu} ({count_xiu/total:.2%})")
else:
    st.info("Chưa có dữ liệu lịch sử.")
