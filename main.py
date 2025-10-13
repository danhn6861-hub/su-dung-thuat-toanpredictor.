import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="🎲 AI Dự đoán Tài Xỉu Nâng Cao", layout="wide")

st.title("🎲 AI Dự đoán Tài Xỉu Nâng Cao")
st.markdown("#### ⚙️ Huấn luyện song song 4 mô hình: XGBoost, CatBoost, RandomForest, LogisticRegression")
st.divider()

# =====================
# ⚙️ THIẾT LẬP THAM SỐ
# =====================
MAX_TRAIN_SAMPLES = 3000  # dữ liệu tối đa để huấn luyện
TEST_SIZE = 0.2
SEED = 42

# =====================
# 📊 TẠO DỮ LIỆU GIẢ LẬP (có thể thay bằng dữ liệu thật)
# =====================
@st.cache_data
def create_data(n_samples=MAX_TRAIN_SAMPLES):
    X = np.random.randn(n_samples, 8)
    y = np.random.randint(0, 2, size=n_samples)
    return X, y

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)

# =====================
# 🧠 KHỞI TẠO CÁC MÔ HÌNH
# =====================
def get_models():
    return {
        "XGBoost": XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.1, n_jobs=-1, verbosity=0, random_state=SEED),
        "CatBoost": CatBoostClassifier(iterations=80, depth=3, learning_rate=0.1, verbose=0, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=80, max_depth=5, n_jobs=-1, random_state=SEED),
        "Logistic": LogisticRegression(max_iter=500, solver="lbfgs", random_state=SEED)
    }

# =====================
# 🚀 HÀM HUẤN LUYỆN SONG SONG
# =====================
@st.cache_resource
def train_all(models, X_train, y_train):
    def train_one(name, model):
        model.fit(X_train, y_train)
        return name, model
    results = Parallel(n_jobs=4)(delayed(train_one)(n, m) for n, m in models.items())
    return dict(results)

# =====================
# 🧩 ENSEMBLE: Weighted Voting
# =====================
def weighted_voting(models, X, weights=None):
    base_probs = {}
    for name, model in models.items():
        probs = model.predict_proba(X)[:, 1]
        base_probs[name] = probs

    common_keys = list(base_probs.keys())
    if weights is None:
        weights = {k: 1.0 / len(common_keys) for k in common_keys}

    combined = np.zeros_like(base_probs[common_keys[0]])
    for k in common_keys:
        combined += base_probs[k] * weights[k]
    combined /= sum(weights.values())
    return combined

# =====================
# 🧠 ENSEMBLE: STACKING META LEARNING
# =====================
def stacking_meta(models, X_train, y_train, X_test, folds=3):
    kf = KFold(n_splits=folds, shuffle=True, random_state=SEED)
    meta_train = np.zeros((len(X_train), len(models)))
    meta_test = np.zeros((len(X_test), len(models)))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        for i, (name, model) in enumerate(models.items()):
            model.fit(X_tr, y_tr)
            meta_train[val_idx, i] = model.predict_proba(X_val)[:, 1]
            meta_test[:, i] += model.predict_proba(X_test)[:, 1] / folds

    meta_model = LogisticRegression(max_iter=500)
    meta_model.fit(meta_train, y_train)
    return meta_model, meta_model.predict_proba(meta_test)[:, 1]

# =====================
# 🎛️ GIAO DIỆN HUẤN LUYỆN
# =====================
if st.button("🚀 Huấn luyện Mô Hình"):
    with st.spinner("🔄 Đang huấn luyện 4 mô hình song song..."):
        models = get_models()
        trained_models = train_all(models, X_train, y_train)

        # Tính độ chính xác từng model
        results = {}
        for name, model in trained_models.items():
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results[name] = round(acc * 100, 2)

        st.success("✅ Huấn luyện hoàn tất!")

        # Voting
        weights = {k: v / sum(results.values()) for k, v in results.items()}
        vote_probs = weighted_voting(trained_models, X_test, weights)
        vote_preds = (vote_probs > 0.5).astype(int)
        acc_vote = round(accuracy_score(y_test, vote_preds) * 100, 2)

        # Stacking
        meta_model, stack_probs = stacking_meta(trained_models, X_train, y_train, X_test)
        stack_preds = (stack_probs > 0.5).astype(int)
        acc_stack = round(accuracy_score(y_test, stack_preds) * 100, 2)

        # Hiển thị kết quả
        st.write("### 📈 Độ chính xác từng mô hình:")
        st.table(pd.DataFrame.from_dict(results, orient="index", columns=["Accuracy (%)"]))

        st.markdown(f"### 🧮 Weighted Voting Accuracy: **{acc_vote}%**")
        st.markdown(f"### 🧠 Stacking Meta-Learning Accuracy: **{acc_stack}%**")

        best_model = max(results, key=results.get)
        st.markdown(f"🏆 Mô hình đơn tốt nhất: **{best_model}** ({results[best_model]}%)")

else:
    st.info("👆 Nhấn **Huấn luyện Mô Hình** để bắt đầu.")
