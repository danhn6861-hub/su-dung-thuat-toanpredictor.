import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

st.set_page_config(page_title="Dự đoán Tài/Xỉu AI - Phiên bản Nâng Cao", layout="wide")

# Disclaimer
st.sidebar.markdown("""
### ⚠️ Lưu Ý
Ứng dụng này chỉ mang tính chất giải trí và tham khảo. Kết quả dự đoán dựa trên lịch sử ngẫu nhiên và không đảm bảo độ chính xác. Không khuyến khích sử dụng cho mục đích cờ bạc hoặc đầu tư thực tế, vì các trò chơi như Tài/Xỉu thường là ngẫu nhiên và có thể dẫn đến rủi ro tài chính.
""")

# ====== Khởi tạo trạng thái ======
if "history" not in st.session_state:
    st.session_state.history = []
if "ai_confidence" not in st.session_state:
    st.session_state.ai_confidence = []
if "models" not in st.session_state:
    st.session_state.models = None
if "ai_last_pred" not in st.session_state:
    st.session_state.ai_last_pred = None
if "undo_stack" not in st.session_state:
    st.session_state.undo_stack = []
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# ====== Hàm tạo đặc trưng ======
def create_features(history, window=6):
    if len(history) < window + 1:
        return np.empty((0, window)), np.empty((0,))
    X = []
    y = []
    for i in range(window, len(history)):
        X.append([1 if x == "Tài" else 0 for x in history[i - window:i]])
        y.append(1 if history[i] == "Tài" else 0)
    return np.array(X), np.array(y)

# ====== Huấn luyện các mô hình với cải thiện ======
@st.cache_resource
def train_models(history_tuple, ai_confidence_tuple, _cache_key):
    history = list(history_tuple)
    ai_confidence = list(ai_confidence_tuple)
    X, y = create_features(history)
    if len(X) < 10:
        st.warning("Cần ít nhất 10 ván để huấn luyện mô hình.")
        return None

    try:
        # Kiểm tra dữ liệu cân bằng
        if np.all(y == 0) or np.all(y == 1):
            st.warning("Dữ liệu không cân bằng (toàn Tài hoặc Xỉu). Mô hình có thể không chính xác.")
            return None

        # Sử dụng KFold với shuffle=False để phù hợp time-series và là partitioner
        n_splits = min(3, len(X) // 4)  # Mỗi split cần ít nhất 4 mẫu
        if n_splits < 2:
            # Fallback: Stacking mà không cross-validation nếu không đủ dữ liệu cho CV
            st.warning("Dữ liệu quá ít để cross-validation, huấn luyện stacking trực tiếp...")
            recent_weight = np.linspace(0.5, 1.0, len(y))
            combined_weight = recent_weight * np.array(ai_confidence[:len(y)]) if len(ai_confidence) >= len(y) else recent_weight

            lr = LogisticRegression().fit(X, y, sample_weight=combined_weight)
            rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y, sample_weight=combined_weight)
            xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss").fit(X, y, sample_weight=combined_weight)

            estimators = [
                ('lr', lr),
                ('rf', rf),
                ('xgb', xgb)
            ]

            stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=None)
            stack.fit(X, y)
            return stack

        tscv = KFold(n_splits=n_splits, shuffle=False)

        # AI Strategy – học trọng số theo thời gian và độ tin cậy
        recent_weight = np.linspace(0.5, 1.0, len(y))
        combined_weight = recent_weight * np.array(ai_confidence[:len(y)]) if len(ai_confidence) >= len(y) else recent_weight

        # Huấn luyện base models với sample_weight
        lr = LogisticRegression().fit(X, y, sample_weight=combined_weight)
        rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y, sample_weight=combined_weight)
        xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss").fit(X, y, sample_weight=combined_weight)

        # Các base models
        estimators = [
            ('lr', lr),
            ('rf', rf),
            ('xgb', xgb)
        ]

        # Stacking classifier
        stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=tscv)
        stack.fit(X, y)

        # Đánh giá mô hình
        if len(X) > 20:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            acc = accuracy_score(y_test, stack.predict(X_test))
            st.info(f"Độ chính xác đánh giá (test set): {acc:.2%}")

        return stack

    except Exception as e:
        st.error(f"Lỗi huấn luyện: {str(e)}. Vui lòng thử nhập thêm dữ liệu hoặc kiểm tra lại.")
        return None

# ====== Hàm phát hiện pattern cải thiện (sử dụng Markov chain đơn giản) ======
def pattern_detector(history, window=6):
    if len(history) < 2:
        return 0.5

    states = {'Tài': 1, 'Xỉu': 0}
    trans = np.zeros((2, 2))
    for i in range(1, len(history)):
        prev = states[history[i-1]]
        curr = states[history[i]]
        trans[prev, curr] += 1

    row_sums = np.sum(trans, axis=1, keepdims=True)
    trans = np.divide(trans, row_sums, where=row_sums != 0)

    last_state = states[history[-1]]
    return trans[last_state, 1]

# ====== Hàm dự đoán ======
def predict_next(models, history):
    if len(history) < 6 or models is None:
        return None, None

    try:
        latest = np.array([[1 if x == "Tài" else 0 for x in history[-6:]]])
        stack_prob = models.predict_proba(latest)[0][1]
        pattern_score = pattern_detector(history)
        final_score = 0.7 * stack_prob + 0.3 * pattern_score
        return {"Stacking Model": stack_prob, "Pattern Detector": pattern_score}, final_score
    except Exception as e:
        st.error(f"Lỗi dự đoán: {str(e)}")
        return None, None

# ====== Hàm thêm kết quả với undo ======
def add_result(result):
    if st.session_state.is_processing:
        return
    if result not in ["Tài", "Xỉu"]:
        st.error(f"Kết quả không hợp lệ: {result}")
        return
    st.session_state.is_processing = True
    try:
        st.session_state.undo_stack.append((st.session_state.history.copy(), st.session_state.ai_confidence.copy()))
        st.session_state.history.append(result)
        if len(st.session_state.history) > 200:
            st.session_state.history = st.session_state.history[-200:]
            st.session_state.ai_confidence = st.session_state.ai_confidence[-200:]
        if st.session_state.ai_last_pred is not None:
            was_correct = (st.session_state.ai_last_pred == result)
            st.session_state.ai_confidence.append(1.2 if was_correct else 0.8)
    finally:
        st.session_state.is_processing = False

# ====== Hàm undo ======
def undo_last():
    if st.session_state.is_processing:
        return
    st.session_state.is_processing = True
    try:
        if st.session_state.undo_stack:
            history, confidence = st.session_state.undo_stack.pop()
            st.session_state.history = history
            st.session_state.ai_confidence = confidence
    finally:
        st.session_state.is_processing = False

# ====== Export/Import lịch sử ======
def export_history():
    df = pd.DataFrame({"Kết quả": st.session_state.history})
    csv = df.to_csv(index=False).encode('utf-8')
    return csv

def import_history(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if "Kết quả" not in df.columns:
                st.error("File CSV phải có cột 'Kết quả'.")
                return
            history = df["Kết quả"].tolist()
            if not all(x in ["Tài", "Xỉu"] for x in history):
                st.error("Dữ liệu trong file CSV chứa giá trị không hợp lệ (chỉ chấp nhận 'Tài' hoặc 'Xỉu').")
                return
            st.session_state.history = history
            st.session_state.ai_confidence = [1.0] * len(history)
            st.session_state.undo_stack = []
            st.success("Đã import lịch sử!")
        except Exception as e:
            st.error(f"Lỗi khi import: {str(e)}")

# ====== Vẽ biểu đồ ======
def plot_history(history):
    if not history:
        return None
    df = pd.DataFrame({"Kết quả": history})
    counts = df["Kết quả"].value_counts(normalize=True) * 100
    fig, ax = plt.subplots()
    counts.plot(kind='bar', ax=ax, color=['green', 'red'])
    ax.set_ylabel("Tỷ lệ (%)")
    ax.set_title("Tỷ lệ Tài/Xỉu trong lịch sử")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

# ====== Giao diện ======
st.title("🎯 AI Dự đoán Tài / Xỉu – Phiên bản Nâng Cao Tự Học")

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown("#### 📊 Kết quả gần đây:")
    if st.session_state.history:
        st.write(" → ".join(st.session_state.history[-30:]))
    else:
        st.info("Chưa có dữ liệu, nhập kết quả để bắt đầu.")

with col2:
    if st.button("🧹 Xóa lịch sử", key="clear_history"):
        confirm_clear = st.checkbox("Xác nhận xóa toàn bộ lịch sử?", key="confirm_clear")
        if confirm_clear:
            st.session_state.history = []
            st.session_state.ai_confidence = []
            st.session_state.undo_stack = []
            st.session_state.models = None
            st.success("Đã xóa toàn bộ lịch sử!")
            st.rerun()

with col3:
    if st.button("↩️ Undo nhập cuối", key="undo_last"):
        undo_last()
        st.success("Đã undo nhập cuối!")
        st.rerun()

# Biểu đồ
if st.session_state.history:
    try:
        img_data = plot_history(st.session_state.history)
        if img_data:
            st.image(f"data:image/png;base64,{img_data}", caption="Biểu đồ tỷ lệ Tài/Xỉu", use_container_width=True)
    except Exception as e:
        st.warning(f"Không thể vẽ biểu đồ: {str(e)}. Vui lòng kiểm tra thư viện matplotlib.")

st.divider()

# Nút nhập kết quả với key unique
col_tai, col_xiu = st.columns(2)
with col_tai:
    if st.button("Nhập Tài", key="add_tai", disabled=st.session_state.is_processing):
        add_result("Tài")
        st.success("Đã thêm Tài!")
        st.rerun()
with col_xiu:
    if st.button("Nhập Xỉu", key="add_xiu", disabled=st.session_state.is_processing):
        add_result("Xỉu")
        st.success("Đã thêm Xỉu!")
        st.rerun()

st.divider()

# Huấn luyện
if st.button("⚙️ Huấn luyện lại từ lịch sử", key="train_models"):
    with st.spinner("Đang huấn luyện các mô hình..."):
        cache_key = str(len(st.session_state.history)) + str(st.session_state.history[-10:])
        st.session_state.models = train_models(tuple(st.session_state.history), tuple(st.session_state.ai_confidence), cache_key)
    if st.session_state.models is not None:
        st.success("✅ Huấn luyện thành công!")

# Dự đoán
if len(st.session_state.history) >= 6:
    if st.session_state.models is None:
        st.info("Vui lòng huấn luyện mô hình trước.")
    else:
        preds, final_score = predict_next(st.session_state.models, st.session_state.history)
        if preds:
            st.session_state.ai_last_pred = "Tài" if final_score >= 0.5 else "Xỉu"
            st.subheader(f"🎯 Dự đoán chung: **{st.session_state.ai_last_pred}** ({final_score:.2%})")
            st.caption("Tổng hợp từ Stacking Model + Pattern Detector:")
            for k, v in preds.items():
                st.write(f"**{k}** → {v:.2%}")
else:
    st.warning("Cần ít nhất 6 ván để bắt đầu dự đoán.")

st.divider()

# Export/Import
col_export, col_import = st.columns(2)
with col_export:
    csv = export_history()
    st.download_button("📥 Export lịch sử (CSV)", csv, f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv", key="export_history")
with col_import:
    uploaded_file = st.file_uploader("📤 Import lịch sử từ CSV", type="csv", key="import_file")
    if uploaded_file:
        import_history(uploaded_file)
