import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, train_test_split, TimeSeriesSplit
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

# ====== Hàm tạo đặc trưng cải tiến - giảm overfitting ======
def create_features_improved(history, window=5):
    if len(history) < window + 1:
        return np.empty((0, window + 2)), np.empty((0,))  # +2 cho features mới
    
    X = []
    y = []
    
    for i in range(window, len(history)):
        # Features cơ bản
        base_features = [1 if x == "Tài" else 0 for x in history[i - window:i]]
        
        # Thêm features thống kê để giảm overfitting
        tai_count = sum(base_features)
        xiu_count = window - tai_count
        tai_ratio = tai_count / window
        
        # Features về biến động (thay đổi liên tục)
        changes = 0
        for j in range(1, len(base_features)):
            if base_features[j] != base_features[j-1]:
                changes += 1
        change_ratio = changes / (window - 1) if window > 1 else 0
        
        # Kết hợp tất cả features
        combined_features = base_features + [tai_ratio, change_ratio]
        
        X.append(combined_features)
        y.append(1 if history[i] == "Tài" else 0)
    
    return np.array(X), np.array(y)

# ====== Pattern detector cải tiến ======
def pattern_detector_improved(history, lookback=8):
    if len(history) < 3:
        return 0.5
    
    # Phân tích đa chiều thay vì chỉ transition đơn giản
    recent = history[-lookback:] if len(history) >= lookback else history
    
    # Tính tỷ lệ Tài/Xỉu gần đây
    tai_recent = sum(1 for x in recent if x == "Tài")
    xiu_recent = len(recent) - tai_recent
    
    # Phát hiện chuỗi
    max_streak = 0
    current_streak = 1
    for i in range(1, len(recent)):
        if recent[i] == recent[i-1]:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 1
    
    # Logic: nếu chuỗi quá dài, khả năng đảo chiều cao hơn
    streak_factor = min(max_streak / 4.0, 1.0)  # Chuỗi 4+ là đáng chú ý
    
    # Cân bằng hơn - tránh thiên vị số đông
    base_prob = tai_recent / len(recent)
    
    # Điều chỉnh dựa trên streak (mean reversion)
    if streak_factor > 0.5:
        adjusted_prob = 1.0 - base_prob  # Thiên về đảo chiều khi streak dài
    else:
        adjusted_prob = 0.5  # Trung lập khi không có streak rõ rệt
    
    return max(0.1, min(0.9, adjusted_prob))  # Giới hạn trong khoảng 10%-90%

# ====== Huấn luyện mô hình cải tiến - tập trung generalization ======
@st.cache_resource
def train_models_improved(history_tuple, ai_confidence_tuple, _cache_key):
    history = list(history_tuple)
    ai_confidence = list(ai_confidence_tuple)
    X, y = create_features_improved(history)
    
    if len(X) < 15:  # Tăng yêu cầu dữ liệu tối thiểu
        st.warning("Cần ít nhất 15 ván để huấn luyện mô hình ổn định.")
        return None

    try:
        # Kiểm tra đa dạng dữ liệu
        unique_classes = len(np.unique(y))
        if unique_classes < 2:
            st.warning("Dữ liệu không đa dạng. Cần cả kết quả Tài và Xỉu.")
            return None

        # Sử dụng mô hình đơn giản hơn để giảm overfitting
        from sklearn.linear_model import LogisticRegressionCV
        
        # Cross-validation cho time series
        tscv = TimeSeriesSplit(n_splits=min(4, len(X)//5))
        
        # Model đơn giản với regularization
        lr = LogisticRegressionCV(
            cv=tscv, 
            random_state=42,
            max_iter=1000,
            class_weight='balanced'  # Cân bằng class imbalance
        )
        
        # Random Forest với parameters giảm overfitting
        rf = RandomForestClassifier(
            n_estimators=30,  # Giảm số cây
            max_depth=5,      # Giới hạn độ sâu
            min_samples_split=10,
            random_state=42,
            class_weight='balanced'
        )
        
        # Huấn luyện với sample_weight nhẹ hơn
        recent_weight = np.linspace(0.3, 1.0, len(y))  # Giảm trọng số gần đây
        combined_weight = recent_weight * np.array(ai_confidence[:len(y)]) if len(ai_confidence) >= len(y) else recent_weight
        
        # Huấn luyện các model
        lr.fit(X, y, sample_weight=combined_weight)
        rf.fit(X, y, sample_weight=combined_weight)
        
        # Voting đơn giản thay vì stacking phức tạp
        voting = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf)],
            voting='soft'
        )
        voting.fit(X, y)
        
        # Đánh giá
        if len(X) > 20:
            # Chia train/test theo thời gian
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            voting.fit(X_train, y_train)
            acc = accuracy_score(y_test, voting.predict(X_test))
            st.info(f"Độ chính xác kiểm tra: {acc:.2%}")
            
            # Kiểm tra overfitting
            train_acc = accuracy_score(y_train, voting.predict(X_train))
            if train_acc - acc > 0.3:  # Chênh lệch lớn -> overfitting
                st.warning("Mô hình có thể bị overfitting. Kết quả dự đoán cần thận trọng.")
        
        return voting

    except Exception as e:
        st.error(f"Lỗi huấn luyện: {str(e)}")
        return None

# ====== Hàm dự đoán cải tiến ======
def predict_next_improved(models, history):
    if len(history) < 5 or models is None:  # Giảm window requirement
        return None, None

    try:
        # Tạo features mới
        X, _ = create_features_improved(history)
        latest = X[-1:].reshape(1, -1) if len(X) > 0 else None
        
        if latest is None:
            return None, None
            
        model_prob = models.predict_proba(latest)[0][1]
        pattern_score = pattern_detector_improved(history)
        
        # Kết hợp cân bằng hơn, ưu tiên pattern khi có streak rõ rệt
        recent_tai_ratio = sum(1 for x in history[-5:] if x == "Tài") / 5
        if abs(recent_tai_ratio - 0.5) > 0.4:  # Nghiêng hẳn 1 phía
            final_score = 0.4 * model_prob + 0.6 * pattern_score
        else:
            final_score = 0.6 * model_prob + 0.4 * pattern_score
            
        return {
            "Model Probability": model_prob, 
            "Pattern Analysis": pattern_score,
            "Recent Balance": recent_tai_ratio
        }, final_score
        
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
    with st.spinner("Đang huấn luyện các mô hình cải tiến..."):
        cache_key = str(len(st.session_state.history)) + str(st.session_state.history[-10:])
        st.session_state.models = train_models_improved(tuple(st.session_state.history), tuple(st.session_state.ai_confidence), cache_key)
    if st.session_state.models is not None:
        st.success("✅ Huấn luyện thành công với phiên bản cải tiến!")

# Dự đoán
if len(st.session_state.history) >= 5:
    if st.session_state.models is None:
        st.info("Vui lòng huấn luyện mô hình trước.")
    else:
        preds, final_score = predict_next_improved(st.session_state.models, st.session_state.history)
        if preds:
            st.session_state.ai_last_pred = "Tài" if final_score >= 0.5 else "Xỉu"
            st.subheader(f"🎯 Dự đoán chung: **{st.session_state.ai_last_pred}** ({final_score:.2%})")
            st.caption("Tổng hợp từ Model + Pattern Analysis (phiên bản cải tiến):")
            for k, v in preds.items():
                if k == "Recent Balance":
                    st.write(f"**{k}** → {v:.2%} Tài trong 5 ván gần nhất")
                else:
                    st.write(f"**{k}** → {v:.2%}")
            
            # Thêm phân tích streak
            recent = st.session_state.history[-5:]
            current_streak = 1
            for i in range(1, len(recent)):
                if recent[i] == recent[i-1]:
                    current_streak += 1
                else:
                    break
            if current_streak >= 3:
                st.info(f"🔍 Đang có chuỗi {recent[0]} {current_streak} ván liên tiếp - Pattern detector đang xem xét khả năng đảo chiều")
else:
    st.warning("Cần ít nhất 5 ván để bắt đầu dự đoán.")

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

# Thông tin về cải tiến
st.sidebar.markdown("""
### 🚀 Phiên bản Cải Tiến
**Chống Overfitting:**
- Features thống kê đa dạng
- Regularization mạnh
- Model đơn giản hơn
- Time-series validation

**Pattern Detection Thông Minh:**
- Phát hiện chuỗi dài
- Mean reversion logic
- Cân bằng tỷ lệ lịch sử
""")
