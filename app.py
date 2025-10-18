import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import random

st.set_page_config(page_title="AI Tài/Xỉu - Chiến Thuật Thông Minh", layout="wide")

# ====== TRIẾT LÝ CHIẾN THUẬT ======
"""
TRIẾT LÝ CHIẾN THUẬT THÔNG MINH:
1. KHÔNG điều chỉnh trọng số dựa trên vài ván thắng/thua ngắn hạn
2. CHỈ thay đổi chiến lược khi có bằng chứng THỐNG KÊ đủ mạnh
3. ƯU TIÊN sự ỔN ĐỊNH thay vì tối ưu hóa liên tục
4. PHÂN BIỆT rõ may mắn ngắn hạn vs kỹ năng thực sự
"""

# ====== KHỞI TẠO TRẠNG THÁI THÔNG MINH ======
if "history" not in st.session_state:
    st.session_state.history = []
if "strategic_memory" not in st.session_state:
    st.session_state.strategic_memory = {
        'model_performance': {'wins': 0, 'total': 0, 'confidence': 0.5},
        'pattern_performance': {'wins': 0, 'total': 0, 'confidence': 0.5},
        'current_phase': 'balanced',  # balanced, tai_streak, xiu_streak, volatile
        'phase_duration': 0,
        'strategic_weights': {'model': 0.6, 'pattern': 0.4},
        'last_weight_adjustment': 0,
        'performance_tracking': []
    }
if "models" not in st.session_state:
    st.session_state.models = None
if "ai_last_pred" not in st.session_state:
    st.session_state.ai_last_pred = None
if "last_prediction_details" not in st.session_state:
    st.session_state.last_prediction_details = None
if "undo_stack" not in st.session_state:
    st.session_state.undo_stack = []

# ====== PHÂN TÍCH THỐNG KÊ THÔNG MINH ======
def analyze_market_phase(history):
    """Phân tích phase thị trường với độ tin cậy thống kê"""
    if len(history) < 20:
        return 'balanced', 0.5
    
    # Phân tích xu hướng ngắn hạn (5 ván)
    short_term = history[-5:] if len(history) >= 5 else history
    tai_short = sum(1 for x in short_term if x == "Tài") / len(short_term)
    
    # Phân tích xu hướng trung hạn (15 ván)
    med_term = history[-15:] if len(history) >= 15 else history
    tai_med = sum(1 for x in med_term if x == "Tài") / len(med_term)
    
    # Phân tích biến động
    changes = sum(1 for i in range(1, len(short_term)) if short_term[i] != short_term[i-1])
    volatility = changes / (len(short_term) - 1) if len(short_term) > 1 else 0.5
    
    # Xác định phase với ngưỡng thống kê
    if abs(tai_med - 0.5) < 0.2 and volatility > 0.6:
        return 'volatile', volatility
    elif tai_med > 0.7:
        return 'tai_streak', tai_med
    elif tai_med < 0.3:
        return 'xiu_streak', 1 - tai_med
    else:
        return 'balanced', 0.5

def should_adjust_weights(strategic_memory, current_phase, phase_confidence):
    """Quyết định THÔNG MINH có nên điều chỉnh trọng số không"""
    memory = strategic_memory
    
    # LUẬT 1: Không điều chỉnh quá thường xuyên
    games_since_last_adjust = len(st.session_state.history) - memory['last_weight_adjustment']
    if games_since_last_adjust < 10:  # Tối thiểu 10 ván giữa các lần điều chỉnh
        return False, "Điều chỉnh quá gần nhau"
    
    # LUẬT 2: Cần đủ dữ liệu thống kê
    min_games_for_adjustment = 30
    if len(st.session_state.history) < min_games_for_adjustment:
        return False, f"Cần ít nhất {min_games_for_adjustment} ván"
    
    # LUẬT 3: Chênh lệch hiệu suất phải đủ lớn và có ý nghĩa thống kê
    model_perf = memory['model_performance']
    pattern_perf = memory['pattern_performance']
    
    if model_perf['total'] < 20 or pattern_perf['total'] < 20:
        return False, "Chưa đủ dữ liệu đánh giá hiệu suất"
    
    model_win_rate = model_perf['wins'] / model_perf['total']
    pattern_win_rate = pattern_perf['wins'] / pattern_perf['total']
    performance_gap = abs(model_win_rate - pattern_win_rate)
    
    # Ngưỡng chênh lệch hiệu suất để điều chỉnh (15%)
    if performance_gap < 0.15:
        return False, f"Chênh lệch hiệu suất {performance_gap:.1%} quá nhỏ"
    
    # LUẬT 4: Phase thị trường ảnh hưởng đến quyết định
    if current_phase == 'volatile' and phase_confidence > 0.7:
        # Trong phase biến động, ưu tiên pattern detection
        return True, "Phase biến động - Ưu tiên pattern"
    elif current_phase in ['tai_streak', 'xiu_streak'] and phase_confidence > 0.7:
        # Trong phase trend rõ, ưu tiên model
        return True, "Phase trend rõ - Ưu tiên model"
    elif performance_gap > 0.2:  # Chênh lệch rất lớn
        return True, f"Chênh lệch hiệu suất lớn: {performance_gap:.1%}"
    
    return False, "Không đủ điều kiện điều chỉnh"

def calculate_strategic_weights(strategic_memory, current_phase):
    """Tính toán trọng số CHIẾN LƯỢC thay vì tự động"""
    memory = strategic_memory
    model_perf = memory['model_performance']
    pattern_perf = memory['pattern_performance']
    
    # Tính win rate với độ tin cậy
    model_win_rate = model_perf['wins'] / model_perf['total'] if model_perf['total'] > 0 else 0.5
    pattern_win_rate = pattern_perf['wins'] / pattern_perf['total'] if pattern_perf['total'] > 0 else 0.5
    
    # Base weights dựa trên hiệu suất
    total_performance = model_win_rate + pattern_win_rate
    if total_performance > 0:
        base_model_weight = model_win_rate / total_performance
        base_pattern_weight = pattern_win_rate / total_performance
    else:
        base_model_weight, base_pattern_weight = 0.6, 0.4
    
    # Điều chỉnh theo phase thị trường
    if current_phase == 'volatile':
        # Phase biến động: pattern detection quan trọng hơn
        final_model_weight = base_model_weight * 0.7
        final_pattern_weight = base_pattern_weight * 1.3
    elif current_phase in ['tai_streak', 'xiu_streak']:
        # Phase trend: model quan trọng hơn
        final_model_weight = base_model_weight * 1.3
        final_pattern_weight = base_pattern_weight * 0.7
    else:
        # Phase cân bằng
        final_model_weight = base_model_weight
        final_pattern_weight = base_pattern_weight
    
    # Chuẩn hóa và đảm bảo trọng số hợp lý
    total = final_model_weight + final_pattern_weight
    model_weight = max(0.3, min(0.8, final_model_weight / total))
    pattern_weight = max(0.2, min(0.7, final_pattern_weight / total))
    
    # Chuẩn hóa lần cuối
    total = model_weight + pattern_weight
    return {
        'model': model_weight / total,
        'pattern': pattern_weight / total
    }

# ====== HỆ THỐNG DỰ ĐOÁN CHIẾN LƯỢC ======
def strategic_prediction_system(models, history):
    """Hệ thống dự đoán với tư duy chiến thuật"""
    if len(history) < 5 or models is None:
        return None, None, "insufficient_data"
    
    try:
        # Phân tích phase thị trường hiện tại
        current_phase, phase_confidence = analyze_market_phase(history)
        
        # Quyết định chiến lược
        should_adjust, reason = should_adjust_weights(
            st.session_state.strategic_memory, current_phase, phase_confidence
        )
        
        if should_adjust:
            new_weights = calculate_strategic_weights(st.session_state.strategic_memory, current_phase)
            st.session_state.strategic_memory['strategic_weights'] = new_weights
            st.session_state.strategic_memory['last_weight_adjustment'] = len(history)
            st.session_state.strategic_memory['last_adjustment_reason'] = reason
        
        # Dự đoán cơ bản
        X, _ = create_features_improved(history)
        if len(X) == 0:
            return None, None, "insufficient_data"
            
        latest = X[-1:].reshape(1, -1)
        model_prob = models.predict_proba(latest)[0][1]
        pattern_prob = intelligent_pattern_detector(history, current_phase)
        
        # Áp dụng trọng số chiến lược
        weights = st.session_state.strategic_memory['strategic_weights']
        final_score = (weights['model'] * model_prob + 
                      weights['pattern'] * pattern_prob)
        
        prediction_details = {
            "Model Probability": model_prob,
            "Pattern Analysis": pattern_prob,
            "Market Phase": current_phase,
            "Phase Confidence": phase_confidence,
            "Strategic Weights": weights.copy(),
            "Adjustment Recommended": should_adjust,
            "Adjustment Reason": reason if should_adjust else "Giữ nguyên chiến lược"
        }
        
        return prediction_details, final_score, "strategic"
        
    except Exception as e:
        st.error(f"Lỗi hệ thống chiến thuật: {str(e)}")
        return None, None, "error"

def intelligent_pattern_detector(history, market_phase):
    """Phát hiện pattern thông minh theo phase thị trường"""
    if len(history) < 5:
        return 0.5
    
    # Phân tích cơ bản
    short_term = history[-5:] if len(history) >= 5 else history
    med_term = history[-10:] if len(history) >= 10 else history
    
    tai_short = sum(1 for x in short_term if x == "Tài") / len(short_term)
    tai_med = sum(1 for x in med_term if x == "Tài") / len(med_term)
    
    # Điều chỉnh logic theo phase
    if market_phase == 'volatile':
        # Phase biến động: mean reversion mạnh
        if tai_short > 0.7:
            return 0.3  # Thiên về Xỉu sau nhiều Tài
        elif tai_short < 0.3:
            return 0.7  # Thiên về Tài sau nhiều Xỉu
        else:
            return 0.5
            
    elif market_phase in ['tai_streak', 'xiu_streak']:
        # Phase trend: follow trend
        return tai_med  # Theo xu hướng trung hạn
        
    else:
        # Phase cân bằng: kết hợp
        streak_length = 1
        for i in range(2, min(6, len(history)) + 1):
            if history[-i] == history[-1]:
                streak_length += 1
            else:
                break
                
        if streak_length >= 3:
            # Chuỗi dài -> mean reversion
            return 0.4 if history[-1] == "Tài" else 0.6
        else:
            return (tai_short * 0.6 + tai_med * 0.4)

# ====== CẬP NHẬT HIỆU SUẤT THÔNG MINH ======
def update_strategic_performance(actual_result, prediction_details):
    """Cập nhật hiệu suất với sự thận trọng"""
    memory = st.session_state.strategic_memory
    predicted_tai = prediction_details['Model Probability'] > 0.5
    pattern_tai = prediction_details['Pattern Analysis'] > 0.5
    
    actual_tai = (actual_result == "Tài")
    
    # Cập nhật hiệu suất model
    memory['model_performance']['total'] += 1
    if predicted_tai == actual_tai:
        memory['model_performance']['wins'] += 1
    
    # Cập nhật hiệu suất pattern
    memory['pattern_performance']['total'] += 1
    if pattern_tai == actual_tai:
        memory['pattern_performance']['wins'] += 1
    
    # Cập nhật phase tracking
    current_phase, _ = analyze_market_phase(st.session_state.history)
    if current_phase == memory['current_phase']:
        memory['phase_duration'] += 1
    else:
        memory['current_phase'] = current_phase
        memory['phase_duration'] = 1
    
    # Lưu tracking hiệu suất
    memory['performance_tracking'].append({
        'game': len(st.session_state.history),
        'model_win_rate': memory['model_performance']['wins'] / memory['model_performance']['total'],
        'pattern_win_rate': memory['pattern_performance']['wins'] / memory['pattern_performance']['total'],
        'phase': current_phase
    })

# ====== CÁC HÀM CƠ BẢN ======
def create_features_improved(history, window=5):
    if len(history) < window + 1:
        return np.empty((0, window + 2)), np.empty((0,))
    
    X = []
    y = []
    
    for i in range(window, len(history)):
        base_features = [1 if x == "Tài" else 0 for x in history[i - window:i]]
        tai_count = sum(base_features)
        tai_ratio = tai_count / window
        
        changes = 0
        for j in range(1, len(base_features)):
            if base_features[j] != base_features[j-1]:
                changes += 1
        change_ratio = changes / (window - 1) if window > 1 else 0
        
        combined_features = base_features + [tai_ratio, change_ratio]
        X.append(combined_features)
        y.append(1 if history[i] == "Tài" else 0)
    
    return np.array(X), np.array(y)

@st.cache_resource
def train_models_improved(history_tuple, _cache_key):
    history = list(history_tuple)
    X, y = create_features_improved(history)
    
    if len(X) < 15:
        st.warning("Cần ít nhất 15 ván để huấn luyện mô hình ổn định.")
        return None

    try:
        tscv = TimeSeriesSplit(n_splits=min(4, len(X)//5))
        
        lr = LogisticRegression(C=0.5, random_state=42, max_iter=1000)
        rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
        
        voting = VotingClassifier(estimators=[('lr', lr), ('rf', rf)], voting='soft')
        voting.fit(X, y)
        
        return voting

    except Exception as e:
        st.error(f"Lỗi huấn luyện: {str(e)}")
        return None

# ====== HÀM THÊM KẾT QUẢ ======
def add_result(result):
    if result not in ["Tài", "Xỉu"]:
        st.error(f"Kết quả không hợp lệ: {result}")
        return
    
    # Lưu trạng thái hiện tại để undo
    st.session_state.undo_stack.append({
        'history': st.session_state.history.copy(),
        'strategic_memory': st.session_state.strategic_memory.copy(),
        'ai_last_pred': st.session_state.ai_last_pred,
        'last_prediction_details': st.session_state.last_prediction_details
    })
    
    # Thêm kết quả mới
    st.session_state.history.append(result)
    
    # Cập nhật hiệu suất nếu có dự đoán trước đó
    if st.session_state.last_prediction_details is not None:
        update_strategic_performance(result, st.session_state.last_prediction_details)

# ====== HÀM UNDO ======
def undo_last():
    if st.session_state.undo_stack:
        last_state = st.session_state.undo_stack.pop()
        st.session_state.history = last_state['history']
        st.session_state.strategic_memory = last_state['strategic_memory']
        st.session_state.ai_last_pred = last_state['ai_last_pred']
        st.session_state.last_prediction_details = last_state['last_prediction_details']

# ====== GIAO DIỆN CHIẾN LƯỢC ======
st.title("🎯 AI Tài/Xỉu - Chiến Thuật Thông Minh & Ổn Định")

# Hiển thị trạng thái chiến lược
col1, col2, col3, col4 = st.columns(4)
with col1:
    phase = st.session_state.strategic_memory['current_phase']
    phase_duration = st.session_state.strategic_memory['phase_duration']
    st.metric("📊 Phase Thị Trường", phase, delta=f"{phase_duration} ván")
with col2:
    model_perf = st.session_state.strategic_memory['model_performance']
    model_win_rate = model_perf['wins'] / model_perf['total'] if model_perf['total'] > 0 else 0
    st.metric("🤖 Model Win Rate", f"{model_win_rate:.1%}")
with col3:
    pattern_perf = st.session_state.strategic_memory['pattern_performance']
    pattern_win_rate = pattern_perf['wins'] / pattern_perf['total'] if pattern_perf['total'] > 0 else 0
    st.metric("🔍 Pattern Win Rate", f"{pattern_win_rate:.1%}")
with col4:
    weights = st.session_state.strategic_memory['strategic_weights']
    adjustment_games = len(st.session_state.history) - st.session_state.strategic_memory.get('last_weight_adjustment', 0)
    st.metric("⚖️ Chiến Lược", f"M:{weights['model']:.0%} P:{weights['pattern']:.0%}", delta=f"{adjustment_games}ván")

# Hiển thị lịch sử gần đây
st.subheader("📊 Lịch sử gần đây")
if st.session_state.history:
    # Hiển thị 20 kết quả gần nhất
    recent_history = st.session_state.history[-20:]
    history_text = " → ".join(recent_history)
    st.write(history_text)
else:
    st.info("Chưa có dữ liệu. Hãy bắt đầu nhập kết quả!")

# Biểu đồ hiệu suất
if st.session_state.strategic_memory['performance_tracking']:
    st.subheader("📈 Biểu Đồ Hiệu Suất Chiến Thuật")
    tracking = st.session_state.strategic_memory['performance_tracking']
    
    games = [x['game'] for x in tracking]
    model_rates = [x['model_win_rate'] for x in tracking]
    pattern_rates = [x['pattern_win_rate'] for x in tracking]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(games, model_rates, label='Model Win Rate', linewidth=2)
    ax.plot(games, pattern_rates, label='Pattern Win Rate', linewidth=2)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Ngưỡng 50%')
    ax.set_ylabel("Tỷ lệ thắng")
    ax.set_xlabel("Số ván")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Nhập liệu
st.divider()
st.subheader("🎮 Nhập kết quả")

col_tai, col_xiu, col_undo = st.columns([1, 1, 1])
with col_tai:
    if st.button("🎲 NHẬP TÀI", key="add_tai", use_container_width=True):
        add_result("Tài")
        st.success("Đã thêm Tài!")
        st.rerun()
        
with col_xiu:
    if st.button("🎲 NHẬP XỈU", key="add_xiu", use_container_width=True):
        add_result("Xỉu")
        st.success("Đã thêm Xỉu!")
        st.rerun()

with col_undo:
    if st.button("↩️ UNDO", key="undo", use_container_width=True):
        undo_last()
        st.success("Đã hoàn tác!")
        st.rerun()

# Huấn luyện và dự đoán
st.divider()
st.subheader("🤖 Huấn luyện AI")

if st.button("🚀 Huấn luyện Hệ Thống", key="train_system", use_container_width=True):
    if len(st.session_state.history) < 15:
        st.warning("Cần ít nhất 15 ván để huấn luyện!")
    else:
        with st.spinner("Đang huấn luyện với chiến lược ổn định..."):
            cache_key = str(len(st.session_state.history)) + str(st.session_state.history[-10:])
            st.session_state.models = train_models_improved(tuple(st.session_state.history), cache_key)
        if st.session_state.models is not None:
            st.success("✅ Hệ thống đã sẵn sàng với chiến lược thông minh!")

# Dự đoán chiến lược
st.divider()
st.subheader("🔮 Dự đoán")

if len(st.session_state.history) >= 5:
    if st.session_state.models is None:
        st.info("Vui lòng huấn luyện mô hình trước khi dự đoán.")
    else:
        pred_details, final_score, strategy = strategic_prediction_system(
            st.session_state.models, st.session_state.history
        )
        
        if pred_details:
            st.session_state.ai_last_pred = "Tài" if final_score >= 0.5 else "Xỉu"
            st.session_state.last_prediction_details = pred_details
            
            confidence = final_score if st.session_state.ai_last_pred == "Tài" else 1 - final_score
            
            st.subheader(f"🎯 Dự Đoán: **{st.session_state.ai_last_pred}** ({confidence:.1%} confidence)")
            
            # Hiển thị phân tích chiến lược
            st.write("**Phân tích chiến thuật:**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📈 Market Phase", f"{pred_details['Market Phase']}", 
                         delta=f"{pred_details['Phase Confidence']:.0%} confidence")
                st.metric("🤖 Model", f"{pred_details['Model Probability']:.1%}")
                st.metric("🔍 Pattern", f"{pred_details['Pattern Analysis']:.1%}")
            
            with col2:
                weights = pred_details['Strategic Weights']
                st.metric("⚖️ Strategic Weights", f"Model: {weights['model']:.0%}")
                st.metric("", f"Pattern: {weights['pattern']:.0%}")
                
                if pred_details['Adjustment Recommended']:
                    st.warning(f"🔧 Đề xuất điều chỉnh: {pred_details['Adjustment Reason']}")
                else:
                    st.info(f"✅ {pred_details['Adjustment Reason']}")
else:
    st.info("Cần ít nhất 5 ván để bắt đầu dự đoán.")

# Panel chiến lược
st.sidebar.markdown("""
### 🧠 Triết Lý Chiến Thuật

**NGUYÊN TẮC VÀNG:**
- ✅ **Ổn định > Tối ưu hóa liên tục**
- ✅ **Thống kê > Cảm tính**
- ✅ **Kiên nhẫn > Vội vàng**

**LUẬT ĐIỀU CHỈNH:**
1. Tối thiểu 10 ván giữa các lần điều chỉnh
2. Cần ít nhất 30 ván để đánh giá hiệu suất  
3. Chênh lệch hiệu suất phải >15%
4. Phase thị trường phải rõ ràng (>70% confidence)

**CHIẾN LƯỢC THEO PHASE:**
- 📊 **Balanced**: Kết hợp cân bằng
- 📈 **Trend**: Ưu tiên model
- 📉 **Volatile**: Ưu tiên pattern
""")

# Hiển thị lịch sử điều chỉnh
if st.sidebar.checkbox("📋 Lịch sử Chiến thuật"):
    st.sidebar.write("**Hiệu suất hiện tại:**")
    st.sidebar.write(f"- Model: {st.session_state.strategic_memory['model_performance']['wins']}/{st.session_state.strategic_memory['model_performance']['total']}")
    st.sidebar.write(f"- Pattern: {st.session_state.strategic_memory['pattern_performance']['wins']}/{st.session_state.strategic_memory['pattern_performance']['total']}")
    
    if 'last_adjustment_reason' in st.session_state.strategic_memory:
        st.sidebar.write(f"**Lần điều chỉnh gần nhất:** {st.session_state.strategic_memory['last_adjustment_reason']}")

# Xóa lịch sử
if st.sidebar.checkbox("🗑️ Xóa toàn bộ lịch sử"):
    if st.sidebar.button("XÁC NHẬN XÓA", type="primary"):
        st.session_state.history = []
        st.session_state.strategic_memory = {
            'model_performance': {'wins': 0, 'total': 0, 'confidence': 0.5},
            'pattern_performance': {'wins': 0, 'total': 0, 'confidence': 0.5},
            'current_phase': 'balanced',
            'phase_duration': 0,
            'strategic_weights': {'model': 0.6, 'pattern': 0.4},
            'last_weight_adjustment': 0,
            'performance_tracking': []
        }
        st.session_state.models = None
        st.session_state.ai_last_pred = None
        st.session_state.last_prediction_details = None
        st.session_state.undo_stack = []
        st.sidebar.success("Đã xóa toàn bộ lịch sử!")
        st.rerun()
