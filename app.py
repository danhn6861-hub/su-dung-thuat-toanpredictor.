import streamlit as st
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import logging
import time

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- KHÔNG CẦN EASYOCR VÀ CẮT ẢNH NỮA ---

# Hàm tính SuperTrend (không đổi)
def calculate_supertrend(highs, lows, closes, period=10, multiplier=3):
    try:
        atr = ta.atr(highs, lows, closes, length=period)
        hl2 = (highs + lows) / 2
        upper = hl2 + (multiplier * atr)
        lower = hl2 - (multiplier * atr)
        return upper, lower 
    except Exception as e:
        logger.error(f"Error in calculate_supertrend: {e}")
        return None, None

# Hàm Huấn luyện Mô hình ML (Sử dụng dữ liệu nhập thủ công)
@st.cache_data
def train_model(entry_price, supertrend_val, ema200_val, rsi_val, macd_val):
    """Tạo dữ liệu giả lập dựa trên giá nhập thủ công để huấn luyện mô hình ML."""
    
    if entry_price is None or entry_price <= 0:
        logger.error("Giá Entry bị thiếu hoặc không hợp lệ.")
        # Trả về các giá trị None/default nếu không thể huấn luyện
        return None, 0.5, 0, None, None, None, None

    np.random.seed(42)
    num_candles = 200 
    
    # Sử dụng giá trị cơ sở (Entry Price) để tạo chuỗi lịch sử giả lập
    base_price = entry_price
    base_volume = 10000 
    
    # Tạo chuỗi giá lịch sử giả lập
    # Bắt đầu chuỗi giá xung quanh giá trị EMA200 (để có mô hình hợp lý)
    # Thêm nhiễu để tạo ra sự biến động tự nhiên
    closes = np.cumsum(np.random.normal(0, base_price * 0.005, num_candles - 1)) + ema200_val * 1.01
    
    # Đảm bảo nến cuối cùng là Entry Price 
    closes = np.append(closes, base_price)
    
    # Giả lập OHL (giả định nến đóng gần với giá trị)
    highs = closes * (1 + np.abs(np.random.normal(0, 0.005, num_candles)))
    lows = closes * (1 - np.abs(np.random.normal(0, 0.005, num_candles)))
    volumes = np.random.uniform(base_volume * 0.5, base_volume * 1.5, num_candles)
    
    # Đảm bảo nến cuối cùng sử dụng giá trị Entry
    highs[-1] = max(base_price * 1.002, base_price)
    lows[-1] = min(base_price * 0.998, base_price)

    df = pd.DataFrame({"high": highs, "low": lows, "close": closes, "volume": volumes})
    
    # TÍNH TOÁN CÁC CHỈ BÁO TỪ DỮ LIỆU GIẢ LẬP
    supertrend_series, _ = calculate_supertrend(df['high'], df['low'], df['close'])
    ema200_series = ta.ema(df['close'], length=200).fillna(method='bfill')
    rsi_series = ta.rsi(df['close'], length=14).fillna(50)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    macd_series = macd['MACD_12_26_9'].fillna(0)
    volatility_series = ta.stdev(df['close'], length=20).fillna(0)

    # ĐIỀU CHỈNH: Ghi đè giá trị cuối cùng của chuỗi giả lập bằng giá trị nhập thủ công
    # Điều này giúp các features price_diff_st và price_diff_ema trong ML khớp với giá trị người dùng nhập
    supertrend_series.iloc[-1] = supertrend_val
    ema200_series.iloc[-1] = ema200_val
    rsi_series.iloc[-1] = rsi_val
    macd_series.iloc[-1] = macd_val
    volatility_final = volatility_series.iloc[-1]

    # Chuẩn bị Dữ liệu cho ML
    features_df = pd.DataFrame({
        'price_diff_st': df['close'] - supertrend_series,
        'price_diff_ema': df['close'] - ema200_series,
        'rsi': rsi_series,
        'macd': macd_series,
        'volume_change': df['volume'].pct_change().fillna(0),
        'volatility': volatility_series
    })
    
    labels = (df['close'].pct_change().shift(-1) > 0).astype(int).iloc[:-1]
    features_df = features_df.iloc[:len(labels)]

    # Huấn luyện và Đánh giá Mô hình ML
    X_train, X_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.2, random_state=42)
    
    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    model = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), param_grid, cv=2)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    
    # Trả về mô hình, độ chính xác, volatility và giá trị chỉ báo người dùng nhập
    return model, acc, volatility_final, supertrend_val, ema200_val, rsi_val, macd_val

# Hàm quyết định giao dịch với tối ưu hóa ML
def decide_trade(entry, supertrend, ema200, rsi, macd_val, model_results):
    try:
        model, acc, volatility_final, _, _, _, _ = model_results
        
        if entry is None:
            return "Giá Entry (Price) không được nhập. Không thể ra lệnh."

        # Chuẩn bị feature hiện tại
        current_features = pd.DataFrame({
            'price_diff_st': [entry - supertrend],
            'price_diff_ema': [entry - ema200],
            'rsi': [rsi],
            'macd': [macd_val],
            'volume_change': [0],
            'volatility': [volatility_final]
        })
        
        # Đảm bảo các chỉ số đầu vào hợp lệ
        if not current_features.isna().all(axis=1).iloc[0]:
            pred = model.predict(current_features)[0]
            prob_win = model.predict_proba(current_features)[0][pred]
        else:
            return "Dữ liệu chỉ báo nhập vào không hợp lệ. Vui lòng kiểm tra lại các trường."
        
        # 6. Ra quyết định Giao dịch Tối ưu (Cải thiện logic và quản lý rủi ro)
        
        # Tín hiệu Mua (LONG)
        if pred == 1 and entry > supertrend and entry > ema200 and rsi > 50 and macd_val > 0:
            edge = prob_win - (1 - prob_win) + 0.1 * (volatility_final / entry)
            risk_pct = max(1, min(5, edge / (1 - prob_win) * 2)) if (1 - prob_win) > 0 else 3
            
            target = entry * (1 + 0.1 * prob_win + 0.05 * edge)
            stop = entry * (1 - 0.02 / (prob_win + 0.1))
            
            return f"LONG tại {entry:.2f} VNDC. Chốt lời tại {target:.2f} VNDC. Stop-loss tại {stop:.2f} VNDC. Rủi ro {risk_pct:.2f}% vốn. Tỉ lệ thắng ước tính: {prob_win*100:.2f}% (Accuracy backtest: {acc*100:.2f}%)."
            
        # Tín hiệu Bán (SHORT)
        elif pred == 0 and entry < supertrend and entry < ema200 and rsi < 50 and macd_val < 0:
            edge = (1 - prob_win) - prob_win + 0.1 * (volatility_final / entry)
            risk_pct = max(1, min(5, edge / prob_win * 2)) if prob_win > 0 else 3
            
            target = entry * (1 - 0.1 * (1 - prob_win) - 0.05 * edge)
            stop = entry * (1 + 0.02 / (1 - prob_win + 0.1))
            
            return f"SHORT tại {entry:.2f} VNDC. Chốt lời tại {target:.2f} VNDC. Stop-loss tại {stop:.2f} VNDC. Rủi ro {risk_pct:.2f}% vốn. Tỉ lệ thắng ước tính: {(1-prob_win)*100:.2f}% (Accuracy backtest: {acc*100:.2f}%)."
        
        # Tín hiệu Chờ Đợi
        else:
            return f"CHỜ ĐỢI. Không có tín hiệu mạnh; thị trường sideway. Tỉ lệ thắng thấp (Dự đoán ML: {prob_win*100:.2f}%). Accuracy backtest: {acc*100:.2f}%."
            
    except Exception as e:
        logger.error(f"Error in decide_trade: {e}")
        return "Lỗi phân tích quyết định. Thử lại."
        
# 7. Giao diện Streamlit
st.set_page_config(page_title="AI Trading Analyzer Pro", layout="wide")
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        font-size: 1.2em;
        font-weight: bold;
    }
    .stSuccess, .stError, .stWarning {
        padding: 20px;
        border-radius: 10px;
        font-size: 1.1em;
    }
    .main-title {
        color: #007bff;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    <h1 class="main-title">🤖 AI Trading Analyzer Pro (Nhập Tay & ML)</h1>
    <p style='text-align: center; color: gray;'>Phân tích dữ liệu chỉ báo thủ công để đưa ra tín hiệu giao dịch tối ưu, bỏ qua lỗi OCR.</p>
""", unsafe_allow_html=True)

st.sidebar.title("Cài Đặt")
st.sidebar.info("Vui lòng nhập các chỉ số kỹ thuật hiện tại của cặp tiền (ví dụ: USELESS/VNDC) để AI phân tích.")

st.subheader("📝 Nhập Dữ Liệu Chỉ Báo Hiện Tại")

# Vùng nhập liệu thủ công
col_input1, col_input2 = st.columns(2)

with col_input1:
    # Đặt giá trị mặc định theo ảnh chụp màn hình cuối cùng (9,087,938)
    entry_price = st.number_input("Giá Entry Hiện Tại (Price)", min_value=0.0, format="%.4f", value=9087938.00)
    supertrend_val = st.number_input("Giá trị SuperTrend (Ví dụ: 9150000.00)", min_value=0.0, format="%.4f", value=9150000.00)
    rsi_val = st.number_input("Giá trị RSI (0 - 100)", min_value=0.0, max_value=100.0, format="%.2f", value=45.0)

with col_input2:
    ema200_val = st.number_input("Giá trị EMA200 (Ví dụ: 8900000.00)", min_value=0.0, format="%.4f", value=8900000.00)
    macd_val = st.number_input("Giá trị MACD (Histogram/MACD Line)", format="%.4f", value=-1000.0)
    
if st.button("🚀 PHÂN TÍCH NÂNG CAO & RA LỆNH", type="primary"):
    
    if entry_price <= 0 or supertrend_val <= 0 or ema200_val <= 0:
        st.error("Giá Entry, SuperTrend và EMA200 phải lớn hơn 0.")
    else:
        progress_bar = st.progress(0, text="Đang xử lý...")
        
        # --- PHASE 1: ML Training and Analysis ---
        with st.spinner("Đang huấn luyện mô hình ML và tính toán độ biến động..."):
            time.sleep(0.5)
            progress_bar.progress(50, text="Đang huấn luyện mô hình ML...")
            
            # Huấn luyện mô hình (chỉ chạy 1 lần)
            # Truyền tất cả các giá trị nhập vào để mô hình tạo dữ liệu giả lập khớp
            model_results = train_model(entry_price, supertrend_val, ema200_val, rsi_val, macd_val)
            
            # Ra quyết định
            decision = decide_trade(entry_price, supertrend_val, ema200_val, rsi_val, macd_val, model_results)
            progress_bar.progress(100)
        
        st.markdown("---")
        st.subheader("🎯 TÍN HIỆU GIAO DỊCH TỪ AI")
        
        # Hiển thị các chỉ báo đã được tính toán/sử dụng
        model, acc, volatility_final, supertrend_final, ema2200_final, rsi_final, macd_final = model_results
        
        st.markdown(f"""
        <div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 15px;'>
            **Dữ liệu Phân Tích (Nhập vào):**<br>
            Giá Entry: **{entry_price:.2f}** VNDC<br>
            SuperTrend: **{supertrend_val:.2f}**<br>
            EMA200: **{ema200_val:.2f}**<br>
            RSI: **{rsi_val:.2f}**<br>
            MACD: **{macd_val:.4f}**<br>
            Độ biến động (Tính toán): **{volatility_final:.2f}**
        </div>
        """, unsafe_allow_html=True)
        
        if "LONG" in decision:
            st.success(f"✅ TÍN HIỆU MUA (LONG)")
            st.markdown(f"**{decision}**")
        elif "SHORT" in decision:
            st.error(f"🔴 TÍN HIỆU BÁN (SHORT)")
            st.markdown(f"**{decision}**")
        else:
            st.warning(f"🟡 TÍN HIỆU CHỜ ĐỢI")
            st.markdown(f"**{decision}**")
        
        st.info("⚠️ Lưu ý: Tín hiệu này dựa trên dữ liệu nhập thủ công và mô hình ML dựa trên dữ liệu giả lập lịch sử. Đây không phải lời khuyên tài chính.")
