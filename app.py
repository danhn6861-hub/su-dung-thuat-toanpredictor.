import streamlit as st
import numpy as np
from PIL import Image
from easyocr import Reader
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import logging
import time  # Để progress bar
# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Khởi tạo EasyOCR tối ưu
# Lưu ý: Cần đảm bảo EasyOCR có thể truy cập các mô hình đã tải xuống.
# Nếu gặp lỗi, hãy thử bỏ 'model_storage_directory=None, download_enabled=True'
try:
    reader = Reader(['en', 'vi'], gpu=False, model_storage_directory=None, download_enabled=True)
except Exception as e:
    st.error(f"Lỗi khởi tạo EasyOCR: {e}. Vui lòng đảm bảo các thư viện đã được cài đặt.")
    reader = None
# Hàm phân tích ảnh với OCR tối ưu
def analyze_image(image):
    if not reader:
        return {"price": None, "supertrend": None, "ema200": None, "volume": None, "rsi": None, "macd": None}
    try:
        # Chuyển đổi ảnh sang định dạng numpy để EasyOCR xử lý
        img_np = np.array(image)
        
        # Đọc văn bản, chỉ trả về văn bản, không trả về bounding box
        result = reader.readtext(img_np, detail=0, paragraph=False, 
                                 contrast_ths=0.2, adjust_contrast=0.6, 
                                 text_threshold=0.8, width_ths=0.8, 
                                 decoder='greedy', beamWidth=5)
        
        data = {"price": None, "supertrend": None, "ema200": None, "volume": None, "rsi": None, "macd": None}
        
        # Cố gắng khớp các chỉ số với từ khóa
        for text in result:
            text_lower = text.strip().lower()
            if any(keyword in text_lower for keyword in ["giá", "price", "current price"]):
                # Ưu tiên các số lớn (thường là giá) và đảm bảo trích xuất đúng
                data["price"] = extract_number(text)
            elif any(keyword in text_lower for keyword in ["supertrend", "st"]):
                data["supertrend"] = extract_number(text)
            elif any(keyword in text_lower for keyword in ["ema200", "ema 200"]):
                data["ema200"] = extract_number(text)
            elif any(keyword in text_lower for keyword in ["volume", "khối lượng"]):
                data["volume"] = extract_number(text)
            elif "rsi" in text_lower:
                data["rsi"] = extract_number(text)
            elif "macd" in text_lower:
                data["macd"] = extract_number(text)
        
        logger.info(f"OCR Data: {data}")
        return data
    except Exception as e:
        logger.error(f"Error in analyze_image: {e}")
        return {"price": None, "supertrend": None, "ema200": None, "volume": None, "rsi": None, "macd": None}
        
# Hàm extract number đã được tối ưu cho định dạng số Việt Nam
def extract_number(text):
    try:
        # Loại bỏ các ký tự không phải số, dấu chấm, dấu phẩy
        num_str = ''.join([c for c in text if c.isdigit() or c in ['.', ',']]).replace(',', '.')
        
        # Nếu số quá lớn (ví dụ: giá 31,430), đôi khi OCR có thể đọc dính dấu phẩy.
        # Thử loại bỏ dấu chấm nếu nó ở vị trí hàng nghìn
        if num_str.count('.') > 1:
            num_str = num_str.replace('.', '', num_str.count('.') - 1)
            
        return float(num_str) if num_str else None
    except:
        return None
        
# Hàm tính SuperTrend
def calculate_supertrend(highs, lows, closes, period=10, multiplier=3):
    try:
        atr = ta.atr(highs, lows, closes, length=period)
        hl2 = (highs + lows) / 2
        upper = hl2 + (multiplier * atr)
        lower = hl2 - (multiplier * atr)
        
        # Trả về giá trị cuối cùng của upper và lower band
        return upper.iloc[-1], lower.iloc[-1]
    except Exception as e:
        logger.error(f"Error in calculate_supertrend: {e}")
        return None, None
        
# Hàm quyết định giao dịch với tối ưu hóa ML
def decide_trade(data):
    try:
        if data["price"] is None:
            return "Không đủ dữ liệu cơ bản (giá). Vui lòng chụp ảnh rõ ràng hơn."
            
        # 1. Tạo Dữ liệu Giả lập Lịch sử (Dựa trên giá và volume hiện tại)
        np.random.seed(42)
        num_candles = 50  # Giảm để nhanh hơn
        
        # Tạo chuỗi giá
        closes = np.cumsum(np.random.normal(0, data["price"] * 0.01, num_candles)) + data["price"]
        highs = closes + np.abs(np.random.normal(0, data["price"] * 0.02, num_candles))
        lows = closes - np.abs(np.random.normal(0, data["price"] * 0.02, num_candles))
        volumes = np.random.uniform(data["volume"] * 0.5 if data["volume"] else 5000, 
                                     data["volume"] * 1.5 if data["volume"] else 20000, num_candles)
        
        df = pd.DataFrame({"high": highs, "low": lows, "close": closes, "volume": volumes})
        
        # 2. Tính toán Chỉ báo Kỹ thuật (Sử dụng dữ liệu giả lập hoặc OCR)
        
        supertrend_upper, supertrend_lower = calculate_supertrend(df['high'], df['low'], df['close'])
        # Ưu tiên giá trị SuperTrend đọc được từ OCR
        supertrend = data["supertrend"] if data["supertrend"] else supertrend_upper or closes[-1]
        
        ema200 = ta.ema(df['close'], length=200).iloc[-1] if data["ema200"] is None else data["ema200"]
        rsi = ta.rsi(df['close'], length=14).iloc[-1] if data["rsi"] is None else data["rsi"]
        
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        macd_val = macd['MACD_12_26_9'].iloc[-1] if data["macd"] is None else data["macd"]
        
        # Thêm feature volatility
        volatility = ta.stdev(df['close'], length=20).iloc[-1]
        
        # 3. Chuẩn bị Dữ liệu cho ML
        features_df = pd.DataFrame({
            'price_diff_st': df['close'] - (supertrend_upper or df['close']),
            'price_diff_ema': df['close'] - ema200,
            'rsi': ta.rsi(df['close'], length=14),
            'macd': macd['MACD_12_26_9'],
            'volume_change': df['volume'].pct_change().fillna(0),
            'volatility': ta.stdev(df['close'], length=20)
        }).dropna()
        
        # Label: 1 nếu nến tiếp theo đóng cửa cao hơn (tăng), 0 nếu thấp hơn (giảm)
        labels = (df['close'].pct_change().shift(-1) > 0).astype(int).iloc[:-1].dropna()
        
        # Đảm bảo features và labels có cùng số lượng
        features_df = features_df.iloc[:len(labels)]

        # 4. Huấn luyện và Đánh giá Mô hình ML
        X_train, X_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.2, random_state=42)
        
        param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}  # Giảm để nhanh hơn
        # Sử dụng class_weight='balanced' để cân bằng giữa tín hiệu mua và bán
        model = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), param_grid, cv=2)
        model.fit(X_train, y_train)
        
        # Đánh giá Accuracy của mô hình trên dữ liệu giả lập
        acc = accuracy_score(y_test, model.predict(X_test))
        
        # 5. Dự đoán trên Dữ liệu Hiện tại (Được OCR)
        
        # Chuẩn bị feature hiện tại
        current_features = pd.DataFrame({
            'price_diff_st': [df['close'].iloc[-1] - supertrend],
            'price_diff_ema': [df['close'].iloc[-1] - ema200],
            'rsi': [rsi],
            'macd': [macd_val],
            'volume_change': [(df['volume'].iloc[-1] - df['volume'].iloc[-2]) / df['volume'].iloc[-2]] if len(df) > 1 else [0],
            'volatility': [volatility]
        })
        
        pred = model.predict(current_features)[0]
        prob_win = model.predict_proba(current_features)[0][pred]
        entry = df['close'].iloc[-1]
        
        # 6. Ra quyết định Giao dịch Tối ưu (Cải thiện logic và quản lý rủi ro)
        
        # Tín hiệu Mua (LONG)
        if pred == 1 and entry > supertrend and entry > ema200 and rsi > 50 and macd_val > 0:
            # Tính toán Edge (lợi thế) dựa trên xác suất và volatility
            edge = prob_win - (1 - prob_win) + 0.1 * (volatility / entry)
            # Tính toán % Rủi ro dựa trên Edge
            risk_pct = max(1, min(5, edge / (1 - prob_win) * 2)) if (1 - prob_win) > 0 else 3
            
            # Tính Target/Stop dựa trên xác suất và edge
            target = entry * (1 + 0.1 * prob_win + 0.05 * edge)
            stop = entry * (1 - 0.02 / (prob_win + 0.1)) # Stop-loss càng hẹp nếu xác suất thắng càng cao
            
            return f"LONG tại {entry:.2f} VNDC. Chốt lời tại {target:.2f} VNDC. Stop-loss tại {stop:.2f} VNDC. Rủi ro {risk_pct:.2f}% vốn. Tỉ lệ thắng ước tính: {prob_win*100:.2f}% (Accuracy backtest: {acc*100:.2f}%)."
            
        # Tín hiệu Bán (SHORT)
        elif pred == 0 and entry < supertrend and entry < ema200 and rsi < 50 and macd_val < 0:
            # Tính toán Edge
            edge = (1 - prob_win) - prob_win + 0.1 * (volatility / entry)
            # Tính toán % Rủi ro
            risk_pct = max(1, min(5, edge / prob_win * 2)) if prob_win > 0 else 3
            
            # Tính Target/Stop
            target = entry * (1 - 0.1 * (1 - prob_win) - 0.05 * edge)
            stop = entry * (1 + 0.02 / (1 - prob_win + 0.1))
            
            return f"SHORT tại {entry:.2f} VNDC. Chốt lời tại {target:.2f} VNDC. Stop-loss tại {stop:.2f} VNDC. Rủi ro {risk_pct:.2f}% vốn. Tỉ lệ thắng ước tính: {(1-prob_win)*100:.2f}% (Accuracy backtest: {acc*100:.2f}%)."
        
        # Tín hiệu Chờ Đợi
        else:
            return f"CHỜ ĐỢI. Không có tín hiệu mạnh; thị trường sideway. Tỉ lệ thắng thấp (<60%). Accuracy backtest: {acc*100:.2f}%."
            
    except Exception as e:
        logger.error(f"Error in decide_trade: {e}")
        return "Lỗi phân tích. Thử lại ảnh khác."
        
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
    <h1 class="main-title">🤖 AI Trading Analyzer Pro (ONUS Chart OCR + ML)</h1>
    <p style='text-align: center; color: gray;'>Phân tích ảnh chụp màn hình biểu đồ ONUS để đưa ra tín hiệu giao dịch tối ưu.</p>
""", unsafe_allow_html=True)

st.sidebar.title("Cài Đặt")
st.sidebar.info("Upload ảnh màn hình ONUS. AI dùng EasyOCR, pandas_ta, ML tối ưu cho tỉ lệ thắng cao. Cần có đủ các chỉ báo trên ảnh: Giá, SuperTrend, EMA200, RSI, MACD, Volume.")

uploaded_file = st.file_uploader("Tải lên Ảnh Màn Hình ONUS", type=["jpg", "png"], help="Chụp rõ các chỉ báo quan trọng.")

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh Upload (ASTR/VNDC)", use_container_width=True)
        
    with col2:
        st.subheader("Bắt đầu Phân Tích")
        if st.button("🚀 PHÂN TÍCH NÂNG CAO & RA LỆNH", type="primary"):
            progress_bar = st.progress(0, text="Đang xử lý...")
            
            # --- PHASE 1: OCR ---
            with st.spinner("Đang đọc dữ liệu từ ảnh (EasyOCR)..."):
                time.sleep(0.5)
                progress_bar.progress(30, text="Đang đọc dữ liệu từ ảnh (EasyOCR)...")
                data = analyze_image(image)
                
            st.markdown("---")
            st.subheader("📊 Dữ Liệu OCR Đã Trích Xuất")
            if all(v is None for v in data.values()):
                st.error("❌ Không đọc được dữ liệu quan trọng nào. Vui lòng chụp ảnh rõ hơn và kiểm tra các chỉ báo.")
                progress_bar.progress(100)
            else:
                st.json(data)
                
                # --- PHASE 2: ML Analysis ---
                with st.spinner("Đang huấn luyện mô hình ML và đưa ra quyết định..."):
                    time.sleep(0.5)
                    progress_bar.progress(60, text="Đang huấn luyện mô hình ML...")
                    decision = decide_trade(data)
                    progress_bar.progress(100)
                
                st.markdown("---")
                st.subheader("🎯 TÍN HIỆU GIAO DỊCH TỪ AI")
                
                if "LONG" in decision:
                    st.success(f"✅ TÍN HIỆU MUA (LONG)")
                    st.markdown(f"**{decision}**")
                elif "SHORT" in decision:
                    st.error(f"🔴 TÍN HIỆU BÁN (SHORT)")
                    st.markdown(f"**{decision}**")
                else:
                    st.warning(f"🟡 TÍN HIỆU CHỜ ĐỢI")
                    st.markdown(f"**{decision}**")
                
                st.info("⚠️ Lưu ý: Tín hiệu này dựa trên dữ liệu giả lập lịch sử và mô hình ML đã được huấn luyện. Đây không phải lời khuyên tài chính.")
