import streamlit as st
import numpy as np
from PIL import Image
from easyocr import Reader
import pandas as pd
import pandas_ta as ta
# Đã sửa lỗi chính tả tại đây: model_model_selection -> model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import logging
import time

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo EasyOCR tối ưu (dùng st.cache_resource để chỉ khởi tạo 1 lần)
@st.cache_resource
def get_ocr_reader():
    try:
        # Cố gắng khởi tạo Reader. Cần đảm bảo các mô hình đã được tải xuống.
        reader = Reader(['en', 'vi'], gpu=False)
        return reader
    except Exception as e:
        st.error(f"Lỗi khởi tạo EasyOCR: {e}. Vui lòng đảm bảo các thư viện đã được cài đặt và các file model đã có.")
        return None

reader = get_ocr_reader()

# Hàm trích xuất số đã được tối ưu cho định dạng số Việt Nam
def extract_number(text):
    """Trích xuất số từ chuỗi văn bản, xử lý định dạng số lớn."""
    try:
        # Loại bỏ các ký tự không phải số, dấu chấm, dấu phẩy
        num_str = ''.join([c for c in text if c.isdigit() or c in ['.', ',']]).replace(',', '.')
        
        # Nếu số quá lớn (ví dụ: giá 31,430), đôi khi OCR có thể đọc dính dấu phẩy.
        # Thử loại bỏ dấu chấm nếu nó ở vị trí hàng nghìn
        if num_str.count('.') > 1:
            # Giữ lại dấu chấm cuối cùng (thường là thập phân) và loại bỏ các dấu chấm khác
            parts = num_str.split('.')
            if len(parts[-1]) != 3: # Nếu phần cuối không phải 3 số, có thể là số thập phân
                num_str = ''.join(parts[:-1]) + '.' + parts[-1]
            else:
                num_str = ''.join(parts) # Nếu là 31.430 -> 31430

        return float(num_str) if num_str else None
    except:
        return None

# Hàm cắt ảnh thông minh để cải thiện chất lượng OCR
def crop_image(image, crop_area):
    """Cắt ảnh theo vùng: 'price' (góc trên bên trái) hoặc 'indicators' (vùng dưới)."""
    width, height = image.size
    
    if crop_area == 'price':
        # Vùng giá (thường ở góc trên bên trái)
        # Tùy chỉnh theo ảnh ONUS: Giới hạn chiều rộng ở 1/4 và chiều cao ở 1/4
        left = 0
        top = 0
        right = width // 3
        bottom = height // 4
    elif crop_area == 'indicators':
        # Vùng RSI, MACD (thường ở 1/3 dưới cùng của màn hình)
        left = 0
        top = height * 2 // 3
        right = width
        bottom = height
    else:
        return image
    
    return image.crop((left, top, right, bottom))

# Hàm phân tích ảnh với OCR tối ưu
def analyze_image(image):
    if not reader:
        return {"price": None, "supertrend": None, "ema200": None, "volume": None, "rsi": None, "macd": None}
    
    data = {"price": None, "supertrend": None, "ema200": None, "volume": None, "rsi": None, "macd": None}

    # 1. OCR Vùng Giá (Tối ưu cho số lớn)
    img_price = crop_image(image, 'price')
    result_price = reader.readtext(np.array(img_price), detail=0, paragraph=False)
    
    for text in result_price:
        text_lower = text.strip().lower()
        num = extract_number(text)
        
        # Trích xuất Giá
        if num is not None and num > 1000: # Giả định giá trị lớn nhất là giá
             if data["price"] is None or num > data["price"]:
                data["price"] = num
        
        # Trích xuất SuperTrend và EMA200 (thường nằm gần giá)
        if data["supertrend"] is None and any(keyword in text_lower for keyword in ["supertrend", "st"]):
            data["supertrend"] = num
        if data["ema200"] is None and any(keyword in text_lower for keyword in ["ema200", "ema 200"]):
            data["ema200"] = num


    # 2. OCR Vùng Chỉ báo (RSI, MACD)
    img_indicators = crop_image(image, 'indicators')
    result_indicators = reader.readtext(np.array(img_indicators), detail=0, paragraph=False)
    
    for text in result_indicators:
        text_lower = text.strip().lower()
        num = extract_number(text)
        
        if "rsi" in text_lower and data["rsi"] is None:
            # RSI thường là số 2 chữ số
            if num is not None and 0 <= num <= 100:
                data["rsi"] = num
        
        # MACD thường có giá trị nhỏ, dương hoặc âm
        elif "macd" in text_lower and data["macd"] is None:
            data["macd"] = num

    # 3. OCR Toàn bộ ảnh (fallback và Volume)
    img_np_full = np.array(image)
    result_full = reader.readtext(img_np_full, detail=0, paragraph=False)

    for text in result_full:
        text_lower = text.strip().lower()
        num = extract_number(text)
        
        if data["volume"] is None and any(keyword in text_lower for keyword in ["volume", "khối lượng"]):
            data["volume"] = num
        
        # Fallback cho giá (nếu chưa tìm thấy)
        if data["price"] is None and num is not None and num > 1000:
            data["price"] = num
        
    logger.info(f"OCR Data: {data}")
    return data
        
# Hàm tính SuperTrend (không đổi)
def calculate_supertrend(highs, lows, closes, period=10, multiplier=3):
    try:
        atr = ta.atr(highs, lows, closes, length=period)
        hl2 = (highs + lows) / 2
        upper = hl2 + (multiplier * atr)
        lower = hl2 - (multiplier * atr)
        return upper.iloc[-1], lower.iloc[-1]
    except Exception as e:
        logger.error(f"Error in calculate_supertrend: {e}")
        return None, None

# Hàm Huấn luyện Mô hình ML (Tối ưu hóa: Dùng st.cache_data để huấn luyện 1 lần)
@st.cache_data
def train_model(data):
    """Tạo dữ liệu giả lập, tính toán features và labels, huấn luyện mô hình."""
    if data["price"] is None:
        return None, 0.5, 0 # Trả về mô hình None, acc thấp và 0 volatility nếu thiếu giá

    np.random.seed(42)
    num_candles = 100 # Tăng số lượng nến giả lập để mô hình học tốt hơn
    
    # Tạo chuỗi giá
    closes = np.cumsum(np.random.normal(0, data["price"] * 0.005, num_candles)) + data["price"]
    highs = closes + np.abs(np.random.normal(0, data["price"] * 0.01, num_candles))
    lows = closes - np.abs(np.random.normal(0, data["price"] * 0.01, num_candles))
    volumes = np.random.uniform(data["volume"] * 0.5 if data["volume"] else 5000, 
                                 data["volume"] * 1.5 if data["volume"] else 20000, num_candles)
    
    df = pd.DataFrame({"high": highs, "low": lows, "close": closes, "volume": volumes})
    
    # Tính toán Chỉ báo Kỹ thuật (Sử dụng dữ liệu giả lập)
    supertrend_upper, _ = calculate_supertrend(df['high'], df['low'], df['close'])
    ema200_series = ta.ema(df['close'], length=200).fillna(method='bfill')
    rsi_series = ta.rsi(df['close'], length=14).fillna(50)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    macd_series = macd['MACD_12_26_9'].fillna(0)
    volatility_series = ta.stdev(df['close'], length=20).fillna(0)

    # Chuẩn bị Dữ liệu cho ML
    features_df = pd.DataFrame({
        'price_diff_st': df['close'] - supertrend_upper,
        'price_diff_ema': df['close'] - ema200_series,
        'rsi': rsi_series,
        'macd': macd_series,
        'volume_change': df['volume'].pct_change().fillna(0),
        'volatility': volatility_series
    })
    
    # Label: 1 nếu nến tiếp theo đóng cửa cao hơn (tăng), 0 nếu thấp hơn (giảm)
    labels = (df['close'].pct_change().shift(-1) > 0).astype(int).iloc[:-1]
    
    # Đảm bảo features và labels có cùng số lượng
    features_df = features_df.iloc[:len(labels)]

    # Huấn luyện và Đánh giá Mô hình ML
    X_train, X_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.2, random_state=42)
    
    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    model = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), param_grid, cv=2)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    
    # Trả về mô hình, độ chính xác và volatility của cây nến cuối cùng
    return model, acc, volatility_series.iloc[-1], supertrend_upper

# Hàm quyết định giao dịch với tối ưu hóa ML
def decide_trade(data, model_acc_vol):
    try:
        model, acc, volatility, supertrend_upper_band = model_acc_vol
        
        if model is None:
             return "Không đủ dữ liệu cơ bản (giá). Vui lòng chụp ảnh rõ ràng hơn."

        # Ưu tiên giá trị OCR, nếu không có thì lấy giá trị giả lập cuối cùng
        supertrend = data["supertrend"] if data["supertrend"] else supertrend_upper_band.iloc[-1]
        ema200 = data["ema200"] if data["ema200"] is not None else data["price"] # Dùng giá nếu EMA không đọc được
        rsi = data["rsi"] if data["rsi"] is not None else 50
        macd_val = data["macd"] if data["macd"] is not None else 0

        entry = data["price"] # Giá entry phải là giá thực tế đọc được
        
        # Chuẩn bị feature hiện tại
        current_features = pd.DataFrame({
            'price_diff_st': [entry - supertrend],
            'price_diff_ema': [entry - ema200],
            'rsi': [rsi],
            'macd': [macd_val],
            'volume_change': [0], # Không tính volume change vì thiếu lịch sử
            'volatility': [volatility]
        })
        
        pred = model.predict(current_features)[0]
        prob_win = model.predict_proba(current_features)[0][pred]
        
        # 6. Ra quyết định Giao dịch Tối ưu (Cải thiện logic và quản lý rủi ro)
        
        # Tín hiệu Mua (LONG)
        if pred == 1 and entry > supertrend and entry > ema200 and rsi > 50 and macd_val > 0:
            edge = prob_win - (1 - prob_win) + 0.1 * (volatility / entry)
            risk_pct = max(1, min(5, edge / (1 - prob_win) * 2)) if (1 - prob_win) > 0 else 3
            
            target = entry * (1 + 0.1 * prob_win + 0.05 * edge)
            stop = entry * (1 - 0.02 / (prob_win + 0.1))
            
            return f"LONG tại {entry:.2f} VNDC. Chốt lời tại {target:.2f} VNDC. Stop-loss tại {stop:.2f} VNDC. Rủi ro {risk_pct:.2f}% vốn. Tỉ lệ thắng ước tính: {prob_win*100:.2f}% (Accuracy backtest: {acc*100:.2f}%)."
            
        # Tín hiệu Bán (SHORT)
        elif pred == 0 and entry < supertrend and entry < ema200 and rsi < 50 and macd_val < 0:
            edge = (1 - prob_win) - prob_win + 0.1 * (volatility / entry)
            risk_pct = max(1, min(5, edge / prob_win * 2)) if prob_win > 0 else 3
            
            target = entry * (1 - 0.1 * (1 - prob_win) - 0.05 * edge)
            stop = entry * (1 + 0.02 / (1 - prob_win + 0.1))
            
            return f"SHORT tại {entry:.2f} VNDC. Chốt lời tại {target:.2f} VNDC. Stop-loss tại {stop:.2f} VNDC. Rủi ro {risk_pct:.2f}% vốn. Tỉ lệ thắng ước tính: {(1-prob_win)*100:.2f}% (Accuracy backtest: {acc*100:.2f}%)."
        
        # Tín hiệu Chờ Đợi
        else:
            return f"CHỜ ĐỢI. Không có tín hiệu mạnh; thị trường sideway. Tỉ lệ thắng thấp (Dự đoán ML: {prob_win*100:.2f}%). Accuracy backtest: {acc*100:.2f}%."
            
    except Exception as e:
        logger.error(f"Error in decide_trade: {e}")
        return "Lỗi phân tích quyết định. Thử lại ảnh khác."
        
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
st.sidebar.info("Upload ảnh màn hình ONUS. AI dùng EasyOCR (cắt ảnh thông minh), pandas_ta, ML tối ưu cho tỉ lệ thắng cao. Cần có đủ các chỉ báo trên ảnh: **Giá**, **SuperTrend**, **EMA200**, **RSI**, **MACD**, **Volume**.")

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
                # Lấy dữ liệu OCR
                data = analyze_image(image)
                
            st.markdown("---")
            st.subheader("📊 Dữ Liệu OCR Đã Trích Xuất")
            if all(v is None for v in data.values()):
                st.error("❌ Không đọc được dữ liệu quan trọng nào. Vui lòng chụp ảnh rõ hơn và kiểm tra các chỉ báo.")
                progress_bar.progress(100)
            else:
                st.json(data)
                
                # --- PHASE 2: ML Training and Analysis ---
                with st.spinner("Đang huấn luyện mô hình ML và đưa ra quyết định..."):
                    time.sleep(0.5)
                    progress_bar.progress(60, text="Đang huấn luyện mô hình ML...")
                    
                    # Huấn luyện mô hình (chỉ chạy 1 lần)
                    model_acc_vol = train_model(data)
                    
                    # Ra quyết định
                    decision = decide_trade(data, model_acc_vol)
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
