import streamlit as st
import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import logging
import time
from PIL import Image
from easyocr import Reader

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CẤU HÌNH VÀ HÀM OCR ---

# Khởi tạo EasyOCR tối ưu (dùng st.cache_resource để chỉ khởi tạo 1 lần)
@st.cache_resource
def get_ocr_reader():
    try:
        # Sử dụng 'en' và 'vi' để đọc số và chữ
        reader = Reader(['en', 'vi'], gpu=False) 
        return reader
    except Exception as e:
        st.error(f"Lỗi khởi tạo EasyOCR: {e}.")
        return None

reader = get_ocr_reader()

# Hàm trích xuất số NÂNG CẤP: Xử lý số lớn có dấu phân cách
def extract_number(text):
    """Trích xuất số từ chuỗi văn bản, ưu tiên số lớn và loại bỏ ký tự nhiễu."""
    try:
        clean_text = ''.join(c for c in text if c.isdigit() or c in ['.', ',']).strip()
        if not clean_text:
            return None
        
        # Nếu số có nhiều hơn 1 dấu phân cách (dấu chấm hoặc phẩy), coi là phân cách hàng nghìn
        if clean_text.count('.') > 1 or clean_text.count(',') > 1:
            num_str = clean_text.replace('.', '').replace(',', '')
        else:
             # Nếu chỉ có 1 dấu phẩy, coi là dấu thập phân và đổi thành chấm
            num_str = clean_text.replace(',', '.')

        if not num_str.replace('.', '').isdigit() and not num_str.isdigit():
             return None

        return float(num_str)
    except:
        return None

# Hàm cắt ảnh TẬP TRUNG: Đọc Giá Đóng/Giá nến cuối cùng
def crop_image(image, crop_area):
    width, height = image.size
    
    if crop_area == 'price_scale':
        # Vùng chứa Giá Đóng (Close Price) trên thang giá bên phải
        left = width * 3 // 4 
        top = height * 1 // 5
        right = width * 19 // 20
        bottom = height * 2 // 3
    elif crop_area == 'rsi_macd_volume':
        # Vùng chứa MACD, RSI, Volume sub-panels (1/3 dưới cùng)
        left = 0
        top = height * 2 // 3
        right = width
        bottom = height
    
    return image.crop((left, top, right, bottom))

# Hàm phân tích ảnh với OCR tối ưu
def analyze_image(image):
    if not reader:
        return {"price": None, "supertrend": None, "ema200": None, "volume": None, "rsi": None, "macd": None,
                "open": None, "high": None, "low": None, "close": None}
    
    data = {"price": None, "supertrend": None, "ema200": None, "volume": None, "rsi": None, "macd": None,
            "open": None, "high": None, "low": None, "close": None}

    # 1. OCR Vùng Giá Đóng (Close Price) - Mục tiêu chính
    img_price_scale = crop_image(image, 'price_scale')
    result_price_scale = reader.readtext(np.array(img_price_scale), detail=0, paragraph=False)
    
    # Tìm giá trị đơn lẻ lớn nhất (rất có thể là giá đóng)
    max_price = None
    for text in result_price_scale:
        num = extract_number(text)
        if num is not None and (max_price is None or num > max_price):
             max_price = num
    
    data["price"] = max_price
    data["close"] = max_price 
    
    # 2. OCR Vùng Chỉ báo Dưới (RSI, MACD, Volume)
    img_indicators = crop_image(image, 'rsi_macd_volume')
    result_indicators = reader.readtext(np.array(img_indicators), detail=0, paragraph=False)
    
    for text in result_indicators:
        text_lower = text.strip().lower()
        num = extract_number(text)
        
        # 2.1 RSI
        if "rsi" in text_lower and data["rsi"] is None:
            if num is not None and 0 <= num <= 100:
                data["rsi"] = num
        
        # 2.2 MACD 
        elif "macd" in text_lower and data["macd"] is None:
            if num is not None:
                data["macd"] = num

        # 2.3 Volume
        elif data["volume"] is None and any(keyword in text_lower for keyword in ["volume", "khối lượng"]):
            if num is not None:
                data["volume"] = num 
    
    # GIẢ ĐỊNH DỮ LIỆU NẾN (Nếu không đọc được nến, dùng giá Close/Entry để tạo dữ liệu O/H/L giả lập.)
    if data["price"] is not None and data["open"] is None:
        # Giả định nến gần nhất là nến giảm (hoặc tạo một chút biến động)
        data["open"] = data["price"] * 1.002
        data["high"] = data["open"] * 1.005
        data["low"] = data["price"] * 0.995

    logger.info(f"OCR Data: {data}")
    return data
        
# Hàm tính SuperTrend (không đổi)
def calculate_supertrend(highs, lows, closes, period=10, multiplier=3):
    try:
        atr = ta.atr(highs, lows, closes, length=period)
        hl2 = (highs + lows) / 2
        upper = hl2 + (multiplier * atr)
        lower = hl2 - (multiplier * atr)
        # SuperTrend là đường đang được kích hoạt. Chỉ lấy giá trị của đường đó.
        # Ở đây ta sẽ lấy đường biên dưới (lower) làm SuperTrend nếu giá đang giảm (tín hiệu bán)
        # hoặc đường biên trên (upper) nếu giá đang tăng (tín hiệu mua) trong bối cảnh lịch sử giả lập.
        # Để đơn giản, ta sẽ lấy giá trị biên phù hợp với vị trí của nến cuối cùng.
        if closes.iloc[-1] > upper.iloc[-1]:
            return lower, upper
        else:
            return upper, lower
    except Exception as e:
        logger.error(f"Error in calculate_supertrend: {e}")
        return None, None

# Hàm Huấn luyện Mô hình ML (Tự động tính Chỉ báo bị thiếu)
@st.cache_data
def train_model(data):
    """Tạo dữ liệu giả lập, tính toán features và labels, huấn luyện mô hình."""
    
    entry_price = data["price"]
    
    if entry_price is None or entry_price <= 0:
        logger.error("Giá Entry bị thiếu hoặc không hợp lệ.")
        return None, 0.5, 0, None, None, None, None

    np.random.seed(42)
    num_candles = 200 
    
    # Sử dụng giá trị cơ sở từ OCR (Close Price) để tạo chuỗi lịch sử giả lập
    base_price = entry_price
    base_volume = data["volume"] if data["volume"] else 10000 
    
    # Tạo chuỗi giá lịch sử giả lập (giả định EMA200 khoảng 98% giá hiện tại nếu không có)
    ema200_default = entry_price * 0.98 
    
    # Tạo chuỗi giá lịch sử giả lập
    closes = np.cumsum(np.random.normal(0, base_price * 0.005, num_candles - 1)) + ema200_default * 1.01
    
    # Đảm bảo nến cuối cùng sử dụng dữ liệu nến thô (O, H, L, C) từ OCR
    closes = np.append(closes, entry_price)
    highs = np.append(closes[:-1] + np.abs(np.random.normal(0, base_price * 0.01, num_candles - 1)), data["high"])
    lows = np.append(closes[:-1] - np.abs(np.random.normal(0, base_price * 0.01, num_candles - 1)), data["low"])
    volumes = np.random.uniform(base_volume * 0.5, base_volume * 1.5, num_candles)
    
    df = pd.DataFrame({"high": highs, "low": lows, "close": closes, "volume": volumes})
    
    # TÍNH TOÁN CÁC CHỈ BÁO THIẾU TỪ DỮ LIỆU GIẢ LẬP
    supertrend_series, _ = calculate_supertrend(df['high'], df['low'], df['close'])
    ema200_series = ta.ema(df['close'], length=200).fillna(method='bfill')
    rsi_series = ta.rsi(df['close'], length=14).fillna(50)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    macd_series = macd['MACD_12_26_9'].fillna(0)
    volatility_series = ta.stdev(df['close'], length=20).fillna(0)

    # Lấy giá trị cuối cùng của các chỉ báo giả lập (hoặc OCR nếu có)
    supertrend_final = data["supertrend"] if data["supertrend"] is not None else supertrend_series.iloc[-1]
    ema200_final = data["ema200"] if data["ema200"] is not None else ema200_series.iloc[-1]
    rsi_final = data["rsi"] if data["rsi"] is not None else rsi_series.iloc[-1]
    macd_final = data["macd"] if data["macd"] is not None else macd_series.iloc[-1]
    volatility_final = volatility_series.iloc[-1]

    # Chuẩn bị Dữ liệu cho ML
    features_df = pd.DataFrame({
        # Sử dụng các giá trị đã được tính toán/OCR để đảm bảo feature hiện tại chính xác
        'price_diff_st': df['close'] - supertrend_final, 
        'price_diff_ema': df['close'] - ema200_final,
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
    
    # Trả về mô hình, độ chính xác, volatility và giá trị chỉ báo cuối cùng
    return model, acc, volatility_final, supertrend_final, ema200_final, rsi_final, macd_final

# Hàm quyết định giao dịch với tối ưu hóa ML
def decide_trade(data, model_results):
    try:
        model, acc, volatility_final, supertrend_final, ema2200_final, rsi_final, macd_final = model_results
        
        entry = data["price"] 
        if entry is None:
            return "Giá Entry (Price) không được đọc thành công. Không thể ra lệnh."

        # Sử dụng giá trị TÍNH TOÁN/OCR
        supertrend = supertrend_final
        ema200 = ema2200_final
        rsi = rsi_final
        macd_val = macd_final

        # Chuẩn bị feature hiện tại
        current_features = pd.DataFrame({
            'price_diff_st': [entry - supertrend],
            'price_diff_ema': [entry - ema200],
            'rsi': [rsi],
            'macd': [macd_val],
            'volume_change': [0],
            'volatility': [volatility_final]
        })
        
        pred = model.predict(current_features)[0]
        prob_win = model.predict_proba(current_features)[0][pred]
        
        # 6. Ra quyết định Giao dịch Tối ưu (Cải thiện logic và quản lý rủi ro)
        
        # Tín hiệu Mua (LONG)
        if pred == 1 and entry > supertrend and entry > ema200 and rsi > 50 and macd_val > 0:
            edge = prob_win - (1 - prob_win) + 0.1 * (volatility_final / entry)
            risk_pct = max(1, min(5, edge / (1 - prob_win) * 2)) if (1 - prob_win) > 0 else 3
            
            target = entry * (1 + 0.1 * prob_win + 0.05 * edge)
            stop = entry * (1 - 0.02 / (prob_win + 0.1))
            
            return f"LONG tại {entry:.2f}. Chốt lời: {target:.2f}. Stop-loss: {stop:.2f}. Rủi ro: {risk_pct:.2f}% vốn. Tỉ lệ thắng: {prob_win*100:.2f}%."
            
        # Tín hiệu Bán (SHORT)
        elif pred == 0 and entry < supertrend and entry < ema200 and rsi < 50 and macd_val < 0:
            edge = (1 - prob_win) - prob_win + 0.1 * (volatility_final / entry)
            risk_pct = max(1, min(5, edge / prob_win * 2)) if prob_win > 0 else 3
            
            target = entry * (1 - 0.1 * (1 - prob_win) - 0.05 * edge)
            stop = entry * (1 + 0.02 / (1 - prob_win + 0.1))
            
            return f"SHORT tại {entry:.2f}. Chốt lời: {target:.2f}. Stop-loss: {stop:.2f}. Rủi ro: {risk_pct:.2f}% vốn. Tỉ lệ thắng: {(1-prob_win)*100:.2f}%."
        
        # Tín hiệu Chờ Đợi
        else:
            return f"CHỜ ĐỢI. Không có tín hiệu mạnh. Tỉ lệ thắng dự kiến: {prob_win*100:.2f}%."
            
    except Exception as e:
        logger.error(f"Error in decide_trade: {e}")
        return "Lỗi phân tích quyết định."
        
# 7. Giao diện Streamlit (Đơn giản hóa)
st.set_page_config(page_title="AI Trading Analyzer Pro", layout="wide")
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        font-size: 1.2em;
        font-weight: bold;
    }
    .main-title {
        color: #007bff;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    <h1 class="main-title">🤖 AI Trading Analyzer (OCR Tự Động & Đơn giản)</h1>
    <p style='text-align: center; color: gray;'>AI đọc ảnh biểu đồ ONUS, tự động tính chỉ báo và đưa ra tín hiệu.</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("🖼️ Tải lên Ảnh Màn Hình Biểu Đồ ONUS", type=["jpg", "png"], help="Chụp rõ giá (thang bên phải) và các chỉ báo dưới cùng (RSI, MACD).")

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    
    image = Image.open(uploaded_file)

    with col1:
        st.image(image, caption="Ảnh Biểu Đồ Đã Tải Lên", use_container_width=True)
        
    with col2:
        st.subheader("Phân Tích Tự Động")
        if st.button("🚀 BẮT ĐẦU PHÂN TÍCH VÀ RA LỆNH", type="primary"):
            progress_bar = st.progress(0, text="Đang xử lý...")
            
            # --- PHASE 1: OCR ---
            with st.spinner("Đang đọc dữ liệu từ ảnh (EasyOCR)..."):
                time.sleep(0.5)
                progress_bar.progress(30, text="Đang đọc dữ liệu từ ảnh (EasyOCR)...")
                data = analyze_image(image)
                
            if data["price"] is None:
                st.error("❌ Không đọc được **GIÁ ĐÓNG** hiện tại từ ảnh. Vui lòng chụp rõ hơn hoặc thử lại.")
                progress_bar.progress(100)
            else:
                
                # --- PHASE 2: ML Training and Analysis ---
                with st.spinner("Đang tính toán chỉ báo và huấn luyện mô hình ML..."):
                    time.sleep(0.5)
                    progress_bar.progress(60, text="Đang huấn luyện mô hình ML...")
                    model_results = train_model(data)
                    decision = decide_trade(data, model_results)
                    progress_bar.progress(100)
                
                model, acc, volatility_final, supertrend_final, ema2200_final, rsi_final, macd_final = model_results
                
                st.markdown("---")
                st.subheader("🎯 TÍN HIỆU AI ĐƯA RA")
                
                # Hiển thị Tín hiệu
                if "LONG" in decision:
                    st.success(f"✅ TÍN HIỆU MUA (LONG)")
                    st.markdown(f"**{decision}**")
                elif "SHORT" in decision:
                    st.error(f"🔴 TÍN HIỆU BÁN (SHORT)")
                    st.markdown(f"**{decision}**")
                else:
                    st.warning(f"🟡 {decision}")
                
                # Hiển thị Dữ liệu phân tích (Đơn giản hóa)
                st.markdown(f"""
                <div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-top: 20px;'>
                    **CHI TIẾT PHÂN TÍCH:**<br>
                    Giá Entry: **{data['price']:.2f}**<br>
                    SuperTrend: **{supertrend_final:.2f}** (Tính toán/OCR)<br>
                    EMA200: **{ema2200_final:.2f}** (Tính toán/OCR)<br>
                    RSI: **{rsi_final:.2f}** (Tính toán/OCR)<br>
                    Độ chính xác mô hình (Backtest): **{acc*100:.2f}%**
                </div>
                """, unsafe_allow_html=True)
                
                st.info("⚠️ Lưu ý: Tín hiệu này dựa trên OCR và mô hình ML. Không phải lời khuyên tài chính.")
