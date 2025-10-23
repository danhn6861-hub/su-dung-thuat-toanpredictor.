# app.py
"""
OKX USDT Coin Scanner — FIXED VERSION (no caching hash errors)
Author: ChatGPT
Features:
- Quét toàn bộ coin /USDT trên OKX
- Nút "Cập nhật coin", "Chạy scan", "Xóa cache"
- Chống rate-limit (retry + delay)
- Phát hiện top 5 tăng mạnh nhất và giảm mạnh nhất
"""

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import math
import traceback

# ----------------- Cấu hình Streamlit -----------------
st.set_page_config(page_title="OKX Coin Scanner", layout="wide")
st.title("🔎 OKX USDT Coin Scanner — Phiên bản ổn định")
st.write("Ứng dụng quét toàn bộ coin USDT trên sàn OKX và hiển thị top coin tăng/giảm tiềm năng nhất.")

# ----------------- Hàm tiện ích -----------------
def safe_sleep(delay):
    """Thêm độ trễ nhỏ để tránh rate-limit"""
    time.sleep(delay + np.random.random() * 0.01)

def retry(func, *args, retries=3, delay=0.3, **kwargs):
    """Thử lại khi lỗi (tối đa 3 lần)"""
    for _ in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception:
            time.sleep(delay)
    return None

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# ----------------- Cache dữ liệu -----------------
@st.cache_data(ttl=600)
def create_exchange():
    """Tạo instance OKX"""
    return ccxt.okx({"enableRateLimit": True})

@st.cache_data(ttl=600)
def load_okx_markets(_exchange):
    """Load danh sách coin (đổi exchange -> _exchange để bỏ hash)"""
    markets = _exchange.load_markets(True)
    symbols = [s for s in markets if s.endswith("/USDT")]
    return sorted(symbols)

# ----------------- Sidebar -----------------
with st.sidebar:
    st.header("⚙️ Cài đặt quét")
    timeframe = st.selectbox("Khung nến", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)
    limit_candles = st.slider("Số nến tải", 50, 1000, 200, step=50)
    max_workers = st.slider("Số luồng quét song song", 1, 20, 10)
    per_delay = st.number_input("Độ trễ giữa các request (giây)", 0.01, 1.0, 0.03)
    st.markdown("---")
    st.header("📈 Tham số chỉ báo")
    ema_short = st.number_input("EMA ngắn", 5, 50, 20)
    ema_long = st.number_input("EMA dài", 10, 200, 50)
    rsi_period = st.number_input("RSI chu kỳ", 5, 50, 14)
    atr_period = st.number_input("ATR chu kỳ", 5, 50, 14)
    st.markdown("---")
    st.header("🔍 Điều khiển")
    btn_update = st.button("🔁 Cập nhật danh sách coin")
    btn_scan = st.button("🚀 Chạy quét")
    btn_clear = st.button("🧹 Xóa cache")

if btn_clear:
    st.cache_data.clear()
    st.success("Cache đã xóa. Reload lại trang.")
    st.stop()

# ----------------- Chuẩn bị dữ liệu -----------------
exchange = create_exchange()

if "symbols" not in st.session_state or btn_update:
    with st.spinner("Đang tải danh sách coin từ OKX..."):
        st.session_state.symbols = load_okx_markets(exchange)
        st.success(f"Đã tải {len(st.session_state.symbols)} coin USDT từ OKX.")

symbols = st.session_state.symbols

st.write(f"Tổng số coin: **{len(symbols)}**")
st.caption("Lưu ý: quá nhiều coin sẽ khiến quá trình quét mất vài chục giây.")

# ----------------- Hàm quét coin -----------------
def analyze_symbol(sym):
    try:
        ticker = retry(exchange.fetch_ticker, sym)
        if not ticker:
            return None
        ohlcv = retry(exchange.fetch_ohlcv, sym, timeframe, limit_candles)
        if not ohlcv or len(ohlcv) < 20:
            return None
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")

        last_close = df["close"].iloc[-1]
        ema_s = ema(df["close"], ema_short).iloc[-1]
        ema_l = ema(df["close"], ema_long).iloc[-1]
        rsi_val = rsi(df["close"], rsi_period).iloc[-1]
        atr_val = atr(df, atr_period).iloc[-1]
        atr_norm = atr_val / (last_close + 1e-9)
        momentum = ticker.get("percentage", 0)

        # Score cho tăng/giảm
        bull_score = 0
        bear_score = 0

        # Tăng khi giá > EMA và RSI trong vùng 50–70
        if last_close > ema_s > ema_l and 50 <= rsi_val <= 70 and momentum > 0:
            bull_score = (rsi_val - 50) + momentum + (ema_s - ema_l) / ema_l * 100
        # Giảm khi giá < EMA và RSI thấp
        if last_close < ema_s < ema_l and rsi_val < 45 and momentum < 0:
            bear_score = (50 - rsi_val) - momentum + (ema_l - ema_s) / ema_s * 100

        return {
            "symbol": sym,
            "last": last_close,
            "rsi": round(rsi_val, 2),
            "atr_norm": round(atr_norm, 5),
            "momentum%": round(momentum, 2),
            "bull_score": round(bull_score, 2),
            "bear_score": round(bear_score, 2),
        }
    except Exception as e:
        return None

# ----------------- Chạy quét -----------------
if btn_scan:
    st.info("⏳ Đang quét toàn bộ coin... hãy chờ vài chục giây.")
    progress = st.progress(0)
    results = []
    total = len(symbols)
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_symbol, s): s for s in symbols}
        for f in as_completed(futures):
            res = f.result()
            done += 1
            progress.progress(done / total)
            if res:
                results.append(res)
            safe_sleep(per_delay)

    if not results:
        st.error("Không thu được dữ liệu nào. Có thể OKX đang giới hạn tạm thời.")
        st.stop()

    df = pd.DataFrame(results)
    df_bull = df.sort_values("bull_score", ascending=False).head(5)
    df_bear = df.sort_values("bear_score", ascending=False).head(5)

    st.success("✅ Quét hoàn tất!")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🚀 Top 5 coin có tiềm năng **TĂNG**")
        st.dataframe(df_bull)
    with col2:
        st.markdown("### 🔻 Top 5 coin có tiềm năng **GIẢM**")
        st.dataframe(df_bear)

    st.markdown("---")
    st.markdown("### 📊 Toàn bộ kết quả")
    st.dataframe(df.sort_values("bull_score", ascending=False))

    # Xuất CSV
    st.download_button("📥 Tải CSV", df.to_csv(index=False).encode("utf-8"), "okx_scan_results.csv", "text/csv")

    # Vẽ chart cho coin chọn
    st.markdown("---")
    coin_view = st.selectbox("Chọn coin để xem biểu đồ", df_bull["symbol"].tolist() + df_bear["symbol"].tolist())
    if coin_view:
        ohlcv = retry(exchange.fetch_ohlcv, coin_view, timeframe, limit_candles)
        if ohlcv:
            d = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
            d["ts"] = pd.to_datetime(d["ts"], unit="ms")
            fig, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(d["ts"], d["close"], label="Giá đóng cửa")
            ax1.plot(d["ts"], ema(d["close"], ema_short), "--", label=f"EMA{ema_short}")
            ax1.plot(d["ts"], ema(d["close"], ema_long), "--", label=f"EMA{ema_long}")
            ax1.legend()
            ax1.set_title(coin_view)
            st.pyplot(fig)
