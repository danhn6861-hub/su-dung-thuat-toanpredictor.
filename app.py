# app.py
"""
OKX USDT Coin Scanner — Demo mô phỏng thí nghiệm
Phiên bản ổn định:
✅ Khắc phục lỗi "Không thu được dữ liệu"
✅ Retry & delay thông minh khi OKX rate-limit
✅ Bộ chỉ báo nâng cấp: EMA + RSI + ADX + Vortex
✅ Chọn top 5 coin tăng mạnh nhất & giảm mạnh nhất
"""

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt

# ----------------- Cấu hình trang -----------------
st.set_page_config(page_title="OKX AI Trend Scanner", layout="wide")
st.title("🤖 OKX AI Trend Scanner (Demo mô phỏng)")
st.caption("Ứng dụng demo mô phỏng phân tích dữ liệu OKX để phát hiện xu hướng mạnh, không dùng cho giao dịch thực.")

# ----------------- Tiện ích -----------------
def safe_sleep(delay):
    time.sleep(delay + np.random.random() * 0.02)

def retry(func, *args, retries=5, delay=0.3, **kwargs):
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception:
            safe_sleep(delay)
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

def true_range(df):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)

def adx(df, period=14):
    tr = true_range(df)
    atr = tr.rolling(period).mean()
    up_move = df["high"] - df["high"].shift()
    down_move = df["low"].shift() - df["low"]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * (pd.Series(plus_dm).rolling(period).sum() / (atr * period + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).sum() / (atr * period + 1e-9))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    adx_val = dx.rolling(period).mean()
    return plus_di, minus_di, adx_val

def vortex(df, period=14):
    vm_plus = (df["high"] - df["low"].shift()).abs()
    vm_minus = (df["low"] - df["high"].shift()).abs()
    tr = true_range(df)
    vi_plus = vm_plus.rolling(period).sum() / tr.rolling(period).sum()
    vi_minus = vm_minus.rolling(period).sum() / tr.rolling(period).sum()
    return vi_plus, vi_minus

# ----------------- Cache dữ liệu -----------------
@st.cache_data(ttl=300)
def create_exchange():
    return ccxt.okx({"enableRateLimit": True})

@st.cache_data(ttl=300)
def load_symbols(_exchange):
    markets = _exchange.load_markets(True)
    syms = [s for s in markets if s.endswith("/USDT")]
    return sorted(list(set(syms)))

# ----------------- Sidebar -----------------
with st.sidebar:
    st.header("⚙️ Cấu hình quét")
    timeframe = st.selectbox("Khung nến", ["5m", "15m", "1h", "4h"], index=2)
    limit_candles = st.slider("Số nến tải mỗi coin", 50, 500, 200)
    threads = st.slider("Số luồng song song", 1, 20, 10)
    delay = st.number_input("Độ trễ mỗi request (giây)", 0.01, 1.0, 0.05)

    st.header("📊 Tham số chỉ báo")
    ema_short = st.number_input("EMA ngắn", 5, 50, 20)
    ema_long = st.number_input("EMA dài", 10, 200, 50)
    rsi_period = st.number_input("RSI chu kỳ", 5, 50, 14)
    adx_period = st.number_input("ADX chu kỳ", 5, 50, 14)
    adx_threshold = st.slider("Ngưỡng ADX xác nhận trend", 10, 60, 25)
    vi_period = st.number_input("Vortex chu kỳ", 5, 50, 14)
    min_volume = st.number_input("Lọc volume tối thiểu", 0.0, 1000000.0, 0.0)
    st.markdown("---")
    btn_update = st.button("🔁 Cập nhật danh sách coin")
    btn_scan = st.button("🚀 Quét dữ liệu")
    btn_clear = st.button("🧹 Xóa cache")

if btn_clear:
    st.cache_data.clear()
    st.success("Cache đã được xóa, tải lại trang.")
    st.stop()

# ----------------- Chuẩn bị dữ liệu -----------------
exchange = create_exchange()
if "symbols" not in st.session_state or btn_update:
    with st.spinner("Đang tải danh sách coin từ OKX..."):
        st.session_state.symbols = load_symbols(exchange)
st.write(f"Tổng số coin khả dụng: {len(st.session_state.symbols)}")

# ----------------- Phân tích coin -----------------
def analyze(symbol):
    try:
        ticker = retry(exchange.fetch_ticker, symbol)
        ohlcv = retry(exchange.fetch_ohlcv, symbol, timeframe, limit_candles)
        if not ohlcv or len(ohlcv) < 30:
            return None
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")

        last = df["close"].iloc[-1]
        vol = df["volume"].tail(20).mean()
        if vol < min_volume:
            return None

        ema_s = ema(df["close"], ema_short).iloc[-1]
        ema_l = ema(df["close"], ema_long).iloc[-1]
        rsi_val = rsi(df["close"], rsi_period).iloc[-1]
        pdi, ndi, adx_val = adx(df, adx_period)
        vi_p, vi_m = vortex(df, vi_period)
        adx_now = adx_val.iloc[-1]
        vi_pn, vi_mn = vi_p.iloc[-1], vi_m.iloc[-1]
        momentum = ticker.get("percentage", 0)

        bull = 0
        bear = 0
        if adx_now > adx_threshold and vi_pn > vi_mn and last > ema_s > ema_l and rsi_val > 50:
            bull = (rsi_val - 50) + (adx_now / 2) + momentum
        if adx_now > adx_threshold and vi_mn > vi_pn and last < ema_s < ema_l and rsi_val < 50:
            bear = (50 - rsi_val) + (adx_now / 2) - momentum

        return {
            "symbol": symbol,
            "last": last,
            "rsi": round(rsi_val, 2),
            "adx": round(adx_now, 2),
            "vi+": round(vi_pn, 2),
            "vi-": round(vi_mn, 2),
            "bull_score": round(bull, 2),
            "bear_score": round(bear, 2),
        }
    except Exception:
        return None

# ----------------- Quét dữ liệu -----------------
if btn_scan:
    st.info("Đang quét dữ liệu, vui lòng chờ...")
    results = []
    progress = st.progress(0)
    total = len(st.session_state.symbols)
    done = 0

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(analyze, s): s for s in st.session_state.symbols}
        for f in as_completed(futures):
            res = f.result()
            done += 1
            progress.progress(done / total)
            if res:
                results.append(res)
            safe_sleep(delay)

    if not results:
        st.error("⚠️ Không thu được dữ liệu nào. Có thể OKX đang quá tải hoặc giới hạn tạm thời. Hãy tăng 'Độ trễ request' lên 0.2s rồi chạy lại.")
        st.stop()

    df = pd.DataFrame(results)
    df_bull = df.sort_values("bull_score", ascending=False).head(5)
    df_bear = df.sort_values("bear_score", ascending=False).head(5)

    st.success("✅ Hoàn tất quét dữ liệu!")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔥 Top 5 coin tiềm năng TĂNG")
        st.dataframe(df_bull)
    with col2:
        st.subheader("❄️ Top 5 coin tiềm năng GIẢM")
        st.dataframe(df_bear)

    st.download_button("📥 Tải kết quả CSV", df.to_csv(index=False).encode("utf-8"), "okx_ai_trend_results.csv", "text/csv")

    # Biểu đồ minh họa
    st.markdown("---")
    coin = st.selectbox("Chọn coin để xem biểu đồ", df_bull["symbol"].tolist() + df_bear["symbol"].tolist())
    if coin:
        data = retry(exchange.fetch_ohlcv, coin, timeframe, limit_candles)
        if data:
            d = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
            d["ts"] = pd.to_datetime(d["ts"], unit="ms")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(d["ts"], d["close"], label="Close", color="blue")
            ax.plot(d["ts"], ema(d["close"], ema_short), "--", label=f"EMA{ema_short}", color="orange")
            ax.plot(d["ts"], ema(d["close"], ema_long), "--", label=f"EMA{ema_long}", color="red")
            ax.legend()
            ax.set_title(f"{coin} - Xu hướng giá ({timeframe})")
            st.pyplot(fig)
