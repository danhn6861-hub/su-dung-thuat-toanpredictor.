# TITAN v7.1 — Smart Early Trend Engine (Onus Optimized)
# - Cải tiến so với v7.0: EMA acceleration, ADX slope, volume pressure,
#   ATR-normalized TQI, multi-timeframe confirm, improved fake-break detection,
#   EMA-angle pullback validation, and lightweight rate-limit friendly design.
# - Mục tiêu: tăng tỷ lệ thắng và phát hiện sớm xu hướng trên Binance Futures / Onus-like adjustments.

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import time, math
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------- UI ----------------
st.set_page_config(page_title="TITAN v7.1 — Smart Early Trend Engine", layout="wide")
st.title("🤖 TITAN v7.1 — Smart Early Trend Engine (Onus Optimized)")
st.caption("Phát hiện xu hướng sớm & mạnh — chuẩn Onus bias/volatility — multi-TF confirm.")

# ---------------- Helpers ----------------
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def true_range(df):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)

def adx(df, period=14):
    tr = true_range(df)
    atr = tr.rolling(period).mean()
    up_move = df['high'] - df['high'].shift()
    down_move = df['low'].shift() - df['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * (pd.Series(plus_dm).rolling(period).sum() / (atr * period + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).sum() / (atr * period + 1e-9))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    return plus_di, minus_di, dx.rolling(period).mean(), atr

def supertrend(df, period=10, multiplier=3.0):
    hl2 = (df['high'] + df['low']) / 2
    tr = true_range(df)
    atr = tr.rolling(period).mean()
    upper = hl2 + (multiplier * atr)
    lower = hl2 - (multiplier * atr)
    trend = pd.Series(index=df.index, dtype='object')
    for i in range(1, len(df)):
        if df['close'].iloc[i] > upper.iloc[i-1]:
            trend.iloc[i] = 'up'
        elif df['close'].iloc[i] < lower.iloc[i-1]:
            trend.iloc[i] = 'down'
        else:
            trend.iloc[i] = trend.iloc[i-1] if i > 0 else 'none'
    return trend

def candle_break_confirm(df, lookback=8, confirm_candles=2):
    # trả về (break_up, break_down)
    if len(df) < lookback + confirm_candles + 1: return False, False
    highs = df['high'].iloc[-(lookback+confirm_candles):-confirm_candles]
    lows = df['low'].iloc[-(lookback+confirm_candles):-confirm_candles]
    resistance, support = highs.max(), lows.min()
    closes = df['close'].iloc[-confirm_candles:]
    return (closes > resistance).any(), (closes < support).any()

def trend_duration(ema20, ema50, ema200):
    long_seq = (ema20 > ema50) & (ema50 > ema200)
    short_seq = (ema20 < ema50) & (ema50 < ema200)
    dur_long, dur_short = 0, 0
    for i in range(1, len(long_seq)+1):
        if long_seq.iloc[-i]: dur_long += 1
        else: break
    for i in range(1, len(short_seq)+1):
        if short_seq.iloc[-i]: dur_short += 1
        else: break
    return dur_long, dur_short

# EMA acceleration (slope acceleration) helper
def ema_acceleration(short_ema, long_ema, lookback=5):
    diff_now = short_ema.iloc[-1] - long_ema.iloc[-1]
    diff_prev = short_ema.iloc[-lookback] - long_ema.iloc[-lookback]
    accel = (diff_now - diff_prev) / (abs(diff_prev) + 1e-9)
    return accel

# volume pressure: hướng của volume (mua/bán)  (thân nến * vol)
def volume_pressure(df, window=10):
    sign = np.sign(df['close'] - df['open'])
    vol_bias = sign * df['vol']
    pressure = vol_bias.rolling(window).sum() / (df['vol'].rolling(window).sum() + 1e-9)
    return pressure

# ema angle (degrees) to check momentum on pullback
def ema_angle(ema_series, lookback=4):
    dy = ema_series.iloc[-1] - ema_series.iloc[-lookback]
    dx = lookback
    angle = math.degrees(math.atan(dy / (abs(ema_series.iloc[-lookback]) + 1e-9))) * (dx)
    return angle

# ---------------- Exchange ----------------
@st.cache_data(ttl=300)
def create_exchange(api_key=None, secret=None):
    ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    ex.options["adjustForTimeDifference"] = True
    if api_key and secret:
        ex.apiKey = api_key; ex.secret = secret
    return ex

@st.cache_data(ttl=300)
def load_symbols(_ex):
    mk = _ex.load_markets(True)
    syms = [s for s in mk if s.endswith('/USDT') and '_' not in s]
    return sorted(syms)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Cấu hình quét")
    timeframe = st.selectbox("Timeframe", ["15m","30m","1h"], 0)
    higher_tf = st.selectbox("Higher TF confirm", ["1h","4h","6h"], 0)
    limit = st.slider("Candles", 150, 800, 400)
    threads = st.slider("Threads", 1, 12, 6)
    delay = st.number_input("Delay (s)", 0.02, 1.0, 0.12)
    top_k = st.slider("Top coins", 1, 10, 5)
    min_vol = st.number_input("Min 24h Quote Vol (USD)", 10000, 5000000, 200000)
    bias = st.number_input("Onus bias (e.g. -0.00065)", -0.01, 0.01, -0.00065, format="%f")
    vol_scale = st.number_input("Onus vol scale (e.g. 0.94)", 0.1, 2.0, 0.94)
    btn_update = st.button("🔁 Reload Symbols")
    btn_scan = st.button("🚀 Scan TITAN v7.1")

ex = create_exchange()
if "symbols" not in st.session_state or btn_update:
    st.session_state.symbols = load_symbols(ex)
st.write(f"Loaded {len(st.session_state.symbols)} futures symbols.")

# ---------------- Multi-TF confirm ----------------
def confirm_tf(ex, sym, base_tf, higher_tf, lookback=200):
    try:
        ohl = ex.fetch_ohlcv(sym, higher_tf, lookback)
        dfh = pd.DataFrame(ohl, columns=["ts","open","high","low","close","vol"]) if len(ohl)>0 else None
        if dfh is None or len(dfh) < 50: return None
        ema50_h = ema(dfh['close'], 50)
        ema200_h = ema(dfh['close'], 200)
        # Up if ema50 over ema200
        return float(ema50_h.iloc[-1]) > float(ema200_h.iloc[-1])
    except Exception:
        return None

# ---------------- Analyzer (enhanced) ----------------
def analyze(sym):
    try:
        t = ex.fetch_ticker(sym)
        # quoteVolume field may differ; fallback
        qv = t.get('quoteVolume') or t.get('info', {}).get('quoteVolume') or 0
        if qv is None: qv = 0
        if qv < min_vol: return None

        ohl = ex.fetch_ohlcv(sym, timeframe, limit)
        df = pd.DataFrame(ohl, columns=["ts","open","high","low","close","vol"]) if len(ohl)>0 else None
        if df is None or len(df) < 120: return None
        df['ts'] = pd.to_datetime(df['ts'], unit='ms') + pd.Timedelta(hours=7)

        # Onus alignment
        mid = df[["open","high","low","close"]].mean(axis=1)
        for c in ["open","high","low","close"]:
            df[c] = mid + (df[c]-mid)*vol_scale
        df[["open","high","low","close"]] *= (1.0 + bias)

        close = df['close']; vol = df['vol']
        ema20, ema50, ema200 = ema(close,20), ema(close,50), ema(close,200)
        rsi_v = rsi(close,14)
        pdi, ndi, adx_v, atr_v = adx(df,14)
        adx_now = float(adx_v.iloc[-1])
        adx_slope = float(adx_v.iloc[-1] - adx_v.iloc[-5]) if len(adx_v) > 5 else 0
        vol_ratio = vol.iloc[-1] / (vol.rolling(20).mean().iloc[-1] + 1e-9)
        vol_press = float(volume_pressure(df, 10).iloc[-1])

        bull, bear = candle_break_confirm(df)
        st_trend = supertrend(df)
        dur_long, dur_short = trend_duration(ema20, ema50, ema200)

        slope = (ema50.iloc[-1] - ema50.iloc[-5]) / (abs(ema50.iloc[-5]) + 1e-9)
        accel = ema_acceleration(ema50, ema200, lookback=5)
        ema_ang = ema_angle(ema20, lookback=4)

        pullback = (ema20.iloc[-3] > ema50.iloc[-3]) and (close.iloc[-1] > ema50.iloc[-1]) and (rsi_v.iloc[-1] > 50)
        # refine pullback check using ema angle and rsi
        pullback_valid = pullback and (ema_ang > 8 and rsi_v.iloc[-1] > 52)

        # improved fake break logic: check ADX slope dropping strongly and small candle body
        body = (df['close'] - df['open']).abs().iloc[-1]
        avg_body = (df['close'] - df['open']).abs().rolling(10).mean().iloc[-1]
        fake_break = (adx_now < 25 and adx_slope < -1.5 and body < 0.6 * (avg_body + 1e-9))

        # ATR percent for normalization
        atr_percent = atr_v.iloc[-1] / (close.iloc[-1] + 1e-9)

        # TQI normalized by ATR% (higher var -> adjust)
        base_tqi = (adx_now * 0.5) + (max(dur_long, dur_short) * 1.2) + (abs(slope) * 500)
        TQI = base_tqi / (atr_percent * 100 + 1e-9)

        if TQI < 35: return None

        # power: combine signals, include volume pressure and acceleration
        power = (rsi_v.iloc[-1] - 50) * 0.35 + adx_now * 0.6 + accel * 140 + vol_ratio * 5 + max(dur_long, dur_short) * 0.9 + vol_press * 20
        # reduce weight if fake_break
        if fake_break: power *= 0.55

        if abs(power) < 45: return None

        direction = None
        if dur_long > 6 and st_trend.iloc[-1] == 'up':
            direction = 'LONG'
        elif dur_short > 6 and st_trend.iloc[-1] == 'down':
            direction = 'SHORT'
        else:
            # early detection: if acceleration high and ema cross trending positive, allow READY signals
            if accel > 0.18 and ema50.iloc[-1] > ema200.iloc[-1] and rsi_v.iloc[-1] > 52:
                direction = 'LONG'
            elif accel < -0.18 and ema50.iloc[-1] < ema200.iloc[-1] and rsi_v.iloc[-1] < 48:
                direction = 'SHORT'
            else:
                return None

        alert = 'CONFIRMED' if (power > 75 and adx_now > 28 and vol_press > 0.15) else ('READY' if power > 60 else 'WATCH')

        # multi-TF confirm: lightweight check (only when CONFIRMED or READY)
        tf_confirm = None
        if alert in ['CONFIRMED', 'READY']:
            tf_confirm = confirm_tf(ex, sym, timeframe, higher_tf)
            # If multi-tf disagree strongly, downrank
            if tf_confirm is False:
                alert = 'WATCH'
                power *= 0.6

        return {
            'symbol': sym, 'direction': direction, 'alert': alert,
            'power': round(power, 1), 'TQI': round(TQI, 1),
            'rsi': round(rsi_v.iloc[-1], 1), 'adx': round(adx_now, 1),
            'dur': max(dur_long, dur_short), 'vol_ratio': round(vol_ratio, 2),
            'vol_pressure': round(vol_press, 3), 'pullback': pullback_valid, 'fake': fake_break,
            'accel': round(accel, 4), 'df': df
        }
    except Exception:
        return None

# ---------------- Scan ----------------
if btn_scan:
    st.info(f"Scanning {timeframe} ... TITAN v7.1 Smart Early Trend Engine")
    results = []; done = 0
    symbols = st.session_state.symbols
    progress = st.progress(0)

    with ThreadPoolExecutor(max_workers=threads) as exe:
        futs = {exe.submit(analyze, s): s for s in symbols}
        for f in as_completed(futs):
            done += 1; progress.progress(done / len(symbols))
            r = f.result()
            if r: results.append(r)
            time.sleep(delay)

    if not results:
        st.warning('Không có coin nào đạt chuẩn xu hướng mạnh.')
        st.stop()

    df = pd.DataFrame(results).sort_values(['alert', 'power'], ascending=[False, False])
    display_cols = ['symbol','alert','direction','power','TQI','rsi','adx','dur','vol_ratio','vol_pressure','pullback','fake','accel']
    st.dataframe(df[display_cols].head(top_k))

    st.subheader('📊 Chi tiết & Biểu đồ')
    for i, row in df.head(top_k).iterrows():
        d = row['df']
        with st.expander(f"{row['symbol']} — {row['alert']} — {row['direction']} — Power {row['power']} — TQI {row['TQI']}"):
            fig, ax = plt.subplots(2,1, figsize=(10,5), gridspec_kw={'height_ratios':[3,1]})
            ax[0].plot(d['ts'], d['close'], label='Price', lw=1)
            ax[0].plot(d['ts'], ema(d['close'],20), label='EMA20')
            ax[0].plot(d['ts'], ema(d['close'],50), label='EMA50')
            ax[0].plot(d['ts'], ema(d['close'],200), label='EMA200')
            ax[0].set_title(f"{row['symbol']} | {row['direction']} | Power {row['power']}")
            ax[0].legend()
            ax[1].plot(d['ts'], rsi(d['close']), lw=1)
            ax[1].axhline(70, ls='--', alpha=0.4)
            ax[1].axhline(30, ls='--', alpha=0.4)
            ax[1].set_title('RSI(14)')
            st.pyplot(fig)

    st.success('✅ Scan done — CONFIRMED > READY > WATCH — phiên bản v7.1 đã tối ưu phát hiện sớm và giảm tín hiệu giả.')

# ---------------- Notes ----------------
st.markdown("""
**Ghi chú:**
- v7.1 tăng nhạy với trend-start bằng EMA acceleration và multi-TF confirm.\
- Kiểm tra kỹ các phép tính với dữ liệu Onus thực tế — một vài sàn trả fields hơi khác (quoteVolume, v.v.).\
- Nếu cần, mình có thể thêm:
  - Orderbook imbalance check (bid/ask volumes),\
  - Time-of-day / session filters,\
  - Live websocket feed để giảm độ trễ khi phát hiện tín hiệu.
""")
