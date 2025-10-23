# app.py
"""
OKX USDT Coin MAX-SAFE Scanner - Streamlit app
- Quét tất cả cặp */USDT trên OKX (spot)
- Button "Cập nhật coin" để reload market list
- Button "Chạy Scan" để quét OHLCV + ticker song song (threadpool) với retry/backoff
- Hiển thị Top 5 bullish & Top 5 bearish theo composite score
- Chart, export CSV, nhiều tuỳ chọn
- Robust: retries, per-request small delay, controlled concurrency, cache (st.cache_data)
"""

import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
import traceback

st.set_page_config(page_title="OKX USDT Coin Scanner — MAX-SAFE", layout="wide", initial_sidebar_state="expanded")
st.title("🔎 OKX USDT Coin Scanner — MAX-SAFE")

# ---------------------------------------------------------------------
# Utilities: indicators, safe fetch, caching
# ---------------------------------------------------------------------

def log(msg):
    st.session_state['log_items'].append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

@st.cache_data(ttl=300)
def create_exchange():
    # create ccxt exchange instance (no api keys, public endpoints only)
    exchange = ccxt.okx({
        "enableRateLimit": True,
        # "rateLimit": 50, # keep default
    })
    return exchange

@st.cache_data(ttl=300)
def load_okx_markets(exchange):
    # load markets once and cache for a while; this returns dict of markets
    markets = exchange.load_markets(True)
    return markets

def safe_sleep(min_delay):
    # small jitter sleep to avoid bursts
    time.sleep(min_delay + np.random.random() * 0.01)

def retry_backoff(func, *args, retries=3, backoff=0.5, **kwargs):
    last_exc = None
    for attempt in range(1, retries+1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            sleep_time = backoff * (2 ** (attempt-1)) + np.random.random()*0.05
            time.sleep(sleep_time)
    # final try one more time
    return func(*args, **kwargs)

def fetch_ticker_safe(exchange, symbol):
    try:
        return retry_backoff(exchange.fetch_ticker, symbol, retries=3, backoff=0.4)
    except Exception as e:
        return None

def fetch_ohlcv_safe(exchange, symbol, timeframe='1h', limit=200):
    try:
        return retry_backoff(exchange.fetch_ohlcv, symbol, timeframe, limit, retries=3, backoff=0.4)
    except Exception as e:
        return None

def ohlcv_to_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    return df

# Indicators
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def sma(series, period):
    return series.rolling(period).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean().replace(0, 1e-9)
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# Composite scoring - returns bullish_score (higher = more likely go up), bearish_score (higher = more likely drop)
def compute_scores(df, ticker_pct_change, params):
    # require enough candles
    if df is None or len(df) < max(params['ema_short'], params['ema_long'], params['rsi_period']) + 10:
        return None, None, {}
    close = df['close']
    last = float(close.iloc[-1])
    ema_s = ema(close, params['ema_short']).iloc[-1]
    ema_l = ema(close, params['ema_long']).iloc[-1]
    rsi_val = float(rsi(close, params['rsi_period']).iloc[-1])
    atr_val = float(atr(df, params['atr_period']).iloc[-1] or 0.0)
    atr_norm = (atr_val / (last + 1e-9)) if last>0 else 0.0
    avg_vol = float(df['volume'].tail(params['vol_lookback']).mean() + 1e-9)
    last_vol = float(df['volume'].iloc[-1] + 1e-9)
    vol_spike = last_vol / avg_vol

    # momentum 24h approx: use ticker_pct_change which usually is 24h percentage
    momentum = float(ticker_pct_change or 0.0) / 100.0  # normalized -5..+50 etc

    # Breakout detection (bullish): price above EMAs + recent close > previous high*? and volume spike
    recent_high = float(df['high'].tail(5).max())
    breakout = 1.0 if (last > recent_high*0.995 and last > ema_s and last > ema_l and vol_spike > params['vol_spike_breakout']) else 0.0

    # Pullback detection (could be bullish after dip): price slightly above ema_s but below ema_l etc (not used much)
    # Bear signals: last < ema_s and ema_s < ema_l and vol_spike>threshold
    bearish_signal = 1.0 if (last < ema_s and ema_s < ema_l and vol_spike > params['vol_spike_breakout']) else 0.0

    # Score components (normalized)
    # momentum_score: scaled tanh
    momentum_score = math.tanh(momentum * params['momentum_scale'])
    # volume_score: favor coins with increasing volume but penalize extremely tiny volumes
    vol_score = math.tanh(math.log1p(avg_vol) / params['vol_scale'])
    # volatility_score: prefer moderate volatility (too high is risky)
    volatility_score = 1 / (1 + atr_norm * params['atr_scale'])

    # RSI factor: favor RSI between 30 and 70 (not overbought/oversold extremes)
    if rsi_val > 80:
        rsi_factor = 0.6
    elif rsi_val > 70:
        rsi_factor = 0.8
    elif rsi_val < 20:
        rsi_factor = 0.8
    else:
        rsi_factor = 1.0

    # bullish_score composition
    bullish = (params['w_momentum'] * momentum_score +
               params['w_volume'] * vol_score +
               params['w_volatility'] * volatility_score +
               params['w_breakout'] * breakout) * rsi_factor

    # bearish_score composition (flip momentum sign, include bearish_signal)
    bearish = (params['w_momentum'] * (-momentum_score) +
               params['w_volume'] * vol_score +
               params['w_volatility'] * (1 - volatility_score) +  # prefer lower volatility for bearish? using invert
               params['w_breakout'] * bearish_signal) * (1.0)  # no rsi factor for bearish

    # Normalize to 0-100
    def scale(x):
        # x typically in roughly -3..3, so map via tanh/v transform
        return float((math.tanh(x) + 1) / 2.0 * 100)

    bull_score = scale(bullish)
    bear_score = scale(bearish)

    meta = {
        'last': last,
        'ema_s': ema_s,
        'ema_l': ema_l,
        'rsi': rsi_val,
        'atr': atr_val,
        'atr_norm': atr_norm,
        'avg_vol': avg_vol,
        'last_vol': last_vol,
        'vol_spike': vol_spike,
        'momentum_pct': momentum * 100,
        'breakout': bool(breakout),
        'bearish_signal': bool(bearish_signal),
    }
    return bull_score, bear_score, meta

# ---------------------------------------------------------------------
# Streamlit UI & main logic
# ---------------------------------------------------------------------

if 'log_items' not in st.session_state:
    st.session_state['log_items'] = []

with st.sidebar:
    st.header("Scan settings")
    timeframe = st.selectbox("Candle timeframe", options=['1m','5m','15m','1h','4h','1d'], index=3)
    limit_candles = st.number_input("Candles to fetch per symbol", min_value=50, max_value=1000, value=200, step=50)
    max_workers = st.slider("Concurrency (threads)", 1, 30, 10)
    per_request_delay = st.number_input("Delay per request (s, jitter added)", min_value=0.0, max_value=1.0, value=0.02, step=0.01)
    st.markdown("---")
    st.header("Filtering")
    min_price = st.number_input("Min price", value=0.0, step=0.0001)
    min_avg_vol = st.number_input("Min average volume (last vol_lookback candles)", value=0.0, step=1.0)
    only_top_by_volume = st.checkbox("Only top N by 24h quoteVolume", value=False)
    if only_top_by_volume:
        top_by_vol = st.number_input("Top N by 24h quoteVolume", min_value=10, max_value=2000, value=400, step=10)
    st.markdown("---")
    st.header("Indicator params")
    params = {
        'ema_short': st.number_input("EMA short span", min_value=5, max_value=50, value=20),
        'ema_long': st.number_input("EMA long span", min_value=20, max_value=200, value=50),
        'rsi_period': st.number_input("RSI period", min_value=5, max_value=50, value=14),
        'atr_period': st.number_input("ATR period", min_value=5, max_value=50, value=14),
        'vol_lookback': st.number_input("Volume lookback (candles)", min_value=3, max_value=200, value=20),
        'vol_spike_breakout': st.number_input("Volume spike threshold for breakout", min_value=1.0, max_value=10.0, value=1.6, step=0.1),
        'momentum_scale': st.number_input("Momentum scale (higher = more sensitive)", min_value=0.1, max_value=10.0, value=1.0),
        'vol_scale': st.number_input("Volume scale (higher = compress vol effect)", min_value=1.0, max_value=100.0, value=10.0),
        'atr_scale': st.number_input("ATR scale", min_value=1.0, max_value=200.0, value=10.0),
        'w_momentum': st.number_input("Weight: momentum", min_value=0.0, max_value=5.0, value=1.0),
        'w_volume': st.number_input("Weight: volume", min_value=0.0, max_value=5.0, value=0.8),
        'w_volatility': st.number_input("Weight: volatility", min_value=0.0, max_value=5.0, value=0.6),
        'w_breakout': st.number_input("Weight: breakout", min_value=0.0, max_value=5.0, value=1.2),
    }

    st.markdown("---")
    st.header("Output & UI")
    top_n = st.number_input("Show top N coins in results", min_value=5, max_value=500, value=50, step=5)
    show_debug = st.checkbox("Show debug logs", value=False)
    st.markdown("**NOTE:** Predictions are probabilistic — not financial advice. Use risk management.")
    st.markdown("---")
    # Buttons handled below outside sidebar to align UX

# Buttons: Update symbols, Run scan, Clear cache
col_btn1, col_btn2, col_btn3 = st.columns([1,1,1])
with col_btn1:
    update_symbols = st.button("🔁 Cập nhật danh sách coin (OKX)")
with col_btn2:
    run_scan = st.button("▶️ Chạy Scan")
with col_btn3:
    clear_cache = st.button("🧹 Xóa cache & reload markets")

if clear_cache:
    try:
        # clear streamlit caches
        st.cache_data.clear()
        st.experimental_rerun()
    except Exception:
        pass

# Create exchange and load markets
exchange = create_exchange()
markets = None
try:
    markets = load_okx_markets(exchange)
except Exception as e:
    st.error(f"Lỗi load markets: {e}")
    st.stop()

# Manage symbols list in session_state for update button
if 'symbols_list' not in st.session_state or update_symbols:
    # Build USDT symbols list
    symbol_keys = []
    for s, m in markets.items():
        try:
            if m.get('spot') and m.get('quote') in ('USDT','USDC') and s.endswith('/USDT'):
                symbol_keys.append(s)
        except Exception:
            continue
    symbol_keys = sorted(list(set(symbol_keys)))
    st.session_state['symbols_list'] = symbol_keys
    log(f"Updated symbols: {len(symbol_keys)} pairs (*/USDT)")
else:
    symbol_keys = st.session_state['symbols_list']

# Option: only top by volume
filtered_symbols = symbol_keys
if only_top_by_volume:
    # compute tickers quoteVolume for top selection (best-effort, sequential; limited to top_by_vol)
    tv = []
    pbar = st.progress(0)
    cnt = len(symbol_keys)
    i = 0
    for s in symbol_keys:
        i += 1
        pbar.progress(int(i/cnt*100))
        try:
            tic = fetch_ticker_safe(exchange, s)
            vol = tic.get('quoteVolume') or tic.get('baseVolume') or 0.0
            tv.append((s, float(vol or 0.0)))
        except Exception:
            tv.append((s, 0.0))
        safe_sleep(per_request_delay)
    pbar.empty()
    tv_sorted = sorted(tv, key=lambda x: x[1], reverse=True)
    top_symbols = [s for s,_ in tv_sorted[:int(top_by_vol)]]
    filtered_symbols = [s for s in symbol_keys if s in top_symbols]
    log(f"Filtered to top {len(filtered_symbols)} symbols by 24h quoteVolume")

st.write(f"**Symbols to scan:** {len(filtered_symbols)} pairs (showing first 50):")
st.write(filtered_symbols[:50])

# If user didn't press run, show info and exit early
if not run_scan:
    st.info("Nhấn 'Chạy Scan' để bắt đầu quét. Bạn có thể tùy chỉnh tham số ở sidebar. Nhấn 'Cập nhật danh sách coin' để refresh danh sách cặp USDT.")
    if show_debug:
        st.markdown("### Log")
        st.write("\n".join(st.session_state['log_items'][-50:]))
    st.stop()

# Now run scan: use threadpool to parallel fetch ticker + ohlcv
st.info("Bắt đầu scan... xin chờ — tiến độ sẽ hiển thị.")
progress_bar = st.progress(0)
placeholder_table = st.empty()

results = []

symbols_to_scan = filtered_symbols.copy()
total = len(symbols_to_scan)
if total == 0:
    st.error("Không có symbol để quét. Hãy cập nhật danh sách coin.")
    st.stop()

# Thread worker function
def worker_scan_symbol(sym, exchange, timeframe, limit_candles, per_request_delay, params):
    out = {'symbol': sym, 'error': None}
    try:
        # 1) fetch ticker
        ticker = fetch_ticker_safe(exchange, sym)
        safe_sleep(per_request_delay)
        if ticker is None:
            out['error'] = 'no_ticker'
            return out

        # 2) fetch ohlcv
        ohlcv = fetch_ohlcv_safe(exchange, sym, timeframe=timeframe, limit=limit_candles)
        safe_sleep(per_request_delay)
        if ohlcv is None:
            out['error'] = 'no_ohlcv'
            return out

        df = ohlcv_to_df(ohlcv)

        # compute 24h change if not provided by ticker
        change_pct = ticker.get('percentage')
        if change_pct is None:
            # attempt estimate: compare last to 24h-ago close if available
            change_pct = None
            try:
                # find candle 24h ago index based on timeframe
                # approximate: if timeframe is '1h', then 24 candles back etc
                tf = timeframe
                mapping = {'1m': 1/60, '5m': 5/60, '15m': 15/60, '1h':1, '4h':4, '1d':24}
                hours_per_candle = mapping.get(tf, 1)
                idx = int(round(24 / hours_per_candle))
                if idx < len(df):
                    prev = df['close'].iloc[-idx-1]
                    change_pct = (df['close'].iloc[-1] - prev) / (prev + 1e-9) * 100
            except Exception:
                change_pct = 0.0

        bull_score, bear_score, meta = compute_scores(df, change_pct, params)
        out.update({
            'last': meta.get('last') if meta else None,
            '24h_change_pct': float(change_pct or 0.0),
            'bull_score': bull_score,
            'bear_score': bear_score,
            'meta': meta,
        })
    except Exception as e:
        out['error'] = str(e)
        out['trace'] = traceback.format_exc()
    return out

# Run threadpool
workers = min(max_workers, total)
results = []
with ThreadPoolExecutor(max_workers=workers) as ex:
    futures = {ex.submit(worker_scan_symbol, sym, exchange, timeframe, int(limit_candles), float(per_request_delay), params): sym for sym in symbols_to_scan}
    done = 0
    for fut in as_completed(futures):
        res = fut.result()
        results.append(res)
        done += 1
        progress_bar.progress(int(done/total*100))
        # update short table preview
        try:
            preview = []
            for r in sorted([x for x in results if x.get('bull_score') is not None], key=lambda y: (y.get('bull_score') or 0), reverse=True)[:10]:
                preview.append([r['symbol'], r.get('last'), round(r.get('24h_change_pct',0),2), round(r.get('bull_score') or 0,2), round(r.get('bear_score') or 0,2)])
            df_preview = pd.DataFrame(preview, columns=['symbol','last','24h%','bull_score','bear_score'])
            placeholder_table.dataframe(df_preview)
        except Exception:
            pass

progress_bar.empty()
st.success(f"Scan hoàn tất — processed {len(results)} symbols")

# Post-process results into DataFrame and apply filters
rows = []
for r in results:
    if r.get('error'):
        continue
    last = r.get('last') or 0.0
    if last < min_price:
        continue
    avg_vol = r.get('meta', {}).get('avg_vol', 0.0)
    if avg_vol < min_avg_vol:
        continue
    rows.append({
        'symbol': r['symbol'],
        'last': last,
        '24h_change_pct': round(r.get('24h_change_pct',0), 3),
        'bull_score': round(r.get('bull_score') or 0, 3),
        'bear_score': round(r.get('bear_score') or 0, 3),
        'vol': round(avg_vol, 4),
        'vol_spike': round(r.get('meta', {}).get('vol_spike', 0.0), 3),
        'rsi': round(r.get('meta', {}).get('rsi', 0.0), 2),
        'atr_norm': round(r.get('meta', {}).get('atr_norm', 0.0), 6),
        'breakout': r.get('meta', {}).get('breakout', False)
    })

df_all = pd.DataFrame(rows)
if df_all.empty:
    st.warning("Không có coin thỏa bộ lọc. Thử giảm filter hoặc tăng timeout/delay.")
    st.stop()

# Rank bullish and bearish
df_bull = df_all.sort_values('bull_score', ascending=False).reset_index(drop=True)
df_bear = df_all.sort_values('bear_score', ascending=False).reset_index(drop=True)

# Show top lists
st.markdown("## 🔺 Top potential *Bullish* (tăng) — Top 5")
top_bull = df_bull.head(5)
st.table(top_bull[['symbol','last','24h_change_pct','bull_score','rsi','vol','vol_spike','breakout']])

st.markdown("## 🔻 Top potential *Bearish* (giảm) — Top 5")
top_bear = df_bear.head(5)
st.table(top_bear[['symbol','last','24h_change_pct','bear_score','rsi','vol','vol_spike','breakout']])

# Full results table
st.markdown("### All scanned results (Top N)")
st.dataframe(df_all.sort_values('bull_score', ascending=False).head(int(top_n)))

# Download CSV
csv = df_all.to_csv(index=False).encode('utf-8')
st.download_button("Tải CSV kết quả scan", data=csv, file_name=f"okx_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime='text/csv')

# Charts for selected coins
st.markdown("### Chart kiểm tra (Top Bull / Top Bear)")
col_left, col_right = st.columns(2)
with col_left:
    sel_bull = st.selectbox("Chọn 1 coin Bull để xem chart", list(top_bull['symbol']), index=0 if len(top_bull)>0 else 0)
with col_right:
    sel_bear = st.selectbox("Chọn 1 coin Bear để xem chart", list(top_bear['symbol']), index=0 if len(top_bear)>0 else 0)

def plot_coin(symbol):
    ohlcv = fetch_ohlcv_safe(exchange, symbol, timeframe=timeframe, limit=max(200, int(limit_candles)))
    if ohlcv is None:
        st.warning("Không lấy được OHLCV cho " + symbol)
        return
    df = ohlcv_to_df(ohlcv)
    fig, axs = plt.subplots(2,1, figsize=(10,5), gridspec_kw={'height_ratios':[3,1]}, sharex=True)
    axs[0].plot(df['ts'], df['close'])
    axs[0].plot(df['ts'], ema(df['close'], params['ema_short']), linestyle='--', alpha=0.8)
    axs[0].plot(df['ts'], ema(df['close'], params['ema_long']), linestyle='--', alpha=0.8)
    axs[0].set_title(symbol + " — Close + EMAs")
    axs[1].bar(df['ts'], df['volume'])
    axs[1].set_title("Volume")
    plt.tight_layout()
    st.pyplot(fig)

if sel_bull:
    plot_coin(sel_bull)
if sel_bear:
    plot_coin(sel_bear)

# Show logs if requested
if show_debug:
    st.markdown("### Debug logs (last 200 lines)")
    st.write("\n".join(st.session_state['log_items'][-200:]))

st.success("Hoàn tất — kiểm tra kỹ trước khi giao dịch. App không đưa ra lời khuyên tài chính.")
