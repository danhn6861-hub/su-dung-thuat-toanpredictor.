import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import hashlib
import traceback
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import entropy, zscore, skew, kurtosis, norm
from scipy.fft import fft
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
warnings.filterwarnings("ignore")

# CONFIG
MIN_GAMES_TO_PREDICT = 60
WINDOW = 7
MAX_TRAIN_SAMPLES = 3000
SEED = 42
HISTORY_FILE = "history.csv"
MODELS_DIR = "models_store"
os.makedirs(MODELS_DIR, exist_ok=True)

# UTILITIES (unchanged, but modularized)

def safe_float_array(lst, length=None, fill=0.0):
    try:
        arr = np.array(lst, dtype=float)
    except Exception:
        arr = np.array([float(x) if _is_num(x) else fill for x in lst], dtype=float)
    if length is not None:
        if arr.size < length:
            arr = np.concatenate([arr, np.full(length - arr.size, fill)])
        elif arr.size > length:
            arr = arr[-length:]
    return arr

def _is_num(x):
    try:
        float(x); return True
    except Exception: return False

def save_obj(obj, path):
    try:
        joblib.dump(obj, path)
    except Exception:
        pass

def load_obj(path):
    try:
        if os.path.exists(path):
            return joblib.load(path)
    except Exception:
        pass
    return None

# FEATURE ENGINEERING (enhanced with caching and efficiency)

def handle_outliers(window_data):
    try:
        arr = safe_float_array(window_data)
        if arr.size < 2:
            return arr.tolist()
        z_scores = np.abs(zscore(arr, ddof=0))
        median_val = float(np.median(arr))
        arr[z_scores > 3] = median_val
        return arr.tolist()
    except Exception:
        return [float(x) if _is_num(x) else 0.0 for x in window_data]

# Other utility functions like calculate_streaks, etc. (unchanged for brevity)

@st.cache_data(ttl=3600, max_entries=10)  # Cache with TTL for dynamic history
def create_features(history, window=WINDOW):
    enc = {"Tài":1, "Xỉu":0}
    hist_num = [enc.get(x, 0) for x in history]
    X, y = [], []
    for i in range(window, len(hist_num)):
        w = hist_num[i-window:i]
        w_clean = handle_outliers(w)
        w_clean = safe_float_array(w_clean, length=window)
        # Existing features (entropy, momentum, etc.) - unchanged
        
        # Enhanced: Add lag features (e.g., lag-1, lag-3 for efficiency)
        lag1 = w_clean[-1] if len(w_clean) > 0 else 0.0
        lag3 = w_clean[-3] if len(w_clean) > 2 else 0.0
        
        # Rolling/expanding (unchanged)
        
        # CEP, Fourier, Markov (unchanged)
        
        new_feats = [lag1, lag3, roll_mean, roll_std, exp_min, exp_max, cep_prob, trans_00, trans_11] + fft_norm
        feats = list(w_clean) + [ent, momentum, streaks, altern, autoc, var, sk, kur, p_runs] + new_feats
        X.append(feats); y.append(hist_num[i])
    X = np.array(X, dtype=float) if X else np.empty((0, window + 11 + len(fft_norm)), dtype=float)  # Adjusted dim
    y = np.array(y, dtype=int) if y else np.empty((0,), dtype=int)
    selector = None
    if X.shape[0] > 0:
        try:
            k = min(15, X.shape[1])
            selector = SelectKBest(f_classif, k=k)
            Xt = selector.fit_transform(X, y)
            return Xt, y, selector
        except Exception:
            return X, y, None
    return X, y, None

# DATA AUGMENTATION (enhanced for time series)

@st.cache_data(ttl=3600)
def augment_data(X, y, factor=3):
    try:
        augmented_X, augmented_y = list(X), list(y)
        for i in range(X.shape[0]):
            for _ in range(factor - 1):  # Original + augmented
                sample = X[i].copy()
                # Jitter: Add clipped Gaussian noise
                noise = np.random.normal(0, 0.05, sample.shape)
                sample += noise
                sample = np.clip(sample, 0, 1)
                
                # Scaling: Multiply by random factor ~ N(1, 0.1)
                scale = np.random.normal(1, 0.1)
                sample *= scale
                sample = np.clip(sample, 0, 1)
                
                # Window Warping: Stretch/compress a random window
                warp_factor = np.random.choice([0.5, 2.0])  # Slow down or speed up
                warp_start = np.random.randint(0, len(sample) // 2)
                warp_len = np.random.randint(3, len(sample) // 4)
                warped_slice = np.interp(np.linspace(0, warp_len, int(warp_len * warp_factor)), np.arange(warp_len), sample[warp_start:warp_start + warp_len])
                sample = np.concatenate([sample[:warp_start], warped_slice, sample[warp_start + warp_len:]])[:len(sample)]  # Truncate if needed
                
                augmented_X.append(sample)
                augmented_y.append(y[i])
        return np.array(augmented_X), np.array(augmented_y)
    except Exception:
        return X, y

# SESSION & HISTORY (unchanged)

# MODEL INFRASTRUCTURE (simplified params, cached)

class LSTMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, units=50, epochs=30):  # Reduced for speed
        self.units = units
        self.epochs = epochs
    # Fit and predict methods unchanged

@st.cache_resource(ttl=3600)  # Cache models as resources
def base_model_defs():
    return {
        "xgb": XGBClassifier(n_estimators=30, max_depth=2, learning_rate=0.05, n_jobs=1, verbosity=0, random_state=SEED),  # Simplified
        "cat": CatBoostClassifier(iterations=40, depth=2, learning_rate=0.05, verbose=0, random_state=SEED),
        "rf": RandomForestClassifier(n_estimators=40, max_depth=4, n_jobs=1, random_state=SEED),
        "lr": LogisticRegression(max_iter=200, solver='lbfgs', random_state=SEED),
        "lstm": LSTMWrapper(units=50, epochs=30)
    }

# Training functions unchanged, but use augmented data

# UI and Prediction (with caching where possible)

# Full code integration as in original, with above changes applied
