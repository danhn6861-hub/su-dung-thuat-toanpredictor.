# app.py — Fusion Pro (Hybrid + Improved v2) — Streamlit Cloud Ready
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io, os, joblib
from datetime import datetime
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, brier_score_loss

# XGBoost optional
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ----------------- Config -----------------
st.set_page_config(page_title="Fusion Pro - AI Tài/Xỉu", layout="wide")
st.title("🔮 Fusion Pro — AI Dự đoán Tài/Xỉu (Hybrid + Improved)")
st.caption("⚠️ Ứng dụng phục vụ học tập và nghiên cứu thuật toán AI, không khuyến khích cờ bạc.")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
MODEL_PATH = "/tmp/fusion_pro_model.joblib"
HISTORY_PATH = "/tmp/fusion_pro_history.csv"

# ----------------- Init Session -----------------
for key, default in {
    "history": [], "ai_conf": [], "models": None,
    "ai_last_pred": None, "undo_stack": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ----------------- Utilities -----------------
def save_state_model(models, path=MODEL_PATH):
    try:
        joblib.dump(models, path)
        return True
    except Exception:
        return False

def load_state_model(path=MODEL_PATH):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None

def export_history_csv_bytes():
    df = pd.DataFrame({"Kết quả": st.session_state.history})
    return df.to_csv(index=False).encode("utf-8")

# ----------------- Feature creation -----------------
def create_features(history, window=6):
    if len(history) <= window:
        return np.empty((0, window + 2)), np.empty((0,))
    X, y = [], []
    for i in range(window, len(history)):
        base = [1 if h == "Tài" else 0 for h in history[i-window:i]]
        tai_ratio = sum(base)/window
        change_ratio = sum(base[j]!=base[j-1] for j in range(1,len(base))) / max(1,window-1)
        streak = 1
        for j in range(len(base)-2, -1, -1):
            if base[j] == base[-1]:
                streak += 1
            else:
                break
        X.append(base + [tai_ratio, change_ratio, streak])
        y.append(1 if history[i]=="Tài" else 0)
    return np.array(X,float), np.array(y,int)

# ----------------- Pattern detector -----------------
def pattern_detector(history, lookback=8):
    if len(history)<3: return 0.5
    recent = history[-lookback:]
    base = [1 if x=="Tài" else 0 for x in recent]
    base_prob = sum(base)/len(base)
    # detect streak
    streak, cur = 1,1
    for i in range(1,len(base)):
        if base[i]==base[i-1]:
            cur+=1; streak=max(streak,cur)
        else: cur=1
    if streak>=4: return 1-base_prob
    return 0.5

# ----------------- Train model -----------------
@st.cache_resource
def train_fusion(history_tuple, ai_conf_tuple, use_xgb=True):
    history, ai_conf = list(history_tuple), list(ai_conf_tuple)
    if len(history)<12: return None
    X,y = create_features(history)
    if X.shape[0]<8 or len(np.unique(y))<2: return None

    weights = np.linspace(0.5,1.2,len(y))
    if ai_conf and len(ai_conf)>=len(y):
        weights *= np.array(ai_conf[-len(y):])
    weights = np.clip(weights,0.3,2.0)

    tscv = TimeSeriesSplit(n_splits=min(4,max(2,len(y)//8)))
    lr = LogisticRegressionCV(cv=tscv,max_iter=1000,class_weight='balanced',random_state=RANDOM_SEED)
    rf = RandomForestClassifier(n_estimators=80,max_depth=6,class_weight='balanced',random_state=RANDOM_SEED)
    learners=[('lr',lr),('rf',rf)]

    xgb=None
    if use_xgb and HAS_XGB:
        try:
            xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                n_estimators=80,random_state=RANDOM_SEED)
            learners.append(('xgb',xgb))
        except: pass

    for _,m in learners:
        try: m.fit(X,y,sample_weight=weights)
        except: m.fit(X,y)

    try:
        calibrated_rf = CalibratedClassifierCV(base_estimator=rf, cv='prefit').fit(X,y)
    except: calibrated_rf = rf

    estimators_voting=[('lr',lr),('rf',calibrated_rf)]
    if xgb: estimators_voting.append(('xgb',xgb))
    voting = VotingClassifier(estimators_voting,voting='soft')
    voting.fit(X,y)

    try:
        stack=StackingClassifier(estimators=learners,final_estimator=LogisticRegression(max_iter=600))
        stack.fit(X,y)
    except: stack=voting

    metrics={}
    if len(X)>20:
        split=int(0.8*len(X))
        X_te,y_te=X[split:],y[split:]
        p=voting.predict(X_te)
        metrics["voting_acc"]=float(accuracy_score(y_te,p))
        metrics["voting_brier"]=float(brier_score_loss(y_te,voting.predict_proba(X_te)[:,1]))
    return {"voting":voting,"stacking":stack,"metrics":metrics}

# ----------------- Prediction -----------------
def predict_fusion(models, history, adjust_strength=0.45, recent_n=20):
    if models is None or len(history)<6: return None,None
    X,_=create_features(history)
    if len(X)==0: return None,None
    latest=X[-1:]
    try: p1=models['voting'].predict_proba(latest)[0][1]
    except: p1=0.5
    try: p2=models['stacking'].predict_proba(latest)[0][1]
    except: p2=p1
    model_prob=np.mean([p1,p2])
    pattern_prob=pattern_detector(history)
    n=min(len(history),recent_n)
    ratio=sum(1 for x in history[-n:] if x=="Tài")/n
    final=(1-adjust_strength)*model_prob + adjust_strength*(0.5*pattern_prob+0.5*ratio)
    return {"VotingProb":p1,"StackingProb":p2,"PatternProb":pattern_prob,"RecentRatio":ratio},float(np.clip(final,0.01,0.99))

# ----------------- History management -----------------
def add_result(res):
    if res not in ["Tài","Xỉu"]: return
    st.session_state.undo_stack.append(st.session_state.history.copy())
    st.session_state.history.append(res)
    if st.session_state.ai_last_pred:
        st.session_state.ai_conf.append(1.1 if st.session_state.ai_last_pred==res else 0.9)
    if len(st.session_state.history)>1000:
        st.session_state.history=st.session_state.history[-1000:]
        st.session_state.ai_conf=st.session_state.ai_conf[-1000:]

def undo():
    if st.session_state.undo_stack:
        st.session_state.history=st.session_state.undo_stack.pop()

# ----------------- Plots -----------------
def plot_history_bar(history):
    if not history: return None
    df=pd.Series(history).value_counts(normalize=True)*100
    fig,ax=plt.subplots(); ax.bar(df.index,df.values)
    ax.set_ylim(0,100); ax.set_ylabel("Tỷ lệ (%)"); ax.set_title("Tỷ lệ Tài/Xỉu")
    buf=io.BytesIO(); fig.tight_layout(); fig.savefig(buf,format="png"); buf.seek(0); plt.close(fig)
    return buf

# ----------------- Load saved -----------------
loaded=load_state_model(MODEL_PATH)
if loaded and not st.session_state.models: st.session_state.models=loaded

if os.path.exists(HISTORY_PATH) and not st.session_state.history:
    try:
        df=pd.read_csv(HISTORY_PATH)
        if "Kết quả" in df.columns:
            st.session_state.history=df["Kết quả"].tolist()
            st.session_state.ai_conf=[1.0]*len(st.session_state.history)
    except: pass

# ----------------- UI -----------------
sidebar=st.sidebar
with sidebar:
    adj_strength=st.slider("⚖️ Pattern vs Model",0.0,1.0,0.45,0.05)
    recent_n=st.number_input("🔢 Recent window",5,100,20,5)
    use_xgb=st.checkbox("Allow XGBoost",False)
    save_model=st.checkbox("💾 Save model",True)

# Top: history
st.subheader("📜 Lịch sử gần nhất")
if st.session_state.history:
    st.write(" → ".join(st.session_state.history[-40:]))
else:
    st.info("Chưa có dữ liệu.")

cols=st.columns(3)
if cols[0].button("➕ Thêm Tài"): add_result("Tài"); st.rerun()
if cols[1].button("➖ Thêm Xỉu"): add_result("Xỉu"); st.rerun()
if cols[2].button("↩️ Undo"): undo(); st.rerun()

st.markdown("---")
left,right=st.columns([2,1])

with left:
    st.subheader("📊 Thống kê & Biểu đồ")
    buf=plot_history_bar(st.session_state.history)
    if buf: st.image(buf,use_column_width=True)
    if st.session_state.history:
        n=min(len(st.session_state.history),recent_n)
        r=sum(1 for x in st.session_state.history[-n:] if x=="Tài")/n
        st.write(f"Tỷ lệ Tài trong {n} ván gần nhất: **{r:.1%}**")

with right:
    st.subheader("⚙️ Huấn luyện & Dự đoán")
    if st.button("🚀 Huấn luyện (Train)"):
        with st.spinner("Đang huấn luyện model..."):
            models=train_fusion(tuple(st.session_state.history),tuple(st.session_state.ai_conf),use_xgb)
            if models is None:
                st.error("Không đủ dữ liệu để huấn luyện.")
            else:
                st.session_state.models=models
                if save_model: save_state_model(models)
                st.success("Huấn luyện hoàn tất ✅")
                if "metrics" in models:
                    st.json(models["metrics"])
                # 🔹 auto predict ngay sau huấn luyện
                preds,final=predict_fusion(models,st.session_state.history,adj_strength,recent_n)
                if preds:
                    label="Tài" if final>=0.5 else "Xỉu"
                    st.metric("🎯 Dự đoán sau huấn luyện",f"{label} ({final*100:.2f}%)")

    st.markdown("---")
    if st.button("🤖 Dự đoán (Predict)"):
        if not st.session_state.models:
            st.warning("Vui lòng huấn luyện model trước.")
        else:
            preds,final=predict_fusion(st.session_state.models,st.session_state.history,adj_strength,recent_n)
            if preds:
                label="Tài" if final>=0.5 else "Xỉu"
                st.metric("🎯 Dự đoán",f"{label} ({final*100:.2f}%)")
                st.write(preds)
                st.session_state.ai_last_pred=label

    st.markdown("---")
    if st.button("💾 Lưu lịch sử"):
        with open(HISTORY_PATH,"wb") as f: f.write(export_history_csv_bytes())
        st.success("Đã lưu lịch sử.")
    st.download_button("📥 Tải CSV",export_history_csv_bytes(),"history.csv","text/csv")

st.markdown("---")
st.caption("© 2025 Fusion Pro — Hybrid + Improved v2 (Streamlit Cloud Optimized)")
