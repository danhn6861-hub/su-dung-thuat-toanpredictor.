import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import io
from datetime import datetime

st.set_page_config(page_title="AI Dự đoán Tài/Xỉu Nâng Cao", layout="wide")

# ===== Sidebar =====
st.sidebar.markdown("""
### ⚠️ Lưu Ý
Ứng dụng này chỉ mang tính chất giải trí và tham khảo. Không khuyến khích sử dụng cho mục đích cờ bạc hoặc đầu tư thực tế.
""")

# ===== Session state =====
for key in ["history", "models", "ai_last_pred", "undo_stack", "pred_history"]:
    if key not in st.session_state:
        st.session_state[key] = []

# ===== Hàm tạo đặc trưng =====
def create_features(history, window=6):
    if len(history) < window + 1:
        return np.empty((0, window)), np.empty((0,))
    X, y = [], []
    for i in range(window, len(history)):
        X.append([1 if x=="Tài" else 0 for x in history[i-window:i]])
        y.append(1 if history[i]=="Tài" else 0)
    return np.array(X), np.array(y)

# ===== Huấn luyện model riêng từng model và Stacking =====
@st.cache_resource
def train_models_individual(history):
    X, y = create_features(history)
    if len(X)<6: return None
    try:
        models = {
            'LogisticRegression': LogisticRegression(max_iter=500, solver='liblinear'),
            'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_leaf=3, random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=50, max_depth=3, learning_rate=0.1)
        }
        # Huấn luyện riêng
        for name, model in models.items():
            model.fit(X, y)
        # Stacking với TimeSeriesSplit
        estimators = [(n,m) for n,m in models.items()]
        tscv = TimeSeriesSplit(n_splits=3)
        stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=tscv)
        stack.fit(X, y)
        models['Stacking'] = stack
        return models
    except Exception as e:
        st.error(f"Lỗi huấn luyện: {str(e)}")
        return None

# ===== Pattern Detector 6 ván =====
def pattern_detector(history, window=6):
    if len(history) < window*2: return 0.5
    states = {'Tài':1, 'Xỉu':0}
    trans = np.zeros((2,2))
    for i in range(1,len(history)):
        prev, curr = states[history[i-1]], states[history[i]]
        trans[prev,curr] +=1
    trans /= np.sum(trans,axis=1,keepdims=True)+1e-6
    last_state = states[history[-1]]
    return trans[last_state,1]

# ===== Pattern Detector 10 ván với trọng số giảm dần =====
def pattern_detector_weighted(history, window=10):
    if len(history)<3:
        return pattern_detector(history)
    actual_window = min(len(history),window)
    recent = history[-actual_window:]
    states = {'Tài':1,'Xỉu':0}
    weights = np.linspace(1.5,0.5,actual_window)
    weighted_sum = sum(weights[i]*states[recent[i]] for i in range(actual_window))
    return weighted_sum/sum(weights)

# ===== Dự đoán từng model + tổng hợp =====
def predict_next(models, history):
    if len(history)<6 or models is None: return None
    latest = np.array([[1 if x=="Tài" else 0 for x in history[-6:]]])
    preds = {}
    for name, model in models.items():
        if hasattr(model,"predict_proba"):
            prob = model.predict_proba(latest)[0][1]
        else:
            prob = float(model.predict(latest)[0])
        preds[name] = prob
    preds['Pattern Detector (6 ván)'] = pattern_detector(history)
    preds['Pattern Detector (10 ván)'] = pattern_detector_weighted(history)
    # Final tổng hợp
    final_score = 0.5*preds.get('Stacking',0.5)+0.25*preds['Pattern Detector (6 ván)']+0.25*preds['Pattern Detector (10 ván)']
    return preds, final_score

# ===== Thêm kết quả + undo + tính accuracy =====
def add_result(result):
    st.session_state.undo_stack.append(st.session_state.history.copy())
    st.session_state.history.append(result)
    if len(st.session_state.history)>200:
        st.session_state.history=st.session_state.history[-200:]
    # Cập nhật pred_history
    if st.session_state.ai_last_pred is not None:
        st.session_state.pred_history.append({'pred':st.session_state.ai_last_pred,'true':result})

def undo_last():
    if st.session_state.undo_stack:
        st.session_state.history = st.session_state.undo_stack.pop()
        if st.session_state.pred_history:
            st.session_state.pred_history.pop()

def get_accuracy():
    if not st.session_state.pred_history: return None
    correct = sum(1 for x in st.session_state.pred_history if x['pred']==x['true'])
    return correct/len(st.session_state.pred_history)

# ===== Export/Import =====
def export_history():
    df = pd.DataFrame({"Kết quả": st.session_state.history})
    return df.to_csv(index=False).encode('utf-8')
def import_history(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.history = df["Kết quả"].tolist()
        st.success("Đã import lịch sử!")

# ===== Biểu đồ dự đoán =====
def plot_prediction(preds):
    fig,ax=plt.subplots()
    names=list(preds.keys())
    values=[preds[n]*100 for n in names]
    colors=['blue','orange','green','purple','red','brown']
    ax.barh(names,values,color=colors[:len(names)])
    ax.set_xlim(0,100)
    ax.set_xlabel("Xác suất Tài (%)")
    ax.set_title("Dự đoán từng model & Pattern Detector")
    buf=io.BytesIO()
    fig.savefig(buf,format='png')
    buf.seek(0)
    st.image(buf)

# ===== Giao diện =====
st.title("🎯 AI Dự đoán Tài/Xỉu – ML + Pattern Detector")

col1,col2 = st.columns(2)
with col1:
    if st.button("Nhập Tài"): add_result("Tài")
    if st.button("Nhập Xỉu"): add_result("Xỉu")
with col2:
    if st.button("↩️ Undo"): undo_last()
    uploaded_file=st.file_uploader("Import CSV",type='csv')
    if uploaded_file: import_history(uploaded_file)
    st.download_button("Export CSV",export_history(),f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

st.divider()

# Hiển thị lịch sử
if st.session_state.history:
    st.write("Lịch sử gần đây:", " → ".join(st.session_state.history[-30:]))

# Huấn luyện model
if st.button("⚙️ Huấn luyện model"):
    st.session_state.models = train_models_individual(st.session_state.history)
    st.success("Đã huấn luyện xong!")

# Dự đoán và hiển thị
if st.session_state.models and len(st.session_state.history)>=6:
    preds, final_score = predict_next(st.session_state.models, st.session_state.history)
    st.session_state.ai_last_pred = "Tài" if final_score>=0.5 else "Xỉu"
    st.subheader(f"🎯 Dự đoán tổng hợp: {st.session_state.ai_last_pred} ({final_score:.2%})")
    st.caption("Dự đoán chi tiết từng model + Pattern Detector")
    for k,v in preds.items():
        st.write(f"**{k}** → {v:.2%} ({'Tài' if v>=0.5 else 'Xỉu'})")
    plot_prediction(preds)

# Hiển thị tỷ lệ thắng dự đoán
acc = get_accuracy()
if acc is not None:
    st.info(f"Tỷ lệ dự đoán đúng sau {len(st.session_state.pred_history)} ván: {acc:.2%}")
