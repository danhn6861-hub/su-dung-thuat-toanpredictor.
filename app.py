import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
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

# ===== Hàm tạo đặc trưng (Cải thiện) =====
def create_features(history, windows=[3,6,9]):
    if len(history) < max(windows) + 1:
        return np.empty((0, len(windows)*2 + 3)), np.empty((0,))
    X, y = [], []
    states = {'Tài':1, 'Xỉu':0}
    for i in range(max(windows), len(history)):
        features = []
        for w in windows:
            recent = [states[x] for x in history[i-w:i]]
            features.extend([np.mean(recent), sum(recent)])  # Trung bình và tổng
        # Thêm đặc trưng mới: streak, ratio dài hạn, entropy
        streak = 1
        for j in range(i-1, max(0, i-11), -1):  # Streak max 10 ván
            if history[j] != history[i-1]: break
            streak += 1
        long_ratio = sum(states[x] for x in history[max(0,i-20):i]) / min(20, i)
        entropy = -np.sum([p * np.log(p+1e-6) for p in [long_ratio, 1-long_ratio]])
        features.extend([streak, long_ratio, entropy])
        X.append(features)
        y.append(states[history[i]])
    return np.array(X), np.array(y)

# ===== Huấn luyện model riêng từng model và Stacking (Cải thiện với CV và validation) =====
@st.cache_resource
def train_models_individual(history):
    X, y = create_features(history)
    if len(X) < 10: return None  # Cần ít nhất 10 mẫu để split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=500, solver='liblinear', class_weight='balanced'),
        'RandomForest': RandomForestClassifier(n_estimators=30, max_depth=3, min_samples_leaf=5, class_weight='balanced', random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=30, max_depth=2, learning_rate=0.05, scale_pos_weight=sum(y==0)/sum(y==1) if sum(y==1)>0 else 1)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        # Kiểm tra CV score để tránh overfit
        cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        st.write(f"{name} CV Accuracy: {np.mean(cv_scores):.2%}")
        # Kiểm tra val accuracy
        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)
        st.write(f"{name} Val Accuracy: {val_acc:.2%}")
        if val_acc < 0.5: st.warning(f"{name} có thể overfit hoặc underfit!")
    
    # Stacking với các model đã huấn luyện
    estimators = [(n,m) for n,m in models.items()]
    stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(class_weight='balanced'), cv=3)
    stack.fit(X_train, y_train)
    models['Stacking'] = stack
    return models

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

# ===== Dự đoán từng model + tổng hợp (Cải thiện với trọng số học được) =====
def predict_next(models, history):
    if len(history) < 10 or models is None: return None, None
    X, y = create_features(history)  # Fixed: assign y
    latest = X[-1].reshape(1, -1)  # Đặc trưng cuối cùng
    
    preds = {}
    for name, model in models.items():
        if hasattr(model,"predict_proba"):
            prob = model.predict_proba(latest)[0][1]
        else:
            prob = float(model.predict(latest)[0])
        preds[name] = prob
    preds['Pattern Detector'] = pattern_detector(history)
    
    # Học trọng số từ data (sử dụng LinearRegression trên preds giả trên train)
    if len(X) > 1 and len(preds) > 1:
        # Tạo ma trận preds cho toàn train (giả lập bằng cách predict trên X)
        model_preds = []
        for n, m in models.items():
            if n != 'Stacking':
                if hasattr(m, 'predict_proba'):
                    model_preds.append(m.predict_proba(X)[:,1])
                else:
                    model_preds.append(m.predict(X))
        pred_matrix = np.array(model_preds).T
        
        # Compute historical patterns correctly
        historical_patterns = []
        max_windows = 9  # From windows=[3,6,9]
        for i in range(max_windows, len(history)):
            hist_slice = history[:i]
            historical_patterns.append(pattern_detector(hist_slice))
        pred_matrix = np.hstack([pred_matrix, np.array(historical_patterns).reshape(-1, 1)])
        
        split = len(X)//2
        if split > 0:
            weight_model = LinearRegression().fit(pred_matrix[:split], y[:split])
            input_vec = [preds['LogisticRegression'], preds['RandomForest'], preds['XGBoost'], preds['Pattern Detector']]
            final_score = weight_model.predict(np.array([input_vec]))[0]
        else:
            final_score = preds.get('Stacking', 0.5)
    else:
        final_score = preds.get('Stacking', 0.5)
    
    return preds, np.clip(final_score, 0, 1)

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
if st.session_state.models and len(st.session_state.history)>=10:
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
