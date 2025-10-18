# =========================================================
# file: app.py
# AI Tài/Xỉu — Fusion Turbo v2 (2025)
# Bản tối ưu: tốc độ cao, không lỗi, tương thích Streamlit mới
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import random
from collections import deque
import plotly.graph_objects as go
import joblib
import tempfile
import io

st.set_page_config(page_title="AI Tài/Xỉu - Fusion Turbo v2", layout="wide")

st.sidebar.markdown("""
### ⚠️ LƯU Ý
Ứng dụng mang tính giải trí. Không dùng để đánh bạc thật.
Phiên bản: **Fusion Turbo v2** — speed-optimized.
""")

# ================= Evolutionary AI =================
class EvolutionaryTaiXiuAI:
    def __init__(self, population_size=200, memory_size=1000, seed=42):
        self.population_size = int(population_size)
        self.memory_size = memory_size
        self.generation = 0
        self.best_fitness = 0.0
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        self.agents = [self._create_agent() for _ in range(self.population_size)]
        self._stacked_cache = None
        self.evolution_history = []
        self.memory = deque(maxlen=memory_size)

    def _create_agent(self):
        return {
            'weights_input': np.random.uniform(-1, 1, (20, 32)).astype(np.float32),
            'weights_hidden': np.random.uniform(-1, 1, (32, 16)).astype(np.float32),
            'weights_output': np.random.uniform(-1, 1, (16, 1)).astype(np.float32),
            'bias_input': np.random.uniform(-0.5, 0.5, 32).astype(np.float32),
            'bias_hidden': np.random.uniform(-0.5, 0.5, 16).astype(np.float32),
            'bias_output': np.random.uniform(-0.1, 0.1, 1).astype(np.float32),
            'fitness': 0.0,
            'age': 0,
            'specialization': random.choice(['pattern', 'momentum', 'cycle', 'random'])
        }

    # =============== Feature Engineering ===============
    def create_advanced_features(self, history):
        if len(history) < 10:
            return np.zeros(20, dtype=np.float32)
        recent = [1 if x == "Tài" else 0 for x in history[-10:]]
        features = []
        features.extend([np.mean(recent), np.std(recent), np.sum(recent)])
        features.extend(self._streak_features(history))
        features.extend(self._markov_features(history))
        features.extend(self._momentum_features(recent))
        while len(features) < 20:
            features.append(0.0)
        return np.array(features[:20], dtype=np.float32)

    def _streak_features(self, history):
        if not history:
            return [0.0, 0.0, 0.0]
        current = 1
        cur_type = history[-1]
        for i in range(len(history)-2, -1, -1):
            if history[i] == cur_type:
                current += 1
            else:
                break
        max_streak = 1
        c = 1
        for i in range(1, len(history)):
            if history[i] == history[i-1]:
                c += 1
                max_streak = max(max_streak, c)
            else:
                c = 1
        return [current, max_streak, np.random.rand()]

    def _markov_features(self, history):
        if len(history) < 3:
            return [0.5, 0.5]
        transitions = {'Tài->Tài': 0, 'Tài->Xỉu': 0, 'Xỉu->Tài': 0, 'Xỉu->Xỉu': 0}
        for i in range(1, len(history)):
            pair = f"{history[i-1]}->{history[i]}"
            if pair in transitions:
                transitions[pair] += 1
        tai_total = transitions['Tài->Tài'] + transitions['Tài->Xỉu']
        xiu_total = transitions['Xỉu->Tài'] + transitions['Xỉu->Xỉu']
        p_tai_tai = transitions['Tài->Tài'] / tai_total if tai_total else 0.5
        p_xiu_tai = transitions['Xỉu->Tài'] / xiu_total if xiu_total else 0.5
        return [p_tai_tai, p_xiu_tai]

    def _momentum_features(self, recent):
        if len(recent) < 4:
            return [0.0, 0.0]
        roc = float(recent[-1] - recent[-4])
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        return [roc, trend]

    # =============== Neural Processing ===============
    def _stack_weights(self):
        if self._stacked_cache is not None:
            return self._stacked_cache
        n = len(self.agents)
        W1 = np.stack([a['weights_input'] for a in self.agents])
        b1 = np.stack([a['bias_input'] for a in self.agents])
        W2 = np.stack([a['weights_hidden'] for a in self.agents])
        b2 = np.stack([a['bias_hidden'] for a in self.agents])
        W3 = np.stack([a['weights_output'] for a in self.agents])
        b3 = np.stack([a['bias_output'] for a in self.agents])
        self._stacked_cache = (W1, b1, W2, b2, W3, b3)
        return self._stacked_cache

    def _vectorized_predict_probs(self, features):
        W1, b1, W2, b2, W3, b3 = self._stack_weights()
        hidden1 = np.tanh(np.einsum('wf,nfd->nwd', features, W1) + b1[:, np.newaxis, :])
        hidden2 = np.tanh(np.einsum('nwd,ndh->nwh', hidden1, W2) + b2[:, np.newaxis, :])
        out = 1 / (1 + np.exp(-(np.einsum('nwh,nho->nwo', hidden2, W3) + b3[:, np.newaxis, :])))
        return np.squeeze(out, axis=2)

    # =============== Training ===============
    def evaluate_agents(self, history):
        if len(history) < 20:
            return
        windows = []
        actuals = []
        for i in range(10, len(history)):
            windows.append(self.create_advanced_features(history[:i]))
            actuals.append(1 if history[i] == "Tài" else 0)
        features = np.stack(windows, axis=0)
        actuals = np.array(actuals)
        probs = self._vectorized_predict_probs(features)
        preds = (probs > 0.5).astype(np.int8)
        acc = np.sum(preds == actuals[np.newaxis, :], axis=1) / len(actuals)
        for i, a in enumerate(self.agents):
            a['fitness'] = float(acc[i])
            a['age'] += 1

    def evolve_population(self, elite_frac=0.2, mutation_rate=0.08):
        self.generation += 1
        self.agents.sort(key=lambda x: x['fitness'], reverse=True)
        self.best_fitness = self.agents[0]['fitness']
        n = len(self.agents)
        elites = [dict(a) for a in self.agents[:max(2, int(n * elite_frac))]]
        new_agents = elites.copy()
        while len(new_agents) < n:
            p1 = random.choice(elites)
            p2 = random.choice(self.agents)
            child = self._crossover(p1, p2)
            self._mutate(child, mutation_rate)
            new_agents.append(child)
        self.agents = new_agents
        self._stacked_cache = None
        self.evolution_history.append({
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': np.mean([a['fitness'] for a in self.agents]),
            'diversity': len(set(a['specialization'] for a in self.agents)) / 4.0
        })

    def _crossover(self, p1, p2):
        child = self._create_agent()
        for key in ['weights_input', 'weights_hidden', 'weights_output']:
            mask = np.random.rand(*p1[key].shape) > 0.5
            child[key] = np.where(mask, p1[key], p2[key])
        return child

    def _mutate(self, agent, rate):
        for key in ['weights_input', 'weights_hidden', 'weights_output']:
            mask = np.random.rand(*agent[key].shape) < rate
            agent[key][mask] += np.random.normal(0, 0.2, mask.sum())
        return agent

    def predict(self, history):
        if len(history) < 10:
            return {"Tài": 0.5, "Xỉu": 0.5}, 0.5, "Chưa đủ dữ liệu"
        features = self.create_advanced_features(history)
        best = max(self.agents, key=lambda a: a['fitness'])
        h1 = np.tanh(np.dot(features, best['weights_input']) + best['bias_input'])
        h2 = np.tanh(np.dot(h1, best['weights_hidden']) + best['bias_hidden'])
        out = 1 / (1 + np.exp(-(np.dot(h2, best['weights_output']) + best['bias_output'])))
        p = float(out.squeeze())
        return {"Tài": p, "Xỉu": 1 - p}, p, best['specialization']

# ================= Session =================
if "history" not in st.session_state:
    st.session_state.history = []
if "evolution_ai" not in st.session_state:
    st.session_state.evolution_ai = EvolutionaryTaiXiuAI()
if "ai_predictions" not in st.session_state:
    st.session_state.ai_predictions = []
if "training_log" not in st.session_state:
    st.session_state.training_log = []

# ================= Utility =================
def add_result(result):
    if result not in ("Tài", "Xỉu"):
        return
    st.session_state.history.append(result)
    if len(st.session_state.history) > 1000:
        st.session_state.history = st.session_state.history[-1000:]
    if st.session_state.ai_predictions:
        last = st.session_state.ai_predictions[-1]
        was_correct = (last['prediction'] == result)
        st.session_state.training_log.append({
            'timestamp': datetime.now(),
            'prediction': last['prediction'],
            'actual': result,
            'correct': was_correct,
            'confidence': last['confidence']
        })

def plot_evolution(history):
    if not history: return None
    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['generation'], y=df['best_fitness'], mode='lines+markers', name='Best'))
    fig.add_trace(go.Scatter(x=df['generation'], y=df['avg_fitness'], mode='lines', name='Average'))
    fig.update_layout(title="Tiến hóa", template='plotly_dark')
    return fig

# ================= UI =================
st.title("🧠 AI Tài/Xỉu — Fusion Turbo v2 (2025)")

# --- Sidebar ---
st.sidebar.header("🎮 Điều khiển AI")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🔄 Huấn luyện 1 thế hệ"):
        st.session_state.evolution_ai.evaluate_agents(st.session_state.history)
        st.session_state.evolution_ai.evolve_population()
        st.rerun()
with col2:
    gens = st.sidebar.number_input("Số thế hệ", 1, 100, 5)
    if st.sidebar.button("⚡ Huấn luyện N thế hệ"):
        for _ in range(int(gens)):
            st.session_state.evolution_ai.evaluate_agents(st.session_state.history)
            st.session_state.evolution_ai.evolve_population()
        st.rerun()

if st.sidebar.button("🧹 Khởi tạo lại AI"):
    st.session_state.evolution_ai = EvolutionaryTaiXiuAI(seed=random.randint(0,99999))
    st.session_state.ai_predictions = []
    st.session_state.training_log = []
    st.rerun()

# --- Stats ---
if st.session_state.evolution_ai.evolution_history:
    latest = st.session_state.evolution_ai.evolution_history[-1]
    st.sidebar.markdown(f"**Thế hệ:** {latest['generation']}")
    st.sidebar.markdown(f"**Fitness tốt nhất:** {latest['best_fitness']:.2%}")
    st.sidebar.markdown(f"**Fitness TB:** {latest['avg_fitness']:.2%}")

# --- Main ---
col_main, col_input = st.columns([2, 1])

with col_main:
    st.subheader("📊 Lịch sử (gần đây)")
    if st.session_state.history:
        display = " ".join(["🟢" if x=="Tài" else "🔴" for x in st.session_state.history[-120:]])
        st.write(display)
    else:
        st.info("Chưa có dữ liệu.")

with col_input:
    st.subheader("🎯 Ghi kết quả")
    if st.button("🎲 Tài"):
        add_result("Tài")
        st.rerun()
    if st.button("🎲 Xỉu"):
        add_result("Xỉu")
        st.rerun()

# --- Prediction ---
st.subheader("🤖 Dự đoán AI")
if len(st.session_state.history) >= 10:
    probs, p_tai, strategy = st.session_state.evolution_ai.predict(st.session_state.history)
    prediction = "Tài" if p_tai > 0.5 else "Xỉu"
    conf = max(p_tai, 1 - p_tai)
    st.session_state.ai_predictions.append({'prediction': prediction, 'confidence': conf})
    c1, c2 = st.columns(2)
    c1.metric("Dự đoán", prediction)
    c2.metric("Độ tin cậy", f"{conf:.1%}")
    figp = go.Figure([go.Bar(x=['Tài','Xỉu'], y=[probs['Tài'], probs['Xỉu']])])
    figp.update_layout(title="Xác suất", template='plotly_dark')
    st.plotly_chart(figp, use_container_width=True)
else:
    st.warning("Cần ít nhất 10 kết quả để AI bắt đầu dự đoán.")

# --- Evolution chart ---
if st.session_state.evolution_ai.evolution_history:
    st.subheader("📈 Tiến hóa & Hiệu suất")
    fig = plot_evolution(st.session_state.evolution_ai.evolution_history)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

# --- Training Log ---
if st.session_state.training_log:
    st.subheader("📋 Nhật ký huấn luyện")
    df = pd.DataFrame(st.session_state.training_log)
    st.dataframe(df.tail(30))
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⤓ Tải nhật ký (CSV)", csv, "training_log.csv")

# --- Reset ---
if st.sidebar.button("🔼 Reset lịch sử"):
    st.session_state.history = []
    st.session_state.ai_predictions = []
    st.session_state.training_log = []
    st.rerun()
