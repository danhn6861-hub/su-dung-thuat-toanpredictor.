# file: app.py
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import random
from collections import deque, Counter
import plotly.graph_objects as go
import joblib
import tempfile
import io

st.set_page_config(page_title="AI Tài/Xỉu - Fusion Turbo v2", layout="wide")

st.sidebar.markdown("""
### ⚠️ LƯU Ý
Ứng dụng mang tính giải trí. Không dùng để đánh bạc thật.
Phiên bản: Fusion Turbo v2 — speed-optimized.
""")

# ================= Evolutionary AI (Optimized) =================
class EvolutionaryTaiXiuAI:
    def __init__(self, population_size=200, memory_size=1000, seed=42):
        self.population_size = int(population_size)
        self.memory_size = memory_size
        self.generation = 0
        self.best_fitness = 0.0
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        # create agents as lists, but we store weights stacked for vectorized forward
        self.agents = [self._create_agent_dict() for _ in range(self.population_size)]
        # also maintain stacked arrays for fast evaluation (created on demand)
        self._stacked_weights_cached = None
        self.evolution_history = []
        self.memory = deque(maxlen=memory_size)

    def _create_agent_dict(self):
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

    # ---------------- feature engineering (unchanged logic) ----------------
    def create_advanced_features(self, history):
        # returns np.array shape (20,), dtype float32
        if len(history) < 10:
            return np.zeros(20, dtype=np.float32)

        recent = [1 if x == "Tài" else 0 for x in history[-10:]]
        features = []
        features.extend([np.mean(recent), np.std(recent), np.sum(recent)])
        streaks = self._calculate_streaks(history[-10:])
        features.extend([streaks['current_streak'], streaks['max_streak'], self._entropy(recent)])
        features.extend(self._technical_indicators(recent))
        features.extend(self._markov_features(history[-15:]))
        features.extend(self._cycle_features(recent))
        features.extend(self._momentum_features(recent))
        while len(features) < 20:
            features.append(0.0)
        return np.array(features[:20], dtype=np.float32)

    def _calculate_streaks(self, history):
        if not history:
            return {'current_streak': 0, 'max_streak': 0}
        current_streak = 1
        current_type = history[-1]
        for i in range(len(history)-2, -1, -1):
            if history[i] == current_type:
                current_streak += 1
            else:
                break
        streaks = []
        count = 1
        for i in range(1, len(history)):
            if history[i] == history[i-1]:
                count += 1
            else:
                streaks.append(count)
                count = 1
        streaks.append(count)
        return {'current_streak': int(current_streak), 'max_streak': int(max(streaks) if streaks else 1)}

    def _entropy(self, sequence):
        if len(sequence) == 0:
            return 0.0
        seq = np.asarray(sequence, dtype=int)
        counts = np.bincount(seq)
        probs = counts / len(seq)
        return float(-np.sum([p * np.log2(p) for p in probs if p > 0]))

    def _technical_indicators(self, recent):
        if len(recent) < 5:
            return [0.0, 0.0, 0.0]
        sma_3 = float(np.mean(recent[-3:]))
        sma_5 = float(np.mean(recent[-5:]))
        momentum = float(recent[-1] - recent[-3]) if len(recent) >= 3 else 0.0
        return [sma_3, sma_5, momentum]

    def _markov_features(self, history):
        if len(history) < 3:
            return [0.5, 0.5]
        transitions = {'Tài->Tài': 0, 'Tài->Xỉu': 0, 'Xỉu->Tài': 0, 'Xỉu->Xỉu': 0}
        for i in range(1, len(history)):
            t = f"{history[i-1]}->{history[i]}"
            if t in transitions:
                transitions[t] += 1
        tai_total = transitions['Tài->Tài'] + transitions['Tài->Xỉu']
        xiu_total = transitions['Xỉu->Tài'] + transitions['Xỉu->Xỉu']
        p_tai_tai = transitions['Tài->Tài'] / tai_total if tai_total > 0 else 0.5
        p_xiu_tai = transitions['Xỉu->Tài'] / xiu_total if xiu_total > 0 else 0.5
        return [float(p_tai_tai), float(p_xiu_tai)]

    def _cycle_features(self, recent):
        if len(recent) < 6:
            return [0.0, 0.0]
        correlations = []
        for cycle_len in range(2, 5):
            if len(recent) >= cycle_len * 2:
                correlations.append(float(self._cycle_correlation(recent, cycle_len)))
            else:
                correlations.append(0.0)
        return correlations[:2]

    def _cycle_correlation(self, sequence, cycle_len):
        cycles = []
        for i in range(0, len(sequence) - cycle_len, cycle_len):
            cycles.append(sequence[i:i+cycle_len])
        if len(cycles) < 2:
            return 0.0
        sims = []
        for i in range(len(cycles)-1):
            a = np.asarray(cycles[i], dtype=float)
            b = np.asarray(cycles[i+1], dtype=float)
            if a.size == b.size and a.size > 1:
                sim = np.corrcoef(a, b)[0,1]
                if not np.isnan(sim):
                    sims.append(sim)
        return float(np.mean(sims)) if sims else 0.0

    def _momentum_features(self, recent):
        if len(recent) < 4:
            return [0.0, 0.0]
        roc = float(recent[-1] - recent[-4]) if len(recent) >= 4 else 0.0
        try:
            trend = float(np.polyfit(range(len(recent)), np.asarray(recent, dtype=float), 1)[0])
        except Exception:
            trend = 0.0
        return [roc, trend]

    # ---------------- vectorized forward for many agents & windows ----------------
    def _stack_weights(self):
        """Create stacked weight arrays for vectorized evaluation.
        Returns shapes:
          W1: (n_agents, 20, 32)
          b1: (n_agents, 32)
          W2: (n_agents, 32, 16)
          b2: (n_agents, 16)
          W3: (n_agents, 16, 1)
          b3: (n_agents, 1)
        """
        if self._stacked_weights_cached is not None:
            return self._stacked_weights_cached
        n = len(self.agents)
        W1 = np.stack([a['weights_input'] for a in self.agents], axis=0)  # (n,20,32)
        b1 = np.stack([a['bias_input'] for a in self.agents], axis=0)     # (n,32)
        W2 = np.stack([a['weights_hidden'] for a in self.agents], axis=0) # (n,32,16)
        b2 = np.stack([a['bias_hidden'] for a in self.agents], axis=0)    # (n,16)
        W3 = np.stack([a['weights_output'] for a in self.agents], axis=0) # (n,16,1)
        b3 = np.stack([a['bias_output'] for a in self.agents], axis=0)    # (n,1)
        self._stacked_weights_cached = (W1, b1, W2, b2, W3, b3)
        return self._stacked_weights_cached

    def _vectorized_predict_probs(self, features):  # features: (n_windows, 20)
        """Return array shape (n_agents, n_windows) of probabilities."""
        # features: (W, 20)
        W1, b1, W2, b2, W3, b3 = self._stack_weights()
        # compute hidden1: einsum -> (n_agents, n_windows, 32)
        # einsum 'wf,naf->nwf' with rename: features 'wf' w windows, f features; W1 'naf' n agents, a features(?)
        # We'll use 'wf,nfd->nwd' pattern:
        hidden1 = np.tanh(np.einsum('wf,nfd->nwd', features, W1) + b1[:, np.newaxis, :])  # (n, W, 32)
        hidden2 = np.tanh(np.einsum('nwd,ndh->nwh', hidden1, W2) + b2[:, np.newaxis, :])  # (n, W, 16)
        out = 1.0 / (1.0 + np.exp(-(np.einsum('nwh,nho->nwo', hidden2, W3) + b3[:, np.newaxis, :])))  # (n, W, 1)
        out = np.squeeze(out, axis=2)  # (n, W)
        return out.astype(np.float32)

    # ---------------- evaluate agents using vectorized ops ----------------
    def evaluate_agents(self, history, max_recent=50):
        """Vectorized evaluation.
        - history: full serial list
        - We use at most last `max_recent` samples (default 50) to build sliding windows.
        """
        if len(history) < 20:
            return

        # Prepare recent history (cap to max_recent for speed)
        recent = history[-max_recent:]
        # Build sliding windows: for i in [10 .. len(recent)-1], we evaluate using window recent[:i]
        windows = []
        actuals = []
        for i in range(10, len(recent)):
            window = recent[:i]
            windows.append(self.create_advanced_features(window))
            actuals.append(1 if recent[i] == "Tài" else 0)
        if not windows:
            return
        features = np.stack(windows, axis=0).astype(np.float32)  # (W,20)
        actuals = np.array(actuals, dtype=np.int8)  # (W,)

        # Compute predictions for all agents and windows at once
        W1, b1, W2, b2, W3, b3 = self._stack_weights()
        # Compute probs: shape (n_agents, n_windows)
        probs = self._vectorized_predict_probs(features)  # (n_agents, W)

        # Convert to binary predictions and compute accuracy per agent
        preds = (probs > 0.5).astype(np.int8)  # (n_agents, W)
        # actuals broadcast: compute correct counts
        correct_counts = np.sum(preds == actuals[np.newaxis, :], axis=1)  # (n_agents,)
        totals = features.shape[0]
        accuracies = correct_counts / totals

        # Apply age bonus and update fitness
        ages = np.array([a['age'] for a in self.agents], dtype=np.float32)
        age_bonus = np.minimum(ages * 0.01, 0.1)
        fitnesses = accuracies + age_bonus

        # Update agents
        for idx, a in enumerate(self.agents):
            a['fitness'] = float(fitnesses[idx])
            a['age'] += 1

        # Invalidate stacked cache since fitness changed (selection may use it) - but weights unchanged
        self._stacked_weights_cached = (W1, b1, W2, b2, W3, b3)

    # ---------------- genetic operators (kept efficient) ----------------
    def evolve_population(self, elite_frac=0.2, mutation_rate=0.08):
        self.generation += 1
        # sort agents by fitness descending
        self.agents.sort(key=lambda x: x['fitness'], reverse=True)
        self.best_fitness = float(self.agents[0]['fitness'])
        n = self.population_size
        elite_count = max(2, int(n * elite_frac))
        elites = [dict(a) for a in self.agents[:elite_count]]
        new_agents = elites.copy()

        # Pre-gather parents for tournament using probabilities to reduce overhead
        while len(new_agents) < n:
            p1 = self._tournament_selection()
            p2 = self._tournament_selection()
            child = self._crossover_fast(p1, p2)
            self._mutate_inplace(child, mutation_rate)
            child['fitness'] = 0.0
            child['age'] = 0
            new_agents.append(child)

        self.agents = new_agents
        # invalidate stacked cache (weights changed)
        self._stacked_weights_cached = None
        # record evolution summary
        self.evolution_history.append({
            'generation': self.generation,
            'best_fitness': float(self.best_fitness),
            'avg_fitness': float(np.mean([a['fitness'] for a in self.agents])),
            'diversity': self._population_diversity()
        })

    def _tournament_selection(self, k=3):
        # simple random sample tournament
        tour = random.sample(self.agents, k)
        return max(tour, key=lambda x: x['fitness'])

    def _crossover_fast(self, p1, p2):
        # uniform crossover for weights/bias with numpy for speed
        child = self._create_agent_dict()
        for key in ['weights_input', 'weights_hidden', 'weights_output']:
            a = p1[key]
            b = p2[key]
            mask = np.random.rand(*a.shape) > 0.5
            child[key] = np.where(mask, a, b).astype(np.float32)
        for key in ['bias_input', 'bias_hidden', 'bias_output']:
            a = p1[key]
            b = p2[key]
            mask = np.random.rand(*a.shape) > 0.5
            child[key] = np.where(mask, a, b).astype(np.float32)
        child['specialization'] = random.choice([p1['specialization'], p2['specialization']])
        return child

    def _mutate_inplace(self, agent, mutation_rate=0.08):
        for key in ['weights_input', 'weights_hidden', 'weights_output']:
            mask = np.random.rand(*agent[key].shape) < mutation_rate
            mutation = np.random.normal(0, 0.25, agent[key].shape).astype(np.float32)
            agent[key] = np.where(mask, agent[key] + mutation, agent[key]).astype(np.float32)
        for key in ['bias_input', 'bias_hidden', 'bias_output']:
            mask = np.random.rand(*agent[key].shape) < mutation_rate
            mutation = np.random.normal(0, 0.08, agent[key].shape).astype(np.float32)
            agent[key] = np.where(mask, agent[key] + mutation, agent[key]).astype(np.float32)
        if random.random() < 0.04:
            agent['specialization'] = random.choice(['pattern', 'momentum', 'cycle', 'random'])
        return agent

    def _population_diversity(self):
        specs = [a['specialization'] for a in self.agents]
        return float(len(set(specs)) / 4.0)

    def predict(self, history):
        # single-agent predict using best current agent
        if len(history) < 10:
            return {"Tài": 0.5, "Xỉu": 0.5}, 0.5, "Chưa đủ dữ liệu"
        features = self.create_advanced_features(history)
        best_agent = max(self.agents, key=lambda x: x['fitness'])
        # simple forward for single agent
        h1 = np.tanh(np.dot(features, best_agent['weights_input']) + best_agent['bias_input'])
        h2 = np.tanh(np.dot(h1, best_agent['weights_hidden']) + best_agent['bias_hidden'])
        out = 1.0 / (1.0 + np.exp(-(np.dot(h2, best_agent['weights_output']) + best_agent['bias_output'])))
        p = float(np.squeeze(out))
        return {"Tài": p, "Xỉu": 1.0 - p}, p, best_agent['specialization']

# ================= App State Init =================
if "history" not in st.session_state:
    st.session_state.history = []
if "evolution_ai" not in st.session_state:
    st.session_state.evolution_ai = EvolutionaryTaiXiuAI(population_size=200, seed=42)
if "ai_predictions" not in st.session_state:
    st.session_state.ai_predictions = []
if "training_log" not in st.session_state:
    st.session_state.training_log = []

# ================= Utilities & UI helpers =================
def add_result(result):
    if result not in ("Tài", "Xỉu"):
        st.error("Kết quả không hợp lệ")
        return
    st.session_state.history.append(result)
    if len(st.session_state.history) > 1000:
        # keep history reasonable
        st.session_state.history = st.session_state.history[-1000:]
    if st.session_state.ai_predictions:
        last = st.session_state.ai_predictions[-1]
        was_correct = (last['prediction'] == result)
        st.session_state.training_log.append({
            'timestamp': datetime.now(),
            'prediction': last['prediction'],
            'actual': result,
            'correct': was_correct,
            'confidence': last['confidence'],
            'strategy': last.get('strategy', '')
        })

def plot_evolution(history):
    if not history:
        return None
    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['generation'], y=df['best_fitness'], mode='lines+markers', name='Best Fitness'))
    fig.add_trace(go.Scatter(x=df['generation'], y=df['avg_fitness'], mode='lines', name='Avg Fitness'))
    fig.update_layout(title='Quá trình tiến hóa (fitness)', xaxis_title='Generation', yaxis_title='Fitness', template='plotly_dark')
    return fig

# ================= UI =================
st.title("🧠 AI Tài/Xỉu — Fusion Turbo v2 (Fast & Balanced)")

# Sidebar controls
st.sidebar.header("🎮 Điều khiển AI")
col_btn = st.sidebar.columns(2)
with col_btn[0]:
    if st.button("🔄 Huấn luyện 1 thế hệ"):
        with st.spinner("Đang huấn luyện 1 thế hệ (vectorized)..."):
            st.session_state.evolution_ai.evaluate_agents(st.session_state.history)
            st.session_state.evolution_ai.evolve_population()
        st.experimental_rerun()
with col_btn[1]:
    # advanced: train multiple gens (fast)
    gens = st.sidebar.number_input("Số thế hệ nhanh", min_value=1, max_value=200, value=5, step=1)
    if st.sidebar.button("⚡ Huấn luyện N thế hệ (fast)"):
        with st.spinner(f"Huấn luyện {gens} thế hệ..."):
            for _ in range(int(gens)):
                st.session_state.evolution_ai.evaluate_agents(st.session_state.history)
                st.session_state.evolution_ai.evolve_population()
        st.experimental_rerun()

if st.sidebar.button("🧹 Khởi tạo lại AI"):
    st.session_state.evolution_ai = EvolutionaryTaiXiuAI(population_size=200, seed=random.randint(0, 999999))
    st.session_state.ai_predictions = []
    st.session_state.training_log = []
    st.sidebar.success("Đã khởi tạo lại AI")

st.sidebar.markdown("---")
st.sidebar.header("📈 Thống kê nhanh")
if st.session_state.evolution_ai.evolution_history:
    latest = st.session_state.evolution_ai.evolution_history[-1]
    st.sidebar.write(f"**Thế hệ:** {latest['generation']}")
    st.sidebar.write(f"**Fitness tốt nhất:** {latest['best_fitness']:.1%}")
    st.sidebar.write(f"**Fitness trung bình:** {latest['avg_fitness']:.1%}")
    st.sidebar.write(f"**Độ đa dạng:** {latest['diversity']:.1%}")
st.sidebar.info(f"Tổng kết quả: {len(st.session_state.history)}")

# Main area
col_main, col_input = st.columns([2, 1])
with col_main:
    st.subheader("📊 Lịch sử (gần đây)")
    if st.session_state.history:
        display = " ".join(["🟢" if x=="Tài" else "🔴" for x in st.session_state.history[-120:]])
        st.write(display)
        tai = st.session_state.history.count("Tài")
        xiu = st.session_state.history.count("Xỉu")
        st.write(f"Tổng: Tài {tai} | Xỉu {xiu} | Tỷ lệ Tài: {tai/len(st.session_state.history):.1%}")
    else:
        st.info("Chưa có dữ liệu. Nhấn nút để thêm kết quả.")

with col_input:
    st.subheader("🎯 Ghi kết quả")
    if st.button("🎲 Tài", use_container_width=True):
        add_result("Tài")
        st.experimental_rerun()
    if st.button("🎲 Xỉu", use_container_width=True):
        add_result("Xỉu")
        st.experimental_rerun()

# Prediction block
st.subheader("🤖 Dự đoán AI")
if len(st.session_state.history) >= 10:
    probs, tai_prob, strategy = st.session_state.evolution_ai.predict(st.session_state.history)
    prediction = "Tài" if tai_prob > 0.5 else "Xỉu"
    confidence = max(tai_prob, 1 - tai_prob)
    st.session_state.ai_predictions.append({'prediction': prediction, 'confidence': confidence, 'strategy': strategy})
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Dự đoán", prediction, delta=f"{confidence:.1%}")
    with c2:
        st.metric("Độ tin cậy", f"{confidence:.1%}")
    with c3:
        names = {'pattern':'📊 Mẫu','momentum':'🚀 Động lực','cycle':'🔄 Chu kỳ','random':'🎲 Ngẫu nhiên'}
        st.metric("Chiến lược", names.get(strategy, strategy))
    # probability bar only (lightweight)
    figp = go.Figure(data=[go.Bar(x=['Tài','Xỉu'], y=[probs['Tài'], probs['Xỉu']])])
    figp.update_layout(title="Xác suất", template='plotly_dark', margin=dict(t=30))
    st.plotly_chart(figp, use_container_width=True)
else:
    st.warning("Cần ít nhất 10 kết quả để AI bắt đầu dự đoán.")

# Evolution visuals (kept but minimal)
if len(st.session_state.history) >= 20 and st.session_state.evolution_ai.evolution_history:
    st.subheader("📈 Tiến hoá & Hiệu suất")
    metrics = st.session_state.evolution_ai.get_performance_metrics()
    if metrics:
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.metric("Thế hệ", metrics['generation'])
        with g2:
            st.metric("Fitness tốt nhất", f"{metrics['best_fitness']:.1%}")
        with g3:
            st.metric("Độ đa dạng", f"{metrics['diversity']:.1%}")
        with g4:
            st.metric("Quần thể", st.session_state.evolution_ai.population_size)
        evo_fig = plot_evolution(st.session_state.evolution_ai.evolution_history)
        if evo_fig:
            st.plotly_chart(evo_fig, use_container_width=True)

# Training log & download
if st.session_state.training_log:
    st.subheader("📋 Nhật ký huấn luyện (gần đây)")
    recent = st.session_state.training_log[-40:]
    for e in reversed(recent):
        icon = "✅" if e['correct'] else "❌"
        ts = e['timestamp'].strftime("%H:%M:%S")
        st.write(f"{icon} {ts} — Dự đoán {e['prediction']} (Tin cậy {e['confidence']:.1%}) — Thực tế {e['actual']}")
    # download
    try:
        df = pd.DataFrame(st.session_state.training_log)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⤓ Tải nhật ký (CSV)", data=csv, file_name="training_log.csv", mime="text/csv")
    except Exception:
        pass

# Agent inspection (optional)
if st.sidebar.checkbox("Hiển thị Agents (Top 8)"):
    agents_sample = sorted(st.session_state.evolution_ai.agents, key=lambda x: x['fitness'], reverse=True)[:8]
    df_agents = pd.DataFrame([{'fitness':a['fitness'],'age':a['age'],'spec':a['specialization']} for a in agents_sample])
    st.sidebar.write(df_agents)

# Save/Load agents
st.sidebar.header("💾 Lưu / Tải AI")
if st.sidebar.button("🔽 Xuất agents (.pkl)"):
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        joblib.dump(st.session_state.evolution_ai.agents, tmp.name)
        tmp.close()
        with open(tmp.name, "rb") as f:
            st.download_button("⤓ Tải file agents", data=f, file_name="agents.pkl")
    except Exception as e:
        st.sidebar.error(f"Lỗi xuất: {e}")

if st.sidebar.button("🔼 Reset lịch sử"):
    st.session_state.history = []
    st.session_state.ai_predictions = []
    st.session_state.training_log = []
    st.experimental_rerun()

st.sidebar.info(f"Tổng số kết quả: {len(st.session_state.history)}")
