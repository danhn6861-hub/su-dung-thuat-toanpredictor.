# app.py
"""
AI Tài/Xỉu — Cấp 3 (Self-Generating Strategies + Evolutionary Improvement)
- Manual train for base ML (Logistic / RF / XGB)
- Strategy population that evolves automatically (and manually via "Evolve now")
- Data stored in st.session_state (session-only)
- Buttons: TÀI / XỈU (record), Huấn luyện lại (train base ML), Evolve now, Reset
"""

import streamlit as st
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import random
import json

# -------------------------
# Config
# -------------------------
WINDOW = 6
RANDOM_STATE = 42
POP_SIZE = 14            # number of strategy agents
EVO_INTERVAL = 5         # auto-evolve every N recorded results (if enabled)
MAX_HISTORY = 300        # cap history length

# -------------------------
# Helpers: safe utils
# -------------------------
def ensure_history_strings():
    # ensure history only contains "Tài" / "Xỉu"
    st.session_state.history = [str(x) for x in st.session_state.history if str(x) in ("Tài","Xỉu")]

def encode(history):
    return [1 if x == "Tài" else 0 for x in history]

def create_Xy(history, window=WINDOW):
    H = encode(history)
    X, y = [], []
    for i in range(len(H)-window):
        X.append(H[i:i+window])
        y.append(H[i+window])
    return np.array(X), np.array(y)

def majority_label_from_bits(bits):
    if len(bits)==0:
        return None
    s = sum(bits)
    return "Tài" if s >= (len(bits)/2) else "Xỉu"

# -------------------------
# Strategy system (agents)
# -------------------------
STR_TYPES = ["trend","contrarian","pattern","probabilistic","fixed"]

def random_strategy():
    """Create a random strategy dict."""
    stype = random.choice(STR_TYPES)
    if stype == "trend":
        param = {"k": random.randint(1, WINDOW)}  # lookback length
    elif stype == "contrarian":
        param = {"k": random.randint(1, WINDOW)}
    elif stype == "pattern":
        param = {"w": random.randint(2, WINDOW), "min_matches": 1}
    elif stype == "probabilistic":
        param = {"use_base": random.choice([True, False]), "bias": random.uniform(0.45,0.55)}
    elif stype == "fixed":
        param = {"fixed": random.choice(["Tài","Xỉu"])}
    return {
        "id": random.randint(1,10**9),
        "type": stype,
        "param": param,
        "fitness": 1.0,    # start neutral
        "wins": 0,
        "trials": 0,
        "last_pred": None
    }

def init_population(n=POP_SIZE):
    return [random_strategy() for _ in range(n)]

def strategy_predict(agent, history, base_probs=None):
    """Return (pred_label, confidence_score[0..1])"""
    stype = agent["type"]
    p = agent["param"]
    if len(history) < 1:
        # no data => random or fixed if fixed
        if stype=="fixed":
            return p["fixed"], 0.6
        return random.choice(["Tài","Xỉu"]), 0.5

    if stype == "trend":
        k = min(p.get("k",1), len(history))
        bits = encode(history[-k:])
        lab = majority_label_from_bits(bits)
        conf = (sum(bits)/k) if lab=="Tài" else (1 - sum(bits)/k)
        return lab, max(0.5, conf)

    if stype == "contrarian":
        k = min(p.get("k",1), len(history))
        bits = encode(history[-k:])
        lab = majority_label_from_bits(bits)
        if lab is None:
            return random.choice(["Tài","Xỉu"]), 0.5
        return ("Xỉu" if lab=="Tài" else "Tài"), 0.5 + 0.25*(1 - abs(sum(bits)/k - 0.5))

    if stype == "pattern":
        w = min(p.get("w",3), len(history))
        if len(history) < w+1:
            return random.choice(["Tài","Xỉu"]), 0.5
        pattern = history[-w:]
        matches = []
        for i in range(len(history)-w):
            if history[i:i+w] == pattern and i+w < len(history):
                matches.append(history[i+w])
        if not matches:
            return random.choice(["Tài","Xỉu"]), 0.5
        cnt = Counter(matches)
        pred = cnt.most_common(1)[0][0]
        prob = cnt[pred] / len(matches)
        return pred, max(0.5, prob)

    if stype == "probabilistic":
        # try to use base_probs if allowed
        bias = p.get("bias",0.5)
        use_base = p.get("use_base", False)
        if use_base and base_probs:
            # base_probs is dict: {"LR":0.6,...} probability of Tài
            vals = [v for v in base_probs.values() if v is not None]
            if vals:
                avg = float(np.mean(vals))
                return ("Tài" if avg >= bias else "Xỉu"), max(0.5, abs(avg - 0.5)+0.5)
        # fallback random with bias
        return ("Tài" if random.random() < bias else "Xỉu"), 0.5

    if stype == "fixed":
        return p["fixed"], 0.6

    return random.choice(["Tài","Xỉu"]), 0.5

# -------------------------
# Evolutionary operators
# -------------------------
def evaluate_and_update_fitness(real_label):
    """After real result recorded, update each agent's fitness and stats."""
    pop = st.session_state.population
    for agent in pop:
        pred = agent.get("last_pred")
        if pred is None:
            agent["trials"] += 0
            continue
        agent["trials"] += 1
        if pred == real_label:
            agent["wins"] += 1
            agent["fitness"] = agent.get("fitness",1.0) * 1.08  # reward
        else:
            agent["fitness"] = agent.get("fitness",1.0) * 0.92  # penalty
        # clamp fitness
        agent["fitness"] = float(max(0.1, min(agent["fitness"], 50.0)))
    # normalize fitness for voting weights optionally
    total = sum(a["fitness"] for a in pop)
    if total > 0:
        for a in pop:
            a["w_norm"] = a["fitness"]/total

def mutate_agent(agent):
    """Return mutated copy of agent (small random change)"""
    new = dict(agent)
    new["id"] = random.randint(1,10**9)
    t = new["type"]
    if t=="trend" or t=="contrarian":
        new["param"] = {"k": max(1, min(WINDOW, new["param"].get("k",1) + random.choice([-1,0,1])))}
    elif t=="pattern":
        neww = max(2, min(WINDOW, new["param"].get("w",3) + random.choice([-1,0,1])))
        new["param"] = {"w": neww, "min_matches":1}
    elif t=="probabilistic":
        nb = new["param"].get("bias",0.5) + random.uniform(-0.05,0.05)
        new["param"] = {"use_base": new["param"].get("use_base",False), "bias": float(max(0.35, min(0.65, nb)))}
    elif t=="fixed":
        new["param"] = {"fixed": random.choice(["Tài","Xỉu"])}
    new["fitness"] = max(0.5, new.get("fitness",1.0) * random.uniform(0.9,1.1))
    new["wins"] = 0
    new["trials"] = 0
    new["last_pred"] = None
    return new

def evolve_population(force_replace=3):
    """Evolve population by replacing worst agents with mutated versions of best ones."""
    pop = st.session_state.population
    # sort by fitness ascending
    pop_sorted = sorted(pop, key=lambda x: x["fitness"])
    k = max(1, force_replace)
    worst = pop_sorted[:k]
    best = pop_sorted[-k:]
    # replace each worst by mutated copy of random best
    for i in range(k):
        donor = random.choice(best)
        mutated = mutate_agent(donor)
        # replace in st.session_state.population by id
        for j, a in enumerate(st.session_state.population):
            if a["id"] == worst[i]["id"]:
                st.session_state.population[j] = mutated
                break
    # small random new agents sometimes
    if random.random() < 0.3:
        idx = random.randrange(len(st.session_state.population))
        st.session_state.population[idx] = random_strategy()
    # normalize fitness
    normalize_population()

def normalize_population():
    total = sum(a["fitness"] for a in st.session_state.population)
    if total <= 0:
        for a in st.session_state.population:
            a["w_norm"]=1.0/len(st.session_state.population)
    else:
        for a in st.session_state.population:
            a["w_norm"] = a["fitness"]/total

# -------------------------
# Base ML training (manual)
# -------------------------
def train_base_models():
    hist = st.session_state.history
    X,y = create_Xy = create_Xy if False else None  # no-op to avoid lint
    X, y = create_Xy_simple(hist := st.session_state.history)  # helper defined below
    if len(X)==0 or len(set(y))<2:
        st.warning("Không đủ dữ liệu đa dạng để huấn luyện base models (cần cả Tài & Xỉu).")
        return
    # train LR, RF, XGB with light config
    lr = LogisticRegression(max_iter=200, solver="liblinear", random_state=RANDOM_STATE)
    lr.fit(X,y)
    rf = RandomForestClassifier(n_estimators=60, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X,y)
    xgb = XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.2, verbosity=0, use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)
    xgb.fit(X,y)
    st.session_state.models["LR"]=lr
    st.session_state.models["RF"]=rf
    st.session_state.models["XGB"]=xgb
    st.success("Huấn luyện base models xong.")
    # update immediate preds
    update_predictions_display()

def create_Xy_simple(history):
    """Return X,y for base models - safe version"""
    H = encode(history)
    X,y = [],[]
    for i in range(len(H)-WINDOW):
        X.append(H[i:i+WINDOW])
        y.append(H[i+WINDOW])
    if len(X)==0:
        return np.array([]), np.array([])
    return np.array(X), np.array(y)

# -------------------------
# Prediction orchestration
# -------------------------
def update_predictions_display():
    """Compute predictions from base models (if present) and from all strategies, and compute ensemble."""
    hist = st.session_state.history
    ensure_history_strings()
    last_window = encode(hist[-WINDOW:]) if len(hist) >= WINDOW else None

    # compute base model probabilities (prob of Tài)
    base_probs = {}
    for key, name in [("LR","LR"),("RF","RF"),("XGB","XGB")]:
        model = st.session_state.models.get(key)
        if model is None or last_window is None:
            base_probs[key] = None
        else:
            try:
                p = model.predict_proba([last_window])[0][1]
                base_probs[key]=float(p)
            except Exception:
                base_probs[key]=None

    # strategy agents predictions
    for agent in st.session_state.population:
        pred, conf = strategy_predict(agent, hist, base_probs)
        agent["last_pred"]=pred
        agent["last_conf"]=conf

    # compute weighted vote among strategies (weights from fitness normalized w_norm)
    wvote = {"Tài":0.0,"Xỉu":0.0}
    for a in st.session_state.population:
        if a.get("last_pred") is None: continue
        w = a.get("w_norm", 1.0/len(st.session_state.population))
        # use confidence to scale
        wvote[a["last_pred"]] += w * a.get("last_conf",0.6)

    # compute base aggregate if available
    base_agg_score = None
    probs = [v for v in base_probs.values() if v is not None]
    if probs:
        avg = float(np.mean(probs))
        base_agg_score = avg  # prob of Tài

    # final ensemble: combine strategy vote and base agg
    # weights: ws (strategies) 0.7, wb (base) 0.3 if base exists
    ws, wb = 0.75, 0.25
    score_tai = 0.0
    score_xiu = 0.0
    # normalize wvote
    total_wvote = wvote["Tài"] + wvote["Xỉu"]
    if total_wvote <= 0:
        # default equal
        strat_tai = 0.5
    else:
        strat_tai = wvote["Tài"]/total_wvote
    strat_xiu = 1 - strat_tai
    score_tai += ws * strat_tai
    score_xiu += ws * strat_xiu
    if base_agg_score is not None:
        score_tai += wb * base_agg_score
        score_xiu += wb * (1-base_agg_score)
    # normalize
    ssum = score_tai+score_xiu
    if ssum>0:
        score_tai/=ssum; score_xiu/=ssum
    final_pred = "Tài" if score_tai>=score_xiu else "Xỉu"

    # save displays
    st.session_state.display = {
        "base_probs": base_probs,
        "strategy_vote": {"Tài":strat_tai,"Xỉu":strat_xiu},
        "final_score": score_tai,
        "final_pred": final_pred
    }

# -------------------------
# Recording a real result
# -------------------------
def record_result(real):
    if real not in ("Tài","Xỉu"):
        return
    # update stats relative to current displayed preds before appending history
    # We compare each agent's last_pred (these are predictions for the round that just finished)
    for a in st.session_state.population:
        lp = a.get("last_pred")
        if lp is None:
            continue
        a["trials"] = a.get("trials",0)+1
        if lp == real:
            a["wins"]= a.get("wins",0)+1
            a["fitness"]= a.get("fitness",1.0)*1.06
        else:
            a["fitness"]= a.get("fitness",1.0)*0.94
        a["fitness"] = float(max(0.1, min(a["fitness"], 50.0)))
    normalize_population()

    # update base stats (if base models predicted before)
    dp = st.session_state.display if "display" in st.session_state else {}
    # append history
    st.session_state.history.append(real)
    if len(st.session_state.history)>MAX_HISTORY:
        st.session_state.history = st.session_state.history[-MAX_HISTORY:]

    # auto-evolve occasionally
    if len(st.session_state.history) % EVO_INTERVAL == 0:
        evolve_population(force_replace=max(1, POP_SIZE//8))

    # update overall displays/predictions for next round (models not retrained automatically)
    update_predictions_display()

# -------------------------
# UI / Init session variables
# -------------------------
if "population" not in st.session_state:
    st.session_state.population = init_population(POP_SIZE)
    normalize_population()
if "models" not in st.session_state:
    st.session_state.models = {"LR":None,"RF":None,"XGB":None,"META":None}
if "preds" not in st.session_state:
    st.session_state.preds = {}
if "display" not in st.session_state:
    st.session_state.display = {"base_probs": {}, "strategy_vote":{"Tài":0.5,"Xỉu":0.5}, "final_score":0.5, "final_pred":"—"}
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# UI Layout
# -------------------------
st.title("🎯 AI Tài/Xỉu — Cấp 3: Self-Generating Strategies & Evolution")
st.write("Manual train base ML; strategy population evolves & adapts automatically. Data stored in session only.")

# control buttons
c1,c2,c3,c4 = st.columns([1,1,1,1])
with c1:
    if st.button("🔴 TÀI"):
        record_result("Tài")
with c2:
    if st.button("🔵 XỈU"):
        record_result("Xỉu")
with c3:
    if st.button("⚙️ Huấn luyện base models (manual)"):
        with st.spinner("Đang huấn luyện base models (LR/RF/XGB)..."):
            train_base_models()
with c4:
    if st.button("🔁 Evolve now"):
        evolve_population(force_replace=max(1,POP_SIZE//6))
        st.success("Population evolved.")

if st.button("🧹 Reset all"):
    # reset everything
    st.session_state.history=[]
    st.session_state.population = init_population(POP_SIZE)
    normalize_population()
    st.session_state.models = {"LR":None,"RF":None,"XGB":None,"META":None}
    st.session_state.display = {"base_probs": {}, "strategy_vote":{"Tài":0.5,"Xỉu":0.5}, "final_score":0.5, "final_pred":"—"}
    st.session_state.ai_history=[]
    st.success("Reset done.")

st.markdown("---")

# history display
st.markdown("### 🧾 Lịch sử (mới nhất bên phải)")
if st.session_state.history:
    safe_hist = [str(x) for x in st.session_state.history[-60:]]
    st.write(" → ".join(safe_hist))
else:
    st.info("Chưa có dữ liệu. Nhấn 'TÀI' hoặc 'XỈU' để bắt đầu.")

st.markdown("---")

# update predictions (display)
update_predictions_display()

# show base model probs
st.markdown("## 🔬 Base models (manual train)")
bp = st.session_state.display.get("base_probs",{})
st.write(f"LR prob(Tài): {bp.get('LR'):.3f}" if bp.get('LR') is not None else "LR: —")
st.write(f"RF prob(Tài): {bp.get('RF'):.3f}" if bp.get('RF') is not None else "RF: —")
st.write(f"XGB prob(Tài): {bp.get('XGB'):.3f}" if bp.get('XGB') is not None else "XGB: —")

st.markdown("---")

# ensemble result
final_pred = st.session_state.display.get("final_pred","—")
final_score = st.session_state.display.get("final_score",0.5)
st.markdown(f"## 🎯 Dự đoán hệ thống: **{final_pred}**  —  Tin cậy: **{final_score:.1%}**")

st.markdown("---")

# show top strategies table
st.markdown("### 🧠 Top strategies (by fitness)")
pop_sorted = sorted(st.session_state.population, key=lambda a: a["fitness"], reverse=True)
top = pop_sorted[:8]
rows=[]
for a in top:
    rows.append({
        "id": a["id"],
        "type": a["type"],
        "param": json.dumps(a["param"]),
        "fitness": f"{a['fitness']:.2f}",
        "wins": a.get("wins",0),
        "trials": a.get("trials",0),
        "last_pred": a.get("last_pred")
    })
st.table(rows)

st.markdown("---")
# population summary
avg_fit = np.mean([a["fitness"] for a in st.session_state.population])
st.write(f"Population size: {len(st.session_state.population)} — Avg fitness: {avg_fit:.2f}")

# debug / logs
if st.checkbox("Show full population (debug)"):
    st.json([{k:v for k,v in a.items() if k in ('id','type','param','fitness','wins','trials','last_pred')} for a in st.session_state.population])

# allow export/import history if wanted
st.markdown("---")
col_e1, col_e2 = st.columns([1,1])
with col_e1:
    if st.button("Export history JSON"):
        st.download_button("Download history.json", data=json.dumps(st.session_state.history), file_name="history.json")
with col_e2:
    uploaded = st.file_uploader("Import history JSON", type=["json"])
    if uploaded:
        try:
            content = json.load(uploaded)
            if isinstance(content, list):
                st.session_state.history = [str(x) for x in content if str(x) in ("Tài","Xỉu")]
                st.success("Imported history.")
                update_predictions_display()
            else:
                st.error("File format invalid.")
        except Exception as e:
            st.error("Cannot parse JSON.")

st.markdown("---")
st.caption("Notes: Strategies evolve automatically every few recorded results and also when you press 'Evolve now'. Base ML training is manual to avoid blocking UI.")
