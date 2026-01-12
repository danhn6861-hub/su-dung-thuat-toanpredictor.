import streamlit as st
import random
import time
import math

# Cấu hình trang tối ưu hiệu năng
st.set_page_config(page_title="Inferno 3D Dice", layout="centered", initial_sidebar_state="collapsed")

# --- QUẢN LÝ SESSION STATE ---
for key, val in {
    'money': 1000, 'history': [], 'start_time': time.time(),
    'bet_info': {"side": None, "amount": 0, "confirmed": False},
    'last_bet': None, 'win_anim': False, 'round_result': None
}.items():
    if key not in st.session_state: st.session_state[key] = val

def get_positions():
    """Thuật toán chống va chạm: Đảm bảo xúc sắc không dính nhau"""
    positions = []
    min_distance = 45  # Khoảng cách tối thiểu để không dính (xúc sắc rộng 40px)
    
    for _ in range(3):
        for _ in range(100): # Thử tối đa 100 lần tìm vị trí trống
            tx = random.randint(-60, 60)
            ty = random.randint(-60, 60)
            
            # Kiểm tra khoảng cách với các viên đã đặt trước đó
            overlap = False
            for p in positions:
                dist = math.sqrt((tx - p['tX'])**2 + (ty - p['tY'])**2)
                if dist < min_distance:
                    overlap = True
                    break
            
            if not overlap:
                positions.append({
                    "tX": tx, "tY": ty, 
                    "rX": random.randint(0, 360), 
                    "rY": random.randint(0, 360)
                })
                break
    # Nếu xui quá không tìm được chỗ (hiếm), trả về vị trí mặc định tán ra
    if len(positions) < 3:
        return [{"tX": -50, "tY": 0, "rX": 0, "rY": 0}, 
                {"tX": 50, "tY": 0, "rX": 0, "rY": 0}, 
                {"tX": 0, "tY": 50, "rX": 0, "rY": 0}]
    return positions

if 'dice_positions' not in st.session_state:
    st.session_state.dice_positions = get_positions()

# --- CSS SIÊU CẤP ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap');
    .stApp { background: #000; font-family: 'Segoe UI', sans-serif; }
    
    /* Hiệu ứng lửa & Thông báo */
    @keyframes fire {
        0% { text-shadow: 0 0 10px #f00, 0 0 20px #ff0; color: #fff; transform: scale(1); }
        50% { text-shadow: 0 0 20px #ff0, 0 0 40px #f00; color: #ffd700; transform: scale(1.1); }
        100% { text-shadow: 0 0 10px #f00, 0 0 20px #ff0; color: #fff; transform: scale(1); }
    }
    .fire-text { animation: fire 0.5s infinite; font-family: 'Bebas Neue'; font-size: 50px !important; text-align: center; }
    .loss-text { color: #555; font-family: 'Bebas Neue'; font-size: 40px; text-align: center; }

    /* Lịch sử hạt tròn */
    .history-container {
        display: flex; flex-wrap: wrap; justify-content: center; gap: 8px;
        margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 20px;
    }
    .dot {
        width: 22px; height: 22px; border-radius: 50%; display: flex;
        align-items: center; justify-content: center; font-size: 12px; font-weight: bold;
    }
    .dot-t { background: #111; color: white; border: 1.5px solid #444; box-shadow: 0 0 5px rgba(255,255,255,0.2); }
    .dot-x { background: #eee; color: black; border: 1.5px solid #999; }

    /* Bàn chơi & Xúc sắc */
    .casino-table {
        background: radial-gradient(circle at center, #1a1a1a 0%, #000 100%);
        border: 2px solid #333; border-radius: 40px; padding: 10px; margin: auto;
    }
    .plate { height: 250px; position: relative; perspective: 1000px; display: flex; align-items: center; justify-content: center; }
    .plate-base { width: 220px; height: 220px; background: #111; border-radius: 50%; border: 10px solid #222; transform: rotateX(30deg); position: absolute; }

    @keyframes physical_toss {
        0% { transform: translate3d(0, 0, 0) rotateX(0); }
        30% { transform: translate3d(var(--tx), -200px, 150px) rotateX(720deg) rotateY(360deg); }
        100% { transform: translate3d(0, 0, 0) rotateX(var(--rx)) rotateY(var(--ry)); }
    }
    .dice { width: 40px; height: 40px; position: absolute; transform-style: preserve-3d; }
    .rolling { animation: physical_toss 0.8s ease-out forwards; }

    .cube-face {
        position: absolute; width: 40px; height: 40px; background: #fff;
        border: 1px solid #ddd; border-radius: 6px; display: flex;
        align-items: center; justify-content: center; font-size: 20px; font-weight: bold;
    }
    .front { transform: rotateY(0deg) translateZ(20px); color: red; }
    .back { transform: rotateY(180deg) translateZ(20px); }
    .right { transform: rotateY(90deg) translateZ(20px); }
    .left { transform: rotateY(-90deg) translateZ(20px); }
    .top { transform: rotateX(90deg) translateZ(20px); }
    .bottom { transform: rotateX(-90deg) translateZ(20px); }

    .stButton>button { border-radius: 15px; height: 55px; font-weight: 700; transition: 0.3s; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
now = time.time()
remaining = max(0, int(25 - (now - st.session_state.start_time)))

c1, c2, c3 = st.columns([1,1,1])
with c1: st.markdown(f"<p style='color:grey;margin:0'>BALANCE</p><h3 style='color:#0f0;margin:0'>${st.session_state.money}</h3>", unsafe_allow_html=True)
with c2: 
    if st.session_state.win_anim:
        st.markdown("<div class='fire-text'>WIN!</div>", unsafe_allow_html=True)
    elif st.session_state.round_result == "LOSE":
        st.markdown("<div class='loss-text'>BET LOST</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h1 style='text-align:center;color:gold;margin:0;font-family:Bebas Neue'>{remaining}s</h1>", unsafe_allow_html=True)
with c3: st.markdown(f"<p style='text-align:right;color:grey;margin:0'>ROADMAP</p><h3 style='text-align:right;color:cyan;margin:0'>{len(st.session_state.history)} VÁN</h3>", unsafe_allow_html=True)

def render_dice_engine(is_rolling=False, results=None):
    html = '<div class="casino-table"><div class="plate"><div class="plate-base"></div>'
    rot_map = {1: [0,0], 2: [0,-90], 3: [0,-180], 4: [0,90], 5: [-90,0], 6: [90,0]}
    for i in range(3):
        pos = st.session_state.dice_positions[i]
        roll_class = "rolling" if is_rolling else ""
        res_rot = rot_map.get(results[i], [pos['rX'], pos['rY']]) if results else [pos['rX'], pos['rY']]
        style = f"left:calc(50% + {pos['tX']}px - 20px); top:calc(50% + {pos['tY']}px - 20px); --tx:{pos['tX']*2}px; --rx:{res_rot[0]}deg; --ry:{res_rot[1]}deg;"
        val = results[i] if results else "?"
        html += f"""<div class="dice {roll_class}" style="{style} transform: rotateX({res_rot[0]}deg) rotateY({res_rot[1]}deg);">
            <div class="cube-face front">{val}</div><div class="cube-face back">6</div>
            <div class="cube-face right">2</div><div class="cube-face left">5</div>
            <div class="cube-face top">3</div><div class="cube-face bottom">4</div>
        </div>"""
    return html + "</div></div>"

placeholder = st.empty()

if remaining > 0:
    st.session_state.win_anim = False
    st.session_state.round_result = None
    placeholder.markdown(render_dice_engine(), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    is_locked = st.session_state.bet_info["confirmed"]
    
    with col1:
        if st.button("TÀI (BLACK)", use_container_width=True, disabled=is_locked, key="btn_t", type="primary" if st.session_state.bet_info["side"]=="T" else "secondary"):
            st.session_state.bet_info["side"] = "T"; st.rerun()
    with col2:
        if st.button("XỈU (WHITE)", use_container_width=True, disabled=is_locked, key="btn_x", type="primary" if st.session_state.bet_info["side"]=="X" else "secondary"):
            st.session_state.bet_info["side"] = "X"; st.rerun()

    if st.session_state.bet_info["side"] and not is_locked:
        st.markdown(f"<p style='text-align:center;color:gold;margin-top:10px'>BET: {st.session_state.bet_info['side']} | ${st.session_state.bet_info['amount']}</p>", unsafe_allow_html=True)
        chips = st.columns(4)
        if chips[0].button("+$10"): st.session_state.bet_info["amount"] += 10; st.rerun()
        if chips[1].button("+$50"): st.session_state.bet_info["amount"] += 50; st.rerun()
        if chips[2].button("+$100"): st.session_state.bet_info["amount"] += 100; st.rerun()
        if chips[3].button("RESET"): st.session_state.bet_info = {"side":None, "amount":0, "confirmed":False}; st.rerun()
        
        if st.button("🔥 CONFIRM", type="primary", use_container_width=True):
            if 0 < st.session_state.bet_info["amount"] <= st.session_state.money:
                st.session_state.bet_info["confirmed"] = True
                st.rerun()
    elif is_locked:
        st.markdown(f"<h3 style='text-align:center;color:#0f0'>CONFIRMED: ${st.session_state.bet_info['amount']} ON {st.session_state.bet_info['side']}</h3>", unsafe_allow_html=True)

    # --- LỊCH SỬ HẠT TRÒN ---
    hist_html = '<div class="history-container">'
    for h in st.session_state.history[-24:]: 
        cls = "dot-t" if h == "T" else "dot-x"
        hist_html += f'<div class="dot {cls}">{h}</div>'
    hist_html += '</div>'
    st.markdown(hist_html, unsafe_allow_html=True)

    time.sleep(1)
    st.rerun()

else:
    # --- MỞ BÁT ---
    dice_res = [random.randint(1, 6) for _ in range(3)]
    total = sum(dice_res)
    result_side = "T" if total >= 11 else "X"
    st.session_state.dice_positions = get_positions() # Tạo vị trí mới không dính nhau
    
    placeholder.markdown(render_dice_engine(is_rolling=True), unsafe_allow_html=True)
    time.sleep(0.8)
    placeholder.markdown(render_dice_engine(results=dice_res), unsafe_allow_html=True)
    
    if st.session_state.bet_info["confirmed"]:
        if st.session_state.bet_info["side"] == result_side:
            win_amt = st.session_state.bet_info["amount"]
            st.session_state.money += win_amt
            st.session_state.win_anim = True
            st.toast(f"WIN +${win_amt}!", icon="🔥")
        else:
            loss_amt = st.session_state.bet_info["amount"]
            st.session_state.money -= loss_amt
            st.session_state.round_result = "LOSE"
            st.toast(f"LOSE -${loss_amt}", icon="💸")
    
    st.session_state.history.append(result_side)
    time.sleep(4)
    st.session_state.start_time = time.time()
    st.session_state.bet_info = {"side": None, "amount": 0, "confirmed": False}
    st.rerun()
