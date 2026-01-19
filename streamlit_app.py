import streamlit as st
import torch
import numpy as np
import os
import sys
import time
from datetime import timedelta
import matplotlib.pyplot as plt
from collections import deque
import folium
from streamlit_folium import st_folium
import altair as alt
import pandas as pd
import random

# Add PokemonRL to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PokemonRL'))

from PokemonRL.src.env.pokemon_env import PokemonSimEnv
from PokemonRL.src.agents.explorer import ExplorerAgent
from PokemonRL.src.agents.tactician import TacticianAgent
from PokemonRL.src.agents.strategist import Strategist

# Page config
st.set_page_config(
    page_title="Pokédex RL Trainer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Pokemon-themed CSS
st.markdown("""
<style>
    /* Pokedex theme */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Sidebar Pokedex style */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #CC0000 0%, #CC0000 60%, #FFFFFF 60%, #FFFFFF 100%);
        border-right: 4px solid #000;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #FFFFFF;
    }
    
    /* Pokedex button in sidebar */
    .pokedex-circle {
        width: 60px;
        height: 60px;
        background: radial-gradient(circle at 30% 30%, #4FC3F7, #0277BD);
        border: 3px solid #000;
        border-radius: 50%;
        margin: 10px auto;
        box-shadow: inset 0 0 10px rgba(255,255,255,0.5);
    }
    
    /* Main content cards */
    .stMetric {
        background: rgba(255, 255, 255, 0.95);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #FFD700;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Combat view */
    .combat-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 20px;
        border-radius: 15px;
        border: 3px solid #000;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
    }
    
    /* HP bars */
    .hp-bar {
        height: 20px;
        background: #ccc;
        border: 2px solid #000;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .hp-fill {
        height: 100%;
        background: linear-gradient(90deg, #00FF00 0%, #FFFF00 50%, #FF0000 100%);
        transition: width 0.3s ease;
    }
    
    /* Buttons Pokemon style */
    .stButton button {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #000;
        font-weight: bold;
        border: 2px solid #000;
        border-radius: 10px;
        padding: 10px 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #FFA500 0%, #FF8C00 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.4);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #FFD700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    /* Alert messages */
    .stAlert {
        border-radius: 10px;
        border: 2px solid #000;
    }
</style>
""", unsafe_allow_html=True)

# Title with Pokemon style
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1 style='font-size: 48px; color: #FFD700; text-shadow: 3px 3px 6px #000;'>
        ⚡ POKÉDEX RL TRAINER ⚡
    </h1>
    <p style='color: #FFF; font-size: 18px;'>Sistema de Entrenamiento con Inteligencia Artificial</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'env' not in st.session_state:
    st.session_state.env = None
if 'explorer' not in st.session_state:
    st.session_state.explorer = None
if 'tactician' not in st.session_state:
    st.session_state.tactician = None
if 'strategist' not in st.session_state:
    st.session_state.strategist = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = {'episodes': [], 'rewards': [], 'epsilons': [], 'gps_coords': [], 'wins': []}
if 'visualization_state' not in st.session_state:
    st.session_state.visualization_state = None
if 'step_count' not in st.session_state:
    st.session_state.step_count = 0
if 'total_reward' not in st.session_state:
    st.session_state.total_reward = 0
if 'done' not in st.session_state:
    st.session_state.done = False
if 'auto_play' not in st.session_state:
    st.session_state.auto_play = False
if 'battle_log' not in st.session_state:
    st.session_state.battle_log = deque(maxlen=10)
if 'last_damage' not in st.session_state:
    st.session_state.last_damage = 0
if 'last_move' not in st.session_state:
    st.session_state.last_move = ""
if 'kpi_metrics' not in st.session_state:
    st.session_state.kpi_metrics = {
        'q_values': [],
        'losses': [],
        'exploration_rate': [],
        'win_rate': []
    }
if 'selected_marker' not in st.session_state:
    st.session_state.selected_marker = None
if 'last_q_values' not in st.session_state:
    st.session_state.last_q_values = None
if 'map_mode' not in st.session_state:
    st.session_state.map_mode = 'MAP'  # 'MAP' or 'COMBAT'

# Sidebar navigation with Pokedex theme
st.sidebar.markdown("""
<div style='text-align: center; padding: 20px 0;'>
    <div class='pokedex-circle'></div>
    <h2 style='color: #FFF; text-shadow: 2px 2px 4px #000;'>POKÉDEX</h2>
    <p style='color: #000; background: #FFF; padding: 5px; border-radius: 5px; margin: 10px;'>
        Sistema de Navegación
    </p>
</div>
""", unsafe_allow_html=True)

mode = st.sidebar.radio(
    "Selecciona el modo:",
    ["Modo Entrenamiento", "Modo Visualización", "Modo Exploración", "Dashboard Estadísticas"],
    label_visibility="collapsed"
)

def initialize_agents():
    """Initialize or get existing agents"""
    if st.session_state.env is None:
        st.session_state.env = PokemonSimEnv(verbose=False)
        st.session_state.explorer = ExplorerAgent(obs_shape=(9, 10, 10), n_actions=4, lr=1e-4)
        st.session_state.tactician = TacticianAgent(input_dim=16, n_actions=5, lr=1e-3)
        st.session_state.strategist = Strategist(st.session_state.env.pokedex)
    return st.session_state.env, st.session_state.explorer, st.session_state.tactician, st.session_state.strategist

def load_checkpoints(explorer, tactician, episode_num=3000):
    """Load trained model checkpoints"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_dir = os.path.join(os.path.dirname(__file__), 'PokemonRL', 'checkpoints')
        
        explorer_path = os.path.join(base_dir, f"explorer_ep{episode_num}.pth")
        tactician_path = os.path.join(base_dir, f"tactician_ep{episode_num}.pth")
        
        if os.path.exists(explorer_path) and os.path.exists(tactician_path):
            explorer.policy_net.load_state_dict(torch.load(explorer_path, map_location=device, weights_only=True))
            tactician.policy_net.load_state_dict(torch.load(tactician_path, map_location=device, weights_only=True))
            return True
        return False
    except Exception as e:
        st.error(f"Error cargando checkpoints: {e}")
        return False

def save_checkpoints(explorer, tactician, episode):
    """Save model checkpoints"""
    try:
        base_dir = os.path.join(os.path.dirname(__file__), 'PokemonRL', 'checkpoints')
        os.makedirs(base_dir, exist_ok=True)
        
        torch.save(explorer.policy_net.state_dict(), 
                  os.path.join(base_dir, f"explorer_ep{episode}.pth"))
        torch.save(tactician.policy_net.state_dict(), 
                  os.path.join(base_dir, f"tactician_ep{episode}.pth"))
        return True
    except Exception as e:
        st.error(f"Error guardando checkpoints: {e}")
        return False

def generate_pokemon_geojson(center_lat=40.4168, center_lon=-3.7038, num_gyms=5, num_wild=10):
    """Generate random GeoJSON data for Pokemon Gyms and Wild Pokemon"""
    gyms = []
    wild_pokemon = []
    
    # Pokemon types for different location types
    location_types = [
        {"type": "park", "pokemon_type": "grass", "icon": "🌳"},
        {"type": "water", "pokemon_type": "water", "icon": "💧"},
        {"type": "city", "pokemon_type": "electric", "icon": "⚡"},
        {"type": "mountain", "pokemon_type": "rock", "icon": "⛰️"},
        {"type": "desert", "pokemon_type": "fire", "icon": "🔥"}
    ]
    
    # Generate gyms
    for i in range(num_gyms):
        # Random offset (approximately 0.01 degrees = ~1km)
        lat_offset = (random.random() - 0.5) * 0.02
        lon_offset = (random.random() - 0.5) * 0.02
        
        location = random.choice(location_types)
        gym = {
            "type": "Feature",
            "properties": {
                "id": f"gym_{i}",
                "name": f"Gimnasio Pokémon {i+1}",
                "location_type": location["type"],
                "pokemon_type": location["pokemon_type"],
                "icon": location["icon"],
                "category": "gym"
            },
            "geometry": {
                "type": "Point",
                "coordinates": [center_lon + lon_offset, center_lat + lat_offset]
            }
        }
        gyms.append(gym)
    
    # Generate wild pokemon
    wild_types = ["fire", "water", "grass", "electric", "rock", "normal"]
    for i in range(num_wild):
        lat_offset = (random.random() - 0.5) * 0.03
        lon_offset = (random.random() - 0.5) * 0.03
        
        pokemon_type = random.choice(wild_types)
        wild = {
            "type": "Feature",
            "properties": {
                "id": f"wild_{i}",
                "name": f"Pokémon Salvaje #{i+1}",
                "pokemon_type": pokemon_type,
                "category": "wild"
            },
            "geometry": {
                "type": "Point",
                "coordinates": [center_lon + lon_offset, center_lat + lat_offset]
            }
        }
        wild_pokemon.append(wild)
    
    return {
        "gyms": {"type": "FeatureCollection", "features": gyms},
        "wild": {"type": "FeatureCollection", "features": wild_pokemon}
    }

def create_folium_map(center_lat=40.4168, center_lon=-3.7038):
    """Create an interactive Folium map with Pokemon markers"""
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    # Generate Pokemon data
    geojson_data = generate_pokemon_geojson(center_lat, center_lon)
    
    # Add gyms to map
    for gym in geojson_data["gyms"]["features"]:
        coords = gym["geometry"]["coordinates"]
        props = gym["properties"]
        
        folium.Marker(
            location=[coords[1], coords[0]],
            popup=folium.Popup(
                f"<b>{props['icon']} {props['name']}</b><br>"
                f"Tipo: {props['location_type']}<br>"
                f"Pokémon: {props['pokemon_type'].upper()}<br>"
                f"<i>Click para batallar</i>",
                max_width=200
            ),
            tooltip=props['name'],
            icon=folium.Icon(color='red', icon='home', prefix='fa')
        ).add_to(m)
    
    # Add wild pokemon to map
    for wild in geojson_data["wild"]["features"]:
        coords = wild["geometry"]["coordinates"]
        props = wild["properties"]
        
        folium.Marker(
            location=[coords[1], coords[0]],
            popup=folium.Popup(
                f"<b>⚡ {props['name']}</b><br>"
                f"Tipo: {props['pokemon_type'].upper()}<br>"
                f"<i>Click para capturar</i>",
                max_width=200
            ),
            tooltip=props['name'],
            icon=folium.Icon(color='green', icon='leaf', prefix='fa')
        ).add_to(m)
    
    return m, geojson_data

# ========== MODO ENTRENAMIENTO ==========
if mode == "Modo Entrenamiento":
    st.header("🎓 Modo Entrenamiento")
    st.write("Entrena los agentes de IA con Reinforcement Learning")
    
    # Training parameters
    col1, col2 = st.columns(2)
    with col1:
        total_episodes = st.number_input("TOTAL_EPISODES", min_value=10, max_value=10000, value=100, step=10)
    with col2:
        batch_size = st.number_input("BATCH_SIZE", min_value=8, max_value=256, value=32, step=8)
    
    max_steps = st.slider("MAX_STEPS por episodio", min_value=50, max_value=500, value=300, step=50)
    
    # Initialize agents button
    if st.button("🔄 Inicializar Agentes"):
        env, explorer, tactician, strategist = initialize_agents()
        # Update batch size
        explorer.batch_size = batch_size
        tactician.batch_size = batch_size
        st.success("✅ Agentes inicializados correctamente")
        st.info(f"Dispositivo: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    
    # Training button
    if st.button("🚀 Iniciar Entrenamiento"):
        if st.session_state.env is None:
            st.error("⚠️ Primero debes inicializar los agentes")
        else:
            env = st.session_state.env
            explorer = st.session_state.explorer
            tactician = st.session_state.tactician
            strategist = st.session_state.strategist
            
            # Clear previous history
            st.session_state.training_history = {'episodes': [], 'rewards': [], 'epsilons': [], 'gps_coords': [], 'wins': []}
            
            # Progress containers
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_container = st.empty()
            chart_container = st.empty()
            
            start_time = time.time()
            
            try:
                for episode in range(1, total_episodes + 1):
                    # Curriculum Learning
                    if episode < total_episodes * 0.2:
                        map_idx = 0
                    elif episode < total_episodes * 0.4:
                        map_idx = np.random.choice([0, 1])
                    elif episode < total_episodes * 0.6:
                        map_idx = np.random.choice([0, 1, 2])
                    elif episode < total_episodes * 0.8:
                        map_idx = np.random.choice([0, 1, 2, 3])
                    else:
                        map_idx = np.random.choice([0, 1, 2, 3, 4])
                    
                    env.current_map_idx = map_idx
                    
                    # Renovar equipo periódicamente
                    if (episode-1) % 10 == 0:
                        all_ids = list(env.pokedex.keys())
                        party_ids = np.random.choice(all_ids, 6, replace=False) if len(all_ids) >= 6 else all_ids
                        strategist.set_party(party_ids)
                    
                    target = np.random.choice(["fire", "water", "grass", "electric", "rock"])
                    best = strategist.build_team(target)
                    
                    env.my_pokemon = best.copy()
                    env.my_pokemon['level'] = 5
                    env.my_pokemon['exp'] = 0
                    
                    state, _ = env.reset()
                    
                    total_reward = 0
                    done = False
                    steps = 0
                    
                    while not done and steps < max_steps:
                        steps += 1
                        
                        if env.mode == "MAP":
                            action = explorer.select_action(state)
                            next_state, reward, done, _, _ = env.step(action)
                            
                            if env.mode == "COMBAT":
                                state = next_state
                                continue
                            
                            explorer.learn(state, action, reward, next_state, done)
                            state = next_state
                            total_reward += reward
                        
                        elif env.mode == "COMBAT":
                            action = tactician.select_action(state)
                            next_state, reward, done, _, _ = env.step(action + 4)
                            
                            if env.mode == "MAP":
                                state = next_state
                                continue
                            
                            tactician.learn(state, action, reward, next_state, done)
                            state = next_state
                            total_reward += reward
                    
                    # Epsilon decay
                    if explorer.epsilon > explorer.epsilon_min:
                        explorer.epsilon *= 0.9995
                    if tactician.epsilon > tactician.epsilon_min:
                        tactician.epsilon *= 0.9993
                    
                    # Store history
                    # Generate random GPS coordinates near Madrid for this episode
                    gps_lat = 40.4168 + (np.random.random() - 0.5) * 0.1
                    gps_lon = -3.7038 + (np.random.random() - 0.5) * 0.1
                    is_win = total_reward > 0  # Simple heuristic for win detection
                    
                    st.session_state.training_history['episodes'].append(episode)
                    st.session_state.training_history['rewards'].append(total_reward)
                    st.session_state.training_history['epsilons'].append(explorer.epsilon)
                    st.session_state.training_history['gps_coords'].append((gps_lat, gps_lon))
                    st.session_state.training_history['wins'].append(is_win)
                    
                    # Update UI every 10 episodes
                    if episode % 10 == 0:
                        progress = episode / total_episodes
                        progress_bar.progress(progress)
                        
                        avg_reward = np.mean(st.session_state.training_history['rewards'][-100:]) if len(st.session_state.training_history['rewards']) >= 100 else np.mean(st.session_state.training_history['rewards'])
                        
                        elapsed = time.time() - start_time
                        eta = (elapsed / episode) * (total_episodes - episode)
                        
                        status_text.text(f"Episodio {episode}/{total_episodes} | Mapa {map_idx} | Reward: {total_reward:.1f} | Avg100: {avg_reward:.1f} | Epsilon: {explorer.epsilon:.3f}")
                        
                        # Metrics
                        with metrics_container.container():
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Episodio", f"{episode}/{total_episodes}")
                            col2.metric("Reward Promedio", f"{avg_reward:.2f}")
                            col3.metric("Epsilon", f"{explorer.epsilon:.3f}")
                            col4.metric("ETA", str(timedelta(seconds=int(eta))))
                        
                        # Chart
                        if len(st.session_state.training_history['episodes']) > 1:
                            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
                            
                            # Rewards
                            ax1.plot(st.session_state.training_history['episodes'], 
                                   st.session_state.training_history['rewards'], alpha=0.3)
                            if len(st.session_state.training_history['rewards']) >= 10:
                                # Moving average - compute only when needed for display
                                window = min(10, len(st.session_state.training_history['rewards']))
                                rewards_array = np.array(st.session_state.training_history['rewards'])
                                rewards_ma = np.convolve(rewards_array, 
                                                        np.ones(window)/window, mode='valid')
                                ax1.plot(range(window, len(rewards_array)+1), 
                                       rewards_ma, 'r-', linewidth=2)
                            ax1.set_xlabel('Episodio')
                            ax1.set_ylabel('Reward')
                            ax1.set_title('Recompensas por Episodio')
                            ax1.grid(True, alpha=0.3)
                            
                            # Epsilon
                            ax2.plot(st.session_state.training_history['episodes'], 
                                   st.session_state.training_history['epsilons'], 'g-')
                            ax2.set_xlabel('Episodio')
                            ax2.set_ylabel('Epsilon')
                            ax2.set_title('Epsilon Decay')
                            ax2.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            chart_container.pyplot(fig)
                            plt.close()
                    
                    # Save checkpoints every 200 episodes
                    if episode % 200 == 0:
                        if save_checkpoints(explorer, tactician, episode):
                            st.success(f"💾 Checkpoint guardado en episodio {episode}")
                
                # Final save
                save_checkpoints(explorer, tactician, total_episodes)
                
                progress_bar.progress(1.0)
                st.success(f"✅ Entrenamiento completado! Tiempo total: {str(timedelta(seconds=int(time.time() - start_time)))}")
                
            except Exception as e:
                st.error(f"❌ Error durante el entrenamiento: {e}")

# ========== MODO VISUALIZACIÓN ==========
elif mode == "Modo Visualización":
    st.markdown("""
    <h2 style='color: #FFD700; text-align: center;'>👁️ MODO COMBATE - VISUALIZACIÓN</h2>
    <p style='color: #FFF; text-align: center;'>Observa las batallas en tiempo real estilo Pokémon</p>
    """, unsafe_allow_html=True)
    
    # Initialize/Load models
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Inicializar Entorno"):
            env, explorer, tactician, strategist = initialize_agents()
            st.success("✅ Entorno inicializado")
    
    with col2:
        checkpoint_episode = st.selectbox("Cargar checkpoint", [3000, 2000, 1000, 200], index=0)
        if st.button("📥 Cargar Modelo Entrenado"):
            if st.session_state.env is None:
                st.error("⚠️ Primero inicializa el entorno")
            else:
                if load_checkpoints(st.session_state.explorer, st.session_state.tactician, checkpoint_episode):
                    st.success(f"✅ Modelo ep{checkpoint_episode} cargado correctamente")
                else:
                    st.warning(f"⚠️ No se encontraron checkpoints para episodio {checkpoint_episode}")
    
    with col3:
        if st.button("🔄 Reiniciar Entorno"):
            if st.session_state.env is None:
                st.error("⚠️ Primero inicializa el entorno")
            else:
                env = st.session_state.env
                strategist = st.session_state.strategist
                
                # Setup team
                all_ids = list(env.pokedex.keys())
                party_ids = np.random.choice(all_ids, 6, replace=False) if len(all_ids) >= 6 else all_ids
                strategist.set_party(party_ids)
                
                target = np.random.choice(["fire", "water", "grass", "electric", "rock"])
                best = strategist.build_team(target)
                
                env.my_pokemon = best.copy()
                env.my_pokemon['level'] = 5
                env.my_pokemon['exp'] = 0
                
                state, _ = env.reset()
                st.session_state.visualization_state = state
                st.session_state.step_count = 0
                st.session_state.total_reward = 0
                st.session_state.done = False
                st.session_state.battle_log.clear()
                
                st.success("✅ Entorno reiniciado")
    
    # Auto-play toggle
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("▶️ MODO AUTO"):
            st.session_state.auto_play = True
    with col2:
        if st.button("⏸️ PAUSAR"):
            st.session_state.auto_play = False
    with col3:
        single_step = st.button("➡️ 1 Paso")
    with col4:
        multiple_steps = st.button("⏩ 10 Pasos")
    
    # Auto-play execution
    if st.session_state.auto_play and st.session_state.env is not None and st.session_state.visualization_state is not None and not st.session_state.done:
        env = st.session_state.env
        explorer = st.session_state.explorer
        tactician = st.session_state.tactician
        state = st.session_state.visualization_state
        
        # Execute one step
        if env.mode == "MAP":
            action = explorer.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            state = next_state
            st.session_state.total_reward += reward
            
        elif env.mode == "COMBAT":
            action, q_values = tactician.select_action(state, return_q_values=True)
            st.session_state.last_q_values = q_values  # Store for XAI visualization
            next_state, reward, done, _, info = env.step(action + 4)
            
            # Log battle information
            if hasattr(env, 'my_pokemon') and hasattr(env, 'enemy_pokemon'):
                move_name = env.my_pokemon.get('active_moves', ['???'])[min(action, 3)]
                st.session_state.last_move = move_name
                if 'damage' in str(info):
                    st.session_state.last_damage = reward
            
            state = next_state
            st.session_state.total_reward += reward
        
        st.session_state.step_count += 1
        st.session_state.done = done
        st.session_state.visualization_state = state
        
        # Auto-rerun with a small delay using st.empty() placeholder technique
        if not st.session_state.done:
            st.rerun()
    
    # Manual step execution
    if st.session_state.env is not None and st.session_state.visualization_state is not None:
        steps_to_take = 0
        if single_step:
            steps_to_take = 1
        elif multiple_steps:
            steps_to_take = 10
        
        if steps_to_take > 0 and not st.session_state.done:
            env = st.session_state.env
            explorer = st.session_state.explorer
            tactician = st.session_state.tactician
            state = st.session_state.visualization_state
            
            for i in range(steps_to_take):
                if st.session_state.done:
                    break
                
                if env.mode == "MAP":
                    action = explorer.select_action(state)
                    next_state, reward, done, _, _ = env.step(action)
                    state = next_state
                    st.session_state.total_reward += reward
                    
                elif env.mode == "COMBAT":
                    action, q_values = tactician.select_action(state, return_q_values=True)
                    st.session_state.last_q_values = q_values  # Store for XAI visualization
                    next_state, reward, done, _, _ = env.step(action + 4)
                    state = next_state
                    st.session_state.total_reward += reward
                
                st.session_state.step_count += 1
                st.session_state.done = done
            
            st.session_state.visualization_state = state
    
    # Display current state
    st.subheader("Estado Actual")
    
    if st.session_state.env is not None:
        env = st.session_state.env
        
        # Status indicators
        st.markdown("---")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        status_color = "🟢" if not st.session_state.done else "🔴"
        auto_status = "▶️ AUTO" if st.session_state.auto_play else "⏸️ PAUSADO"
        
        col1.markdown(f"**Estado:** {status_color}")
        col2.markdown(f"**Modo:** {'⚔️ COMBATE' if env.mode == 'COMBAT' else '🗺️ EXPLORACIÓN'}")
        col3.markdown(f"**Auto:** {auto_status}")
        col4.markdown(f"**Pasos:** {st.session_state.step_count}")
        col5.markdown(f"**Reward:** {st.session_state.total_reward:.1f}")
        
        # Combat visualization (Pokemon style)
        if env.mode == "COMBAT" and hasattr(env, 'my_pokemon') and hasattr(env, 'enemy_pokemon'):
            st.markdown("---")
            st.markdown("""
            <div style='text-align: center; padding: 10px; background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                        border-radius: 15px; border: 3px solid #000; margin: 20px 0;'>
                <h3 style='color: #000;'>⚔️ ¡BATALLA POKÉMON! ⚔️</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Battle display - Two columns for each Pokemon
            col_enemy, col_player = st.columns(2)
            
            with col_enemy:
                st.markdown(f"""
                <div style='background: rgba(255,100,100,0.3); padding: 15px; border-radius: 10px; border: 2px solid #8B0000;'>
                    <h3 style='color: #8B0000; text-align: center;'>🔥 RIVAL</h3>
                    <h2 style='text-align: center;'>{env.enemy_pokemon.get('name', 'Unknown').upper()}</h2>
                    <p style='text-align: center; font-size: 18px;'>Nv. {env.enemy_pokemon.get('level', 1)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Enemy HP bar
                if hasattr(env, 'enemy_hp') and hasattr(env, 'max_hp_enemy'):
                    hp_percent = max(0, min(100, (env.enemy_hp / env.max_hp_enemy) * 100))
                    hp_color = "#00FF00" if hp_percent > 50 else ("#FFFF00" if hp_percent > 20 else "#FF0000")
                    
                    st.markdown(f"""
                    <div style='margin: 10px 0;'>
                        <div style='background: #333; border: 2px solid #000; border-radius: 10px; padding: 3px;'>
                            <div style='background: {hp_color}; width: {hp_percent}%; height: 25px; border-radius: 7px; 
                                        transition: width 0.3s ease;'></div>
                        </div>
                        <p style='text-align: center; font-weight: bold; margin-top: 5px;'>
                            HP: {int(env.enemy_hp)}/{int(env.max_hp_enemy)}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Enemy info
                types = env.enemy_pokemon.get('types', [])
                type_badges = ' '.join([f"<span style='background: #666; color: #FFF; padding: 3px 8px; border-radius: 5px; margin: 2px;'>{t.upper()}</span>" for t in types])
                st.markdown(f"<p style='text-align: center;'>{type_badges}</p>", unsafe_allow_html=True)
                
                if env.enemy_pokemon.get('ability'):
                    st.markdown(f"<p style='text-align: center; font-size: 12px;'>⚡ {env.enemy_pokemon['ability']}</p>", unsafe_allow_html=True)
            
            with col_player:
                st.markdown(f"""
                <div style='background: rgba(100,100,255,0.3); padding: 15px; border-radius: 10px; border: 2px solid #00008B;'>
                    <h3 style='color: #00008B; text-align: center;'>💙 TU POKÉMON</h3>
                    <h2 style='text-align: center;'>{env.my_pokemon.get('name', 'Unknown').upper()}</h2>
                    <p style='text-align: center; font-size: 18px;'>Nv. {env.my_pokemon.get('level', 1)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Player HP bar
                if hasattr(env, 'my_hp') and hasattr(env, 'max_hp_my'):
                    hp_percent = max(0, min(100, (env.my_hp / env.max_hp_my) * 100))
                    hp_color = "#00FF00" if hp_percent > 50 else ("#FFFF00" if hp_percent > 20 else "#FF0000")
                    
                    st.markdown(f"""
                    <div style='margin: 10px 0;'>
                        <div style='background: #333; border: 2px solid #000; border-radius: 10px; padding: 3px;'>
                            <div style='background: {hp_color}; width: {hp_percent}%; height: 25px; border-radius: 7px; 
                                        transition: width 0.3s ease;'></div>
                        </div>
                        <p style='text-align: center; font-weight: bold; margin-top: 5px;'>
                            HP: {int(env.my_hp)}/{int(env.max_hp_my)}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Player info
                types = env.my_pokemon.get('types', [])
                type_badges = ' '.join([f"<span style='background: #4169E1; color: #FFF; padding: 3px 8px; border-radius: 5px; margin: 2px;'>{t.upper()}</span>" for t in types])
                st.markdown(f"<p style='text-align: center;'>{type_badges}</p>", unsafe_allow_html=True)
                
                if env.my_pokemon.get('ability'):
                    st.markdown(f"<p style='text-align: center; font-size: 12px;'>⚡ {env.my_pokemon['ability']}</p>", unsafe_allow_html=True)
            
            # Moves display
            st.markdown("### 🎯 Movimientos Disponibles")
            moves = env.my_pokemon.get('active_moves', [])
            cols = st.columns(4)
            for i, move in enumerate(moves[:4]):
                with cols[i]:
                    st.markdown(f"""
                    <div style='background: #FFD700; padding: 10px; border-radius: 8px; border: 2px solid #000; 
                                text-align: center; min-height: 60px;'>
                        <p style='font-weight: bold; color: #000; margin: 0;'>{move.upper()}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Battle log and XAI
            st.markdown("### 📜 Registro de Batalla")
            log_container = st.container()
            with log_container:
                if st.session_state.last_move:
                    st.info(f"💥 {env.my_pokemon.get('name', 'Pokémon')} usó **{st.session_state.last_move}**!")
                
                # XAI: Q-Values Visualization
                if st.session_state.last_q_values is not None:
                    st.markdown("#### 🧠 IA Explicable (XAI) - Análisis de Decisión")
                    
                    moves = env.my_pokemon.get('active_moves', ['Movimiento 1', 'Movimiento 2', 'Movimiento 3', 'Movimiento 4'])
                    q_vals = st.session_state.last_q_values
                    
                    # Create DataFrame for visualization
                    q_data = []
                    for i in range(min(4, len(moves))):
                        move_name = moves[i] if i < len(moves) else f"Movimiento {i+1}"
                        q_value = q_vals.get(i, 0) if q_vals else 0
                        q_data.append({'Movimiento': move_name, 'Q-Value': q_value})
                    
                    df_q = pd.DataFrame(q_data)
                    
                    # Create bar chart with Altair
                    chart = alt.Chart(df_q).mark_bar().encode(
                        x=alt.X('Movimiento:N', title='Movimiento'),
                        y=alt.Y('Q-Value:Q', title='Valor Q (Esperado)'),
                        color=alt.condition(
                            alt.datum['Q-Value'] == alt.expr.max(df_q['Q-Value']),
                            alt.value('#FFD700'),  # Gold for best action
                            alt.value('#4169E1')   # Blue for others
                        ),
                        tooltip=['Movimiento', 'Q-Value']
                    ).properties(
                        width=600,
                        height=300,
                        title='Comparación de Q-Values por Movimiento'
                    )
                    
                    st.altair_chart(chart, use_container_width=True)
                    
                    # Textual explanation
                    if q_vals and len(q_data) > 0:
                        best_idx = max(range(len(q_data)), key=lambda i: q_data[i]['Q-Value'])
                        best_move = q_data[best_idx]['Movimiento']
                        best_q = q_data[best_idx]['Q-Value']
                        
                        # Find second best
                        sorted_q = sorted(q_data, key=lambda x: x['Q-Value'], reverse=True)
                        second_best = sorted_q[1] if len(sorted_q) > 1 else sorted_q[0]
                        
                        explanation = f"""
                        **Explicación de la Decisión:**
                        
                        El agente eligió **{best_move}** porque su valor esperado (Q) es **{best_q:.2f}**, 
                        mientras que **{second_best['Movimiento']}** tiene un valor de **{second_best['Q-Value']:.2f}**.
                        
                        Los Q-Values representan la recompensa esperada de cada acción. El agente selecciona 
                        el movimiento con el mayor Q-Value para maximizar la efectividad en batalla.
                        """
                        st.info(explanation)
                elif st.session_state.last_move:
                    st.warning("🔍 Modo exploración activo - Q-Values no disponibles (decisión aleatoria)")
                
                st.markdown("""
                <div style='background: rgba(0,0,0,0.7); color: #FFF; padding: 15px; border-radius: 10px; 
                            border: 2px solid #FFD700; max-height: 200px; overflow-y: auto;'>
                    <p>⚔️ La batalla continúa...</p>
                    <p>🎲 Las IAs están tomando decisiones estratégicas</p>
                    <p>📊 Analizando efectividad de tipos...</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Map visualization (only when not in combat)
        elif env.mode == "MAP":
            st.markdown("---")
            st.markdown("### 🗺️ EXPLORACIÓN DEL MAPA")
            
            if hasattr(env, 'grid') and hasattr(env, 'player_pos'):
                # Create a visual representation of the map
                fig, ax = plt.subplots(figsize=(10, 10))
                fig.patch.set_facecolor('#2a5298')
                
                # Colors for different tiles
                colors = {
                    0: [0.9, 0.9, 0.85],   # Path (light beige)
                    1: [0.3, 0.3, 0.3],    # Wall (dark gray)
                    2: [0.2, 0.7, 0.2],    # Grass (green)
                    9: [1.0, 0.84, 0.0]    # Goal (gold)
                }
                
                # Create RGB image
                grid_visual = np.zeros((10, 10, 3))
                for i in range(10):
                    for j in range(10):
                        tile_value = env.grid[i][j]
                        grid_visual[i, j] = colors.get(tile_value, [0, 0, 0])
                
                # Mark player position
                py, px = env.player_pos
                grid_visual[py, px] = [0, 0.8, 1]  # Cyan for player
                
                ax.imshow(grid_visual, interpolation='nearest')
                ax.set_title(f"Mapa {env.current_map_idx + 1} - Jugador en ({py}, {px})", 
                           fontsize=16, color='white', fontweight='bold')
                ax.set_xticks(range(10))
                ax.set_yticks(range(10))
                ax.grid(True, alpha=0.3, color='white', linewidth=2)
                ax.set_facecolor('#1e3c72')
                
                st.pyplot(fig)
                plt.close()
                
                # Map legend
                col1, col2, col3, col4 = st.columns(4)
                col1.markdown("🟦 **Jugador**")
                col2.markdown("🟩 **Hierba** (encuentros)")
                col3.markdown("⬜ **Camino**")
                col4.markdown("🟨 **Meta**")
        
        else:
            st.info("🎮 Esperando inicialización del entorno...")

# ========== MODO EXPLORACIÓN (FOLIUM MAP) ==========
elif mode == "Modo Exploración":
    st.markdown("""
    <h2 style='color: #FFD700; text-align: center;'>🗺️ MODO EXPLORACIÓN - MAPA INTERACTIVO</h2>
    <p style='color: #FFF; text-align: center;'>Explora el mundo real y encuentra Pokémon salvajes y gimnasios</p>
    """, unsafe_allow_html=True)
    
    # Initialize agents if not done
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Inicializar Sistema"):
            env, explorer, tactician, strategist = initialize_agents()
            st.success("✅ Sistema inicializado")
    
    with col2:
        checkpoint_episode = st.selectbox("Cargar checkpoint", [3000, 2000, 1000], index=0, key="exp_checkpoint")
        if st.button("📥 Cargar Modelo", key="exp_load"):
            if st.session_state.env is None:
                st.error("⚠️ Primero inicializa el sistema")
            else:
                if load_checkpoints(st.session_state.explorer, st.session_state.tactician, checkpoint_episode):
                    st.success(f"✅ Modelo ep{checkpoint_episode} cargado")
                else:
                    st.warning(f"⚠️ No se encontraron checkpoints para episodio {checkpoint_episode}")
    
    st.markdown("---")
    
    # Create and display Folium map
    st.subheader("🌍 Mapa del Mundo Pokémon")
    st.write("Haz clic en los marcadores para iniciar batallas o capturar Pokémon")
    
    # Create the map
    folium_map, geojson_data = create_folium_map()
    
    # Display the map and capture interactions
    map_data = st_folium(
        folium_map,
        width=1000,
        height=600,
        key="folium_map"
    )
    
    # Handle marker click
    if map_data and map_data.get('last_object_clicked'):
        clicked = map_data['last_object_clicked']
        
        # Check if this is a new click
        if clicked != st.session_state.selected_marker:
            st.session_state.selected_marker = clicked
            
            # Find which marker was clicked
            clicked_lat = clicked.get('lat')
            clicked_lng = clicked.get('lng')
            
            if clicked_lat and clicked_lng:
                # Search in gyms
                selected_location = None
                for gym in geojson_data["gyms"]["features"]:
                    coords = gym["geometry"]["coordinates"]
                    if abs(coords[1] - clicked_lat) < 0.0001 and abs(coords[0] - clicked_lng) < 0.0001:
                        selected_location = gym["properties"]
                        selected_location['category'] = 'gym'
                        break
                
                # Search in wild pokemon
                if not selected_location:
                    for wild in geojson_data["wild"]["features"]:
                        coords = wild["geometry"]["coordinates"]
                        if abs(coords[1] - clicked_lat) < 0.0001 and abs(coords[0] - clicked_lng) < 0.0001:
                            selected_location = wild["properties"]
                            selected_location['category'] = 'wild'
                            break
                
                if selected_location:
                    # Transition to combat mode
                    st.session_state.map_mode = 'COMBAT'
                    
                    # Initialize environment if needed
                    if st.session_state.env is None:
                        st.warning("⚠️ Inicializando entorno automáticamente...")
                        env, explorer, tactician, strategist = initialize_agents()
                    
                    env = st.session_state.env
                    strategist = st.session_state.strategist
                    
                    # Setup team based on location type
                    all_ids = list(env.pokedex.keys())
                    if len(all_ids) >= 6:
                        party_ids = np.random.choice(all_ids, 6, replace=False)
                    else:
                        party_ids = all_ids
                    strategist.set_party(party_ids)
                    
                    # Determine enemy type based on location
                    enemy_type = selected_location.get('pokemon_type', 'normal')
                    best = strategist.build_team(enemy_type)
                    
                    env.my_pokemon = best.copy()
                    env.my_pokemon['level'] = 5
                    env.my_pokemon['exp'] = 0
                    
                    # Reset environment to start combat
                    state, _ = env.reset()
                    env.mode = "COMBAT"  # Force combat mode
                    
                    st.session_state.visualization_state = state
                    st.session_state.step_count = 0
                    st.session_state.total_reward = 0
                    st.session_state.done = False
                    
                    st.success(f"⚔️ ¡Batalla iniciada contra {selected_location['name']}! (Tipo: {enemy_type.upper()})")
                    st.info("👉 Cambia al 'Modo Visualización' para ver la batalla")
    
    # Display selected marker info
    if st.session_state.selected_marker:
        st.markdown("---")
        st.subheader("📍 Ubicación Seleccionada")
        marker_info = st.session_state.selected_marker
        st.json(marker_info)
    
    # Legend
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🏠 Gimnasios Pokémon (Rojo)**
        - Batallas de nivel avanzado
        - Basados en el tipo del lugar
        - Recompensas especiales
        """)
    with col2:
        st.markdown("""
        **🍃 Pokémon Salvajes (Verde)**
        - Encuentros aleatorios
        - Variedad de tipos
        - Oportunidad de captura
        """)

# ========== DASHBOARD ESTADÍSTICAS ==========
elif mode == "Dashboard Estadísticas":
    st.markdown("""
    <h2 style='color: #FFD700; text-align: center;'>📊 DASHBOARD DE ESTADÍSTICAS</h2>
    <p style='color: #FFF; text-align: center;'>Analiza el rendimiento de tus agentes en el mundo real</p>
    """, unsafe_allow_html=True)
    
    # Check if we have training data
    if not st.session_state.training_history.get('episodes') or len(st.session_state.training_history['episodes']) == 0:
        st.warning("⚠️ No hay datos de entrenamiento disponibles. Primero ejecuta un entrenamiento en 'Modo Entrenamiento'.")
        st.info("💡 El dashboard mostrará estadísticas una vez que hayas entrenado el modelo.")
    else:
        history = st.session_state.training_history
        
        # Summary metrics
        st.subheader("📈 Resumen General")
        col1, col2, col3, col4 = st.columns(4)
        
        total_episodes = len(history['episodes'])
        avg_reward = np.mean(history['rewards']) if history['rewards'] else 0
        total_wins = sum(history.get('wins', []))
        win_rate = (total_wins / total_episodes * 100) if total_episodes > 0 else 0
        
        col1.metric("Total Episodios", total_episodes)
        col2.metric("Reward Promedio", f"{avg_reward:.2f}")
        col3.metric("Victorias", total_wins)
        col4.metric("Tasa de Victoria", f"{win_rate:.1f}%")
        
        st.markdown("---")
        
        # Heatmap of battle wins
        st.subheader("🗺️ Mapa de Calor - Zonas de Victoria")
        st.write("Visualización de las ubicaciones donde el agente ha ganado más batallas")
        
        if history.get('gps_coords') and history.get('wins'):
            # Create folium map for heatmap
            center_lat = 40.4168
            center_lon = -3.7038
            
            heatmap_m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles='OpenStreetMap'
            )
            
            # Add markers for wins (green) and losses (red)
            for i, (coords, is_win) in enumerate(zip(history['gps_coords'], history['wins'])):
                if coords and len(coords) == 2:
                    lat, lon = coords
                    episode = history['episodes'][i]
                    reward = history['rewards'][i]
                    
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=5,
                        popup=f"Ep {episode}: {'Victoria' if is_win else 'Derrota'} (R: {reward:.1f})",
                        color='green' if is_win else 'red',
                        fill=True,
                        fillOpacity=0.6
                    ).add_to(heatmap_m)
            
            # Display the heatmap
            st_folium(heatmap_m, width=1000, height=500, key="heatmap")
        else:
            st.info("📍 Datos GPS no disponibles. Los datos de GPS se generan durante el entrenamiento.")
        
        st.markdown("---")
        
        # Scatter plot: Distance vs Win Rate
        st.subheader("📍 Distancia vs Tasa de Victoria")
        st.write("Relación entre la distancia a la ubicación base y la tasa de victorias")
        
        if history.get('gps_coords') and history.get('wins'):
            # Calculate distances from center
            center_lat = 40.4168
            center_lon = -3.7038
            
            distances = []
            win_rates_by_distance = []
            
            # Group by distance buckets
            for coords, is_win in zip(history['gps_coords'], history['wins']):
                if coords and len(coords) == 2:
                    lat, lon = coords
                    # Simple distance calculation (Euclidean approximation)
                    dist = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2) * 111  # Convert to km
                    distances.append(dist)
            
            # Create scatter data
            scatter_data = pd.DataFrame({
                'Episodio': history['episodes'][:len(distances)],
                'Distancia (km)': distances,
                'Victoria': [1 if w else 0 for w in history['wins'][:len(distances)]],
                'Reward': history['rewards'][:len(distances)]
            })
            
            # Create scatter plot with Altair
            scatter_chart = alt.Chart(scatter_data).mark_circle(size=60).encode(
                x=alt.X('Distancia (km):Q', title='Distancia desde Centro (km)'),
                y=alt.Y('Reward:Q', title='Reward Total'),
                color=alt.Color('Victoria:N', scale=alt.Scale(domain=[0, 1], range=['red', 'green']), 
                               legend=alt.Legend(title='Resultado')),
                tooltip=['Episodio', 'Distancia (km)', 'Victoria', 'Reward']
            ).properties(
                width=800,
                height=400,
                title='Dispersión: Distancia vs Rendimiento'
            )
            
            st.altair_chart(scatter_chart, use_container_width=True)
            
            # Statistics by distance
            st.markdown("#### 📊 Estadísticas por Distancia")
            
            # Bucket distances
            scatter_data['Distancia_Bucket'] = pd.cut(scatter_data['Distancia (km)'], bins=5, labels=['Muy Cerca', 'Cerca', 'Medio', 'Lejos', 'Muy Lejos'])
            
            stats_by_dist = scatter_data.groupby('Distancia_Bucket').agg({
                'Victoria': ['sum', 'count', 'mean'],
                'Reward': 'mean'
            }).round(2)
            
            stats_by_dist.columns = ['Victorias', 'Total Batallas', 'Win Rate', 'Reward Promedio']
            stats_by_dist['Win Rate'] = (stats_by_dist['Win Rate'] * 100).round(1)
            
            st.dataframe(stats_by_dist, use_container_width=True)
        else:
            st.info("📍 Datos GPS no disponibles. Los datos de GPS se generan durante el entrenamiento.")
        
        st.markdown("---")
        
        # Additional analytics
        st.subheader("📉 Análisis Temporal")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Reward over time
            reward_df = pd.DataFrame({
                'Episodio': history['episodes'],
                'Reward': history['rewards']
            })
            
            reward_chart = alt.Chart(reward_df).mark_line(point=True).encode(
                x='Episodio:Q',
                y='Reward:Q',
                tooltip=['Episodio', 'Reward']
            ).properties(
                width=400,
                height=300,
                title='Evolución del Reward'
            )
            
            st.altair_chart(reward_chart, use_container_width=True)
        
        with col2:
            # Win rate over time (moving average)
            if history.get('wins'):
                window = min(20, len(history['wins']))
                wins_array = np.array([1 if w else 0 for w in history['wins']])
                
                if len(wins_array) >= window:
                    win_rate_ma = np.convolve(wins_array, np.ones(window)/window, mode='valid')
                    
                    winrate_df = pd.DataFrame({
                        'Episodio': history['episodes'][window-1:],
                        'Win Rate (%)': win_rate_ma * 100
                    })
                    
                    winrate_chart = alt.Chart(winrate_df).mark_line(color='green').encode(
                        x='Episodio:Q',
                        y=alt.Y('Win Rate (%):Q', scale=alt.Scale(domain=[0, 100])),
                        tooltip=['Episodio', 'Win Rate (%)']
                    ).properties(
                        width=400,
                        height=300,
                        title=f'Tasa de Victoria (Media móvil {window} ep.)'
                    )
                    
                    st.altair_chart(winrate_chart, use_container_width=True)
                else:
                    st.info("Necesitas más episodios para calcular la media móvil")

# Footer with Pokemon theme
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background: #FFF; padding: 15px; border-radius: 10px; border: 2px solid #000; color: #000;'>
    <h4 style='color: #CC0000; text-align: center;'>POKÉDEX v2.0</h4>
    <p style='font-size: 12px; margin: 5px 0;'><b>ExplorerAgent:</b> CNN para exploración</p>
    <p style='font-size: 12px; margin: 5px 0;'><b>TacticianAgent:</b> DQN para combate</p>
    <p style='font-size: 12px; margin: 5px 0;'><b>Strategist:</b> Sistema experto</p>
    <p style='text-align: center; margin-top: 10px; font-size: 11px;'>
        🎮 Entrenamiento con Deep RL<br>
        ⚡ Powered by PyTorch
    </p>
</div>
""", unsafe_allow_html=True)
