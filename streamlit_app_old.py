import streamlit as st
import torch
import numpy as np
import os
import sys
import time
from datetime import timedelta
import matplotlib.pyplot as plt
from collections import deque

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
    st.session_state.training_history = {'episodes': [], 'rewards': [], 'epsilons': []}
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
    ["Modo Entrenamiento", "Modo Visualización"],
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
            st.session_state.training_history = {'episodes': [], 'rewards': [], 'epsilons': []}
            
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
                    st.session_state.training_history['episodes'].append(episode)
                    st.session_state.training_history['rewards'].append(total_reward)
                    st.session_state.training_history['epsilons'].append(explorer.epsilon)
                    
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
            action = tactician.select_action(state)
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
                    action = tactician.select_action(state)
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
            
            # Battle log
            st.markdown("### 📜 Registro de Batalla")
            log_container = st.container()
            with log_container:
                if st.session_state.last_move:
                    st.info(f"💥 {env.my_pokemon.get('name', 'Pokémon')} usó **{st.session_state.last_move}**!")
                
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
