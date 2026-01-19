import streamlit as st
import torch
import numpy as np
import os
import sys
import time
from datetime import timedelta
import matplotlib.pyplot as plt

# Add PokemonRL to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PokemonRL'))

from PokemonRL.src.env.pokemon_env import PokemonSimEnv
from PokemonRL.src.agents.explorer import ExplorerAgent
from PokemonRL.src.agents.tactician import TacticianAgent
from PokemonRL.src.agents.strategist import Strategist

# Page config
st.set_page_config(
    page_title="Pokemoncito - RL Training & Visualization",
    page_icon="🎮",
    layout="wide"
)

# Title
st.title("🎮 Pokemoncito - Reinforcement Learning Simulator")

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

# Sidebar navigation
st.sidebar.title("Navegación")
mode = st.sidebar.radio("Selecciona el modo:", ["Modo Entrenamiento", "Modo Visualización"])

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
            explorer.policy_net.load_state_dict(torch.load(explorer_path, map_location=device))
            tactician.policy_net.load_state_dict(torch.load(tactician_path, map_location=device))
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
                                # Moving average
                                window = min(10, len(st.session_state.training_history['rewards']))
                                rewards_ma = np.convolve(st.session_state.training_history['rewards'], 
                                                        np.ones(window)/window, mode='valid')
                                ax1.plot(range(window, len(st.session_state.training_history['rewards'])+1), 
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
    st.header("👁️ Modo Visualización")
    st.write("Visualiza el comportamiento del agente entrenado")
    
    # Initialize/Load models
    col1, col2 = st.columns(2)
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
    
    # Reset environment
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
            
            st.success("✅ Entorno reiniciado")
    
    # Step controls
    st.subheader("Control de Pasos")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        single_step = st.button("➡️ 1 Paso")
    with col2:
        multiple_steps = st.button("⏩ 10 Pasos")
    with col3:
        auto_steps = st.button("▶️ 50 Pasos (Auto)")
    
    # Execute steps
    if st.session_state.env is not None and st.session_state.visualization_state is not None:
        steps_to_take = 0
        if single_step:
            steps_to_take = 1
        elif multiple_steps:
            steps_to_take = 10
        elif auto_steps:
            steps_to_take = 50
        
        if steps_to_take > 0 and not st.session_state.done:
            env = st.session_state.env
            explorer = st.session_state.explorer
            tactician = st.session_state.tactician
            state = st.session_state.visualization_state
            
            progress_placeholder = st.empty()
            
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
                
                if steps_to_take > 1:
                    progress_placeholder.text(f"Ejecutando paso {i+1}/{steps_to_take}")
            
            st.session_state.visualization_state = state
            if steps_to_take > 1:
                progress_placeholder.empty()
    
    # Display current state
    st.subheader("Estado Actual")
    
    if st.session_state.env is not None:
        env = st.session_state.env
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Modo", env.mode)
        col2.metric("Pasos", st.session_state.step_count)
        col3.metric("Reward Total", f"{st.session_state.total_reward:.2f}")
        col4.metric("Estado", "Terminado" if st.session_state.done else "En Progreso")
        
        # Map visualization
        st.subheader("🗺️ Mapa")
        if hasattr(env, 'grid') and hasattr(env, 'player_pos'):
            # Create a visual representation of the map
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Colors for different tiles
            colors = {
                0: [0.8, 0.8, 0.7],   # Path (beige)
                1: [0.4, 0.4, 0.4],   # Wall (gray)
                2: [0.2, 0.6, 0.2],   # Grass (green)
                9: [1.0, 0.84, 0.0]   # Goal (gold)
            }
            
            # Create RGB image
            grid_visual = np.zeros((10, 10, 3))
            for i in range(10):
                for j in range(10):
                    tile_value = env.grid[i][j]
                    grid_visual[i, j] = colors.get(tile_value, [0, 0, 0])
            
            # Mark player position
            py, px = env.player_pos
            grid_visual[py, px] = [0, 1, 1]  # Cyan for player
            
            ax.imshow(grid_visual)
            ax.set_title(f"Mapa {env.current_map_idx} - Posición del Jugador: ({py}, {px})")
            ax.set_xticks(range(10))
            ax.set_yticks(range(10))
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
        
        # Combat information
        if env.mode == "COMBAT":
            st.subheader("⚔️ Información de Combate")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Tu Pokémon:**")
                if hasattr(env, 'my_pokemon') and env.my_pokemon:
                    st.write(f"Nombre: {env.my_pokemon.get('name', 'Unknown')}")
                    st.write(f"Nivel: {env.my_pokemon.get('level', 1)}")
                    st.write(f"Tipos: {', '.join(env.my_pokemon.get('types', []))}")
                    if hasattr(env, 'my_hp') and hasattr(env, 'max_hp_my'):
                        hp_percent = (env.my_hp / env.max_hp_my) * 100
                        st.progress(hp_percent / 100)
                        st.write(f"HP: {int(env.my_hp)}/{int(env.max_hp_my)}")
                    if env.my_pokemon.get('ability'):
                        st.write(f"Habilidad: {env.my_pokemon['ability']}")
                    if env.my_pokemon.get('held_item'):
                        st.write(f"Objeto: {env.my_pokemon['held_item']}")
            
            with col2:
                st.write("**Pokémon Enemigo:**")
                if hasattr(env, 'enemy_pokemon') and env.enemy_pokemon:
                    st.write(f"Nombre: {env.enemy_pokemon.get('name', 'Unknown')}")
                    st.write(f"Nivel: {env.enemy_pokemon.get('level', 1)}")
                    st.write(f"Tipos: {', '.join(env.enemy_pokemon.get('types', []))}")
                    if hasattr(env, 'enemy_hp') and hasattr(env, 'max_hp_enemy'):
                        hp_percent = (env.enemy_hp / env.max_hp_enemy) * 100
                        st.progress(hp_percent / 100)
                        st.write(f"HP: {int(env.enemy_hp)}/{int(env.max_hp_enemy)}")
                    if env.enemy_pokemon.get('ability'):
                        st.write(f"Habilidad: {env.enemy_pokemon['ability']}")
                    if env.enemy_pokemon.get('held_item'):
                        st.write(f"Objeto: {env.enemy_pokemon['held_item']}")
        else:
            st.info("No hay batalla activa en este momento")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Pokemoncito v2.0**

Un simulador de Pokémon con Deep Reinforcement Learning.

- ExplorerAgent: CNN para exploración
- TacticianAgent: DQN para combate
- Strategist: Sistema experto
""")
