import torch
import numpy as np
import os
import time
from datetime import timedelta
from src.env.pokemon_env import PokemonSimEnv
from src.agents.explorer import ExplorerAgent
from src.agents.tactician import TacticianAgent
from src.agents.strategist import Strategist

print("🧪 Testing timing and GPU detection features...")

# Test GPU detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"🎮 GPU DETECTADA: {torch.cuda.get_device_name(0)}")
    print(f"   Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print(f"💻 Usando CPU (para usar GPU, instala PyTorch con soporte CUDA)")

# Initialize environment and agents
env = PokemonSimEnv(verbose=False)
explorer = ExplorerAgent(obs_shape=(9, 10, 10), n_actions=4, lr=1e-4)
tactician = TacticianAgent(input_dim=16, n_actions=5, lr=1e-3)
strategist = Strategist(env.pokedex)

print(f"\nDispositivo Explorer: {explorer.device}")
print(f"Dispositivo Tactician: {tactician.device}")

# Test timing with 5 episodes
EPISODES = 5
start_time = time.time()
episode_times = []

for episode in range(1, EPISODES + 1):
    episode_start = time.time()
    
    # Quick episode simulation
    all_ids = list(env.pokedex.keys())
    party_ids = np.random.choice(all_ids, 6, replace=False) if len(all_ids) >= 6 else all_ids
    strategist.set_party(party_ids)
    
    target = np.random.choice(["fire", "water", "grass"])
    best = strategist.build_team(target)
    
    env.my_pokemon = best.copy()
    env.my_pokemon['level'] = 5
    state, _ = env.reset()
    
    # Run 20 steps
    for _ in range(20):
        if env.mode == "MAP":
            action = explorer.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            if env.mode != "COMBAT":
                explorer.learn(state, action, reward, next_state, done)
            state = next_state
        elif env.mode == "COMBAT":
            action = tactician.select_action(state)
            next_state, reward, done, _, _ = env.step(action + 4)
            if env.mode != "MAP":
                tactician.learn(state, action, reward, next_state, done)
            state = next_state
    
    episode_time = time.time() - episode_start
    episode_times.append(episode_time)
    
    # Calculate ETA
    avg_episode_time = np.mean(episode_times)
    remaining_episodes = EPISODES - episode
    eta_seconds = avg_episode_time * remaining_episodes
    eta_str = str(timedelta(seconds=int(eta_seconds)))
    elapsed_str = str(timedelta(seconds=int(time.time() - start_time)))
    
    print(f"\n✅ Episodio {episode}/{EPISODES}")
    print(f"   Tiempo del episodio: {episode_time:.2f}s")
    print(f"   ⏱️  Tiempo transcurrido: {elapsed_str}")
    print(f"   🔮 ETA restante: {eta_str}")
    print(f"   📈 Velocidad: {1/avg_episode_time:.2f} ep/s")

total_time = time.time() - start_time
print(f"\n🎉 Test completado en {total_time:.2f}s")
print(f"   Tiempo promedio por episodio: {np.mean(episode_times):.2f}s")
