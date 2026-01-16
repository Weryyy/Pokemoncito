import torch
import numpy as np
import os
import sys
from src.env.pokemon_env import PokemonSimEnv
from src.agents.explorer import ExplorerAgent
from src.agents.tactician import TacticianAgent
from src.agents.strategist import Strategist

# --- AJUSTES PARA VERSIÓN CON MEMORIA ---
EPISODES = 5000        # Aumentamos duración
MAX_STEPS = 600        # Damos más tiempo por episodio
SAVE_INTERVAL = 200    # Guardamos menos a menudo para no llenar el disco

def save_checkpoint(explorer, tactician, episode):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "checkpoints")
    if not os.path.exists(path): os.makedirs(path)
    
    torch.save(explorer.policy_net.state_dict(), os.path.join(path, f"explorer_ep{episode}.pth"))
    torch.save(tactician.policy_net.state_dict(), os.path.join(path, f"tactician_ep{episode}.pth"))
    print(f"💾 CHECKPOINT GUARDADO: Episodio {episode}")

def train():
    print("🚀 INICIANDO ENTRENAMIENTO PRO (CON MEMORIA)...")
    env = PokemonSimEnv(verbose=False)
    
    # --- LA LÍNEA MÁGICA ---
    # Cambiamos 3 por 9. Esto conecta los "ojos" nuevos a la red neuronal.
    explorer = ExplorerAgent(obs_shape=(9, 10, 10), n_actions=4, lr=1e-4)
    
    tactician = TacticianAgent(input_dim=10, n_actions=5, lr=1e-3)
    strategist = Strategist(env.pokedex)

    try: 
        for episode in range(1, EPISODES + 1):
            
            # Curriculum Learning Ajustado
            if episode < 1000: map_idx = 0
            elif episode < 2000: map_idx = np.random.choice([0, 1])
            elif episode < 3500: map_idx = np.random.choice([0, 1, 2, 3])
            else: map_idx = np.random.choice([0, 1, 2, 3, 4])
            env.current_map_idx = map_idx
            
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
            
            while not done and steps < MAX_STEPS:
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

            # Decaimiento más lento para asegurar que aprende bien
            if explorer.epsilon > 0.05: explorer.epsilon *= 0.9997
            if tactician.epsilon > 0.05: tactician.epsilon *= 0.9995

            if episode % 10 == 0:
                print(f"Ep {episode}/{EPISODES} | Mapa {map_idx} | R: {total_reward:.1f} | Eps: {explorer.epsilon:.2f}")

            if episode % SAVE_INTERVAL == 0:
                save_checkpoint(explorer, tactician, episode)

    except KeyboardInterrupt:
        print("\n🛑 GUARDANDO...")
        save_checkpoint(explorer, tactician, episode)
        sys.exit(0)

    save_checkpoint(explorer, tactician, EPISODES)

if __name__ == "__main__":
    train()