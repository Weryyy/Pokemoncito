import torch
import numpy as np
import os
import sys
from src.env.pokemon_env import PokemonSimEnv
from src.agents.explorer import ExplorerAgent
from src.agents.tactician import TacticianAgent
from src.agents.strategist import Strategist

# --- AJUSTES MEJORADOS PARA ENTRENAMIENTO ---
EPISODES = 3000        # Más episodios para mejor convergencia
MAX_STEPS = 300
SAVE_INTERVAL = 200

def save_checkpoint(explorer, tactician, episode):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "checkpoints")
    if not os.path.exists(path): os.makedirs(path)
    
    # Guardar tanto policy_net como target_net
    torch.save(explorer.policy_net.state_dict(), os.path.join(path, f"explorer_ep{episode}.pth"))
    torch.save(explorer.target_net.state_dict(), os.path.join(path, f"explorer_target_ep{episode}.pth"))
    torch.save(tactician.policy_net.state_dict(), os.path.join(path, f"tactician_ep{episode}.pth"))
    torch.save(tactician.target_net.state_dict(), os.path.join(path, f"tactician_target_ep{episode}.pth"))
    print(f"💾 CHECKPOINT GUARDADO: Episodio {episode}")

def train():
    print("🚀 INICIANDO ENTRENAMIENTO MEJORADO (Anti-Overfitting)...")
    env = PokemonSimEnv(verbose=False)
    
    # Inicializar agentes con arquitecturas mejoradas
    explorer = ExplorerAgent(obs_shape=(9, 10, 10), n_actions=4, lr=1e-4)
    tactician = TacticianAgent(input_dim=16, n_actions=5, lr=1e-3)  # Actualizado a 16
    strategist = Strategist(env.pokedex)

    best_reward = -float('inf')
    rewards_history = []

    try: 
        for episode in range(1, EPISODES + 1):
            
            # Curriculum Learning más suave
            if episode < 500: map_idx = 0
            elif episode < 1000: map_idx = np.random.choice([0, 1])
            elif episode < 1500: map_idx = np.random.choice([0, 1, 2])
            elif episode < 2000: map_idx = np.random.choice([0, 1, 2, 3])
            else: map_idx = np.random.choice([0, 1, 2, 3, 4])
            
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

            # Decaimiento epsilon más controlado
            if explorer.epsilon > explorer.epsilon_min: 
                explorer.epsilon *= 0.9995
            if tactician.epsilon > tactician.epsilon_min: 
                tactician.epsilon *= 0.9993
            
            rewards_history.append(total_reward)

            if episode % 10 == 0:
                avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
                print(f"Ep {episode}/{EPISODES} | Mapa {map_idx} | R: {total_reward:.1f} | Avg100: {avg_reward:.1f} | Eps: {explorer.epsilon:.3f}")
                
                # Guardar mejor modelo
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    save_checkpoint(explorer, tactician, episode)

            if episode % SAVE_INTERVAL == 0:
                save_checkpoint(explorer, tactician, episode)

    except KeyboardInterrupt:
        print("\n🛑 GUARDANDO...")
        save_checkpoint(explorer, tactician, episode)
        sys.exit(0)

    save_checkpoint(explorer, tactician, EPISODES)
    print(f"\n✅ ENTRENAMIENTO COMPLETADO! Mejor recompensa promedio: {best_reward:.2f}")

if __name__ == "__main__":
    train()