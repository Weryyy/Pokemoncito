import torch
import numpy as np
import os
import sys
from src.env.pokemon_env import PokemonSimEnv
from src.agents.explorer import ExplorerAgent
from src.agents.tactician import TacticianAgent
from src.agents.strategist import Strategist

# Entrenamiento corto para generar checkpoints demo
EPISODES = 200
MAX_STEPS = 150
SAVE_INTERVAL = 100

def save_checkpoint(explorer, tactician, episode):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "checkpoints")
    if not os.path.exists(path): os.makedirs(path)
    
    torch.save(explorer.policy_net.state_dict(), os.path.join(path, f"explorer_ep{episode}.pth"))
    torch.save(tactician.policy_net.state_dict(), os.path.join(path, f"tactician_ep{episode}.pth"))
    print(f"💾 CHECKPOINT GUARDADO: Episodio {episode}")

def train():
    print("🚀 INICIANDO ENTRENAMIENTO DEMO (200 episodios)...")
    env = PokemonSimEnv(verbose=False)
    
    explorer = ExplorerAgent(obs_shape=(9, 10, 10), n_actions=4, lr=1e-4)
    tactician = TacticianAgent(input_dim=16, n_actions=5, lr=1e-3)
    strategist = Strategist(env.pokedex)

    rewards_history = []

    for episode in range(1, EPISODES + 1):
        
        # Curriculum simple
        if episode < 100: map_idx = 0
        else: map_idx = np.random.choice([0, 1])
        
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

        if explorer.epsilon > explorer.epsilon_min: 
            explorer.epsilon *= 0.9995
        if tactician.epsilon > tactician.epsilon_min: 
            tactician.epsilon *= 0.9993
        
        rewards_history.append(total_reward)

        if episode % 20 == 0:
            avg_reward = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history)
            print(f"Ep {episode}/{EPISODES} | Mapa {map_idx} | R: {total_reward:.1f} | Avg: {avg_reward:.1f} | Eps: {explorer.epsilon:.3f}")

        if episode % SAVE_INTERVAL == 0:
            save_checkpoint(explorer, tactician, episode)

    save_checkpoint(explorer, tactician, EPISODES)
    print(f"\n✅ ENTRENAMIENTO DEMO COMPLETADO!")
    print(f"   Explorer buffer: {len(explorer.replay_buffer)} experiencias")
    print(f"   Tactician buffer: {len(tactician.replay_buffer)} experiencias")

if __name__ == "__main__":
    train()
