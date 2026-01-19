import torch
import numpy as np
import os
from src.env.pokemon_env import PokemonSimEnv
from src.agents.explorer import ExplorerAgent
from src.agents.tactician import TacticianAgent
from src.agents.strategist import Strategist

print("🧪 Testing training for 10 episodes...")
env = PokemonSimEnv(verbose=False)

explorer = ExplorerAgent(obs_shape=(9, 10, 10), n_actions=4, lr=1e-4)
tactician = TacticianAgent(input_dim=16, n_actions=5, lr=1e-3)
strategist = Strategist(env.pokedex)

EPISODES = 10
MAX_STEPS = 50

for episode in range(1, EPISODES + 1):
    env.current_map_idx = 0
    
    if (episode-1) % 5 == 0:
        all_ids = list(env.pokedex.keys())
        party_ids = np.random.choice(all_ids, 6, replace=False) if len(all_ids) >= 6 else all_ids
        strategist.set_party(party_ids)

    target = np.random.choice(["fire", "water", "grass"])
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
    
    print(f"✅ Episode {episode}/{EPISODES} | Steps: {steps} | Reward: {total_reward:.2f}")

print("\n🎉 Training test completed successfully!")
print(f"Explorer buffer: {len(explorer.replay_buffer)} experiences")
print(f"Tactician buffer: {len(tactician.replay_buffer)} experiences")
