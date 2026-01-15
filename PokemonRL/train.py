import torch
import numpy as np
import os
import sys
from src.env.pokemon_env import PokemonSimEnv
from src.agents.explorer import ExplorerAgent
from src.agents.tactician import TacticianAgent
from src.agents.strategist import Strategist

EPISODES = 500
MAX_STEPS = 1000
SAVE_INTERVAL = 20  # cambiar esto a cada 50


def save_checkpoint(explorer, tactician, episode):
    # RUTA ABSOLUTA BLINDADA
    # Obtenemos la carpeta donde está ESTE archivo train.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "checkpoints")

    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(explorer.policy_net.state_dict(),
               os.path.join(path, f"explorer_ep{episode}.pth"))
    torch.save(tactician.policy_net.state_dict(),
               os.path.join(path, f"tactician_ep{episode}.pth"))
    print(f"💾 CHECKPOINT GUARDADO EN: {path}")


def train():
    print("Inicializando entorno (MODO TURBO)...")
    # verbose=False desactiva los prints del entorno
    env = PokemonSimEnv(verbose=False)

    explorer = ExplorerAgent(obs_shape=(3, 10, 10), n_actions=4, lr=1e-4)
    tactician = TacticianAgent(input_dim=10, n_actions=5, lr=1e-3)
    strategist = Strategist(env.pokedex)

    print(f"--- INICIANDO ENTRENAMIENTO DE {EPISODES} EPISODIOS ---")

    party_ids = []
    episode = 0

    try:
        for episode in range(1, EPISODES + 1):

            # Rotación equipo
            if (episode - 1) % 10 == 0:
                all_ids = list(env.pokedex.keys())
                party_ids = np.random.choice(all_ids, size=6, replace=False) if len(
                    all_ids) >= 6 else all_ids
                strategist.set_party(party_ids)

            # Selección Pokemon
            possible_types = ["fire", "water",
                              "grass", "electric", "rock", "psychic"]
            my_best_pokemon = strategist.build_team(
                target_type=np.random.choice(possible_types))
            env.my_pokemon = my_best_pokemon.copy()

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

            # Feedback reducido para que vaya rápido
            print(
                f"Ep {episode}/{EPISODES} | R: {total_reward:.1f} | Pasos: {steps}")

            explorer.decay_epsilon()
            tactician.decay_epsilon()

            if episode % SAVE_INTERVAL == 0:
                save_checkpoint(explorer, tactician, episode)

    except KeyboardInterrupt:
        print("\n🛑 GUARDANDO Y SALIENDO...")
        save_checkpoint(explorer, tactician, episode)
        sys.exit(0)

    print("\n✅ FIN.")
    save_checkpoint(explorer, tactician, EPISODES)


if __name__ == "__main__":
    train()
