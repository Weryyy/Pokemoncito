import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from src.env.pokemon_env import PokemonSimEnv
from src.agents.explorer import ExplorerAgent
from src.agents.tactician import TacticianAgent
from src.agents.strategist import Strategist


def play():
    # 1. Configuración Gráfica
    plt.ion()  # Modo interactivo
    fig, (ax_map, ax_battle) = plt.subplots(1, 2, figsize=(12, 6))

    # 2. Cargar Entorno y Agentes
    env = PokemonSimEnv()

    # IMPORTANTE: Asegúrate de que las dimensiones coinciden con train.py
    explorer = ExplorerAgent(obs_shape=(3, 10, 10), n_actions=4)
    tactician = TacticianAgent(input_dim=10, n_actions=5)
    strategist = Strategist(env.pokedex)

    # 3. Cargar los Pesos Entrenados (Checkpoints)
    # Busca en tu carpeta 'checkpoints' el número más alto que tengas
    # Ej: explorer_ep500.pth
    try:
        episode_num = 500  # <--- CAMBIA ESTO POR EL QUE TENGAS
        explorer.policy_net.load_state_dict(torch.load(
            f"checkpoints/explorer_ep{episode_num}.pth"))
        tactician.policy_net.load_state_dict(torch.load(
            f"checkpoints/tactician_ep{episode_num}.pth"))
        print(f"✅ Modelos del episodio {episode_num} cargados.")
    except FileNotFoundError:
        print("❌ No se encontraron modelos entrenados. Jugando con cerebros aleatorios.")

    # Poner modelos en modo evaluación (no aprenden, solo actúan)
    explorer.policy_net.eval()
    tactician.policy_net.eval()

    # Configurar un equipo aleatorio de 6 para la demo
    all_ids = list(env.pokedex.keys())
    if all_ids:
        party = np.random.choice(all_ids, 6, replace=False)
        strategist.set_party(party)
        # Elegir pokemon inicial
        best_mon = strategist.build_team("normal")  # Default
        env.my_pokemon = best_mon.copy()

    state, _ = env.reset()
    done = False

    print("\n--- INICIANDO VISUALIZACIÓN ---")

    while not done:
        # --- LÓGICA DE DECISIÓN (Igual que train.py pero sin learn) ---
        if env.mode == "MAP":
            # Usar Explorador
            # Nota: select_action espera un array, no un tensor directo a veces, depende de tu impl
            # Aquí forzamos una selección 'greedy' (sin aleatoriedad)
            with torch.no_grad():
                state_t = torch.FloatTensor(
                    state).unsqueeze(0).to(explorer.device)
                action = explorer.policy_net(state_t).argmax().item()

            next_state, reward, done, _, _ = env.step(action)

            if env.mode == "COMBAT":
                state = next_state
                continue
            state = next_state

        elif env.mode == "COMBAT":
            # Usar Táctico
            with torch.no_grad():
                state_t = torch.FloatTensor(state).to(tactician.device)
                if state_t.dim() == 1:
                    state_t = state_t.unsqueeze(0)
                action = tactician.policy_net(state_t).argmax().item()

            env_action = action + 4
            next_state, reward, done, _, _ = env.step(env_action)

            if env.mode == "MAP":
                state = next_state
                continue
            state = next_state

        # --- RENDERIZADO GRÁFICO ---
        render_frame(env, ax_map, ax_battle)
        plt.pause(0.1)  # Velocidad del juego (0.1 = rápido, 0.5 = lento)

    plt.ioff()
    plt.show()


def render_frame(env, ax1, ax2):
    ax1.clear()
    ax2.clear()

    # --- DIBUJAR MAPA (Izquierda) ---
    # Convertimos la matriz del entorno a imagen visual
    # Canal 0: nada, Canal 1: Muros, Canal 2: Hierba
    # Reconstruimos una matriz 2D para mostrar
    grid_display = np.zeros((10, 10))

    # Pintamos hierba (verde claro = 0.5)
    # Suponiendo que tu env pone hierba como valor 2 en algún lugar
    # Como tu env es simple, usaremos posiciones conocidas o aleatorias si no guardas el grid
    # Para visualizar, pintamos el grid que tenga el env si es accesible, o uno fake

    # Truco: Usar el state del mapa (3, 10, 10)
    if env.mode == "MAP":
        map_data = env.map_state
        # Canal 0 es 'Jugador' si lo implementaste así, si no lo pintamos manual
        player_y, player_x = env.player_pos

        # Fondo blanco
        display = np.ones((10, 10, 3))

        # Pintar Jugador (Rojo)
        display[player_y, player_x] = [1, 0, 0]

        # Meta (Dorado) - Asumimos pos 9,9
        display[9, 9] = [1, 0.8, 0]

        ax1.imshow(display)
        ax1.set_title("Exploración")
    else:
        ax1.text(0.5, 0.5, "EN COMBATE", ha='center', va='center', fontsize=20)
        ax1.set_title("Exploración Pausada")

    # --- DIBUJAR COMBATE (Derecha) ---
    if env.mode == "COMBAT":
        # Extraer datos
        p1_name = env.my_pokemon['name']
        p2_name = env.enemy_pokemon['name']

        # Calcular porcentajes de vida
        hp_p1 = env.my_hp / env.max_hp_my if env.max_hp_my > 0 else 0
        hp_p2 = env.enemy_hp / env.max_hp_enemy if env.max_hp_enemy > 0 else 0

        # Barras de vida
        ax2.bar([0], [hp_p1], color='green' if hp_p1 >
                0.5 else 'red', label='Tú')
        ax2.bar([1], [hp_p2], color='blue', label='Enemigo')

        ax2.set_ylim(0, 1)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels([p1_name, p2_name])
        ax2.set_title(f"Batalla: {p1_name} vs {p2_name}")
        ax2.text(0, hp_p1 + 0.05, f"{int(env.my_hp)}", ha='center')
        ax2.text(1, hp_p2 + 0.05, f"{int(env.enemy_hp)}", ha='center')

    else:
        ax2.text(0.5, 0.5, "Esperando oponente...", ha='center')
        ax2.set_title("Estado de Batalla")


if __name__ == "__main__":
    play()
