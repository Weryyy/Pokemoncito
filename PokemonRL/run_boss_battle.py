import time
import torch
import sys
import os

# Ajustar path para ejecución desde la raíz del proyecto
sys.path.append(os.getcwd())

# --- IMPORTS CORREGIDOS SEGÚN TUS ARCHIVOS ---
from src.env.pokemon_env import PokemonSimEnv  # Nombre correcto de la clase del entorno
from src.game_manager import GameManager
from src.agents.strategist import Strategist
from src.agents.tactician import TacticianAgent # Nombre correcto: TacticianAgent
from src.agents.explorer import ExplorerAgent   # Nombre correcto: ExplorerAgent

def main():
    print("--- INICIANDO SISTEMA DE BATALLA DE GIMNASIO (FIXED) ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # 1. Inicializar Entorno
    # Usamos PokemonSimEnv como está definido en tu archivo
    env = PokemonSimEnv(verbose=True)
    
    # 2. Inicializar Agentes con Dimensiones Manuales
    # Tu entorno no expone n_states_combat, así que lo definimos según _get_combat_state (tamaño 10)
    TACTICIAN_INPUT_DIM = 10 
    TACTICIAN_N_ACTIONS = 4   # 4 movimientos posibles
    
    # El Explorer usa un stack de frames (9 canales, 10x10) según _get_stacked_state
    EXPLORER_OBS_SHAPE = (9, 10, 10)
    EXPLORER_N_ACTIONS = 4

    print("Inicializando Cerebros...")
    
    # Strategist necesita la pokedex del env
    strategist = Strategist(env.pokedex)
    
    # TacticianAgent (Nombre corregido)
    tactician = TacticianAgent(TACTICIAN_INPUT_DIM, TACTICIAN_N_ACTIONS)
    
    # ExplorerAgent (Nombre corregido)
    explorer = ExplorerAgent(EXPLORER_OBS_SHAPE, EXPLORER_N_ACTIONS)

    # Opcional: Cargar modelos si tienes checkpoints
    # if os.path.exists("checkpoints/tactician_ep2000.pth"):
    #     tactician.policy_net.load_state_dict(torch.load("checkpoints/tactician_ep2000.pth"))

    # 3. Inicializar Game Manager
    manager = GameManager(env, strategist, tactician, explorer)
    
    # Preparamos el juego (Crea el equipo del jugador)
    manager.init_game()

    # ---------------------------------------------------------
    # 4. CONFIGURACIÓN DEL JEFE FINAL (GYM LEADER)
    # ---------------------------------------------------------
    # IDs de la Gen 1 para el equipo del Boss
    boss_team_ids = ["6", "9", "3", "149", "143", "150"] # Charizard, Blastoise, Venusaur, Dragonite, Snorlax, Mewtwo
    
    print(f"\n⚡ GENERANDO EQUIPO DEL LÍDER DE GIMNASIO ⚡")
    manager.gym_team = [] 
    
    for pid in boss_team_ids:
        # Creamos los pokemon del boss a nivel 65
        boss_mon = strategist.prepare_pokemon(pid, level=65)
        if boss_mon:
            manager.gym_team.append(boss_mon)
            print(f" - [Rival] {boss_mon['name']} (Nv. {boss_mon['level']}) preparado.")

    # Buff al jugador para que sea una pelea justa
    print("\n🛡️ ENTRENANDO EQUIPO DEL JUGADOR (MODO PRUEBA) 🛡️")
    for i, p in enumerate(manager.my_team):
        # Subimos al jugador a nivel 62
        new_p = strategist.prepare_pokemon(p['id'], level=62)
        if new_p:
            manager.my_team[i] = new_p
            print(f" - [Tú] {new_p['name']} subido a Nv. {new_p['level']}")
    
    # Actualizar referencia del activo
    manager.update_active_pokemon()

    # ---------------------------------------------------------
    # 5. GAME LOOP (MODO JEFE)
    # ---------------------------------------------------------
    # Forzamos el inicio de la batalla de jefe
    manager.start_boss_battle()

    print("\n--- ¡COMIENZA EL COMBATE! ---\n")
    game_running = True
    
    try:
        while game_running:
            # Solo ejecutamos lógica de combate
            if manager.boss_mode:
                manager.combat_logic()
                time.sleep(1.0) # Pausa para leer logs
            else:
                # Si boss_mode se vuelve False, significa que ganaste o perdiste
                print("\n🏁 La secuencia de batalla ha terminado.")
                game_running = False

    except SystemExit:
        print("\n--- FIN DE LA SIMULACIÓN ---")
    except KeyboardInterrupt:
        print("\n--- INTERRUMPIDO POR EL USUARIO ---")
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()