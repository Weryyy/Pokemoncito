import time
import os
import numpy as np

# Simulamos la clase Entorno que definimos antes


class PokemonSimEnv:
    def __init__(self):
        self.width = 10
        self.height = 10
        self.player_pos = [0, 0]  # [y, x]
        self.goal_pos = [8, 8]
        # 0: Suelo, 1: Muro, 2: Hierba, 9: Meta
        self.grid = np.zeros((10, 10), dtype=int)
        self._generate_map()
        self.in_combat = False

    def _generate_map(self):
        # Ponemos algunos obstáculos y hierba
        self.grid[2:4, 2:4] = 1  # Muro
        self.grid[5:8, 5:8] = 2  # Hierba alta
        self.grid[self.goal_pos[0], self.goal_pos[1]] = 9  # Meta

    def render(self):
        # Limpiar consola (funciona en Windows y Linux/Mac)
        os.system('cls' if os.name == 'nt' else 'clear')

        if self.in_combat:
            print("\n" + "="*20)
            print("   ¡EN COMBATE!   ")
            print("   [Agente Tactico Activo]")
            print("="*20 + "\n")
            return

        print(f"--- MODO EXPLORACIÓN ---")
        print(f"Posición: {self.player_pos}")

        # Dibujado ASCII
        for y in range(self.height):
            row_str = ""
            for x in range(self.width):
                if y == self.player_pos[0] and x == self.player_pos[1]:
                    row_str += "🤖 "  # Nuestro Agente
                elif self.grid[y][x] == 0:
                    row_str += ".  "
                elif self.grid[y][x] == 1:
                    row_str += "🧱 "
                elif self.grid[y][x] == 2:
                    row_str += "🌿 "  # Hierba
                elif self.grid[y][x] == 9:
                    row_str += "🏆 "
            print(row_str)

    def step(self, action):
        # Simulación muy básica del movimiento
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Arriba, Abajo, Izq, Der
        dy, dx = moves[action]

        new_y = max(0, min(9, self.player_pos[0] + dy))
        new_x = max(0, min(9, self.player_pos[1] + dx))

        cell_type = self.grid[new_y][new_x]

        if cell_type == 1:  # Choca con muro
            return -1, False  # Castigo leve

        self.player_pos = [new_y, new_x]

        if cell_type == 2:  # Hierba
            # 30% probabilidad de combate
            if np.random.rand() < 0.3:
                self.in_combat = True
                return 0, False  # Entra en combate

        if cell_type == 9:  # Gana
            return 100, True

        return -0.1, False  # Penalización de tiempo

# --- INICIO DEL JUEGO (Simulación) ---


env = PokemonSimEnv()
done = False

print("Iniciando Simulación de Agentes...")
time.sleep(2)

while not done:
    env.render()

    if env.in_combat:
        # AQUÍ ACTÚA EL AGENTE DE COMBATE
        print(">> Agente Táctico calculando mejor ataque...")
        # action = combat_cnn.predict(state)
        time.sleep(1)  # Simular tiempo de "pensar"
        print(">> ¡Charizard usó Lanzallamas!")

        # Simulamos que gana el combate rápido para seguir explorando
        env.in_combat = False
        # Pausa para que veas el combate
        input("Presiona Enter para continuar...")

    else:
        # AQUÍ ACTÚA EL AGENTE EXPLORADOR
        # action = exploration_cnn.predict(map_state)

        # Simulamos una acción aleatoria inteligente (fake AI para el ejemplo)
        action = np.random.choice([0, 1, 2, 3])

        reward, done = env.step(action)
        time.sleep(0.5)  # Velocidad de visualización

print("¡El Agente ha llegado a la meta!")
