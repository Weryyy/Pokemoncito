import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import os
from src.env.battle_engine import BattleEngine


class PokemonSimEnv(gym.Env):
    def __init__(self, verbose=False):  # <--- NUEVO PARÁMETRO
        super(PokemonSimEnv, self).__init__()
        self.verbose = verbose  # Si es False, no imprime nada

        # Cargar datos
        try:
            # Ruta absoluta segura para leer el JSON
            base_dir = os.path.dirname(
                os.path.dirname(os.path.dirname(__file__)))
            data_path = os.path.join(base_dir, 'data', 'pokedex.json')
            with open(data_path, 'r') as f:
                self.pokedex = json.load(f)
        except FileNotFoundError:
            self.pokedex = {}

        # Definir Grid si no existe
        self.grid = np.zeros((10, 10), dtype=int)
        self.grid[9][9] = 9

        self.observation_space = spaces.Box(
            low=0, high=2, shape=(3, 10, 10), dtype=np.float32)
        self.action_space = spaces.Discrete(9)

        self.mode = "MAP"
        self.state = None
        self.my_pokemon = None
        self.enemy_pokemon = None

    def log(self, message):
        """Solo imprime si estamos en modo verbose (visual)"""
        if self.verbose:
            print(message)

    def reset(self, seed=None):
        super().reset(seed=seed)
        if not hasattr(self, 'grid') or self.grid is None:
            self.grid = np.zeros((10, 10), dtype=int)
            self.grid[9][9] = 9

        self.map_state = np.zeros((3, 10, 10), dtype=np.float32)
        self.player_pos = [0, 0]

        if self.my_pokemon is None:
            if self.pokedex:
                self.my_pokemon = self.pokedex.get("4", {}).copy()

        if self.my_pokemon:
            self.max_hp_my = self.my_pokemon['stats']['hp'] * 2 + 110
            self.my_hp = self.max_hp_my

        self.mode = "MAP"
        return self.map_state, {}

    def step(self, action):
        if self.mode == "MAP":
            return self._step_map(action)
        else:
            return self._step_combat(action)

    def _step_map(self, action):
        if action >= 4:
            return self.map_state, -10, False, False, {}

        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dy, dx = moves.get(action, (0, 0))

        ny, nx = self.player_pos[0] + dy, self.player_pos[1] + dx

        if 0 <= ny < 10 and 0 <= nx < 10:
            tile = self.grid[ny][nx]
            if tile == 1:  # Muro
                return self.map_state, -0.5, False, False, {}

            self.player_pos = [ny, nx]

            if tile == 9:  # Meta
                self.log("¡Meta alcanzada!")
                return self.map_state, 100, True, False, {}

            elif tile == 2:  # Hierba
                if np.random.rand() < 0.2:
                    self.mode = "COMBAT"
                    if self.pokedex:
                        eid = str(np.random.randint(1, 152))
                        self.enemy_pokemon = self.pokedex.get(
                            eid, self.my_pokemon).copy()
                    else:
                        self.enemy_pokemon = self.my_pokemon.copy()

                    self.max_hp_my = self.my_pokemon['stats']['hp'] * 2 + 110
                    self.max_hp_enemy = self.enemy_pokemon['stats']['hp'] * 2 + 110
                    self.my_hp, self.enemy_hp = self.max_hp_my, self.max_hp_enemy

                    self.log(f"¡Combate contra {self.enemy_pokemon['name']}!")
                    return self._get_combat_state(), 0, False, False, {}
        else:
            return self.map_state, -1.0, False, False, {}  # Fuera mapa

        return self.map_state, -0.1, False, False, {}

    def _step_combat(self, action):
        if action < 4:
            return self._get_combat_state(), -10, False, False, {}

        attack_types = ["normal", "fire", "water", "grass", "electric"]
        idx = action - 4
        if idx >= len(attack_types):
            idx = 0

        dmg, _ = BattleEngine.calculate_damage(
            self.my_pokemon, self.enemy_pokemon, 60, attack_types[idx])
        self.enemy_hp -= dmg

        if self.enemy_hp <= 0:
            self.mode = "MAP"
            self.log("¡Victoria!")
            return self.map_state, 50, False, False, {}

        enemy_type = self.enemy_pokemon['types'][0]
        dmg_r, _ = BattleEngine.calculate_damage(
            self.enemy_pokemon, self.my_pokemon, 50, enemy_type)
        self.my_hp -= dmg_r

        if self.my_hp <= 0:
            self.log("¡Derrota!")
            return self._get_combat_state(), -50, True, False, {}

        return self._get_combat_state(), dmg*0.1, False, False, {}

    def _get_combat_state(self):
        state = np.zeros(10, dtype=np.float32)
        if self.max_hp_my > 0:
            state[0] = self.my_hp / self.max_hp_my
        state[1] = self.my_pokemon['stats']['attack'] / 200.0
        state[2] = self.my_pokemon['stats']['defense'] / 200.0
        if self.max_hp_enemy > 0:
            state[5] = self.enemy_hp / self.max_hp_enemy
        state[6] = self.enemy_pokemon['stats']['attack'] / 200.0
        state[7] = self.enemy_pokemon['stats']['defense'] / 200.0
        return state
