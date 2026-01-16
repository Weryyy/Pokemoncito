import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import os
from src.env.battle_engine import BattleEngine
from src.env.maps import ALL_MAPS

class PokemonSimEnv(gym.Env):
    def __init__(self, verbose=False):
        super(PokemonSimEnv, self).__init__()
        self.verbose = verbose
        
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            with open(os.path.join(base_dir, 'data', 'pokedex.json'), 'r') as f:
                self.pokedex = json.load(f)
        except:
            self.pokedex = {}

        self.observation_space = spaces.Box(low=0, high=2, shape=(3, 10, 10), dtype=np.float32)
        self.action_space = spaces.Discrete(9)
        
        self.current_map_idx = 0
        self.grid = np.array(ALL_MAPS[0])
        self.mode = "MAP"
        self.my_pokemon = None

    def log(self, msg):
        if self.verbose: print(msg)

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Cargar el mapa que toque
        self.grid = np.array(ALL_MAPS[self.current_map_idx])
        self.map_state = np.zeros((3, 10, 10), dtype=np.float32)
        self.player_pos = [0, 0]
        
        # --- FIX CRÍTICO: Inicialización correcta del Pokemon ---
        if self.my_pokemon is None:
            if self.pokedex:
                self.my_pokemon = self.pokedex.get("4", {}).copy()
            else:
                self.my_pokemon = {"name": "BugMon", "stats": {"hp":40, "attack":40, "defense":40}, "types":["normal"]}

        # Asegurar campos RPG
        if 'level' not in self.my_pokemon: self.my_pokemon['level'] = 5
        if 'exp' not in self.my_pokemon: self.my_pokemon['exp'] = 0

        # Calcular HP Real actual
        stats = BattleEngine.get_stats_at_level(self.my_pokemon, self.my_pokemon['level'])
        self.max_hp_my = stats['hp']
        self.my_hp = self.max_hp_my
        # -----------------------------------------------------

        self.mode = "MAP"
        return self.map_state, {}

    def step(self, action):
        if self.mode == "MAP": return self._step_map(action)
        else: return self._step_combat(action)

    def _step_map(self, action):
        if action >= 4: return self.map_state, -10, False, False, {}

        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dy, dx = moves.get(action, (0, 0))
        ny, nx = self.player_pos[0] + dy, self.player_pos[1] + dx
        
        if 0 <= ny < 10 and 0 <= nx < 10:
            tile = self.grid[ny][nx]
            if tile == 1: # Muro
                return self.map_state, -0.5, False, False, {}
            
            self.player_pos = [ny, nx]
            
            if tile == 9: # Meta
                self.log(f"¡Mapa {self.current_map_idx+1} Completado!")
                # Recompensa alta si ya tenemos nivel decente
                reward = 100 if self.my_pokemon['level'] >= 25 else 20
                return self.map_state, reward, True, False, {}
            
            elif tile == 2: # Hierba
                if np.random.rand() < 0.2:
                    self.mode = "COMBAT"
                    self._generate_wild_enemy()
                    self.log(f"¡{self.enemy_pokemon['name']} Nvl {self.enemy_pokemon['level']} salvaje!")
                    return self._get_combat_state(), 0, False, False, {}
        else:
            return self.map_state, -1.0, False, False, {}

        return self.map_state, -0.1, False, False, {}

    def _generate_wild_enemy(self):
        # Nivel depende del mapa
        min_lvl = 3 + (self.current_map_idx * 5)
        max_lvl = 6 + (self.current_map_idx * 5)
        level = np.random.randint(min_lvl, max_lvl + 1)
        
        if self.pokedex:
            eid = str(np.random.randint(1, 152))
            self.enemy_pokemon = self.pokedex.get(eid, self.my_pokemon).copy()
        else:
            self.enemy_pokemon = self.my_pokemon.copy()
            
        self.enemy_pokemon['level'] = level
        stats = BattleEngine.get_stats_at_level(self.enemy_pokemon, level)
        self.max_hp_enemy = stats['hp']
        self.enemy_hp = self.max_hp_enemy

    def _step_combat(self, action):
        if action < 4: return self._get_combat_state(), -10, False, False, {}
        
        attack_types = ["normal", "fire", "water", "grass", "electric"]
        idx = action - 4
        if idx >= len(attack_types): idx = 0
        
        # 1. Mi turno
        dmg, _ = BattleEngine.calculate_damage(self.my_pokemon, self.enemy_pokemon, 60, attack_types[idx])
        self.enemy_hp -= dmg
        
        if self.enemy_hp <= 0:
            self.mode = "MAP"
            exp_gain = BattleEngine.get_exp_reward(self.enemy_pokemon)
            self.my_pokemon['exp'] += exp_gain
            msg = f"Ganaste +{exp_gain} EXP."
            
            if self.my_pokemon['exp'] >= 100:
                self.my_pokemon['level'] += 1
                self.my_pokemon['exp'] = 0
                # Recalcular stats al subir nivel
                stats = BattleEngine.get_stats_at_level(self.my_pokemon, self.my_pokemon['level'])
                self.max_hp_my = stats['hp']
                self.my_hp = self.max_hp_my
                msg += f" ¡NIVEL {self.my_pokemon['level']}!"

            # Recompensa alta por farmear si soy nivel bajo
            combat_reward = 80 if self.my_pokemon['level'] < 30 else 10
            return self.map_state, combat_reward, False, False, {"log": msg}
            
        # 2. Turno Rival
        enemy_type = self.enemy_pokemon['types'][0]
        dmg_r, _ = BattleEngine.calculate_damage(self.enemy_pokemon, self.my_pokemon, 40, enemy_type)
        self.my_hp -= dmg_r
        
        if self.my_hp <= 0:
            return self._get_combat_state(), -50, True, False, {"log": "Te debilitaste..."}

        return self._get_combat_state(), dmg*0.05, False, False, {}

    def _get_combat_state(self):
        state = np.zeros(10, dtype=np.float32)
        if self.max_hp_my > 0: state[0] = self.my_hp / self.max_hp_my
        
        my_stats = BattleEngine.get_stats_at_level(self.my_pokemon, self.my_pokemon['level'])
        state[1] = my_stats['attack'] / 300.0
        state[2] = my_stats['defense'] / 300.0
        
        if self.max_hp_enemy > 0: state[5] = self.enemy_hp / self.max_hp_enemy
        
        en_stats = BattleEngine.get_stats_at_level(self.enemy_pokemon, self.enemy_pokemon['level'])
        state[6] = en_stats['attack'] / 300.0
        state[7] = en_stats['defense'] / 300.0
        
        return state