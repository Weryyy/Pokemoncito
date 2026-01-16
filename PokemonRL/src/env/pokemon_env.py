import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import os
from collections import deque # <--- IMPORTANTE: Para la memoria
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

        # --- FRAME STACKING ---
        # Ahora el estado es 9 capas (3 frames x 3 canales)
        self.stack_size = 3
        self.frame_stack = deque(maxlen=self.stack_size)
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(9, 10, 10), dtype=np.float32)
        self.action_space = spaces.Discrete(4) 
        
        self.current_map_idx = 0
        self.grid = np.array(ALL_MAPS[0])
        self.mode = "MAP"
        self.my_pokemon = None

    def log(self, msg):
        if self.verbose: print(msg)

    def _get_map_state(self):
        """Genera UN solo frame (3 capas)"""
        state = np.zeros((3, 10, 10), dtype=np.float32)
        state[0] = (self.grid == 1).astype(np.float32) # Muros
        
        mask_grass = (self.grid == 2).astype(np.float32) * 0.5
        mask_goal = (self.grid == 9).astype(np.float32) * 1.0
        state[1] = mask_grass + mask_goal # Interés
        
        py, px = self.player_pos
        state[2][py][px] = 1.0 # Jugador
        return state

    def _get_stacked_state(self):
        """Devuelve los últimos 3 frames concatenados (9 capas)"""
        # Convertimos la deque en un array numpy (3, 3, 10, 10) -> (9, 10, 10)
        stack = np.array(self.frame_stack)
        return stack.reshape(-1, 10, 10)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.grid = np.array(ALL_MAPS[self.current_map_idx])
        self.player_pos = [0, 0]
        
        if self.my_pokemon is None:
            if self.pokedex: self.my_pokemon = self.pokedex.get("4", {}).copy()
            else: self.my_pokemon = {"name": "BugMon", "stats": {"hp":40,"atk":40,"def":40}, "types":["normal"]}

        if 'level' not in self.my_pokemon: self.my_pokemon['level'] = 5
        if 'exp' not in self.my_pokemon: self.my_pokemon['exp'] = 0

        stats = BattleEngine.get_stats_at_level(self.my_pokemon, self.my_pokemon['level'])
        self.max_hp_my = stats['hp']
        self.my_hp = self.max_hp_my
        self.mode = "MAP"
        
        # --- LLENAR PILA INICIAL ---
        initial_frame = self._get_map_state()
        for _ in range(self.stack_size):
            self.frame_stack.append(initial_frame)
            
        return self._get_stacked_state(), {}

    def step(self, action):
        if self.mode == "MAP": return self._step_map(action)
        else: return self._step_combat(action)

    def _step_map(self, action):
        if action >= 4: 
            return self._get_stacked_state(), -0.1, False, False, {}

        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dy, dx = moves.get(action, (0, 0))
        ny, nx = self.player_pos[0] + dy, self.player_pos[1] + dx
        
        step_reward = -0.01 
        
        if 0 <= ny < 10 and 0 <= nx < 10:
            tile = self.grid[ny][nx]
            
            if tile == 1: # Choque
                self.frame_stack.append(self._get_map_state()) # Actualizamos visión (ve el muro)
                return self._get_stacked_state(), -0.5, False, False, {}
            
            self.player_pos = [ny, nx]
            
            # --- RECOMPENSAS OPTIMIZADAS ---
            if tile == 9: # META
                self.log(f"¡Mapa Completado!")
                # Si soy nivel bajo, llegar a la meta da poco (prefiero farmear)
                # Si soy nivel alto, llegar a la meta da MUCHO (quiero avanzar)
                reward = 500 if self.my_pokemon['level'] >= 25 else 50
                self.frame_stack.append(self._get_map_state())
                return self._get_stacked_state(), reward, True, False, {}
            
            elif tile == 2: # HIERBA
                # Pequeño premio por pisar hierba si necesito nivel
                if self.my_pokemon['level'] < 25:
                    step_reward = 0.1 
                
                if np.random.rand() < 0.2: # Combate
                    self.mode = "COMBAT"
                    self._generate_wild_enemy()
                    self.log(f"¡{self.enemy_pokemon['name']} salvaje!")
                    self.frame_stack.append(self._get_map_state())
                    return self._get_combat_state(), 0, False, False, {}
        else:
            self.frame_stack.append(self._get_map_state())
            return self._get_stacked_state(), -0.5, False, False, {}

        self.frame_stack.append(self._get_map_state())
        return self._get_stacked_state(), step_reward, False, False, {}

    def _generate_wild_enemy(self):
        min_lvl = 3 + (self.current_map_idx * 5)
        max_lvl = 6 + (self.current_map_idx * 5)
        level = np.random.randint(min_lvl, max_lvl + 1)
        if self.pokedex:
            eid = str(np.random.randint(1, 152))
            self.enemy_pokemon = self.pokedex.get(eid, self.my_pokemon).copy()
        else: self.enemy_pokemon = self.my_pokemon.copy()
        self.enemy_pokemon['level'] = level
        stats = BattleEngine.get_stats_at_level(self.enemy_pokemon, level)
        self.max_hp_enemy = stats['hp']
        self.enemy_hp = self.max_hp_enemy

    def _step_combat(self, action):
        if action < 4: return self._get_combat_state(), -0.5, False, False, {}
        idx = action - 4
        dmg, _ = BattleEngine.calculate_damage(self.my_pokemon, self.enemy_pokemon, 60, ["normal","fire","water","grass","electric"][idx] if idx<5 else "normal")
        self.enemy_hp -= dmg
        
        if self.enemy_hp <= 0:
            self.mode = "MAP"
            exp = BattleEngine.get_exp_reward(self.enemy_pokemon)
            self.my_pokemon['exp'] += exp
            msg = f"Ganaste +{exp} XP"
            if self.my_pokemon['exp'] >= 100:
                self.my_pokemon['level'] += 1; self.my_pokemon['exp'] = 0
                s = BattleEngine.get_stats_at_level(self.my_pokemon, self.my_pokemon['level'])
                self.max_hp_my = s['hp']; self.my_hp = s['hp']
                msg += " ¡NIVEL UP!"
            
            # --- RECOMPENSA DE COMBATE ---
            # Muy alta si soy nivel bajo (quiero farmear)
            combat_reward = 100 if self.my_pokemon['level'] < 30 else 20
            
            # ¡IMPORTANTE! Al volver al mapa, añadimos el frame actual a la pila
            self.frame_stack.append(self._get_map_state()) 
            return self._get_stacked_state(), combat_reward, False, False, {"log": msg}
            
        enemy_type = self.enemy_pokemon['types'][0]
        dmg_r, _ = BattleEngine.calculate_damage(self.enemy_pokemon, self.my_pokemon, 40, enemy_type)
        self.my_hp -= dmg_r
        
        if self.my_hp <= 0: return self._get_combat_state(), -50, True, False, {"log": "Debilitado..."}
        return self._get_combat_state(), dmg*0.05, False, False, {}

    def _get_combat_state(self):
        # El estado de combate sigue siendo 1D (no usa frame stacking por ahora, no hace falta para táctica simple)
        state = np.zeros(10, dtype=np.float32)
        if self.max_hp_my > 0: state[0] = self.my_hp / self.max_hp_my
        ms = BattleEngine.get_stats_at_level(self.my_pokemon, self.my_pokemon['level'])
        state[1] = ms['attack']/300.0; state[2] = ms['defense']/300.0
        if self.max_hp_enemy > 0: state[5] = self.enemy_hp / self.max_hp_enemy
        es = BattleEngine.get_stats_at_level(self.enemy_pokemon, self.enemy_pokemon['level'])
        state[6] = es['attack']/300.0; state[7] = es['defense']/300.0
        return state