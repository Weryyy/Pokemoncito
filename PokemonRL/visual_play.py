import pygame
import torch
import numpy as np
import sys
import os
import requests
import io
import time
from collections import deque
from src.env.pokemon_env import PokemonSimEnv
from src.agents.explorer import ExplorerAgent
from src.agents.tactician import TacticianAgent
from src.agents.strategist import Strategist
from src.env.battle_engine import BattleEngine
from src.env.maps import ALL_MAPS

# --- CONFIGURACIÓN ---
CELL_SIZE = 50
GRID_W, GRID_H = 10, 10
SCREEN_W = GRID_W * CELL_SIZE
SCREEN_H = GRID_H * CELL_SIZE + 220 
FPS = 60
STEP_DELAY = 50 
BOSS_DELAY = 500

C_BG = (20, 20, 20)
C_WALL = (100, 100, 100)
C_GRASS = (50, 150, 50)
C_PATH = (200, 200, 180)
C_PLAYER = (0, 255, 255) # CIAN BRILLANTE (Para que se vea bien)
C_GOAL = (255, 215, 0)
C_TEXT = (255, 255, 255)
C_BLOCKED = (0, 0, 0) 
C_VISITED = (50, 50, 50) # GRIS OSCURO (Rastro de migas)

GYM_LEADER_TEAM_IDS = ["18", "65", "112", "59", "103", "6"] 

class SpriteManager:
    def __init__(self):
        self.cache = {}
        self.sprite_dir = "data/sprites"
        os.makedirs(self.sprite_dir, exist_ok=True)
        self.default = pygame.Surface((150, 150))
        self.default.fill((50,50,50))

    def get_sprite(self, pokemon_data):
        if not pokemon_data: return self.default
        pid = str(pokemon_data['id'])
        url = pokemon_data.get('sprite')
        if pid in self.cache: return self.cache[pid]
        path = os.path.join(self.sprite_dir, f"{pid}.png")
        try:
            if os.path.exists(path):
                img = pygame.image.load(path)
            elif url:
                r = requests.get(url)
                img = pygame.image.load(io.BytesIO(r.content))
                pygame.image.save(img, path)
            else: return self.default
            img = pygame.transform.scale(img, (180, 180))
            self.cache[pid] = img
            return img
        except: return self.default

class GameVisualizer:
    def __init__(self, episode_to_load=2000):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Pokémon AI: Gameplay Completo")
        self.clock = pygame.time.Clock()
        self.font_log = pygame.font.SysFont("Consolas", 14)
        self.font_ui = pygame.font.SysFont("Arial", 20, bold=True)
        self.big_font = pygame.font.SysFont("Arial", 40, bold=True)
        
        self.env = PokemonSimEnv(verbose=True) 
        self.sprites = SpriteManager()
        
        self.explorer = ExplorerAgent(obs_shape=(9, 10, 10), n_actions=4)
        self.tactician = TacticianAgent(input_dim=10, n_actions=5)
        self.strategist = Strategist(self.env.pokedex)
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(base_dir, "checkpoints")
        exp_path = os.path.join(checkpoint_dir, f"explorer_ep{episode_to_load}.pth")
        tac_path = os.path.join(checkpoint_dir, f"tactician_ep{episode_to_load}.pth")
        
        try:
            self.explorer.policy_net.load_state_dict(torch.load(exp_path))
            self.tactician.policy_net.load_state_dict(torch.load(tac_path))
            print(f"✅ Cerebros cargados desde: {checkpoint_dir}")
        except FileNotFoundError:
            print(f"❌ ERROR: No se encuentra el archivo en: {exp_path}")
        except Exception as e:
            print(f"⚠ Error de carga: {e}")
            
        self.explorer.policy_net.eval()
        self.tactician.policy_net.eval()
        
        self.current_level_idx = 0
        self.battle_log = ["¡Iniciando aventura RPG!"]
        self.boss_mode = False
        self.my_full_team = [] 
        self.enemy_gym_team = []
        
        # --- MEMORIA DE NAVEGACIÓN ---
        self.wall_hits = {}       
        self.blocked_cells = set() 
        self.visit_counts = {} 
        self.action_history = deque(maxlen=15)
        
        self.setup_rpg_teams()
        self.load_level(0)

    def setup_rpg_teams(self):
        all_ids = list(self.env.pokedex.keys())
        party_ids = np.random.choice(all_ids, 6, replace=False) if len(all_ids)>6 else all_ids
        self.strategist.set_party(party_ids)
        for pid in party_ids:
            if pid in self.env.pokedex:
                p = self.env.pokedex[pid].copy()
                p['level'] = 15 # Subimos nivel inicial para aguantar más
                p['exp'] = 0
                p['stats'] = BattleEngine.get_stats_at_level(p, 15)
                self.my_full_team.append(p)
        self.update_active_pokemon()

        for pid in GYM_LEADER_TEAM_IDS:
            if pid in self.env.pokedex:
                p = self.env.pokedex[pid].copy()
                p['level'] = 50
                p['stats'] = BattleEngine.get_stats_at_level(p, 50)
                self.enemy_gym_team.append(p)

    def update_active_pokemon(self):
        """Busca el primer Pokémon vivo del equipo"""
        for p in self.my_full_team:
            if p['stats']['hp'] > 0:
                self.env.my_pokemon = p.copy()
                self.env.my_hp = p['stats']['hp']
                self.env.max_hp_my = p['stats']['hp']
                return True
        return False # Todos muertos

    def respawn_player(self):
        """Reinicia el nivel y cura al equipo"""
        print("💀 ¡EQUIPO DEBILITADO! Volviendo al Centro Pokémon...")
        self.battle_log.append("¡EQUIPO DEBILITADO! Respawn...")
        
        # Curar equipo
        for p in self.my_full_team:
            p['stats'] = BattleEngine.get_stats_at_level(p, p['level'])
        
        self.update_active_pokemon()
        
        # Resetear Mapa y Memorias (Importante olvidar bloqueos antiguos)
        self.env.reset()
        self.env.grid = np.array(ALL_MAPS[self.current_level_idx])
        self.wall_hits.clear()
        self.blocked_cells.clear()
        self.visit_counts.clear()
        self.action_history.clear()
        time.sleep(1) # Pequeña pausa dramática

    def check_team_health(self):
        """Verifica si el Pokémon activo ha muerto y saca al siguiente"""
        if self.env.my_hp <= 0:
            fainted_name = self.env.my_pokemon['name']
            print(f"❌ {fainted_name} se debilitó.")
            self.battle_log.append(f"{fainted_name} cayó!")
            
            # Actualizar HP en la lista maestra
            for p in self.my_full_team:
                if p['name'] == fainted_name:
                    p['stats']['hp'] = 0
            
            # Intentar sacar al siguiente
            if self.update_active_pokemon():
                print(f"⚡ ¡Adelante {self.env.my_pokemon['name']}!")
                self.battle_log.append(f"¡Go {self.env.my_pokemon['name']}!")
            else:
                self.respawn_player()

    def load_level(self, idx):
        if idx >= len(ALL_MAPS):
            self.start_boss_battle()
            return
        self.current_level_idx = idx
        self.env.current_map_idx = idx
        self.env.grid = np.array(ALL_MAPS[idx])
        self.env.reset() 
        self.env.grid = np.array(ALL_MAPS[idx]) 
        self.battle_log.append(f"--- NIVEL {idx + 1} ---")
        
        self.wall_hits.clear()
        self.blocked_cells.clear()
        self.visit_counts.clear() 
        self.action_history.clear()
        
        print(f"\n🗺 CARGANDO NIVEL {idx + 1}")

    def start_boss_battle(self):
        self.boss_mode = True
        self.env.mode = "COMBAT"
        self.battle_log.append("!!! ALERTA: LÍDER DE GIMNASIO !!!")
        print("\n🏆 ¡HA LLEGADO EL MOMENTO! BATALLA FINAL CONTRA EL LÍDER.")
        self.boss_pokemon_idx = 0
        self.env.enemy_pokemon = self.enemy_gym_team[0].copy()
        self.env.max_hp_enemy = self.env.enemy_pokemon['stats']['hp']
        self.env.enemy_hp = self.env.max_hp_enemy
        
        # El estratega elige el mejor pokemon vivo para empezar
        candidates = [p for p in self.my_full_team if p['stats']['hp'] > 0]
        if candidates:
            self.env.my_pokemon = candidates[0].copy()
            self.env.my_hp = self.env.my_pokemon['stats']['hp']
            self.env.max_hp_my = self.env.my_hp

    def draw_game(self):
        self.screen.fill(C_BG)
        if self.boss_mode:
            self.draw_battle_scene()
            txt = self.big_font.render("COMBATE DE GIMNASIO", True, C_GOAL)
            self.screen.blit(txt, (SCREEN_W//2 - 200, 20))
            my_alive = sum(1 for p in self.my_full_team if p['stats']['hp'] > 0)
            en_alive = len(self.enemy_gym_team) - self.boss_pokemon_idx
            self.screen.blit(self.font_ui.render(f"Tu Equipo: {my_alive}/6", True, C_TEXT), (50, 100))
            self.screen.blit(self.font_ui.render(f"Rival: {en_alive}/6", True, C_TEXT), (SCREEN_W - 200, 100))
        elif self.env.mode == "MAP":
            self.draw_grid()
            py, px = self.env.player_pos
            pygame.draw.circle(self.screen, C_PLAYER, (px*CELL_SIZE+25, py*CELL_SIZE+25), 15)
        elif self.env.mode == "COMBAT":
            self.draw_grid()
            s = pygame.Surface((SCREEN_W, SCREEN_H)); s.set_alpha(200); s.fill((0,0,0))
            self.screen.blit(s, (0,0))
            self.draw_battle_scene()
        self.draw_logs()
        pygame.display.flip()

    def draw_grid(self):
        for y in range(GRID_H):
            for x in range(GRID_W):
                rect = (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                val = self.env.grid[y][x]
                
                color = C_PATH
                if val == 1: color = C_WALL
                elif val == 2: color = C_GRASS
                elif val == 9: color = C_GOAL
                
                if (y, x) in self.blocked_cells:
                    color = C_BLOCKED
                
                pygame.draw.rect(self.screen, color, rect)
                
                visits = self.visit_counts.get((y,x), 0)
                if visits > 0 and val != 1:
                    # Rastro GRIS OSCURO sutil
                    alpha = min(150, visits * 30) 
                    s = pygame.Surface((CELL_SIZE, CELL_SIZE))
                    s.set_alpha(alpha)
                    s.fill(C_VISITED) 
                    self.screen.blit(s, (x*CELL_SIZE, y*CELL_SIZE))

                pygame.draw.rect(self.screen, (0,0,0), rect, 1)

    def draw_battle_scene(self):
        my_img = self.sprites.get_sprite(self.env.my_pokemon)
        en_img = self.sprites.get_sprite(self.env.enemy_pokemon)
        self.screen.blit(en_img, (SCREEN_W - 220, 50))
        self.screen.blit(my_img, (50, 250))
        self.draw_hp_bar(50, 230, self.env.my_hp, self.env.max_hp_my, f"{self.env.my_pokemon['name']} Lv{self.env.my_pokemon.get('level', '?')}")
        self.draw_hp_bar(SCREEN_W - 220, 30, self.env.enemy_hp, self.env.max_hp_enemy, f"{self.env.enemy_pokemon['name']} Lv{self.env.enemy_pokemon.get('level', '?')}")

    def draw_hp_bar(self, x, y, curr, max_hp, name):
        pct = max(0, curr / max_hp) if max_hp > 0 else 0
        col = (0, 255, 0) if pct > 0.5 else (255, 0, 0)
        pygame.draw.rect(self.screen, (50,50,50), (x, y, 200, 20))
        pygame.draw.rect(self.screen, col, (x, y, 200 * pct, 20))
        txt = self.font_ui.render(name, True, C_TEXT)
        self.screen.blit(txt, (x, y - 25))

    def draw_logs(self):
        base_y = GRID_H * CELL_SIZE + 20
        pygame.draw.line(self.screen, C_TEXT, (0, base_y), (SCREEN_W, base_y))
        for i, line in enumerate(self.battle_log[-6:]):
            txt = self.font_log.render(line, True, C_TEXT)
            self.screen.blit(txt, (10, base_y + 10 + i*20))

    def get_target_pos(self, action):
        y, x = self.env.player_pos
        if action == 0: y -= 1 
        elif action == 1: y += 1 
        elif action == 2: x -= 1 
        elif action == 3: x += 1 
        return y, x

    def run(self):
        print("📺 VENTANA GRÁFICA INICIADA (GAMEPLAY COMPLETO)...")
        running = True
        last_step_time = pygame.time.get_ticks()
        self.screen.fill(C_BG)
        pygame.display.flip()
        
        while running:
            self.clock.tick(FPS)
            current_time = pygame.time.get_ticks()
            for e in pygame.event.get():
                if e.type == pygame.QUIT: running = False

            self.draw_game()
            needed_delay = STEP_DELAY
            if self.env.mode == "COMBAT" or self.boss_mode: needed_delay = BOSS_DELAY

            if current_time - last_step_time > needed_delay:
                last_step_time = current_time
                
                # --- VERIFICACIÓN DE VIDA (GLOBAL) ---
                # Hacemos esto ANTES de cualquier acción para asegurar que no hay zombies
                self.check_team_health()

                if self.boss_mode:
                    if self.env.enemy_hp <= 0:
                        self.battle_log.append(f"Rival {self.env.enemy_pokemon['name']} cayó!")
                        self.boss_pokemon_idx += 1
                        if self.boss_pokemon_idx >= len(self.enemy_gym_team):
                            self.battle_log.append("¡¡¡ERES EL CAMPEÓN!!!")
                            print("🏆 ¡¡¡CAMPEÓN DE LA LIGA POKÉMON!!! 🏆")
                            self.draw_game(); pygame.display.flip(); pygame.time.delay(5000); running = False
                            continue
                        self.env.enemy_pokemon = self.enemy_gym_team[self.boss_pokemon_idx].copy()
                        self.env.max_hp_enemy = self.env.enemy_pokemon['stats']['hp']
                        self.env.enemy_hp = self.env.max_hp_enemy
                        continue
                    
                    with torch.no_grad():
                        st = self.env._get_combat_state()
                        st_t = torch.FloatTensor(st).unsqueeze(0).to(self.tactician.device)
                        action = self.tactician.policy_net(st_t).argmax().item()
                    _, _, _, _, info = self.env.step(action + 4)
                    if "log" in info: self.battle_log.append(info["log"])
                    continue

                if self.env.mode == "MAP":
                    move_names = ["ARRIBA", "ABAJO", "IZQUIERDA", "DERECHA"]
                    curr_pos = tuple(self.env.player_pos)
                    self.visit_counts[curr_pos] = self.visit_counts.get(curr_pos, 0) + 1
                    
                    with torch.no_grad():
                        st_t = torch.FloatTensor(self.env._get_stacked_state()).unsqueeze(0).to(self.explorer.device)
                        q_vals = self.explorer.policy_net(st_t)
                        q_vals_np = q_vals.cpu().numpy()[0].copy()
                        
                        for a in range(4):
                            ty, tx = self.get_target_pos(a)
                            coord = (ty, tx)
                            if coord in self.blocked_cells:
                                q_vals_np[a] = -99999
                                continue
                            visits = self.visit_counts.get(coord, 0)
                            if visits > 0:
                                penalty = (visits ** 2) * 1.5 
                                q_vals_np[a] -= penalty
                        
                        action = np.argmax(q_vals_np)
                        
                        # ANTI-SPAM
                        self.action_history.append(action)
                        if len(self.action_history) == 15 and len(set(self.action_history)) == 1:
                             print("😵 SPAM DETECTADO: ¡Basta de pulsar el mismo botón!")
                             action = np.random.randint(0,4)
                             self.action_history.clear()

                        # DETECTOR PAREDES
                        ty, tx = self.get_target_pos(action)
                        if 0 <= tx < GRID_W and 0 <= ty < GRID_H and self.env.grid[ty][tx] == 1:
                            coord = (ty, tx)
                            current_hits = self.wall_hits.get(coord, 0) + 1
                            self.wall_hits[coord] = current_hits
                            print(f"💥 Golpe #{current_hits} en pared {coord}")
                            if current_hits >= 2:
                                print(f"🚫 BLOQUEANDO {coord}")
                                self.blocked_cells.add(coord)
                                q_vals_np[action] = -99999
                                action = np.argmax(q_vals_np)

                        print(f"🚶 Agente: {move_names[action]}")

                    _, _, done, _, info = self.env.step(action)
                    if done: 
                        self.battle_log.append("¡Nivel Superado!")
                        self.load_level(self.current_level_idx + 1)
                    if self.env.mode == "COMBAT": 
                        self.battle_log.append(f"¡{self.env.enemy_pokemon['name']} salvaje!")
                        
                elif self.env.mode == "COMBAT":
                    with torch.no_grad():
                        st = self.env._get_combat_state()
                        st_t = torch.FloatTensor(st).unsqueeze(0).to(self.tactician.device)
                        action = self.tactician.policy_net(st_t).argmax().item()
                    _, _, _, _, info = self.env.step(action + 4)
                    if "log" in info: self.battle_log.append(info["log"])
                    
                    if self.env.mode == "MAP":
                        current_name = self.env.my_pokemon['name']
                        for p in self.my_full_team:
                            if p['name'] == current_name:
                                p['level'] = self.env.my_pokemon['level']
                                p['exp'] = self.env.my_pokemon['exp']
                                p['stats'] = BattleEngine.get_stats_at_level(p, p['level'])
        pygame.quit()

if __name__ == "__main__":
    game = GameVisualizer(episode_to_load=2000) 
    game.run()