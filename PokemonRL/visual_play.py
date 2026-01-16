import pygame
import torch
import numpy as np
import sys
import os
import requests
import io
import time
from src.env.pokemon_env import PokemonSimEnv
from src.agents.explorer import ExplorerAgent
from src.agents.tactician import TacticianAgent
from src.agents.strategist import Strategist
from src.env.battle_engine import BattleEngine
from src.env.maps import ALL_MAPS

CELL_SIZE = 50
GRID_W, GRID_H = 10, 10
SCREEN_W = GRID_W * CELL_SIZE
SCREEN_H = GRID_H * CELL_SIZE + 200 
FPS = 60
STEP_DELAY = 150
BOSS_DELAY = 500

C_BG = (20, 20, 20)
C_WALL = (100, 100, 100)
C_GRASS = (50, 150, 50)
C_PATH = (200, 200, 180)
C_PLAYER = (255, 50, 50)
C_GOAL = (255, 215, 0)
C_TEXT = (255, 255, 255)

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
    def __init__(self, episode_to_load=3000):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Pokémon AI: RPG Adventure")
        self.clock = pygame.time.Clock()
        self.font_log = pygame.font.SysFont("Consolas", 14)
        self.font_ui = pygame.font.SysFont("Arial", 20, bold=True)
        self.big_font = pygame.font.SysFont("Arial", 40, bold=True)
        
        # --- ACTIVAR LOGS AQUÍ ---
        self.env = PokemonSimEnv(verbose=True) # <--- verbose=True para ver todo en consola
        self.sprites = SpriteManager()
        
        self.explorer = ExplorerAgent(obs_shape=(3, 10, 10), n_actions=4)
        self.tactician = TacticianAgent(input_dim=10, n_actions=5)
        self.strategist = Strategist(self.env.pokedex)
        
        # Cargar Checkpoints (Ruta Absoluta Blindada)
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
        self.setup_rpg_teams()
        self.load_level(0)

    def setup_rpg_teams(self):
        all_ids = list(self.env.pokedex.keys())
        party_ids = np.random.choice(all_ids, 6, replace=False) if len(all_ids)>6 else all_ids
        self.strategist.set_party(party_ids)
        for pid in party_ids:
            if pid in self.env.pokedex:
                p = self.env.pokedex[pid].copy()
                p['level'] = 5
                p['exp'] = 0
                p['stats'] = BattleEngine.get_stats_at_level(p, 5)
                self.my_full_team.append(p)
        self.env.my_pokemon = self.my_full_team[0].copy()
        self.env.my_hp = self.env.my_pokemon['stats']['hp']
        self.env.max_hp_my = self.env.my_hp

        for pid in GYM_LEADER_TEAM_IDS:
            if pid in self.env.pokedex:
                p = self.env.pokedex[pid].copy()
                p['level'] = 50
                p['stats'] = BattleEngine.get_stats_at_level(p, 50)
                self.enemy_gym_team.append(p)

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
        best = self.strategist.build_team(self.env.enemy_pokemon['types'][0])
        real_best = next((p for p in self.my_full_team if p['name'] == best['name']), self.my_full_team[0])
        self.env.my_pokemon = real_best.copy()
        self.env.max_hp_my = real_best['stats']['hp']
        self.env.my_hp = self.env.max_hp_my

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
                pygame.draw.rect(self.screen, color, rect)
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

    def run(self):
        print("📺 VENTANA GRÁFICA INICIADA...")
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
                if self.boss_mode:
                    if self.env.my_hp <= 0:
                        self.battle_log.append(f"{self.env.my_pokemon['name']} cayó!")
                        print(f"❌ {self.env.my_pokemon['name']} ha sido debilitado.")
                        for p in self.my_full_team:
                            if p['name'] == self.env.my_pokemon['name']: p['stats']['hp'] = 0
                        candidates = [p for p in self.my_full_team if p['stats']['hp'] > 0]
                        if not candidates:
                            self.battle_log.append("¡TE HAN DERROTADO!")
                            print("☠ GAME OVER - Sin Pokémon restantes.")
                            pygame.display.flip(); pygame.time.delay(3000); running = False
                            continue
                        self.strategist.current_party = {str(p['id']): p for p in self.my_full_team}
                        next_mon = self.strategist.build_team(self.env.enemy_pokemon['types'][0])
                        if next_mon['stats']['hp'] <= 0: next_mon = candidates[0]
                        self.env.my_pokemon = next_mon.copy()
                        self.env.max_hp_my = self.env.my_pokemon['stats']['hp']
                        self.env.my_hp = self.env.max_hp_my
                        continue
                    if self.env.enemy_hp <= 0:
                        self.battle_log.append(f"Rival {self.env.enemy_pokemon['name']} cayó!")
                        print(f"✅ ¡Rival {self.env.enemy_pokemon['name']} derrotado!")
                        self.boss_pokemon_idx += 1
                        if self.boss_pokemon_idx >= len(self.enemy_gym_team):
                            self.battle_log.append("¡¡¡ERES EL CAMPEÓN!!!")
                            print("🏆 ¡¡¡CAMPEÓN DE LA LIGA POKÉMON!!! 🏆")
                            self.draw_game(); pygame.display.flip(); pygame.time.delay(5000); running = False
                            continue
                        self.env.enemy_pokemon = self.enemy_gym_team[self.boss_pokemon_idx].copy()
                        self.env.max_hp_enemy = self.env.enemy_pokemon['stats']['hp']
                        self.env.enemy_hp = self.env.max_hp_enemy
                        print(f"⚡ Rival saca a {self.env.enemy_pokemon['name']}")
                        continue
                    with torch.no_grad():
                        st = self.env._get_combat_state()
                        st_t = torch.FloatTensor(st).unsqueeze(0).to(self.tactician.device)
                        action = self.tactician.policy_net(st_t).argmax().item()
                    _, _, _, _, info = self.env.step(action + 4)
                    if "log" in info: self.battle_log.append(info["log"])
                    continue

                if self.env.mode == "MAP":
                    with torch.no_grad():
                        st_t = torch.FloatTensor(self.env.map_state).unsqueeze(0).to(self.explorer.device)
                        q_vals = self.explorer.policy_net(st_t)
                        action = q_vals.argmax().item()
                        
                        # LOG DE MOVIMIENTO
                        move_names = ["ARRIBA", "ABAJO", "IZQUIERDA", "DERECHA"]
                        print(f"🚶 Explorador decide: {move_names[action]}") # Descomentar si quieres spam de logs
                        
                    _, _, done, _, info = self.env.step(action)
                    if done: 
                        self.battle_log.append("¡Nivel Superado!")
                        self.load_level(self.current_level_idx + 1)
                    if self.env.mode == "COMBAT": 
                        self.battle_log.append(f"¡{self.env.enemy_pokemon['name']} salvaje!")
                        print(f"⚠ ¡COMBATE SALVAJE! vs {self.env.enemy_pokemon['name']}")
                        
                elif self.env.mode == "COMBAT":
                    with torch.no_grad():
                        st = self.env._get_combat_state()
                        st_t = torch.FloatTensor(st).unsqueeze(0).to(self.tactician.device)
                        action = self.tactician.policy_net(st_t).argmax().item()
                    _, _, _, _, info = self.env.step(action + 4)
                    if "log" in info: 
                        self.battle_log.append(info["log"])
                        print(f"⚔ {info['log']}")
                    
                    if self.env.mode == "MAP":
                        current_name = self.env.my_pokemon['name']
                        for p in self.my_full_team:
                            if p['name'] == current_name:
                                p['level'] = self.env.my_pokemon['level']
                                p['exp'] = self.env.my_pokemon['exp']
                                p['stats'] = BattleEngine.get_stats_at_level(p, p['level'])
        pygame.quit()

if __name__ == "__main__":
    game = GameVisualizer(episode_to_load=3000)
    game.run()