import pygame
import os
import torch
import sys
from src.env.pokemon_env import PokemonSimEnv
from src.agents.explorer import ExplorerAgent
from src.agents.tactician import TacticianAgent
from src.agents.strategist import Strategist
from src.game_manager import GameManager  # <--- IMPORTAMOS EL NUEVO MANAGER

# CONFIG
CELL_SIZE = 50
GRID_W, GRID_H = 10, 10
SCREEN_W = GRID_W * CELL_SIZE
SCREEN_H = GRID_H * CELL_SIZE + 250
FPS = 60
STEP_DELAY = 15

# COLORES
C_BG = (20, 20, 20)
C_WALL = (100, 100, 100)
C_GRASS = (50, 150, 50)
C_PATH = (200, 200, 180)
C_PLAYER = (0, 255, 255) 
C_TEXT = (255, 255, 255)
C_BLOCKED = (0, 0, 0)
C_VISITED = (50, 50, 50)
C_PANEL = (30, 30, 40)
C_GOAL = (255, 215, 0)

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
        if pid in self.cache: return self.cache[pid]
        try:
            path = f"data/sprites/{pid}.png"
            if os.path.exists(path):
                img = pygame.image.load(path)
                img = pygame.transform.scale(img, (180, 180))
                self.cache[pid] = img
                return img
        except: pass
        return self.default

class GameRenderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Pokémon AI: Modular")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16, bold=True)
        self.sprites = SpriteManager()
        
        # INICIALIZAR SISTEMA
        self.env = PokemonSimEnv(verbose=False)
        self.explorer = ExplorerAgent((9, 10, 10), 4)
        self.tactician = TacticianAgent(10, 5)
        self.strategist = Strategist(self.env.pokedex)
        
        # Cargar Pesos
        try:
            # Obtenemos la ruta absoluta de la carpeta donde está este script (visual_play.py)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Construimos la ruta a la carpeta checkpoints
            ckpt_path = os.path.join(base_dir, "checkpoints")

            # Cargamos los archivos
            self.explorer.policy_net.load_state_dict(torch.load(os.path.join(ckpt_path, "explorer_ep2000.pth")))
            self.tactician.policy_net.load_state_dict(torch.load(os.path.join(ckpt_path, "tactician_ep2000.pth")))
            
            print(f"✅ ¡CEREBROS CARGADOS! Leyendo de: {ckpt_path}")
        except Exception as e: 
            print(f"⚠ ERROR CARGANDO PESOS: {e}")
            print("⚠ Usando IA aleatoria (Esto explica por qué spamean Leer/Malicioso)")
        # MANAGER
        self.manager = GameManager(self.env, self.strategist, self.tactician, self.explorer)
        self.manager.init_game()

    def run(self):
        running = True
        last_step = 0
        
        while running:
            self.clock.tick(FPS)
            for e in pygame.event.get():
                if e.type == pygame.QUIT: running = False
            
            now = pygame.time.get_ticks()
            if now - last_step > STEP_DELAY:
                last_step = now
                if self.env.mode == "MAP":
                    self.manager.map_logic()
                else:
                    self.manager.combat_logic()
            
            self.draw()
            pygame.display.flip()
        
        pygame.quit()

    def draw(self):
        self.screen.fill(C_BG)
        
        # 1. MAPA / COMBATE
        if self.env.mode == "MAP":
            self.draw_grid()
            y, x = self.env.player_pos
            pygame.draw.circle(self.screen, C_PLAYER, (x*CELL_SIZE+25, y*CELL_SIZE+25), 18)
        else:
            self.draw_grid() # Fondo
            s = pygame.Surface((SCREEN_W, 500)); s.set_alpha(200); s.fill((0,0,0))
            self.screen.blit(s, (0,0))
            self.draw_battle()
            
        # 2. UI
        self.draw_ui()

    def draw_grid(self):
        for y in range(GRID_H):
            for x in range(GRID_W):
                rect = (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                val = self.env.grid[y][x]
                col = C_PATH
                if val == 1: col = C_WALL
                elif val == 2: col = C_GRASS
                elif val == 9: col = C_GOAL
                if (y, x) in self.manager.blocked_cells: col = C_BLOCKED
                
                # Resaltar Farming
                if self.manager.farming_mode and val == 2:
                    pygame.draw.rect(self.screen, (255, 215, 0), rect, 3)
                
                pygame.draw.rect(self.screen, col, rect)
                
                vis = self.manager.visit_counts.get((y,x), 0)
                if vis > 0 and val != 1:
                    s = pygame.Surface((CELL_SIZE, CELL_SIZE))
                    s.set_alpha(min(150, vis*20))
                    s.fill(C_VISITED)
                    self.screen.blit(s, (x*50, y*50))
                pygame.draw.rect(self.screen, (0,0,0), rect, 1)

    def draw_battle(self):
        # Sprites
        my_img = self.sprites.get_sprite(self.env.my_pokemon)
        en_img = self.sprites.get_sprite(self.env.enemy_pokemon)
        self.screen.blit(en_img, (320, 50))
        self.screen.blit(my_img, (50, 250))
        
        # Barras
        self.draw_hp(300, 20, self.env.enemy_pokemon, self.env.enemy_hp, self.env.max_hp_enemy)
        self.draw_hp(50, 220, self.env.my_pokemon, self.env.my_hp, self.env.max_hp_my)

    def draw_hp(self, x, y, p, curr, max_hp):
        pygame.draw.rect(self.screen, C_PANEL, (x, y, 180, 55), border_radius=5)
        txt = self.font.render(f"{p['name']} Lv{p['level']}", True, C_TEXT)
        self.screen.blit(txt, (x+10, y+5))
        
        pct = max(0, curr/max_hp)
        col = (0, 255, 0) if pct > 0.5 else (255, 0, 0)
        pygame.draw.rect(self.screen, (50,50,50), (x+10, y+30, 160, 10))
        pygame.draw.rect(self.screen, col, (x+10, y+30, 160*pct, 10))

    def draw_ui(self):
        rect = pygame.Rect(0, 500, SCREEN_W, SCREEN_H-500)
        pygame.draw.rect(self.screen, C_PANEL, rect)
        pygame.draw.line(self.screen, (255,255,255), (0, 500), (SCREEN_W, 500), 2)
        
        # LOGS
        for i, l in enumerate(self.manager.logs):
            col = (200, 200, 200)
            if "Ganaste" in l or "NIVEL" in l: col = (100, 255, 100)
            elif "cayó" in l: col = (255, 100, 100)
            elif "aprender" in l: col = (100, 200, 255)
            self.screen.blit(self.font.render(l, True, col), (20, 510+i*18))
            
        # EQUIPO
        for i, p in enumerate(self.manager.my_team):
            if i > 5: break
            hp = p['stats']['hp']
            col = (0, 255, 0) if hp > 0 else (100, 100, 100)
            prefix = "▶ " if p == self.env.my_pokemon else ""
            if p == self.manager.get_weakest() and self.manager.farming_mode: prefix = "★ "
            
            self.screen.blit(self.font.render(f"{prefix}{p['name'][:8]} {hp}", True, col), (350, 510+i*20))

if __name__ == "__main__":
    game = GameRenderer()
    game.run()