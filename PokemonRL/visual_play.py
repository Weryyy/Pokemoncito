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

# --- CONFIGURACIÓN ---
CELL_SIZE = 50
GRID_W, GRID_H = 10, 10
SCREEN_W = GRID_W * CELL_SIZE
SCREEN_H = GRID_H * CELL_SIZE + 200
FPS = 60
STEP_DELAY = 150

# Colores
C_BG = (20, 20, 20)
C_WALL = (100, 100, 100)
C_GRASS = (50, 150, 50)
C_PATH = (200, 200, 180)
C_PLAYER = (255, 50, 50)
C_GOAL = (255, 215, 0)

# --- DEFINICIÓN DE LOS 5 NIVELES ---
# 0: Camino, 1: Muro, 2: Hierba (Peligro), 9: Meta
MAPS = [
    # NIVEL 1: Pueblo Paleta (Sencillo)
    [
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 2, 2, 0, 0, 0, 0, 1, 0, 1],
        [1, 2, 2, 0, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 2, 2, 2, 1, 0, 0, 0, 9, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    # NIVEL 2: Ruta 1 (Más hierba)
    [
        [0, 0, 1, 2, 2, 2, 1, 1, 1, 1],
        [1, 0, 1, 2, 2, 2, 2, 2, 2, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 2, 1],
        [1, 1, 1, 2, 1, 1, 1, 0, 2, 1],
        [1, 2, 2, 2, 2, 2, 1, 0, 0, 1],
        [1, 2, 1, 1, 1, 0, 0, 0, 0, 1],
        [1, 2, 2, 2, 1, 0, 1, 1, 0, 1],
        [1, 1, 1, 2, 1, 0, 2, 2, 0, 1],
        [1, 9, 0, 0, 0, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    # NIVEL 3: Bosque Verde (Laberinto)
    [
        [0, 0, 1, 2, 2, 1, 2, 2, 1, 1],
        [1, 0, 1, 0, 0, 1, 0, 0, 2, 1],
        [1, 0, 0, 0, 1, 1, 1, 0, 1, 1],
        [1, 1, 1, 2, 2, 2, 2, 0, 0, 1],
        [1, 0, 0, 0, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 2, 2, 2, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 9, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    # NIVEL 4: Cueva Diglett (Estrecho)
    [
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 2, 2, 1, 0, 1, 0, 1],
        [1, 0, 1, 2, 2, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 9, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    # NIVEL 5: Calle Victoria (Final)
    [
        [0, 0, 0, 2, 2, 2, 0, 0, 0, 1],
        [1, 1, 0, 2, 2, 2, 0, 1, 1, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 9, 1, 1, 1, 0, 1],  # La meta está en el centro rodeada
        [1, 0, 1, 0, 0, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 2, 2, 2, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
]

# EQUIPO DEL LÍDER DE GIMNASIO (Gary/Blue) - IDs de la Pokedex
GYM_LEADER_TEAM_IDS = ["18", "65", "112", "59", "103", "6"]
# Pidgeot, Alakazam, Rhydon, Arcanine, Exeggutor, Charizard


class SpriteManager:
    def __init__(self):
        self.cache = {}
        self.sprite_dir = "data/sprites"
        os.makedirs(self.sprite_dir, exist_ok=True)
        self.default = pygame.Surface((150, 150))
        self.default.fill((50, 50, 50))

    def get_sprite(self, pokemon_data):
        if not pokemon_data:
            return self.default
        pid = str(pokemon_data['id'])
        url = pokemon_data.get('sprite')

        if pid in self.cache:
            return self.cache[pid]

        path = os.path.join(self.sprite_dir, f"{pid}.png")
        try:
            if os.path.exists(path):
                img = pygame.image.load(path)
            elif url:
                print(f"📥 Descargando sprite {pokemon_data['name']}...")
                r = requests.get(url)
                img = pygame.image.load(io.BytesIO(r.content))
                pygame.image.save(img, path)
            else:
                return self.default

            img = pygame.transform.scale(img, (180, 180))
            self.cache[pid] = img
            return img
        except:
            return self.default


class GameVisualizer:
    def __init__(self, episode_to_load=500):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Pokémon AI: Liga Pokémon")
        self.clock = pygame.time.Clock()
        self.font_log = pygame.font.SysFont("Consolas", 14)
        self.font_ui = pygame.font.SysFont("Arial", 20, bold=True)
        self.big_font = pygame.font.SysFont("Arial", 40, bold=True)

        self.env = PokemonSimEnv(verbose=True)
        self.sprites = SpriteManager()

        # Agentes
        self.explorer = ExplorerAgent(obs_shape=(3, 10, 10), n_actions=4)
        self.tactician = TacticianAgent(input_dim=10, n_actions=5)
        self.strategist = Strategist(self.env.pokedex)

        # Cargar Checkpoints
        try:
            self.explorer.policy_net.load_state_dict(torch.load(
                f"checkpoints/explorer_ep{episode_to_load}.pth"))
            self.tactician.policy_net.load_state_dict(torch.load(
                f"checkpoints/tactician_ep{episode_to_load}.pth"))
            print("✅ Cerebros cargados.")
        except:
            print("⚠ Usando cerebros aleatorios.")

        self.explorer.policy_net.eval()
        self.tactician.policy_net.eval()

        # Estado Juego
        self.current_level_idx = 0
        self.battle_log = ["¡Iniciando aventura!"]
        self.boss_mode = False

        # Preparar mi equipo de 6
        self.my_full_team = []  # Lista de diccionarios de pokemon
        self.prepare_my_team()

        # Preparar equipo rival (Jefe)
        self.enemy_gym_team = []
        for pid in GYM_LEADER_TEAM_IDS:
            if pid in self.env.pokedex:
                p = self.env.pokedex[pid].copy()
                p['stats']['hp'] = p['stats']['hp'] * 2 + 110  # Max HP Real
                self.enemy_gym_team.append(p)

        self.load_level(0)

    def prepare_my_team(self):
        # Elegimos 6 al azar para la aventura
        all_ids = list(self.env.pokedex.keys())
        party_ids = np.random.choice(all_ids, 6, replace=False)
        self.strategist.set_party(party_ids)

        # Convertimos IDs a objetos Pokemon completos
        for pid in party_ids:
            if pid in self.env.pokedex:
                p = self.env.pokedex[pid].copy()
                p['stats']['hp'] = p['stats']['hp'] * 2 + 110  # Max HP
                self.my_full_team.append(p)

        # Empezamos con el primero
        self.env.my_pokemon = self.my_full_team[0].copy()
        self.env.my_hp = self.env.my_pokemon['stats']['hp']
        self.env.max_hp_my = self.env.my_hp

    def load_level(self, idx):
        if idx >= len(MAPS):
            self.start_boss_battle()
            return

        self.current_level_idx = idx
        self.env.grid = np.array(MAPS[idx])
        self.env.reset()  # Resetea posición
        # Re-aplicar grid tras reset
        self.env.grid = np.array(MAPS[idx])
        self.battle_log.append(f"--- NIVEL {idx + 1} ---")

    def start_boss_battle(self):
        self.boss_mode = True
        self.env.mode = "COMBAT"
        self.battle_log.append("!!! ALERTA: LÍDER DE GIMNASIO !!!")

        # Sacar primer pokemon rival
        self.boss_pokemon_idx = 0
        self.env.enemy_pokemon = self.enemy_gym_team[0].copy()
        self.env.enemy_hp = self.env.enemy_pokemon['stats']['hp']
        self.env.max_hp_enemy = self.env.enemy_hp

        # Sacar mi mejor pokemon contra ese
        best = self.strategist.build_team(self.env.enemy_pokemon['types'][0])
        self.env.my_pokemon = best.copy()
        self.env.max_hp_my = best['stats']['hp'] * 2 + 110
        self.env.my_hp = self.env.max_hp_my

    def draw_game(self):
        self.screen.fill(C_BG)

        # MODO JEFE (Solo combate)
        if self.boss_mode:
            self.draw_battle_scene()
            # UI Especial
            txt = self.big_font.render("COMBATE DE GIMNASIO", True, C_GOAL)
            self.screen.blit(txt, (SCREEN_W//2 - 150, 20))

            # Pokeballs restantes
            my_alive = sum(
                1 for p in self.my_full_team if p['stats']['hp'] > 0)
            en_alive = len(self.enemy_gym_team) - self.boss_pokemon_idx
            self.screen.blit(self.font_ui.render(
                f"Tu Equipo: {my_alive}/6", True, C_TEXT), (50, 100))
            self.screen.blit(self.font_ui.render(
                f"Rival: {en_alive}/6", True, C_TEXT), (SCREEN_W - 200, 100))

            self.draw_logs()
            pygame.display.flip()
            return

        # MODO MAPA NORMAL
        if self.env.mode == "MAP":
            self.draw_grid()
            # Jugador
            py, px = self.env.player_pos
            pygame.draw.circle(self.screen, C_PLAYER,
                               (px*CELL_SIZE+25, py*CELL_SIZE+25), 15)

        elif self.env.mode == "COMBAT":
            self.draw_grid()  # Fondo
            s = pygame.Surface((SCREEN_W, SCREEN_H))
            s.set_alpha(200)
            s.fill((0, 0, 0))
            self.screen.blit(s, (0, 0))
            self.draw_battle_scene()

        self.draw_logs()
        pygame.display.flip()

    def draw_grid(self):
        for y in range(GRID_H):
            for x in range(GRID_W):
                rect = (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                val = self.env.grid[y][x]
                color = C_PATH
                if val == 1:
                    color = C_WALL
                elif val == 2:
                    color = C_GRASS
                elif val == 9:
                    color = C_GOAL
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)

    def draw_battle_scene(self):
        # Sprites
        my_img = self.sprites.get_sprite(self.env.my_pokemon)
        en_img = self.sprites.get_sprite(self.env.enemy_pokemon)

        self.screen.blit(en_img, (SCREEN_W - 220, 50))
        self.screen.blit(my_img, (50, 250))

        # Barras Vida
        self.draw_hp_bar(50, 230, self.env.my_hp,
                         self.env.max_hp_my, self.env.my_pokemon['name'])
        self.draw_hp_bar(SCREEN_W - 220, 30, self.env.enemy_hp,
                         self.env.max_hp_enemy, self.env.enemy_pokemon['name'])

    def draw_hp_bar(self, x, y, curr, max_hp, name):
        pct = max(0, curr / max_hp)
        col = (0, 255, 0) if pct > 0.5 else (255, 0, 0)
        pygame.draw.rect(self.screen, (50, 50, 50), (x, y, 200, 20))
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
        running = True
        while running:
            self.clock.tick(FPS)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False

            self.draw_game()

            # --- LÓGICA JEFE ---
            if self.boss_mode:
                pygame.time.delay(800)  # Lento para emoción

                # 1. Comprobar muertes
                if self.env.my_hp <= 0:
                    self.battle_log.append(
                        f"{self.env.my_pokemon['name']} debilitado!")
                    # Buscar otro vivo
                    alive = [p for p in self.my_full_team if p['stats']
                             ['hp'] > 0]  # NO TOCAR self.my_full_team
                    # Buscar en la lista de pokemon originales, no en la copia del env
                    # Pequeño hack: marcar como muerto en la lista original
                    for p in self.my_full_team:
                        if p['name'] == self.env.my_pokemon['name']:
                            p['stats']['hp'] = 0  # Marcar muerto

                    # Buscar siguiente
                    candidates = [
                        p for p in self.my_full_team if p['stats']['hp'] > 0]

                    if not candidates:
                        self.battle_log.append("¡PERDISTE LA LIGA!")
                        pygame.time.delay(3000)
                        running = False
                        continue

                    # Estratega elige al siguiente mejor
                    self.strategist.current_party = {
                        str(p['id']): p for p in self.my_full_team}
                    next_mon = self.strategist.build_team(
                        self.env.enemy_pokemon['types'][0])

                    # Si devuelve uno muerto (fallback), cogemos el primero vivo a mano
                    if next_mon['stats']['hp'] <= 0:
                        next_mon = candidates[0]

                    self.env.my_pokemon = next_mon.copy()
                    # Ya viene con vida max/actual sincronizada
                    self.env.max_hp_my = self.env.my_pokemon['stats']['hp']
                    self.env.my_hp = self.env.max_hp_my
                    self.battle_log.append(
                        f"¡Adelante {self.env.my_pokemon['name']}!")
                    continue

                if self.env.enemy_hp <= 0:
                    self.battle_log.append(
                        f"Rival {self.env.enemy_pokemon['name']} debilitado!")
                    self.boss_pokemon_idx += 1
                    if self.boss_pokemon_idx >= len(self.enemy_gym_team):
                        self.battle_log.append("¡¡¡CAMPEÓN DE LA LIGA!!!")
                        self.draw_game()
                        pygame.time.delay(5000)
                        running = False
                        continue

                    self.env.enemy_pokemon = self.enemy_gym_team[self.boss_pokemon_idx].copy(
                    )
                    self.env.enemy_hp = self.env.enemy_pokemon['stats']['hp']
                    self.env.max_hp_enemy = self.env.enemy_hp
                    self.battle_log.append(
                        f"Rival saca a {self.env.enemy_pokemon['name']}")
                    continue

                # 2. Turno Combate
                with torch.no_grad():
                    st = self.env._get_combat_state()
                    st_t = torch.FloatTensor(st).unsqueeze(
                        0).to(self.tactician.device)
                    action = self.tactician.policy_net(st_t).argmax().item()

                # Ejecutar paso combate manual (sin env.step para control total)
                res, rew, done, _, info = self.env.step(action + 4)
                if "log" in info:
                    self.battle_log.append(info["log"])
                continue

            # --- LÓGICA MAPA NORMAL ---
            if self.env.mode == "MAP":
                pygame.time.delay(STEP_DELAY)
                with torch.no_grad():
                    st_t = torch.FloatTensor(self.env.map_state).unsqueeze(
                        0).to(self.explorer.device)
                    action = self.explorer.policy_net(st_t).argmax().item()

                _, _, done, _, info = self.env.step(action)
                if done:  # Meta alcanzada
                    self.battle_log.append("¡Nivel Completado!")
                    self.load_level(self.current_level_idx + 1)

                if self.env.mode == "COMBAT":
                    pygame.time.delay(500)
                    self.battle_log.append(
                        f"¡{self.env.enemy_pokemon['name']} salvaje!")

            elif self.env.mode == "COMBAT":
                pygame.time.delay(STEP_DELAY * 3)
                with torch.no_grad():
                    st = self.env._get_combat_state()
                    st_t = torch.FloatTensor(st).unsqueeze(
                        0).to(self.tactician.device)
                    action = self.tactician.policy_net(st_t).argmax().item()

                _, _, done, _, info = self.env.step(action + 4)
                if "log" in info:
                    self.battle_log.append(info["log"])

                if self.env.mode == "MAP":
                    self.battle_log.append("Victoria. +Exp")

        pygame.quit()


if __name__ == "__main__":
    # ¡PON AQUÍ TU EPISODIO GUARDADO!
    game = GameVisualizer(episode_to_load=500)
    game.run()
