import numpy as np
import time
from collections import deque
from src.env.battle_engine import BattleEngine
from src.env.maps import ALL_MAPS
import sys
import torch

# Configuración de Niveles
LEVEL_GATES = {0: 10, 1: 20, 2: 30, 3: 40, 4: 55}
WILD_ENCOUNTERS = {
    0: ['19', '16', '13', '10', '29', '32'],
    1: ['41', '27', '50', '74', '46', '23'],
    2: ['60', '72', '129', '118', '54'],
    3: ['92', '96', '109', '88', '114'],
    4: ['147', '148', '142', '115', '113']
}
GYM_LEADER_TEAM_IDS = ["18", "65", "112", "59", "103", "6"]

class GameManager:
    def __init__(self, env, strategist, tactician, explorer):
        self.env = env
        self.strategist = strategist
        self.tactician = tactician
        self.explorer = explorer
        
        # Estado del RPG
        self.my_team = []
        self.gym_team = []
        self.current_level_idx = 0
        self.potions = 10
        self.farming_mode = True
        self.boss_mode = False
        self.boss_idx = 0
        
        # Logs y Memoria
        self.logs = deque(maxlen=8)
        self.logs.append("Sistema Modular Iniciado")
        self.wall_hits = {}
        self.blocked_cells = set()
        self.visit_counts = {}
        self.action_history = deque(maxlen=15)

    def log(self, msg):
        self.logs.append(msg)
        print(msg)

    def init_game(self):
        # 1. Crear Equipo
        all_ids = list(self.env.pokedex.keys())
        party_ids = np.random.choice(all_ids, 6, replace=False) if len(all_ids)>6 else all_ids
        self.strategist.set_party(party_ids)
        
        self.my_team = []
        for pid in party_ids:
            if str(pid) in self.env.pokedex:
                self.my_team.append(self.strategist.prepare_pokemon(pid, level=5))
        
        # 2. Crear Boss
        self.gym_team = []
        for pid in GYM_LEADER_TEAM_IDS:
            if str(pid) in self.env.pokedex:
                self.gym_team.append(self.strategist.prepare_pokemon(pid, level=60))
        
        self.load_level(0)

    def load_level(self, idx):
        if idx >= len(ALL_MAPS):
            self.start_boss_battle()
            return
            
        self.current_level_idx = idx
        self.env.current_map_idx = idx
        self.env.grid = np.array(ALL_MAPS[idx])
        self.env.reset()
        
        self.heal_team()
        self.wall_hits.clear(); self.blocked_cells.clear(); self.visit_counts.clear()
        self.farming_mode = True
        self.log(f"--- NIVEL {idx + 1} ---")

    def heal_team(self):
        self.potions = 10
        for i, p in enumerate(self.my_team):
            # Regenerar manteniendo XP
            new_p = self.strategist.prepare_pokemon(p['id'], p['level'])
            new_p['exp'] = p['exp']
            self.my_team[i] = new_p
        self.update_active_pokemon()

    def update_active_pokemon(self):
        # Busca el primero vivo para ponerlo de activo
        for p in self.my_team:
            if p['stats']['hp'] > 0:
                self.env.my_pokemon = p
                self.env.my_hp = p['stats']['hp']
                self.env.max_hp_my = p['stats']['hp']
                return True
        return False

    def get_weakest(self):
        """Devuelve el miembro más débil (vivo o muerto)"""
        return min(self.my_team, key=lambda x: x['level'])

    def get_strongest_alive(self):
        alive = [p for p in self.my_team if p['stats']['hp'] > 0]
        if not alive: return None
        return max(alive, key=lambda x: x['level'])

    def switch_pokemon(self, new_pokemon):
        # Guardar estado del actual antes de cambiar
        for i, p in enumerate(self.my_team):
            if p['name'] == self.env.my_pokemon['name']:
                self.my_team[i]['stats']['hp'] = self.env.my_hp
                break
        
        self.env.my_pokemon = new_pokemon
        self.env.my_hp = new_pokemon['stats']['hp']
        self.env.max_hp_my = new_pokemon['stats']['hp']
        self.log(f"🔄 Cambio: Entra {new_pokemon['name']}")

    # --- CEREBRO DE COMBATE ---
    def combat_logic(self):
        if self.env.enemy_hp <= 0:
            self.handle_victory()
            return

        if self.env.my_hp <= 0:
            self.handle_faint()
            return

        # 1. Lógica de Entrenamiento (Solo en Farming, No Boss)
        if self.farming_mode and not self.boss_mode:
            weakest = self.get_weakest()
            strongest = self.get_strongest_alive()
            active = self.env.my_pokemon
            
            # A) Si el débil está vivo y NO está peleando -> SACARLO
            if weakest['stats']['hp'] > 0 and active['name'] != weakest['name']:
                self.log(f"🎓 ¡Hora de aprender, {weakest['name']}!")
                self.switch_pokemon(weakest)
                self.enemy_turn() # Pierde turno
                return

            # B) Si el débil está peleando y peligra (<50%) -> SACAR AL FUERTE
            current_pct = self.env.my_hp / self.env.max_hp_my
            if active['name'] == weakest['name'] and current_pct < 0.5:
                if strongest and strongest['name'] != active['name']:
                    self.log(f"🛡️ ¡Ayuda {strongest['name']}!")
                    self.switch_pokemon(strongest)
                    self.enemy_turn()
                    return

        # 2. Pociones
        if self.potions > 0:
            for p in self.my_team:
                # Curar solo si no es el activo (para no perder tempo) o si es crítico
                limit = p['stats']['hp'] < (p['stats']['hp'] * 0.4) # Esto está mal, arreglamos
                # Necesitamos Max HP real
                real_max = BattleEngine.get_stats_at_level(p, p['level'])['hp']
                if p['stats']['hp'] > 0 and p['stats']['hp'] < real_max * 0.4:
                    heal = int(real_max * 0.5)
                    p['stats']['hp'] = min(real_max, p['stats']['hp'] + heal)
                    self.potions -= 1
                    self.log(f"💊 Poción a {p['name']}")
                    if p == self.env.my_pokemon: self.env.my_hp = p['stats']['hp']
                    self.enemy_turn()
                    return

        # 3. Ataque (Tactician)
        import torch
        with torch.no_grad():
            st = self.env._get_combat_state()
            st_t = torch.FloatTensor(st).unsqueeze(0).to(self.tactician.device)
            action = self.tactician.policy_net(st_t).argmax().item()
        
        self.player_attack(action)

    def player_attack(self, action_idx):
        if action_idx > 3: action_idx = 0
        moves = self.env.my_pokemon.get('active_moves', ['tackle'])
        move = moves[action_idx] if action_idx < len(moves) else moves[0]
        
        dmg, msg = BattleEngine.calculate_damage(self.env.my_pokemon, self.env.enemy_pokemon, move)
        self.env.enemy_hp -= dmg
        self.log(f"Tú: {move} -{dmg} {msg}")
        
        if self.env.enemy_hp > 0:
            self.enemy_turn()

    def enemy_turn(self):
        moves = self.env.enemy_pokemon.get('active_moves', ['tackle'])
        move = np.random.choice(moves)
        dmg, msg = BattleEngine.calculate_damage(self.env.enemy_pokemon, self.env.my_pokemon, move)
        self.env.my_hp -= dmg
        self.log(f"Rival: {move} -{dmg}")

    def handle_victory(self):
        # Repartir XP al más débil siempre
        receiver = self.get_weakest()
        xp, leveled, _ = BattleEngine.gain_experience(receiver, self.env.enemy_pokemon['level'])
        
        self.log(f"Ganaste! {receiver['name']} +{xp} XP")
        
        if leveled:
            new_lvl = receiver['level']
            self.log(f"🎉 ¡{receiver['name']} SUBE A NIVEL {new_lvl}!")
            # Actualizar datos del pokemon
            new_p = self.strategist.prepare_pokemon(receiver['id'], new_lvl)
            new_p['exp'] = receiver['exp']
            
            # Actualizar en la lista
            for i, p in enumerate(self.my_team):
                if p['name'] == receiver['name']:
                    self.my_team[i] = new_p
            
            # Si era el activo, actualizar entorno
            if self.env.my_pokemon['name'] == receiver['name']:
                self.env.my_pokemon = new_p
                self.env.my_hp = new_p['stats']['hp']
                self.env.max_hp_my = new_p['stats']['hp']
                
        if self.boss_mode:
            self.boss_idx += 1
            if self.boss_idx >= len(self.gym_team):
                self.log("🏆 ¡CAMPEÓN DE LA LIGA!")
                self.log("Has derrotado a todo el equipo.")
                # Aquí podrías reiniciar el juego o cerrarlo
                time.sleep(5)
                sys.exit()
            else:
                # Siguiente Pokemon del Boss
                next_mon = self.gym_team[self.boss_idx]
                self.env.enemy_pokemon = next_mon
                self.env.max_hp_enemy = next_mon['stats']['hp']
                self.env.enemy_hp = self.env.max_hp_enemy
                self.log(f"⚔ El Líder cambia a {next_mon['name']}!")
        else:
            self.env.mode = "MAP"

    def handle_faint(self):
        name = self.env.my_pokemon['name']
        self.log(f"❌ {name} cayó!")
        
        # Marcar muerto
        for i, p in enumerate(self.my_team):
            if p['name'] == name: self.my_team[i]['stats']['hp'] = 0
            
        if self.update_active_pokemon():
            self.log(f"¡Ve {self.env.my_pokemon['name']}!")
        else:
            self.log("💀 WIPEOUT. Reiniciando...")
            self.heal_team()
            self.env.reset()
            self.env.grid = np.array(ALL_MAPS[self.current_level_idx])
            self.farming_mode = True

    # --- CEREBRO DE MAPA ---
    def map_logic(self):
        # 1. Chequeo de Farmeo
        req = LEVEL_GATES.get(self.current_level_idx, 99)
        # Modo Farmeo activo si CUALQUIERA del equipo es menor al requisito
        self.farming_mode = any(p['level'] < req for p in self.my_team)
        
        # 2. Explorer decide (Torch)
        
        with torch.no_grad():
            st_t = torch.FloatTensor(self.env._get_stacked_state()).unsqueeze(0).to(self.explorer.device)
            q_vals = self.explorer.policy_net(st_t).cpu().numpy()[0].copy()
            
            # MÁSCARAS LÓGICAS
            y, x = self.env.player_pos
            
            for a in range(4):
                # Calcular coord destino
                ty, tx = y, x
                if a==0: ty-=1
                elif a==1: ty+=1
                elif a==2: tx-=1
                elif a==3: tx+=1
                
                coord = (ty, tx)
                
                # Paredes / Bloqueos
                if coord in self.blocked_cells: 
                    q_vals[a] = -99999
                    continue
                
                # Zanahoria (Hierba)
                if 0 <= tx < 10 and 0 <= ty < 10:
                    is_grass = (self.env.grid[ty][tx] == 2)
                    if self.farming_mode and is_grass:
                        q_vals[a] += 5000
                    if not self.farming_mode and is_grass:
                        q_vals[a] -= 5000 # Repelente
                        
                # Aburrimiento
                vis = self.visit_counts.get(coord, 0)
                if vis > 0: q_vals[a] -= (vis**2) * 5
            
            action = np.argmax(q_vals)
            
            # Anti-Bucle
            self.action_history.append(action)
            if len(self.action_history) >= 10 and len(set(self.action_history)) == 1:
                action = np.random.randint(0,4)
                self.action_history.clear()
                
            # Ejecutar
            self.process_map_action(action)

    def trigger_final_boss(self):
            """Prepara y lanza la batalla final de gimnasio"""
            self.boss_mode = True
            self.env.mode = "COMBAT"
            self.farming_mode = False # Desactivar farmeo
            
            self.log("⚡ -------------------------------- ⚡")
            self.log("⚠ ¡ALERTA! LÍDER DE GIMNASIO DESAFÍA ⚠")
            self.log("⚡ -------------------------------- ⚡")

            # 1. Curar al jugador para que sea justo
            self.heal_team()
            
            # 2. Generar el equipo del Boss (Nivel 65 Hardcore)
            self.gym_team = []
            for pid in GYM_LEADER_TEAM_IDS: # <----------cambio aqui por si acaso
                # Usamos el strategist para crear el pokemon
                boss_mon = self.strategist.prepare_pokemon(pid, level=65)
                if boss_mon:
                    # Pequeño buff de stats para hacerlo 'Boss'
                    boss_mon['stats']['hp'] = int(boss_mon['stats']['hp'] * 1.2)
                    boss_mon['max_hp'] = boss_mon['stats']['hp'] # Hack visual
                    self.gym_team.append(boss_mon)

            # 3. Configurar el primer enfrentamiento
            self.boss_idx = 0
            first_boss = self.gym_team[0]
            
            self.env.enemy_pokemon = first_boss
            self.env.max_hp_enemy = first_boss['stats']['hp']
            self.env.enemy_hp = first_boss['stats']['hp']
            
            self.log(f"🔥 El Líder envía a {first_boss['name']} (Nv. {first_boss['level']})")
    def process_map_action(self, action):
        # Pre-check de colisión para memoria
        y, x = self.env.player_pos
        ty, tx = y, x
        if action==0: ty-=1
        elif action==1: ty+=1
        elif action==2: tx-=1
        elif action==3: tx+=1

        # logica de muros
        if 0 <= tx < 10 and 0 <= ty < 10 and self.env.grid[ty][tx] == 1:
            self.wall_hits[(ty, tx)] = self.wall_hits.get((ty, tx), 0) + 1
            if self.wall_hits[(ty, tx)] >= 2:
                self.blocked_cells.add((ty, tx))
        
        self.visit_counts[(y, x)] = self.visit_counts.get((y, x), 0) + 1

        # Ejecutar paso en el entorno
        _, _, done, _, _ = self.env.step(action)
        
        # --- AQUÍ ESTÁ EL CAMBIO IMPORTANTE ---
        if done: # Se pisó la meta (Casilla valor 9)
            if not self.farming_mode:
                # Comprobamos si es el último mapa
                if self.current_level_idx + 1 >= len(ALL_MAPS):
                    # SI YA NO HAY MAPAS -> JEFE FINAL
                    self.trigger_final_boss()
                else:
                    # SI HAY MAPA -> SIGUIENTE NIVEL
                    self.log("¡NIVEL SUPERADO!")
                    self.load_level(self.current_level_idx + 1)
            else:
                self.log("⛔ ¡Nivel bajo! Volviendo al inicio...")
                self.env.reset()
                self.env.grid = np.array(ALL_MAPS[self.current_level_idx])
                self.wall_hits.clear(); self.blocked_cells.clear()
        if self.env.mode == "COMBAT" and not self.boss_mode:
                    if not self.farming_mode:
                        self.env.mode = "MAP" # Repelente si estamos buscando la salida
                        self.log("Repelente usado.")
                    else:
                        wild = self.generate_wild_pokemon()
                        self.log(f"¡{wild['name']} salvaje!")
                        
    def generate_wild_pokemon(self):
        base = 3 + (self.current_level_idx * 10)
        lvl = np.random.randint(base, base+4)
        pool = WILD_ENCOUNTERS.get(self.current_level_idx, ['1'])
        pid = np.random.choice(pool)
        
        wild = self.strategist.prepare_pokemon(pid, lvl)
        if not wild: wild = self.strategist.prepare_pokemon('1', lvl)
        
        self.env.enemy_pokemon = wild
        self.env.max_hp_enemy = wild['stats']['hp']
        self.env.enemy_hp = wild['stats']['hp']
        return wild

    def start_boss_battle(self):
        self.boss_mode = True
        self.env.mode = "COMBAT"
        self.log("🔥 ¡JEFE FINAL! 🔥")
        self.heal_team()
        self.boss_idx = 0
        self.env.enemy_pokemon = self.gym_team[0]
        self.env.max_hp_enemy = self.env.enemy_pokemon['stats']['hp']
        self.env.enemy_hp = self.env.max_hp_enemy