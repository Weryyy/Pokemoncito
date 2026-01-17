import numpy as np
import json
import os
import random
from src.env.battle_engine import BattleEngine, EFFECTS_DB  # <--- Importamos la Lista Blanca

class Strategist:
    def __init__(self, pokedex):
        self.pokedex = pokedex
        self.current_party = {} 
        self.moves_db = {}
        self.load_moves_db()

    def load_moves_db(self):
        """
        Carga la base de datos detallada de movimientos (moves.json).
        Busca en varias rutas para asegurar que lo encuentra.
        """
        posibles_rutas = [
            'moves.json',
            'data/moves.json',
            '../data/moves.json',
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'moves.json')
        ]

        for path in posibles_rutas:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        self.moves_db = json.load(f)
                    print(f"🧠 ESTRATEGA: Base de datos de movimientos cargada desde {path} ({len(self.moves_db)} ataques).")
                    return
                except Exception as e:
                    print(f"⚠ Error leyendo {path}: {e}")

        print("⚠ ESTRATEGA: No se encontró 'moves.json'. El sistema usará movimientos básicos.")
        self.moves_db = {}

    def set_party(self, party_ids):
        """Define el equipo actual."""
        self.current_party = {
            str(pid): self.pokedex[str(pid)]
            for pid in party_ids
            if str(pid) in self.pokedex
        }
        names = [p['name'] for p in self.current_party.values()]
        print(f"\n🎒 ESTRATEGA: Nuevo equipo asignado: {names}")

    def select_moves(self, pokemon, level):
        """
        Elige 4 movimientos aplicando la LISTA BLANCA y el CAPADO POR NIVEL.
        """
        possible_moves_names = pokemon.get('moves', [])
        valid_moves = []
        
        # --- 1. DEFINIR LÍMITE DE PODER POR NIVEL ---
        # Nivel 5 -> Max 55 (Placaje, Ascuas)
        # Nivel 50 -> Max 145 (Hiperrayo)
        max_power_allowed = 45 + (level * 2.0)

        for move_name in possible_moves_names:
            if move_name in self.moves_db:
                move_data = self.moves_db[move_name]
                power = move_data.get('power')
                
                if power is None: power = 0
                
                # --- 2. FILTRO DE LISTA BLANCA (WHITELIST) ---
                
                # A) Es un ataque de daño válido para el nivel
                is_valid_damage = (power > 0 and power <= max_power_allowed)
                
                # B) Es un ataque de estado soportado por nuestro motor
                is_supported_status = (power == 0 and move_name in EFFECTS_DB)
                
                # Solo pasa si cumple A o B
                if is_valid_damage or is_supported_status:
                    valid_moves.append(move_name)

        # Fallback si no hay nada válido
        if not valid_moves:
            return ['tackle', 'struggle']

        # --- 3. SELECCIÓN ESTRATÉGICA ---
        # Priorizar STAB (Same Type Attack Bonus)
        stab_moves = [m for m in valid_moves if self.moves_db[m]['type'] in pokemon['types']]
        other_moves = [m for m in valid_moves if m not in stab_moves]
        
        chosen = []
        
        # Intentamos coger los 2 mejores de STAB (ordenados por poder)
        if stab_moves:
            # Ordenar por poder para coger los más fuertes disponibles
            stab_moves.sort(key=lambda x: (self.moves_db[x].get('power') or 0), reverse=True)
            chosen.extend(stab_moves[:2])
        
        # Rellenamos los huecos con el resto (barajados para variedad)
        remaining_slots = 4 - len(chosen)
        if remaining_slots > 0:
            # Quitamos los que ya elegimos del pool de otros
            pool = [m for m in valid_moves if m not in chosen]
            
            # Intentar meter al menos un movimiento de estado si hay hueco
            status_moves = [m for m in pool if self.moves_db[m].get('power', 0) == 0]
            if status_moves and remaining_slots >= 1:
                chosen.append(random.choice(status_moves))
                remaining_slots -= 1
                pool = [m for m in pool if m not in chosen] # Actualizar pool

            # Rellenar el resto al azar
            if pool and remaining_slots > 0:
                chosen.extend(np.random.choice(pool, size=min(remaining_slots, len(pool)), replace=False))

        return list(chosen)

    def prepare_pokemon(self, pid, level):
        """
        Prepara un Pokémon completo con stats calculados y movimientos filtrados.
        """
        pid = str(pid)
        if pid not in self.pokedex:
            return None
            
        p = self.pokedex[pid].copy()
        
        # Establecer Nivel
        p['level'] = level
        p['exp'] = 0
        
        # Calcular Stats
        p['stats'] = BattleEngine.get_stats_at_level(p, level)
        
        # Elegir Movimientos
        p['active_moves'] = self.select_moves(p, level)
        
        return p

    def build_team(self, target_type, team_size=1):
        """
        Elige el mejor Pokémon disponible para contrarrestar un tipo.
        """
        candidates = []
        pool = self.current_party if self.current_party else self.pokedex

        print(f"🧠 Estratega: Analizando opciones contra {target_type.upper()}...")

        for p_id, data in pool.items():
            # Saltar muertos
            if 'stats' in data and data['stats'].get('hp', 0) <= 0:
                continue

            score = 0
            
            # 1. Ventaja Ofensiva (Tipos)
            # Simplificación: Miramos los tipos base del Pokémon
            for t in data['types']:
                if t in BattleEngine.TYPE_CHART:
                    mult = BattleEngine.TYPE_CHART[t].get(target_type, 1.0)
                    if mult > 1.2: score += 20
                    elif mult < 0.8: score -= 10
            
            # 2. Ventaja Defensiva (Resistencia)
            if target_type in BattleEngine.TYPE_CHART:
                chart = BattleEngine.TYPE_CHART[target_type]
                damage_taken = 1.0
                for my_t in data['types']:
                    damage_taken *= chart.get(my_t, 1.0)
                
                if damage_taken < 1.0: score += 25 # ¡Resisto!
                elif damage_taken > 1.0: score -= 30 # ¡Me duele!

            # 3. Nivel (Fuerza bruta)
            score += data.get('level', 1)

            candidates.append((score, data))

        # Ordenar por puntuación
        candidates.sort(key=lambda x: x[0], reverse=True)

        if not candidates:
            # Si todos están muertos, devolvemos el primero aunque sea cadáver (se gestionará fuera)
            return list(pool.values())[0]

        best_mon = candidates[0][1]
        print(f"🧠 Estratega: Elijo a {best_mon['name']} (Score: {candidates[0][0]:.1f})")
        return best_mon