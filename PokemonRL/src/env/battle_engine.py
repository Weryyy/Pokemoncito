import numpy as np
import json
import os
import random

# Intentar cargar moves.json
try:
    # Ajusta la ruta según tu estructura de carpetas
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(base_path, 'data', 'moves.json')
    with open(path, 'r') as f:
        MOVES_DB = json.load(f)
except:
    MOVES_DB = {}

STRUGGLE = {"type": "normal", "power": 50, "accuracy": 100, "class": "physical"}

# --- LISTA BLANCA DE EFECTOS SOPORTADOS ---
# Solo los movimientos que estén aquí O tengan daño > 0 serán usados.
EFFECTS_DB = {
    # BUFFS (Usuario)
    'swords-dance': {'target': 'self', 'stat': 'attack', 'stage': 2, 'msg': '¡Subió mucho su ATAQUE!'},
    'growth':       {'target': 'self', 'stat': 'special-attack', 'stage': 1, 'msg': '¡Subió su AT. ESP!'},
    'defense-curl': {'target': 'self', 'stat': 'defense', 'stage': 1, 'msg': '¡Subió su DEFENSA!'},
    'hardening':    {'target': 'self', 'stat': 'defense', 'stage': 1, 'msg': '¡Subió su DEFENSA!'},
    'agility':      {'target': 'self', 'stat': 'speed', 'stage': 2, 'msg': '¡Subió mucho su VELOCIDAD!'},
    'recover':      {'target': 'self', 'heal': 0.5, 'msg': '¡Recuperó salud!'},
    'soft-boiled':  {'target': 'self', 'heal': 0.5, 'msg': '¡Recuperó salud!'},
    'protect':      {'target': 'self', 'special': 'protect', 'msg': '¡Se protegió!'},
    'focus-energy': {'target': 'self', 'special': 'crit', 'msg': '¡Se está concentrando!'},

    # DEBUFFS (Rival)
    'growl':        {'target': 'enemy', 'stat': 'attack', 'stage': -1, 'msg': '¡Bajó el ATAQUE del rival!'},
    'tail-whip':    {'target': 'enemy', 'stat': 'defense', 'stage': -1, 'msg': '¡Bajó la DEFENSA del rival!'},
    'leer':         {'target': 'enemy', 'stat': 'defense', 'stage': -1, 'msg': '¡Bajó la DEFENSA del rival!'},
    'screech':      {'target': 'enemy', 'stat': 'defense', 'stage': -2, 'msg': '¡Bajó mucho la DEFENSA rival!'},
    'sand-attack':  {'target': 'enemy', 'stat': 'accuracy', 'stage': -1, 'msg': '¡Bajó la PRECISIÓN rival!'},
    'string-shot':  {'target': 'enemy', 'stat': 'speed', 'stage': -1, 'msg': '¡Bajó la VELOCIDAD rival!'},
    'smokescreen':  {'target': 'enemy', 'stat': 'accuracy', 'stage': -1, 'msg': '¡Bajó la PRECISIÓN rival!'},

    # ESTADOS ALTERADOS
    'thunder-wave': {'target': 'enemy', 'status': 'PAR', 'msg': '¡El rival está paralizado!'},
    'glare':        {'target': 'enemy', 'status': 'PAR', 'msg': '¡El rival está paralizado!'},
    'stun-spore':   {'target': 'enemy', 'status': 'PAR', 'msg': '¡El rival está paralizado!'},
    'toxic':        {'target': 'enemy', 'status': 'PSN', 'msg': '¡El rival está gravemente envenenado!'},
    'poison-powder':{'target': 'enemy', 'status': 'PSN', 'msg': '¡El rival está envenenado!'},
    'poison-gas':   {'target': 'enemy', 'status': 'PSN', 'msg': '¡El rival está envenenado!'},
    'hypnosis':     {'target': 'enemy', 'status': 'SLP', 'msg': '¡El rival se durmió!'},
    'sleep-powder': {'target': 'enemy', 'status': 'SLP', 'msg': '¡El rival se durmió!'},
    'sing':         {'target': 'enemy', 'status': 'SLP', 'msg': '¡El rival se durmió!'},
    'will-o-wisp':  {'target': 'enemy', 'status': 'BRN', 'msg': '¡El rival se quemó!'},
    'confuse-ray':  {'target': 'enemy', 'special': 'confuse', 'msg': '¡El rival está confuso!'},
    'supersonic':   {'target': 'enemy', 'special': 'confuse', 'msg': '¡El rival está confuso!'},
}

class BattleEngine:
    TYPE_CHART = {
        'normal': {'rock': 0.5, 'ghost': 0, 'steel': 0.5},
        'fire': {'fire': 0.5, 'water': 0.5, 'grass': 2.0, 'ice': 2.0, 'bug': 2.0, 'rock': 0.5, 'dragon': 0.5, 'steel': 2.0},
        'water': {'fire': 2.0, 'water': 0.5, 'grass': 0.5, 'ground': 2.0, 'rock': 2.0, 'dragon': 0.5},
        'electric': {'water': 2.0, 'electric': 0.5, 'grass': 0.5, 'ground': 0, 'flying': 2.0, 'dragon': 0.5},
        'grass': {'fire': 0.5, 'water': 2.0, 'grass': 0.5, 'poison': 0.5, 'ground': 2.0, 'flying': 0.5, 'bug': 0.5, 'rock': 2.0, 'dragon': 0.5, 'steel': 0.5},
        'ice': {'fire': 0.5, 'water': 0.5, 'grass': 2.0, 'ice': 0.5, 'ground': 2.0, 'flying': 2.0, 'dragon': 2.0, 'steel': 0.5},
        'fighting': {'normal': 2.0, 'ice': 2.0, 'poison': 0.5, 'flying': 0.5, 'psychic': 0.5, 'bug': 0.5, 'rock': 2.0, 'ghost': 0, 'dark': 2.0, 'steel': 2.0, 'fairy': 0.5},
        'poison': {'grass': 2.0, 'poison': 0.5, 'ground': 0.5, 'rock': 0.5, 'ghost': 0.5, 'steel': 0, 'fairy': 2.0},
        'ground': {'fire': 2.0, 'electric': 2.0, 'grass': 0.5, 'poison': 2.0, 'flying': 0, 'bug': 0.5, 'rock': 2.0, 'steel': 2.0},
        'flying': {'electric': 0.5, 'grass': 2.0, 'fighting': 2.0, 'bug': 2.0, 'rock': 0.5, 'steel': 0.5},
        'psychic': {'fighting': 2.0, 'poison': 2.0, 'psychic': 0.5, 'dark': 0, 'steel': 0.5},
        'bug': {'fire': 0.5, 'grass': 2.0, 'fighting': 0.5, 'poison': 2.0, 'flying': 0.5, 'psychic': 2.0, 'ghost': 0.5, 'dark': 2.0, 'steel': 0.5, 'fairy': 0.5},
        'rock': {'fire': 2.0, 'ice': 2.0, 'fighting': 0.5, 'ground': 0.5, 'flying': 2.0, 'bug': 2.0, 'steel': 0.5},
        'ghost': {'normal': 0, 'psychic': 2.0, 'ghost': 2.0, 'dark': 0.5},
        'dragon': {'dragon': 2.0, 'steel': 0.5, 'fairy': 0},
        'dark': {'fighting': 0.5, 'psychic': 2.0, 'ghost': 2.0, 'dark': 0.5, 'fairy': 0.5},
        'steel': {'fire': 0.5, 'water': 0.5, 'electric': 0.5, 'ice': 2.0, 'rock': 2.0, 'steel': 0.5, 'fairy': 2.0},
        'fairy': {'fire': 0.5, 'fighting': 2.0, 'poison': 0.5, 'dragon': 2.0, 'dark': 2.0, 'steel': 0.5}
    }

    @staticmethod
    def get_moves_for_level(pokemon, level):
        """Devuelve 4 movimientos válidos usando el Estratega (se llama desde fuera normalmente)"""
        # Este método es un fallback, la lógica principal está en Strategist
        return ['tackle']

    @staticmethod
    def calculate_damage(attacker, defender, move_name):
        move = MOVES_DB.get(move_name, STRUGGLE)
        
        # 1. ESTADOS QUE IMPIDEN MOVERSE
        status = attacker.get('status_condition')
        if status == 'SLP':
            if random.random() < 0.6: return 0, f"{attacker['name']} duerme."
            else: attacker['status_condition'] = None; return 0, f"¡{attacker['name']} despertó!"
        if status == 'PAR' and random.random() < 0.25:
            return 0, f"{attacker['name']} está paralizado."
        if status == 'FRZ' and random.random() < 0.8:
            return 0, f"{attacker['name']} está congelado."

        # 2. PROTECCIÓN
        if defender.get('is_protected', False) and move.get('power', 0) > 0:
            defender['is_protected'] = False
            return 0, f"¡{defender['name']} se protegió!"
        defender['is_protected'] = False 

        # 3. APLICAR EFECTO (Si no tiene daño)
        if move.get('power', 0) == 0:
            return BattleEngine.apply_effect(attacker, defender, move_name)

        # 4. CÁLCULO DE DAÑO
        level = attacker.get('level', 5)
        
        # Stages (-6 a +6) -> Multiplicador
        # Fórmula simple: (2 + stage) / 2 para positivos, 2 / (2 + abs(stage)) para negativos
        atk_st = attacker.get('modifiers', {}).get('attack', 0)
        def_st = defender.get('modifiers', {}).get('defense', 0)
        
        atk_mult = (2 + atk_st) / 2 if atk_st >= 0 else 2 / (2 + abs(atk_st))
        def_mult = (2 + def_st) / 2 if def_st >= 0 else 2 / (2 + abs(def_st))

        att_stat = attacker['stats'].get('attack', 10) * atk_mult
        def_stat = defender['stats'].get('defense', 10) * def_mult
        
        if move.get('class') == 'special':
            # Simplificación: Usamos special-attack sin stages por ahora
            att_stat = attacker['stats'].get('special-attack', 10) 
            def_stat = defender['stats'].get('special-defense', 10)

        stab = 1.5 if move['type'] in attacker['types'] else 1.0
        
        multiplier = 1.0
        for dt in defender['types']:
            m_type = move['type']
            dt_name = dt.lower()
            if m_type in BattleEngine.TYPE_CHART and dt_name in BattleEngine.TYPE_CHART[m_type]:
                multiplier *= BattleEngine.TYPE_CHART[m_type][dt_name]
        
        critical = 1.5 if random.random() < 0.06 else 1.0
        random_factor = random.uniform(0.85, 1.0)

        damage = (((2 * level / 5 + 2) * move.get('power', 40) * (att_stat / def_stat)) / 50 + 2)
        damage *= stab * multiplier * critical * random_factor
        
        # Estado Quemado reduce ataque físico a la mitad
        if attacker.get('status_condition') == 'BRN' and move.get('class') == 'physical':
            damage *= 0.5

        msg = ""
        if multiplier > 1.2: msg = "(Eficaz!)"
        elif multiplier < 0.8: msg = "(No eficaz)"
        
        return int(max(1, damage)), msg

    @staticmethod
    def apply_effect(attacker, defender, move_name):
        if move_name not in EFFECTS_DB:
            # Si el movimiento es de estado pero NO está en nuestra DB, falla.
            return 0, "¡Pero falló! (No implementado)"
            
        effect = EFFECTS_DB[move_name]
        target = attacker if effect['target'] == 'self' else defender
        
        if 'modifiers' not in target: target['modifiers'] = {}
        
        if 'heal' in effect:
            max_hp = BattleEngine.get_stats_at_level(target, target['level'])['hp']
            heal = int(max_hp * effect['heal'])
            target['stats']['hp'] = min(max_hp, target['stats']['hp'] + heal)
            return 0, f"¡Recuperó {heal} PS!"

        if 'special' in effect and effect['special'] == 'protect':
            target['is_protected'] = True
            return 0, "¡Se prepara para protegerse!"

        if 'stat' in effect:
            stat = effect['stat']
            stage = effect['stage']
            curr = target['modifiers'].get(stat, 0)
            target['modifiers'][stat] = max(-6, min(6, curr + stage))
            return 0, effect['msg']

        if 'status' in effect:
            if target.get('status_condition'):
                return 0, "¡Falló!"
            target['status_condition'] = effect['status']
            return 0, effect['msg']

        return 0, "..."

    @staticmethod
    def get_stats_at_level(pokemon, level):
        stats = {}
        for s, val in pokemon['stats'].items():
            # Fórmula aproximada
            if s == 'hp':
                stats[s] = int((val * 2 * level) / 100 + level + 10)
            else:
                stats[s] = int((val * 2 * level) / 100 + 5)
        return stats

    @staticmethod
    def gain_experience(winner, loser_level):
        xp_gain = loser_level * 500 # XP Turbo
        winner['exp'] = winner.get('exp', 0) + xp_gain
        
        leveled_up = False
        old_stats = winner['stats'].copy()
        
        # Curva de nivel simple
        while winner['exp'] >= winner['level'] * 100:
            winner['exp'] -= winner['level'] * 100
            winner['level'] += 1
            leveled_up = True
            winner['stats'] = BattleEngine.get_stats_at_level(winner, winner['level'])
            winner['status_condition'] = None # Curar estado al subir nivel
            winner['modifiers'] = {}
            
        return xp_gain, leveled_up, old_stats