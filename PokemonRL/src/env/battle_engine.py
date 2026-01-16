import numpy as np
from src.env.moves_data import MOVES_DB, STRUGGLE

class BattleEngine:
    TYPE_CHART = {
        'normal': {'rock': 0.5, 'ghost': 0},
        'fire': {'fire': 0.5, 'water': 0.5, 'grass': 2.0, 'ice': 2.0, 'bug': 2.0, 'rock': 0.5, 'dragon': 0.5},
        'water': {'fire': 2.0, 'water': 0.5, 'grass': 0.5, 'ground': 2.0, 'rock': 2.0, 'dragon': 0.5},
        'electric': {'water': 2.0, 'electric': 0.5, 'grass': 0.5, 'ground': 0, 'flying': 2.0, 'dragon': 0.5},
        'grass': {'fire': 0.5, 'water': 2.0, 'grass': 0.5, 'poison': 0.5, 'ground': 2.0, 'flying': 0.5, 'bug': 0.5, 'rock': 2.0, 'dragon': 0.5},
        'ice': {'fire': 0.5, 'water': 0.5, 'grass': 2.0, 'ice': 0.5, 'ground': 2.0, 'flying': 2.0, 'dragon': 2.0},
        'fighting': {'normal': 2.0, 'ice': 2.0, 'poison': 0.5, 'flying': 0.5, 'psychic': 0.5, 'bug': 0.5, 'rock': 2.0, 'ghost': 0},
        'poison': {'grass': 2.0, 'poison': 0.5, 'ground': 0.5, 'rock': 0.5, 'ghost': 0.5},
        'ground': {'fire': 2.0, 'electric': 2.0, 'grass': 0.5, 'poison': 2.0, 'flying': 0, 'bug': 0.5, 'rock': 2.0},
        'flying': {'electric': 0.5, 'grass': 2.0, 'fighting': 2.0, 'bug': 2.0, 'rock': 0.5},
        'psychic': {'fighting': 2.0, 'poison': 2.0, 'psychic': 0.5},
        'bug': {'fire': 0.5, 'grass': 2.0, 'fighting': 0.5, 'poison': 2.0, 'flying': 0.5, 'psychic': 2.0, 'ghost': 0.5},
        'rock': {'fire': 2.0, 'ice': 2.0, 'fighting': 0.5, 'ground': 0.5, 'flying': 2.0, 'bug': 2.0},
        'ghost': {'normal': 0, 'psychic': 2.0, 'ghost': 2.0},
        'dragon': {'dragon': 2.0}
    }

    @staticmethod
    def get_type_effectiveness(move_type, defender_types):
        multiplier = 1.0
        if move_type not in BattleEngine.TYPE_CHART:
            return 1.0
        for dt in defender_types:
            dt = dt.lower()
            if dt in BattleEngine.TYPE_CHART[move_type]:
                multiplier *= BattleEngine.TYPE_CHART[move_type][dt]
        return multiplier

    @staticmethod
    def calculate_damage(attacker, defender, move_name):
        move = MOVES_DB.get(move_name, STRUGGLE)
        level = attacker.get('level', 5)
        
        att_stat = attacker['stats'].get('attack', 10)
        def_stat = defender['stats'].get('defense', 10)
        
        stab = 1.5 if move['type'] in attacker['types'] else 1.0
        effectiveness = BattleEngine.get_type_effectiveness(move['type'], defender['types'])
        critical = 1.5 if np.random.rand() < 0.0625 else 1.0
        random_factor = np.random.uniform(0.85, 1.0)

        damage = (((2 * level / 5 + 2) * move['power'] * (att_stat / def_stat)) / 50 + 2)
        damage *= stab * effectiveness * critical * random_factor
        
        return int(max(1, damage)), effectiveness

    @staticmethod
    def get_stats_at_level(pokemon, level):
        stats = {}
        # Hacemos que la subida de stats sea MÁS notable para la demo
        for s, val in pokemon['stats'].items():
            if s == 'hp':
                stats[s] = int((val * 3 * level) / 100 + level + 10) # x3 en lugar de x2
            else:
                stats[s] = int((val * 3 * level) / 100 + 5)
        return stats

    @staticmethod
    def gain_experience(winner, loser_level):
        # --- MODIFICACIÓN TURBO ---
        # XP = Nivel Enemigo * 80 (Subirán casi 1 nivel por combate)
        xp_gain = loser_level * 80 
        
        winner['exp'] = winner.get('exp', 0) + xp_gain
        needed_xp = winner['level'] * 100 # Umbral
        
        leveled_up = False
        old_stats = winner['stats'].copy() # Guardar para comparar
        
        while winner['exp'] >= needed_xp:
            winner['exp'] -= needed_xp
            winner['level'] += 1
            leveled_up = True
            
            # Recalcular stats inmediatamente
            new_stats = BattleEngine.get_stats_at_level(winner, winner['level'])
            winner['stats'] = new_stats
            
            needed_xp = winner['level'] * 100
            
        return xp_gain, leveled_up, old_stats