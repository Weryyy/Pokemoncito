import random

class BattleEngine:
    TYPE_CHART = {
        "normal": {"rock": 0.5, "ghost": 0.0},
        "fire": {"fire": 0.5, "water": 0.5, "grass": 2.0, "bug": 2.0, "rock": 0.5, "dragon": 0.5, "ice": 2.0},
        "water": {"fire": 2.0, "water": 0.5, "grass": 0.5, "rock": 2.0, "ground": 2.0, "dragon": 0.5},
        "electric": {"water": 2.0, "electric": 0.5, "grass": 0.5, "ground": 0.0, "flying": 2.0, "dragon": 0.5},
        "grass": {"fire": 0.5, "water": 2.0, "grass": 0.5, "poison": 0.5, "ground": 2.0, "flying": 0.5, "bug": 0.5, "rock": 2.0},
        "ice": {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 0.5, "ground": 2.0, "flying": 2.0, "dragon": 2.0},
        "fighting": {"normal": 2.0, "ice": 2.0, "rock": 2.0, "dark": 2.0},
        "poison": {"grass": 2.0},
        "ground": {"fire": 2.0, "electric": 2.0, "poison": 2.0, "rock": 2.0},
        "flying": {"grass": 2.0, "fighting": 2.0, "bug": 2.0},
        "psychic": {"fighting": 2.0, "poison": 2.0},
        "bug": {"grass": 2.0, "psychic": 2.0},
        "rock": {"fire": 2.0, "ice": 2.0, "flying": 2.0, "bug": 2.0},
        "ghost": {"ghost": 2.0, "psychic": 2.0},
        "dragon": {"dragon": 2.0}
    }

    @staticmethod
    def get_multiplier(attack_type, defender_types):
        """Calcula la efectividad (x2, x0.5, etc.)"""
        multiplier = 1.0
        for dtype in defender_types:
            if attack_type in BattleEngine.TYPE_CHART:
                multiplier *= BattleEngine.TYPE_CHART[attack_type].get(dtype, 1.0)
        return multiplier

    @staticmethod
    def get_stats_at_level(pokemon_data, level):
        """Calcula los stats reales (HP, Atk, Def) basados en el nivel"""
        base = pokemon_data.get('stats', {'hp': 40, 'attack': 40, 'defense': 40})
        hp = int(((2 * base.get('hp', 40) * level) / 100) + level + 10)
        atk = int(((2 * base.get('attack', 40) * level) / 100) + 5)
        dfs = int(((2 * base.get('defense', 40) * level) / 100) + 5)
        return {'hp': hp, 'attack': atk, 'defense': dfs}

    @staticmethod
    def calculate_damage(attacker, defender, move_power=40, move_type="normal"):
        lvl_a = attacker.get('level', 5)
        lvl_d = defender.get('level', 5)
        
        stats_a = BattleEngine.get_stats_at_level(attacker, lvl_a)
        stats_d = BattleEngine.get_stats_at_level(defender, lvl_d)
        
        atk = stats_a['attack']
        defense = stats_d['defense']
        if defense < 1: defense = 1

        type_mult = BattleEngine.get_multiplier(move_type, defender.get('types', []))
        stab = 1.5 if move_type in attacker.get('types', []) else 1.0
        rand = random.uniform(0.85, 1.0)

        base_damage = (((2 * lvl_a / 5 + 2) * atk * move_power / defense) / 50) + 2
        final_damage = int(base_damage * type_mult * stab * rand)
        
        return final_damage, type_mult

    @staticmethod
    def get_exp_reward(defeated_pokemon):
        lvl = defeated_pokemon.get('level', 5)
        return lvl * 15