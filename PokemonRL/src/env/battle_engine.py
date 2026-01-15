import random


class BattleEngine:
    # Tabla de efectividad simplificada (1.0 es neutro)
    # Formato: ATACANTE: {DEFENSOR: MULTIPLICADOR}
    TYPE_CHART = {
        "normal": {"rock": 0.5, "ghost": 0.0},
        "fire": {"fire": 0.5, "water": 0.5, "grass": 2.0, "bug": 2.0, "ice": 2.0, "rock": 0.5, "dragon": 0.5},
        "water": {"fire": 2.0, "water": 0.5, "grass": 0.5, "ground": 2.0, "rock": 2.0, "dragon": 0.5},
        "electric": {"water": 2.0, "electric": 0.5, "grass": 0.5, "ground": 0.0, "flying": 2.0, "dragon": 0.5},
        "grass": {"fire": 0.5, "water": 2.0, "grass": 0.5, "poison": 0.5, "ground": 2.0, "flying": 0.5, "bug": 0.5, "rock": 2.0, "dragon": 0.5},
        "ice": {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 0.5, "ground": 2.0, "flying": 2.0, "dragon": 2.0},
        "fighting": {"normal": 2.0, "ice": 2.0, "rock": 2.0, "dark": 2.0, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "ghost": 0.0},
        "poison": {"grass": 2.0, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5},
        "ground": {"fire": 2.0, "electric": 2.0, "grass": 0.5, "poison": 2.0, "flying": 0.0, "bug": 0.5, "rock": 2.0},
        "flying": {"electric": 0.5, "grass": 2.0, "fighting": 2.0, "bug": 2.0, "rock": 0.5},
        "psychic": {"fighting": 2.0, "poison": 2.0, "psychic": 0.5, "dark": 0.0},
        "bug": {"fire": 0.5, "grass": 2.0, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2.0, "ghost": 0.5},
        "rock": {"fire": 2.0, "ice": 2.0, "fighting": 0.5, "ground": 0.5, "flying": 2.0, "bug": 2.0},
        "ghost": {"normal": 0.0, "psychic": 2.0, "ghost": 2.0, "dark": 0.5},
        "dragon": {"dragon": 2.0}
    }

    @staticmethod
    def get_multiplier(attack_type, defender_types):
        multiplier = 1.0
        for dtype in defender_types:
            # Buscamos en la tabla, si no existe la relación asumimos 1.0
            if attack_type in BattleEngine.TYPE_CHART:
                multiplier *= BattleEngine.TYPE_CHART[attack_type].get(
                    dtype, 1.0)
        return multiplier

    @staticmethod
    def calculate_damage(attacker, defender, move_power=40, move_type="normal"):
        """
        Calcula el daño usando la fórmula de Gen 1 simplificada.
        """
        # 1. Obtener Stats
        level = 50  # Estandarizamos nivel 50

        # Manejo seguro de diccionarios (por si la API trajo datos raros)
        a_stats = attacker.get('stats', {})
        d_stats = defender.get('stats', {})

        # Usamos Attack vs Defense por defecto (simplificación física/especial)
        atk = a_stats.get('attack', 10)
        defense = d_stats.get('defense', 10)

        # Evitar división por cero
        if defense < 1:
            defense = 1

        # 2. Calcular Multiplicadores
        type_mult = BattleEngine.get_multiplier(
            move_type, defender.get('types', []))

        # STAB (Same Type Attack Bonus) - Si el pokemon usa ataque de su tipo pega 50% más
        stab = 1.5 if move_type in attacker.get('types', []) else 1.0

        # Variación aleatoria (Roll) entre 0.85 y 1.0
        random_mult = random.uniform(0.85, 1.0)

        # 3. Fórmula
        # Daño = ((((2 * Lvl / 5 + 2) * Atk * Power / Def) / 50) + 2) * Mods
        base_damage = (((2 * level / 5 + 2) * atk *
                       move_power / defense) / 50) + 2

        final_damage = base_damage * type_mult * stab * random_mult

        return int(final_damage), type_mult
