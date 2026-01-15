import numpy as np
import random
from src.env.battle_engine import BattleEngine


class Strategist:
    def __init__(self, pokedex):
        self.pokedex = pokedex
        self.current_party = {}  # <--- CAMBIO 1: Variable para el equipo de 6

    def set_party(self, party_ids):
        """
        Define el equipo de 6 pokemons disponibles actualmente.
        Llamado desde train.py cada 10 episodios.
        """
        # Filtramos la pokedex completa para quedarnos solo con los IDs que nos pasan
        self.current_party = {
            pid: self.pokedex[pid]
            for pid in party_ids
            if pid in self.pokedex
        }

        # Feedback visual
        names = [p['name'] for p in self.current_party.values()]
        print(f"\n🎒 ESTRATEGA: Nuevo equipo asignado: {names}")

    def build_team(self, target_type, team_size=1):
        """
        Selecciona el mejor Pokémon disponible en el EQUIPO ACTUAL (current_party)
        para vencer al tipo objetivo.
        """
        candidates = []

        # <--- CAMBIO 3: Decidir dónde buscar
        # Si tenemos un equipo definido, buscamos ahí. Si no, usamos toda la pokedex (por seguridad).
        pool = self.current_party if self.current_party else self.pokedex

        print(
            f"🧠 Estratega: Buscando en mi equipo de {len(pool)} la mejor opción contra {target_type.upper()}...")

        for p_id, data in pool.items():
            # Saltamos datos corruptos
            if 'stats' not in data or 'types' not in data:
                continue

            score = 0

            # 1. Ventaja Ofensiva (¿Mis ataques le duelen?)
            my_types = data['types']
            atk_advantage = 0
            for t in my_types:
                mult = BattleEngine.get_multiplier(t, [target_type])
                if mult > 1.0:
                    atk_advantage += 10
                if mult < 1.0:
                    atk_advantage -= 5

            # 2. Ventaja Defensiva (¿Sus ataques me duelen?)
            def_mult = BattleEngine.get_multiplier(target_type, my_types)
            def_score = 0
            if def_mult < 1.0:
                def_score += 10
            if def_mult > 1.0:
                def_score -= 10

            # 3. Stats Base
            stats = data['stats']
            power_score = (stats.get('hp', 0) + stats.get('attack',
                           0) + stats.get('defense', 0)) / 10.0

            # PUNTUACIÓN FINAL
            score = atk_advantage + def_score + power_score

            candidates.append((score, data))

        # Ordenar por puntuación (de mayor a menor)
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Seguridad por si la lista está vacía
        if not candidates:
            # Devuelve un pokemon cualquiera si falla la lógica
            return list(self.pokedex.values())[0]

        # Elegir los top 'team_size'
        best_team = [c[1] for c in candidates[:team_size]]

        print(
            f"🧠 Estratega: He seleccionado a {best_team[0]['name']} (Score: {candidates[0][0]:.1f})")
        return best_team[0]
