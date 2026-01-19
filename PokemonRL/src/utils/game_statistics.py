"""
Sistema de estadísticas y tracking de partidas
"""
import json
import os
from datetime import datetime
from collections import defaultdict
import uuid


class GameStatistics:
    """Clase para trackear estadísticas de una partida"""
    
    def __init__(self, game_id=None):
        self.game_id = game_id or str(uuid.uuid4())[:8]
        self.start_time = datetime.now()
        self.end_time = None
        
        # Estadísticas generales
        self.total_steps = 0
        self.total_damage_dealt = 0
        self.total_damage_received = 0
        self.total_exp_gained = 0
        
        # Combates
        self.battles_won = 0
        self.battles_lost = 0
        self.wild_encounters = 0
        self.boss_battles = 0
        
        # Mapas
        self.maps_completed = []
        self.current_map = 0
        self.time_per_map = {}
        self.map_start_time = datetime.now()
        
        # Movimientos
        self.moves_used = defaultdict(int)
        self.move_effectiveness = defaultdict(lambda: {'hits': 0, 'damage': 0})
        
        # Pokemon
        self.pokemon_used = []
        self.pokemon_stats = {}  # Stats finales de cada pokemon
        self.level_ups = 0
        
        # Eventos especiales
        self.critical_hits = 0
        self.status_inflicted = defaultdict(int)
        self.items_used = defaultdict(int)
        
        # Progresión
        self.progression_log = []
        
    def log_event(self, event_type, data):
        """Registra un evento en el log de progresión"""
        self.progression_log.append({
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'data': data
        })
    
    def start_map(self, map_idx):
        """Registra inicio de un nuevo mapa"""
        self.current_map = map_idx
        self.map_start_time = datetime.now()
        self.log_event('map_start', {'map': map_idx})
    
    def complete_map(self, map_idx):
        """Registra completitud de un mapa"""
        time_taken = (datetime.now() - self.map_start_time).total_seconds()
        self.maps_completed.append(map_idx)
        self.time_per_map[map_idx] = time_taken
        self.log_event('map_complete', {
            'map': map_idx,
            'time': time_taken
        })
    
    def log_battle_start(self, enemy_name, enemy_level, is_boss=False):
        """Registra inicio de batalla"""
        self.wild_encounters += 1
        if is_boss:
            self.boss_battles += 1
        self.log_event('battle_start', {
            'enemy': enemy_name,
            'level': enemy_level,
            'is_boss': is_boss
        })
    
    def log_battle_end(self, won, exp_gained=0):
        """Registra fin de batalla"""
        if won:
            self.battles_won += 1
            self.total_exp_gained += exp_gained
        else:
            self.battles_lost += 1
        
        self.log_event('battle_end', {
            'won': won,
            'exp': exp_gained
        })
    
    def log_move(self, move_name, damage, effectiveness=""):
        """Registra uso de movimiento"""
        self.moves_used[move_name] += 1
        self.total_damage_dealt += damage
        self.move_effectiveness[move_name]['hits'] += 1
        self.move_effectiveness[move_name]['damage'] += damage
        
        if 'crítico' in effectiveness.lower():
            self.critical_hits += 1
        
        self.total_steps += 1
    
    def log_damage_received(self, damage):
        """Registra daño recibido"""
        self.total_damage_received += damage
    
    def log_level_up(self, pokemon_name, new_level):
        """Registra subida de nivel"""
        self.level_ups += 1
        self.log_event('level_up', {
            'pokemon': pokemon_name,
            'level': new_level
        })
    
    def log_status(self, status_name):
        """Registra estado alterado infligido"""
        self.status_inflicted[status_name] += 1
    
    def log_item_use(self, item_name):
        """Registra uso de objeto"""
        self.items_used[item_name] += 1
    
    def set_pokemon_team(self, team):
        """Registra equipo pokemon y sus stats finales"""
        self.pokemon_used = [p['name'] for p in team]
        self.pokemon_stats = {
            p['name']: {
                'level': p['level'],
                'hp': p['stats']['hp'],
                'types': p.get('types', []),
                'ability': p.get('ability', ''),
                'held_item': p.get('held_item', '')
            }
            for p in team
        }
    
    def finalize(self):
        """Finaliza la partida y calcula estadísticas finales"""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        return {
            'game_id': self.game_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': duration,
            'total_steps': self.total_steps,
            'total_damage_dealt': self.total_damage_dealt,
            'total_damage_received': self.total_damage_received,
            'total_exp_gained': self.total_exp_gained,
            'battles_won': self.battles_won,
            'battles_lost': self.battles_lost,
            'wild_encounters': self.wild_encounters,
            'boss_battles': self.boss_battles,
            'maps_completed': self.maps_completed,
            'time_per_map': self.time_per_map,
            'moves_used': dict(self.moves_used),
            'move_effectiveness': {k: dict(v) for k, v in self.move_effectiveness.items()},
            'pokemon_used': self.pokemon_used,
            'pokemon_stats': self.pokemon_stats,
            'level_ups': self.level_ups,
            'critical_hits': self.critical_hits,
            'status_inflicted': dict(self.status_inflicted),
            'items_used': dict(self.items_used),
            'win_rate': self.battles_won / max(1, self.battles_won + self.battles_lost),
            'avg_damage_per_battle': self.total_damage_dealt / max(1, self.battles_won + self.battles_lost),
            'progression_log': self.progression_log[-50:]  # Últimos 50 eventos
        }
    
    def save(self, base_dir='game_statistics'):
        """Guarda las estadísticas en un archivo JSON"""
        os.makedirs(base_dir, exist_ok=True)
        
        # Crear estructura de carpetas por fecha
        date_folder = self.start_time.strftime('%Y-%m-%d')
        date_path = os.path.join(base_dir, date_folder)
        os.makedirs(date_path, exist_ok=True)
        
        # Guardar archivo
        filename = f"game_{self.game_id}_{self.start_time.strftime('%H%M%S')}.json"
        filepath = os.path.join(date_path, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.finalize(), f, indent=2)
        
        return filepath


class StatisticsManager:
    """Gestor para cargar y analizar múltiples partidas"""
    
    @staticmethod
    def load_all_games(base_dir='game_statistics'):
        """Carga todas las partidas guardadas"""
        games = []
        
        if not os.path.exists(base_dir):
            return games
        
        for date_folder in os.listdir(base_dir):
            date_path = os.path.join(base_dir, date_folder)
            if not os.path.isdir(date_path):
                continue
            
            for filename in os.listdir(date_path):
                if filename.endswith('.json'):
                    filepath = os.path.join(date_path, filename)
                    try:
                        with open(filepath, 'r') as f:
                            game_data = json.load(f)
                            game_data['date_folder'] = date_folder
                            games.append(game_data)
                    except Exception as e:
                        print(f"Error loading {filepath}: {e}")
        
        # Ordenar por fecha
        games.sort(key=lambda x: x['start_time'], reverse=True)
        return games
    
    @staticmethod
    def get_summary_stats(games):
        """Calcula estadísticas agregadas de múltiples partidas"""
        if not games:
            return {}
        
        return {
            'total_games': len(games),
            'total_battles': sum(g['battles_won'] + g['battles_lost'] for g in games),
            'total_wins': sum(g['battles_won'] for g in games),
            'total_losses': sum(g['battles_lost'] for g in games),
            'avg_win_rate': sum(g['win_rate'] for g in games) / len(games),
            'total_damage_dealt': sum(g['total_damage_dealt'] for g in games),
            'total_damage_received': sum(g['total_damage_received'] for g in games),
            'total_exp_gained': sum(g['total_exp_gained'] for g in games),
            'total_maps_completed': sum(len(g['maps_completed']) for g in games),
            'avg_duration': sum(g['duration_seconds'] for g in games) / len(games),
            'total_critical_hits': sum(g['critical_hits'] for g in games),
            'total_level_ups': sum(g['level_ups'] for g in games),
        }
    
    @staticmethod
    def get_most_used_moves(games, top_n=10):
        """Obtiene los movimientos más usados"""
        all_moves = defaultdict(int)
        for game in games:
            for move, count in game['moves_used'].items():
                all_moves[move] += count
        
        return sorted(all_moves.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    @staticmethod
    def get_most_used_pokemon(games, top_n=10):
        """Obtiene los pokemon más usados"""
        all_pokemon = defaultdict(int)
        for game in games:
            for pokemon in game['pokemon_used']:
                all_pokemon[pokemon] += 1
        
        return sorted(all_pokemon.items(), key=lambda x: x[1], reverse=True)[:top_n]
