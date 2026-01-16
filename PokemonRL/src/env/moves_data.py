# src/env/moves_data.py

MOVES_DB = {
    # --- NORMAL ---
    "tackle": {"type": "normal", "power": 40, "accuracy": 1.0},
    "scratch": {"type": "normal", "power": 40, "accuracy": 1.0},
    "cut": {"type": "normal", "power": 50, "accuracy": 0.95},
    "slam": {"type": "normal", "power": 80, "accuracy": 0.75},
    "headbutt": {"type": "normal", "power": 70, "accuracy": 1.0},
    "body-slam": {"type": "normal", "power": 85, "accuracy": 1.0},
    "hyper-beam": {"type": "normal", "power": 150, "accuracy": 0.9},
    "quick-attack": {"type": "normal", "power": 40, "accuracy": 1.0},
    "bite": {"type": "normal", "power": 60, "accuracy": 1.0},
    
    # --- FUEGO ---
    "ember": {"type": "fire", "power": 40, "accuracy": 1.0},
    "flamethrower": {"type": "fire", "power": 90, "accuracy": 1.0},
    "fire-blast": {"type": "fire", "power": 110, "accuracy": 0.85},
    "fire-punch": {"type": "fire", "power": 75, "accuracy": 1.0},
    "fire-spin": {"type": "fire", "power": 35, "accuracy": 0.85},

    # --- AGUA ---
    "water-gun": {"type": "water", "power": 40, "accuracy": 1.0},
    "bubble-beam": {"type": "water", "power": 65, "accuracy": 1.0},
    "hydro-pump": {"type": "water", "power": 110, "accuracy": 0.8},
    "surf": {"type": "water", "power": 90, "accuracy": 1.0},
    "waterfall": {"type": "water", "power": 80, "accuracy": 1.0},
    "crabhammer": {"type": "water", "power": 100, "accuracy": 0.9},

    # --- PLANTA ---
    "vine-whip": {"type": "grass", "power": 45, "accuracy": 1.0},
    "razor-leaf": {"type": "grass", "power": 55, "accuracy": 0.95},
    "solar-beam": {"type": "grass", "power": 120, "accuracy": 1.0},
    "mega-drain": {"type": "grass", "power": 40, "accuracy": 1.0},
    "petal-dance": {"type": "grass", "power": 120, "accuracy": 1.0},

    # --- ELÉCTRICO ---
    "thunder-shock": {"type": "electric", "power": 40, "accuracy": 1.0},
    "thunderbolt": {"type": "electric", "power": 90, "accuracy": 1.0},
    "thunder": {"type": "electric", "power": 110, "accuracy": 0.7},
    "thunder-punch": {"type": "electric", "power": 75, "accuracy": 1.0},

    # --- HIELO ---
    "ice-beam": {"type": "ice", "power": 90, "accuracy": 1.0},
    "blizzard": {"type": "ice", "power": 110, "accuracy": 0.7},
    "aurora-beam": {"type": "ice", "power": 65, "accuracy": 1.0},

    # --- LUCHA ---
    "karate-chop": {"type": "fighting", "power": 50, "accuracy": 1.0},
    "submission": {"type": "fighting", "power": 80, "accuracy": 0.8},
    "seismic-toss": {"type": "fighting", "power": 60, "accuracy": 1.0}, # Simplificado

    # --- VENENO ---
    "sludge": {"type": "poison", "power": 65, "accuracy": 1.0},
    "acid": {"type": "poison", "power": 40, "accuracy": 1.0},
    "sludge-bomb": {"type": "poison", "power": 90, "accuracy": 1.0},

    # --- TIERRA ---
    "earthquake": {"type": "ground", "power": 100, "accuracy": 1.0},
    "dig": {"type": "ground", "power": 80, "accuracy": 1.0},
    "bone-club": {"type": "ground", "power": 65, "accuracy": 0.85},

    # --- VOLADOR ---
    "fly": {"type": "flying", "power": 90, "accuracy": 0.95},
    "peck": {"type": "flying", "power": 35, "accuracy": 1.0},
    "drill-peck": {"type": "flying", "power": 80, "accuracy": 1.0},
    "wing-attack": {"type": "flying", "power": 60, "accuracy": 1.0},

    # --- PSÍQUICO ---
    "psychic": {"type": "psychic", "power": 90, "accuracy": 1.0},
    "psybeam": {"type": "psychic", "power": 65, "accuracy": 1.0},
    "confusion": {"type": "psychic", "power": 50, "accuracy": 1.0},

    # --- BICHO ---
    "twineedle": {"type": "bug", "power": 25, "accuracy": 1.0},
    "pin-missile": {"type": "bug", "power": 25, "accuracy": 0.95},

    # --- ROCA ---
    "rock-throw": {"type": "rock", "power": 50, "accuracy": 0.9},
    "rock-slide": {"type": "rock", "power": 75, "accuracy": 0.9},

    # --- FANTASMA ---
    "lick": {"type": "ghost", "power": 30, "accuracy": 1.0},
    "shadow-ball": {"type": "ghost", "power": 80, "accuracy": 1.0}, # Gen 2 pero clásico

    # --- DRAGÓN ---
    "dragon-rage": {"type": "dragon", "power": 40, "accuracy": 1.0}, # Simplificado
}

# Movimiento por defecto si algo falla (Combate)
STRUGGLE = {"name": "struggle", "type": "normal", "power": 50, "accuracy": 1.0}