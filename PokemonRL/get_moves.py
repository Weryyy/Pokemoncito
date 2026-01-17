import requests
import json
import os
import time

# Configuración
POKEDEX_FILE = 'PokemonRL\data\pokedex.json'
OUTPUT_FILE = 'data/moves.json'
os.makedirs('data', exist_ok=True)

def fetch_all_moves():
    # 1. Cargar lista de movimientos únicos desde tu Pokedex
    print("📖 Leyendo Pokedex...")
    with open(POKEDEX_FILE, 'r') as f:
        pokedex = json.load(f)
    
    unique_moves = set()
    for p in pokedex.values():
        for m in p.get('moves', []):
            unique_moves.add(m)
    
    print(f"🔍 Encontrados {len(unique_moves)} movimientos únicos.")
    
    # 2. Descargar datos de la API
    moves_db = {}
    total = len(unique_moves)
    
    for i, move_name in enumerate(unique_moves):
        # Pequeño cache local por si tienes que reiniciar
        if move_name in moves_db: continue
            
        print(f"[{i+1}/{total}] Descargando: {move_name}...")
        try:
            url = f"https://pokeapi.co/api/v2/move/{move_name}"
            r = requests.get(url)
            if r.status_code == 200:
                data = r.json()
                moves_db[move_name] = {
                    "type": data['type']['name'],
                    "power": data['power'] if data['power'] else 0, # Movimientos de estado tienen 0
                    "accuracy": data['accuracy'] if data['accuracy'] else 100,
                    "pp": data['pp'],
                    "class": data['damage_class']['name'] # physical, special, status
                }
            else:
                print(f"❌ Error al descargar {move_name}")
        except Exception as e:
            print(f"⚠ Excepción: {e}")
        
        # Respetar a la API (anti-ban)
        # time.sleep(0.05) 

    # 3. Guardar JSON
    print(f"💾 Guardando base de datos en {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(moves_db, f, indent=4)
    print("✅ ¡Completado!")

if __name__ == "__main__":
    fetch_all_moves()