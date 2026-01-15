import requests
import json
import os
import time


class PokeDownloader:
    def __init__(self):
        # Endpoint oficial
        self.base_url = "https://pokeapi.co/api/v2/"
        # Ruta para guardar en PokemonRL/data/pokedex.json
        self.data_path = os.path.join(os.path.dirname(
            os.path.dirname(os.path.dirname(__file__))), 'data')
        os.makedirs(self.data_path, exist_ok=True)

    def fetch_all_gen1(self):
        """Descarga los primeros 151 Pokémon y los guarda."""
        pokemon_db = {}
        total_count = 151

        print(
            f"--- Iniciando descarga de la Generación 1 ({total_count} Pokémon) ---")
        print("Esto puede tardar unos 2-3 minutos dependiendo de tu conexión.\n")

        for poke_id in range(1, total_count + 1):
            try:
                # Petición a la API
                response = requests.get(f"{self.base_url}pokemon/{poke_id}")

                if response.status_code == 200:
                    data = response.json()
                    name = data['name'].capitalize()

                    # Extraer stats normalizados (aprox 0-1 para la Red Neuronal)
                    # Base stats máximos en Gen 1 rondan 250 (Chansey HP), dividimos por 255
                    stats_raw = {s['stat']['name']: s['base_stat']
                                 for s in data['stats']}

                    # Extraer solo los nombres de los movimientos (para el Estratega)
                    moves = [m['move']['name'] for m in data['moves']]

                    pokemon_db[data['id']] = {
                        "id": data['id'],
                        "name": name,
                        "types": [t['type']['name'] for t in data['types']],
                        "stats": {
                            "hp": stats_raw.get('hp', 0),
                            "attack": stats_raw.get('attack', 0),
                            "defense": stats_raw.get('defense', 0),
                            "special-attack": stats_raw.get('special-attack', 0),
                            "special-defense": stats_raw.get('special-defense', 0),
                            "speed": stats_raw.get('speed', 0)
                        },
                        "moves": moves,  # Guardamos la lista de ataques posibles
                        "sprite": data['sprites']['front_default']
                    }

                    # Barra de progreso simple
                    print(f"[{poke_id}/{total_count}] {name} descargado ✅")
                else:
                    print(
                        f"[{poke_id}/{total_count}] Error HTTP {response.status_code} ❌")

            except Exception as e:
                print(f"Error descargando ID {poke_id}: {e}")

            # Pausa pequeña para no saturar la API
            # time.sleep(0.1)

        # Guardar en JSON local
        output_file = os.path.join(self.data_path, 'pokedex.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pokemon_db, f, indent=4)

        print(f"\n¡Éxito! Base de datos guardada en: {output_file}")
        print(f"Tamaño total de la Pokedex: {len(pokemon_db)} entradas.")


if __name__ == "__main__":
    downloader = PokeDownloader()
    downloader.fetch_all_gen1()
