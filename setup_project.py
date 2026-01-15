import os
import json

# Definimos la estructura del proyecto
structure = {
    "PokemonRL": {
        "data": {
            "maps": {},
            "pokedex.json": "[]",  # Se llenará luego con el data_loader
            "moves.json": "[]"
        },
        "src": {
            "__init__.py": "",
            "env": {
                "__init__.py": "",
                "pokemon_env.py": "",  # Ver código abajo
                "battle_engine.py": ""
            },
            "models": {
                "__init__.py": "",
                "cnn_map.py": "",
                "dqn_combat.py": ""
            },
            "agents": {
                "__init__.py": "",
                "explorer.py": "",
                "tactician.py": "",
                "strategist.py": ""
            },
            "utils": {
                "__init__.py": "",
                "data_loader.py": ""  # Aquí irá la lógica de la API
            }
        },
        "checkpoints": {},
        "train.py": "",
        "play.py": "",
        "requirements.txt": "torch\ngymnasium\nnumpy\nrequests\nmatplotlib"
    }
}


def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Creado archivo: {path}")


# Ejecutar creación
create_structure(".", structure)
print("\n¡Estructura de carpetas PokemonRL creada con éxito!")
