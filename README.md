# 🎮 Pokemoncito - Simulador de Pokémon con IA

Un simulador de Pokémon con **Reinforcement Learning** (Deep Q-Networks) que entrena agentes de IA para explorar mapas y combatir de forma autónoma. El proyecto incluye un sistema de combate completo con habilidades, objetos, y mecánicas reales de Pokémon.

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Requisitos del Sistema](#-requisitos-del-sistema)
- [Instalación](#-instalación)
- [Uso](#-uso)
  - [Entrenar las IAs](#1-entrenar-las-ias)
  - [Visualizar el Juego](#2-visualizar-el-juego)
  - [Otros Modos de Juego](#3-otros-modos-de-juego)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Resolución de Problemas](#-resolución-de-problemas)
- [Documentación Adicional](#-documentación-adicional)

## 🌟 Características

### Sistema de IA con Deep Reinforcement Learning
- **ExplorerAgent**: Red neuronal convolucional (CNN) para explorar mapas
- **TacticianAgent**: Red neuronal DQN para tomar decisiones de combate
- **Strategist**: Sistema experto para selección de Pokémon y movimientos óptimos

### Mecánicas de Pokémon Reales
- ✅ **Habilidades**: Overgrow, Blaze, Torrent, Intimidate, Levitate, etc.
- ✅ **Objetos Equipados**: Sitrus Berry, Leftovers, Choice Band, Focus Sash
- ✅ **Sistema de Tipos**: 18 tipos con efectividades correctas
- ✅ **Golpes Críticos**: Sistema de probabilidad 6.25% base
- ✅ **Estados Alterados**: Parálisis, Quemadura, Veneno, Dormir
- ✅ **Sistema de Experiencia**: Subida de nivel y aprendizaje de movimientos

### Técnicas Modernas de RL
- Experience Replay Buffer (10,000 experiencias)
- Target Networks para estabilidad
- Gradient Clipping
- Dropout y BatchNorm para prevenir overfitting
- Exploración basada en curiosidad

## 💻 Requisitos del Sistema

### Requisitos Mínimos
- **Sistema Operativo**: Windows 10/11, Linux, macOS
- **Python**: 3.8 o superior
- **RAM**: 4 GB
- **Espacio en disco**: 500 MB

### Requisitos Recomendados
- **GPU**: NVIDIA con soporte CUDA (para entrenamiento más rápido)
- **RAM**: 8 GB o más
- **CPU**: Procesador de 4 núcleos o más

## 🔧 Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/Weryyy/Pokemoncito.git
cd Pokemoncito
```

### 2. Instalar Dependencias

#### En Windows:
```bash
cd PokemonRL
pip install -r requirements.txt
```

#### En Linux/macOS:
```bash
cd PokemonRL
pip3 install -r requirements.txt
```

### 3. (Opcional) Instalar PyTorch con Soporte GPU

Si tienes una GPU NVIDIA y quieres entrenar 10-100x más rápido:

```bash
# Para CUDA 11.8 (mayoría de GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Para CUDA 12.1 (GPUs más nuevas)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Nota**: Consulta [GPU_SETUP.md](GPU_SETUP.md) para más detalles sobre configuración de GPU.

## 🎮 Uso

### 1. Entrenar las IAs

Para entrenar los cerebros de las IAs desde cero (toma ~2-4 horas en CPU, ~15-30 minutos en GPU):

```bash
cd PokemonRL
python train.py
```

Esto generará checkpoints en `PokemonRL/checkpoints/` cada 200 episodios:
- `explorer_ep1000.pth`, `explorer_ep2000.pth`, `explorer_ep3000.pth`
- `tactician_ep1000.pth`, `tactician_ep2000.pth`, `tactician_ep3000.pth`

**Progreso del entrenamiento:**
```
Ep 200/3000 | Mapa 1 | R: -25.3 | Avg100: -28.1 | Eps: 0.952
⏱️  Tiempo: 0:05:23 | ETA: 1:15:45 | Ep/s: 0.62
```

### 2. Visualizar el Juego

Una vez entrenado (o usando los checkpoints incluidos), ejecuta la visualización interactiva:

```bash
cd PokemonRL
python visual_play.py
```

Esto abrirá una ventana de Pygame mostrando:
- **Exploración del mapa**: El agente Explorer navegando
- **Combates**: Batallas automáticas con la IA Tactician
- **Estadísticas**: HP, movimientos, tipos, efectividad
- **Log de eventos**: Registro en tiempo real de acciones

**Controles:**
- La IA juega automáticamente
- Cierra la ventana para terminar

### 3. Archivos de Prueba

**Nota**: Los archivos `play.py`, `Juego.py` y `run_boss_battle.py` son scripts experimentales de prueba y no están en mantenimiento activo. Se recomienda usar únicamente `train.py` y `visual_play.py` para la funcionalidad principal del proyecto.

## 📁 Estructura del Proyecto

```
Pokemoncito/
├── PokemonRL/              # Directorio principal del proyecto
│   ├── src/
│   │   ├── agents/         # Agentes de IA
│   │   │   ├── explorer.py     # Agente de exploración (CNN)
│   │   │   ├── tactician.py    # Agente de combate (DQN)
│   │   │   └── strategist.py   # Sistema experto
│   │   ├── env/           # Entorno de simulación
│   │   │   ├── pokemon_env.py      # Entorno principal
│   │   │   ├── battle_engine.py    # Motor de combate
│   │   │   ├── maps.py             # Definiciones de mapas
│   │   │   └── moves_data.py       # Datos de movimientos
│   │   ├── models/        # Arquitecturas de redes neuronales
│   │   │   ├── cnn_map.py          # CNN para mapas
│   │   │   └── dqn_combat.py       # DQN para combate
│   │   └── game_manager.py   # Gestor principal del juego
│   ├── data/
│   │   ├── moves.json      # Base de datos de movimientos
│   │   └── sprites/        # Sprites de Pokémon
│   ├── checkpoints/        # Modelos entrenados
│   ├── train.py           # ⭐ Script principal de entrenamiento
│   ├── visual_play.py     # ⭐ Visualización principal (Pygame)
│   ├── play.py            # [Experimental] Script de prueba
│   ├── run_boss_battle.py # [Experimental] Script de prueba
│   └── Juego.py           # [Experimental] Script de prueba
├── README.md              # Este archivo
├── GPU_SETUP.md           # Guía de configuración GPU
└── MEJORAS_REALIZADAS.md  # Registro de mejoras técnicas
```

## 🛠️ Resolución de Problemas

### Error: "Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False"

**Causa**: Los modelos fueron entrenados en GPU pero estás ejecutando en CPU.

**Solución**: Este error ya fue corregido en la versión actual. Si aún lo ves, actualiza el código:

```python
# En visual_play.py y play.py, los pesos ahora se cargan con:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.load("checkpoints/explorer_ep3000.pth", map_location=device)
```

### Error: "ModuleNotFoundError: No module named 'pygame'"

**Solución**: Instala las dependencias:

```bash
pip install -r PokemonRL/requirements.txt
```

### Error: "FileNotFoundError: checkpoints/explorer_ep3000.pth"

**Causa**: No has entrenado los modelos o no existen los checkpoints.

**Solución**:
1. Entrena los modelos ejecutando `python train.py`
2. O descarga checkpoints pre-entrenados si están disponibles
3. El juego seguirá funcionando con IA aleatoria si no hay checkpoints (verás un mensaje: "⚠ Usando IA aleatoria")

### Entrenamiento Muy Lento

**Solución**:

1. **Usar GPU** (Mejora más significativa):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```
   Consulta [GPU_SETUP.md](GPU_SETUP.md) para más detalles.

2. **Usar menos episodios** (para pruebas rápidas):
   Edita `train.py` y cambia:
   ```python
   total_episodes = 1000  # En lugar de 3000
   ```

3. **Usar Google Colab** (GPU gratis):
   - Sube el proyecto a Google Drive
   - Abre un notebook en [Colab](https://colab.research.google.com/)
   - Activa GPU: Runtime → Change runtime type → GPU
   - Ejecuta el entrenamiento

### La IA Spamea Movimientos de Estado (Leer/Malicioso)

**Causa**: Los modelos no están cargados o están sin entrenar.

**Solución**: 
1. Asegúrate de ver el mensaje "✅ ¡CEREBROS CARGADOS!"
2. Si ves "⚠ Usando IA aleatoria", necesitas entrenar con `python train.py`
3. La IA aleatoria tiende a usar movimientos de estado porque son opciones válidas

### Ventana de Pygame se Cierra Inmediatamente

**Solución**: 
- Verifica que los archivos `data/moves.json` existan
- Ejecuta desde el directorio `PokemonRL/`:
  ```bash
  cd PokemonRL
  python visual_play.py
  ```

### Error: "CUDA out of memory"

**Solución**:
1. Reduce el `batch_size` en `explorer.py` y `tactician.py` (línea ~12)
2. Cierra otros programas que usen GPU
3. Usa CPU en su lugar (es más lento pero funciona)

## 📚 Documentación Adicional

- **[MEJORAS_REALIZADAS.md](MEJORAS_REALIZADAS.md)**: Detalles técnicos completos de todas las mejoras implementadas
- **[GPU_SETUP.md](GPU_SETUP.md)**: Guía detallada para configuración y uso de GPU

## 🎯 Características Técnicas

### Agente Explorer (Exploración de Mapas)
- **Arquitectura**: CNN con 3.47M parámetros
- **Capas**: 32→64→64 convolucionales + 512→256 fully connected
- **Input**: Stack de 9 frames (3 mapas × 3 canales)
- **Output**: 4 acciones (arriba, abajo, izquierda, derecha)
- **Técnicas**: Dropout (0.2), BatchNorm, Gradient Clipping

### Agente Tactician (Combate)
- **Arquitectura**: DQN con 27K parámetros
- **Capas**: 128→128→64 fully connected
- **Input**: Vector de 16 características (HP, stats, habilidades, objetos, niveles)
- **Output**: 5 acciones (4 movimientos + huir)
- **Técnicas**: Dropout (0.3), Experience Replay, Target Networks

### Sistema Strategist (Decisiones de Alto Nivel)
- Selección de Pokémon basada en tipos
- Evaluación de efectividad de movimientos
- Gestión de equipo y curación
- Aplicación de habilidades y objetos

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Haz fork del repositorio
2. Crea una rama para tu feature (`git checkout -b feature/mi-feature`)
3. Commit tus cambios (`git commit -m 'Añadir mi-feature'`)
4. Push a la rama (`git push origin feature/mi-feature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## 🙏 Créditos

- **Datos de Pokémon**: [PokeAPI](https://pokeapi.co/)
- **Framework de RL**: PyTorch
- **Sprites**: Sprites oficiales de Pokémon (solo para uso educativo)

## 📧 Contacto

Para preguntas, sugerencias o reportar problemas, abre un issue en GitHub.

---

**Versión**: 2.0  
**Última actualización**: Enero 2026
