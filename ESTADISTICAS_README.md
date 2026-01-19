# 📊 Sistema de Estadísticas de Pokemoncito

## Descripción

El nuevo sistema de estadísticas de Pokemoncito permite trackear y analizar todas tus partidas con un dashboard interactivo lleno de gráficas y KPIs.

## ¿Qué se Registra?

### Estadísticas de Combate
- **Batallas totales**: Ganadas y perdidas
- **Daño**: Total infligido y recibido
- **Win Rate**: Porcentaje de victorias
- **Golpes críticos**: Número total de hits críticos
- **Estados alterados**: Parálisis, quemadura, veneno, etc.

### Progresión
- **Mapas completados**: Con tiempo por mapa
- **Experiencia ganada**: Total de XP acumulada
- **Subidas de nivel**: Cantidad de level ups
- **Duración de partida**: Tiempo total jugado

### Movimientos y Pokémon
- **Movimientos más usados**: Frecuencia y efectividad
- **Daño promedio por movimiento**
- **Equipo Pokémon usado**: Niveles, stats finales
- **Log de eventos**: Historia detallada de la partida

## Cómo Usar

### 1. Jugar una Partida

Las estadísticas se guardan automáticamente al completar una partida (llegar al boss final):

```bash
cd PokemonRL
python visual_play.py
```

Al derrotar al líder de gimnasio final, las estadísticas se guardarán automáticamente en:
```
game_statistics/
├── 2026-01-19/
│   ├── game_abc12345_180230.json
│   └── game_def67890_184512.json
└── 2026-01-20/
    └── game_ghi09876_103045.json
```

### 2. Ver el Dashboard

Ejecuta el dashboard de Streamlit:

```bash
# Desde la raíz del proyecto
streamlit run streamlit_app.py
```

El dashboard se abrirá en tu navegador en `http://localhost:8501`

### 3. Explorar las Estadísticas

El dashboard tiene 4 secciones principales:

#### 📈 Resumen General
- KPIs globales de todas las partidas
- Gráficas de evolución de win rate
- Daño infligido vs recibido
- Top 10 movimientos más usados
- Top 10 Pokémon más usados

#### 🎯 Partida Individual
- Análisis detallado de una partida específica
- Tiempo por mapa completado
- Efectividad de movimientos
- Equipo Pokémon usado
- Log de eventos cronológico

#### 📊 Análisis Comparativo
- Comparación entre múltiples partidas
- Gráfica de radar multidimensional
- Tabla comparativa completa
- Tendencias de rendimiento

#### 🏆 Rankings
- Mejor Win Rate
- Mayor daño infligido
- Más golpes críticos
- Partida más rápida
- Más experiencia ganada
- Más subidas de nivel

## Estructura de Datos

Cada partida se guarda en formato JSON con la siguiente estructura:

```json
{
  "game_id": "abc12345",
  "start_time": "2026-01-19T18:02:30",
  "end_time": "2026-01-19T18:45:12",
  "duration_seconds": 2562,
  "total_damage_dealt": 15420,
  "total_damage_received": 8350,
  "battles_won": 45,
  "battles_lost": 3,
  "win_rate": 0.9375,
  "maps_completed": [0, 1, 2, 3, 4],
  "time_per_map": {
    "0": 180.5,
    "1": 245.2,
    "2": 312.8,
    "3": 425.1,
    "4": 398.4
  },
  "moves_used": {
    "thunderbolt": 87,
    "flamethrower": 65,
    "earthquake": 42
  },
  "pokemon_used": ["Pikachu", "Charizard", "Blastoise"],
  "critical_hits": 23,
  "level_ups": 18
}
```

## Personalización

### Filtros
- **Por fecha**: Selecciona fechas específicas para analizar
- **Multi-selección**: Compara varias fechas a la vez

### Exportación
Los archivos JSON se pueden:
- Compartir con otros jugadores
- Importar en otras herramientas de análisis
- Procesar con scripts personalizados

## Tips y Consejos

1. **Completa partidas hasta el final**: Las estadísticas solo se guardan al derrotar al boss final
2. **Juega varias partidas**: El dashboard es más útil con múltiples partidas para comparar
3. **Experimenta con diferentes estrategias**: Compara tu rendimiento con diferentes equipos Pokémon
4. **Analiza tus movimientos**: Descubre qué movimientos son más efectivos en combate

## Solución de Problemas

### No aparecen partidas en el dashboard
- Asegúrate de haber completado al menos una partida hasta el boss final
- Verifica que existe la carpeta `game_statistics/` en la raíz del proyecto
- Recarga el dashboard después de jugar una nueva partida

### Errores al ejecutar el dashboard
```bash
# Instala las dependencias necesarias
cd PokemonRL
pip install -r requirements.txt
```

### Las gráficas no se ven bien
- Usa un navegador moderno (Chrome, Firefox, Edge)
- Asegúrate de tener Plotly instalado: `pip install plotly`

## Mejoras Futuras

Posibles características a añadir:
- Exportación de reportes en PDF
- Comparación con partidas de otros jugadores
- Predicción de win rate basada en historial
- Análisis de efectividad por tipo de Pokémon
- Heatmaps de exploración de mapas

## Créditos

Sistema de estadísticas implementado como parte de Pokemoncito RL v2.1
