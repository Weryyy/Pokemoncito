# Nuevas Funcionalidades - Pokédex RL Trainer

Este documento describe las nuevas funcionalidades implementadas en la aplicación Pokédex RL Trainer.

## 🗺️ Modo Exploración con Mapa Interactivo

### Descripción
Un mapa interactivo real usando Folium que muestra gimnasios Pokémon y Pokémon salvajes en ubicaciones GPS reales alrededor de Madrid.

### Características
- **Mapa Base:** Centrado en Madrid (coordenadas: 40.4168, -3.7038)
- **Gimnasios Pokémon (Marcadores Rojos):**
  - 5 gimnasios distribuidos aleatoriamente
  - Cada gimnasio tiene un tipo asociado (parque→grass, agua→water, ciudad→electric, montaña→rock, desierto→fire)
  - Click para iniciar batalla
  
- **Pokémon Salvajes (Marcadores Verdes):**
  - 10 ubicaciones de Pokémon salvajes
  - Tipos aleatorios
  - Click para capturar/batallar

### Cómo Usar
1. Selecciona "Modo Exploración" en la barra lateral
2. Opcional: Inicializa el sistema y carga un modelo entrenado
3. Haz click en cualquier marcador del mapa
4. El sistema iniciará automáticamente una batalla
5. Cambia a "Modo Visualización" para ver la batalla

### Integración con el Juego
- Al hacer click en un marcador, `st.session_state.mode` cambia de 'MAP' a 'COMBAT'
- El enemigo se configura automáticamente según el tipo del lugar
- La interfaz se actualiza para mostrar la pantalla de combate

## 🧠 IA Explicable (XAI) - Análisis de Decisiones

### Descripción
Visualización de los Q-Values (valores de utilidad) que el agente considera al tomar decisiones en combate.

### Características
- **Gráfico de Barras:** Compara los Q-Values de los 4 movimientos disponibles
- **Código de Colores:**
  - 🟨 Oro: Mejor movimiento (el elegido)
  - 🟦 Azul: Otros movimientos
  
- **Explicación Textual Automática:**
  ```
  El agente eligió [Movimiento A] porque su valor esperado (Q) es X,
  mientras que [Movimiento B] es Y.
  ```

### Cómo Ver
1. Inicializa el entorno en "Modo Visualización"
2. Carga un modelo entrenado
3. Ejecuta pasos hasta entrar en combate
4. En la sección "📜 Registro de Batalla", verás:
   - Gráfico de barras con Q-Values
   - Explicación textual de la decisión

### Detalles Técnicos
- Los Q-Values solo se muestran cuando el agente está en modo "explotación" (epsilon bajo)
- Durante exploración aleatoria, se muestra un aviso
- Usa Altair para gráficos interactivos

## 📊 Dashboard de Estadísticas

### Descripción
Panel completo para analizar el rendimiento de los agentes con datos geoespaciales.

### Características

#### 1. Resumen General
- Total de episodios
- Reward promedio
- Total de victorias
- Tasa de victoria (%)

#### 2. Mapa de Calor
- Visualiza ubicaciones de batallas
- 🟢 Verde: Victorias
- 🔴 Rojo: Derrotas
- Mapa interactivo con popups informativos

#### 3. Gráfico de Dispersión
- **Eje X:** Distancia desde el centro (km)
- **Eje Y:** Reward total
- **Color:** Victoria (verde) / Derrota (rojo)
- Tooltip con información detallada

#### 4. Estadísticas por Distancia
Tabla con 5 categorías de distancia:
- Muy Cerca
- Cerca
- Medio
- Lejos
- Muy Lejos

Para cada categoría:
- Número de victorias
- Total de batallas
- Win Rate (%)
- Reward promedio

#### 5. Análisis Temporal
- **Gráfico 1:** Evolución del reward por episodio
- **Gráfico 2:** Tasa de victoria (media móvil)

### Cómo Usar
1. Entrena el modelo en "Modo Entrenamiento"
2. Selecciona "Dashboard Estadísticas" en la barra lateral
3. Explora las diferentes visualizaciones

### Datos GPS
- Se generan automáticamente durante el entrenamiento
- Coordenadas aleatorias alrededor del centro de Madrid
- Rango: ±0.1 grados (~11 km)
- Pueden reemplazarse con GPS real en futuras versiones

## 🔧 Cambios Técnicos

### Dependencias Nuevas
```
folium - Mapas interactivos
streamlit-folium - Integración Folium-Streamlit
altair - Gráficos interactivos
```

### Modificaciones en `tactician.py`
```python
# Antes
action = tactician.select_action(state)

# Ahora (compatible con versión anterior)
action = tactician.select_action(state)  # Sigue funcionando
action, q_values = tactician.select_action(state, return_q_values=True)  # Nueva opción
```

### Nuevos Campos en `session_state`
- `gps_coords`: Lista de coordenadas (lat, lon) por episodio
- `wins`: Lista de booleanos indicando victoria/derrota
- `selected_marker`: Último marcador clickeado en el mapa
- `last_q_values`: Q-values del último movimiento (para XAI)
- `map_mode`: Estado del modo de mapa ('MAP' o 'COMBAT')

## 📋 Requisitos del Sistema

- Python 3.8+
- Todas las dependencias en `requirements.txt`
- Conexión a internet (para tiles de mapa)
- Navegador web moderno

## 🎮 Flujo de Uso Recomendado

1. **Entrenar:** Ejecuta entrenamiento para generar datos
2. **Explorar:** Usa el mapa para seleccionar ubicaciones
3. **Visualizar:** Observa batallas con explicaciones XAI
4. **Analizar:** Revisa estadísticas en el dashboard

## 🐛 Solución de Problemas

### El mapa no se carga
- Verifica conexión a internet
- Comprueba que folium esté instalado: `pip install folium`

### No hay datos en el Dashboard
- Primero ejecuta un entrenamiento en "Modo Entrenamiento"
- Los datos se generan durante el entrenamiento

### Q-Values no aparecen
- Asegúrate de estar en "Modo Visualización"
- Verifica que el modelo esté cargado
- Los Q-Values solo aparecen en modo combate
- Durante exploración aleatoria no hay Q-Values (decisión aleatoria)

## 🚀 Próximas Mejoras Sugeridas

- [ ] Integración con GPS real del dispositivo
- [ ] Más tipos de ubicaciones (bosque, cueva, playa, etc.)
- [ ] Historial de batallas por ubicación
- [ ] Exportar estadísticas a CSV/JSON
- [ ] Comparación entre diferentes modelos entrenados
- [ ] Filtros temporales en el dashboard
- [ ] Análisis de efectividad de tipos

## 📖 Referencias

- [Folium Documentation](https://python-visualization.github.io/folium/)
- [Streamlit-Folium](https://github.com/randyzwitch/streamlit-folium)
- [Altair Documentation](https://altair-viz.github.io/)
- [Q-Learning y Q-Values](https://en.wikipedia.org/wiki/Q-learning)

---

**Versión:** 2.0  
**Última actualización:** 2026-01-19  
**Autor:** Pokédex RL Team
