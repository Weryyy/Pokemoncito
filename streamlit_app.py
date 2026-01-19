"""
Dashboard de Estadísticas de Pokemoncito
Visualización de KPIs y estadísticas de partidas
"""
import streamlit as st
import json
import os
import sys
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add PokemonRL to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PokemonRL'))

from PokemonRL.src.utils.game_statistics import StatisticsManager

# Page config
st.set_page_config(
    page_title="📊 Pokemoncito - Dashboard de Estadísticas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.95);
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #FFD700;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    h1, h2, h3 {
        color: #FFD700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .stat-card {
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.1), rgba(255, 215, 0, 0.05));
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #FFD700;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1 style='font-size: 48px; color: #FFD700; text-shadow: 3px 3px 6px #000;'>
        📊 POKEMONCITO - DASHBOARD DE ESTADÍSTICAS
    </h1>
    <p style='color: #FFF; font-size: 18px;'>Análisis de Partidas y KPIs</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("""
<div style='text-align: center; padding: 20px 0;'>
    <h2 style='color: #FFD700;'>📁 NAVEGACIÓN</h2>
</div>
""", unsafe_allow_html=True)

# Load games
base_dir = os.path.join(os.path.dirname(__file__), 'game_statistics')
games = StatisticsManager.load_all_games(base_dir)

if not games:
    st.warning("⚠️ No hay partidas guardadas aún. Juega una partida con `visual_play.py` para ver estadísticas.")
    st.info("""
    **¿Cómo generar estadísticas?**
    
    1. Ejecuta el juego: `cd PokemonRL && python visual_play.py`
    2. Completa una partida hasta el boss final
    3. Las estadísticas se guardarán automáticamente
    4. Recarga este dashboard para ver los datos
    """)
    st.stop()

# Sidebar filters
st.sidebar.markdown("### 🎮 Filtros")

# Date folders
date_folders = sorted(list(set([g['date_folder'] for g in games])), reverse=True)
selected_dates = st.sidebar.multiselect(
    "Seleccionar fechas:",
    options=date_folders,
    default=date_folders[:1] if date_folders else []
)

# Filter games
if selected_dates:
    filtered_games = [g for g in games if g['date_folder'] in selected_dates]
else:
    filtered_games = games

st.sidebar.markdown(f"**Total partidas:** {len(filtered_games)}")

# Navigation
page = st.sidebar.radio(
    "Sección:",
    ["📈 Resumen General", "🎯 Partida Individual", "📊 Análisis Comparativo", "🏆 Rankings"]
)

# ========== RESUMEN GENERAL ==========
if page == "📈 Resumen General":
    st.header("📈 Resumen General de Partidas")
    
    if not filtered_games:
        st.warning("No hay partidas en el rango seleccionado")
        st.stop()
    
    # Summary stats
    summary = StatisticsManager.get_summary_stats(filtered_games)
    
    # KPIs principales
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "🎮 Total Partidas",
            f"{summary['total_games']}"
        )
    
    with col2:
        st.metric(
            "⚔️ Batallas Totales",
            f"{summary['total_battles']}",
            delta=f"W: {summary['total_wins']} / L: {summary['total_losses']}"
        )
    
    with col3:
        st.metric(
            "📊 Win Rate Promedio",
            f"{summary['avg_win_rate']*100:.1f}%"
        )
    
    with col4:
        st.metric(
            "💥 Daño Total",
            f"{summary['total_damage_dealt']:,.0f}",
            delta=f"-{summary['total_damage_received']:,.0f} recibido"
        )
    
    with col5:
        st.metric(
            "⭐ Experiencia Total",
            f"{summary['total_exp_gained']:,.0f}"
        )
    
    # Segunda fila de KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🗺️ Mapas Completados", f"{summary['total_maps_completed']}")
    
    with col2:
        avg_duration_min = summary['avg_duration'] / 60
        st.metric("⏱️ Duración Promedio", f"{avg_duration_min:.1f} min")
    
    with col3:
        st.metric("💫 Golpes Críticos", f"{summary['total_critical_hits']}")
    
    with col4:
        st.metric("📈 Subidas de Nivel", f"{summary['total_level_ups']}")
    
    st.markdown("---")
    
    # Gráficas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Evolución del Win Rate")
        
        # Win rate por partida
        win_rates = [g['win_rate'] * 100 for g in filtered_games]
        game_ids = [f"Game {i+1}" for i in range(len(filtered_games))]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=game_ids,
            y=win_rates,
            mode='lines+markers',
            name='Win Rate',
            line=dict(color='#FFD700', width=3),
            marker=dict(size=10)
        ))
        fig.update_layout(
            title="Win Rate por Partida",
            xaxis_title="Partida",
            yaxis_title="Win Rate (%)",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚔️ Batallas Ganadas vs Perdidas")
        
        wins = [g['battles_won'] for g in filtered_games]
        losses = [g['battles_lost'] for g in filtered_games]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Victorias',
            x=game_ids,
            y=wins,
            marker_color='#00FF00'
        ))
        fig.add_trace(go.Bar(
            name='Derrotas',
            x=game_ids,
            y=losses,
            marker_color='#FF0000'
        ))
        fig.update_layout(
            title="Victorias vs Derrotas por Partida",
            xaxis_title="Partida",
            yaxis_title="Cantidad",
            barmode='group',
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tercera fila de gráficas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💥 Daño Infligido y Recibido")
        
        damage_dealt = [g['total_damage_dealt'] for g in filtered_games]
        damage_received = [g['total_damage_received'] for g in filtered_games]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Daño Infligido',
            x=game_ids,
            y=damage_dealt,
            marker_color='#FFA500'
        ))
        fig.add_trace(go.Bar(
            name='Daño Recibido',
            x=game_ids,
            y=damage_received,
            marker_color='#FF6347'
        ))
        fig.update_layout(
            title="Daño por Partida",
            xaxis_title="Partida",
            yaxis_title="Daño Total",
            barmode='group',
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⏱️ Duración de Partidas")
        
        durations = [g['duration_seconds'] / 60 for g in filtered_games]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=game_ids,
            y=durations,
            mode='lines+markers',
            name='Duración',
            line=dict(color='#4169E1', width=3),
            marker=dict(size=10),
            fill='tozeroy'
        ))
        fig.update_layout(
            title="Duración de Partidas (minutos)",
            xaxis_title="Partida",
            yaxis_title="Minutos",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Movimientos más usados
    st.markdown("---")
    st.subheader("🎯 Top 10 Movimientos Más Usados")
    
    most_used_moves = StatisticsManager.get_most_used_moves(filtered_games, top_n=10)
    
    if most_used_moves:
        moves_df = pd.DataFrame(most_used_moves, columns=['Movimiento', 'Usos'])
        
        fig = px.bar(
            moves_df,
            x='Usos',
            y='Movimiento',
            orientation='h',
            title="Movimientos Más Utilizados",
            color='Usos',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Pokemon más usados
    st.markdown("---")
    st.subheader("🌟 Top 10 Pokémon Más Usados")
    
    most_used_pokemon = StatisticsManager.get_most_used_pokemon(filtered_games, top_n=10)
    
    if most_used_pokemon:
        pokemon_df = pd.DataFrame(most_used_pokemon, columns=['Pokémon', 'Partidas'])
        
        fig = px.pie(
            pokemon_df,
            values='Partidas',
            names='Pokémon',
            title="Distribución de Pokémon Usados",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

# ========== PARTIDA INDIVIDUAL ==========
elif page == "🎯 Partida Individual":
    st.header("🎯 Análisis de Partida Individual")
    
    if not filtered_games:
        st.warning("No hay partidas en el rango seleccionado")
        st.stop()
    
    # Selector de partida
    game_options = [f"{g['game_id']} - {g['start_time'][:19]}" for g in filtered_games]
    selected_game_idx = st.selectbox("Seleccionar partida:", range(len(filtered_games)), 
                                      format_func=lambda x: game_options[x])
    
    game = filtered_games[selected_game_idx]
    
    # Info general
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📅 Información General")
        st.write(f"**ID:** {game['game_id']}")
        st.write(f"**Inicio:** {game['start_time'][:19]}")
        st.write(f"**Fin:** {game['end_time'][:19]}")
        st.write(f"**Duración:** {game['duration_seconds']/60:.1f} min")
    
    with col2:
        st.markdown("### ⚔️ Estadísticas de Combate")
        st.write(f"**Batallas:** {game['battles_won'] + game['battles_lost']}")
        st.write(f"**Victorias:** {game['battles_won']}")
        st.write(f"**Derrotas:** {game['battles_lost']}")
        st.write(f"**Win Rate:** {game['win_rate']*100:.1f}%")
    
    with col3:
        st.markdown("### 💪 Rendimiento")
        st.write(f"**Daño Total:** {game['total_damage_dealt']:,.0f}")
        st.write(f"**Daño Recibido:** {game['total_damage_received']:,.0f}")
        st.write(f"**Críticos:** {game['critical_hits']}")
        st.write(f"**Nivel Ups:** {game['level_ups']}")
    
    st.markdown("---")
    
    # Mapas completados
    st.subheader("🗺️ Progresión de Mapas")
    
    if game['maps_completed']:
        maps_data = []
        for map_idx in game['maps_completed']:
            time_taken = game['time_per_map'].get(str(map_idx), 0)
            maps_data.append({
                'Mapa': f"Mapa {map_idx + 1}",
                'Tiempo (seg)': time_taken
            })
        
        maps_df = pd.DataFrame(maps_data)
        
        fig = px.bar(
            maps_df,
            x='Mapa',
            y='Tiempo (seg)',
            title="Tiempo por Mapa",
            color='Tiempo (seg)',
            color_continuous_scale='Sunset'
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No se completaron mapas en esta partida")
    
    # Movimientos usados
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Movimientos Utilizados")
        
        if game['moves_used']:
            moves_df = pd.DataFrame([
                {'Movimiento': move, 'Usos': count}
                for move, count in sorted(game['moves_used'].items(), 
                                         key=lambda x: x[1], reverse=True)[:10]
            ])
            
            fig = px.bar(
                moves_df,
                x='Movimiento',
                y='Usos',
                title="Top 10 Movimientos",
                color='Usos',
                color_continuous_scale='Blues'
            )
            fig.update_layout(template="plotly_dark", height=400)
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💥 Efectividad de Movimientos")
        
        if game['move_effectiveness']:
            effectiveness_data = []
            for move, stats in game['move_effectiveness'].items():
                if stats['hits'] > 0:
                    avg_damage = stats['damage'] / stats['hits']
                    effectiveness_data.append({
                        'Movimiento': move,
                        'Daño Promedio': avg_damage,
                        'Total Hits': stats['hits']
                    })
            
            if effectiveness_data:
                eff_df = pd.DataFrame(effectiveness_data).sort_values('Daño Promedio', ascending=False).head(10)
                
                fig = px.scatter(
                    eff_df,
                    x='Total Hits',
                    y='Daño Promedio',
                    size='Daño Promedio',
                    text='Movimiento',
                    title="Daño Promedio vs Usos",
                    color='Daño Promedio',
                    color_continuous_scale='Reds'
                )
                fig.update_traces(textposition='top center')
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    # Equipo Pokemon
    st.markdown("---")
    st.subheader("🌟 Equipo Pokémon")
    
    if game['pokemon_stats']:
        pokemon_data = []
        for name, stats in game['pokemon_stats'].items():
            pokemon_data.append({
                'Pokémon': name,
                'Nivel': stats['level'],
                'HP': stats['hp'],
                'Tipos': ', '.join(stats['types']),
                'Habilidad': stats.get('ability', 'N/A'),
                'Objeto': stats.get('held_item', 'N/A')
            })
        
        pokemon_df = pd.DataFrame(pokemon_data)
        st.dataframe(pokemon_df, use_container_width=True)
    
    # Progresión de eventos
    st.markdown("---")
    st.subheader("📜 Log de Eventos (Últimos 20)")
    
    if game['progression_log']:
        for event in game['progression_log'][-20:]:
            event_type = event['type']
            timestamp = event['timestamp']
            data = event['data']
            
            if event_type == 'battle_start':
                st.info(f"⚔️ [{timestamp[11:19]}] Batalla vs {data.get('enemy', 'Unknown')} (Nv. {data.get('level', '?')})")
            elif event_type == 'battle_end':
                if data.get('won'):
                    st.success(f"✅ [{timestamp[11:19]}] Victoria! (+{data.get('exp', 0)} XP)")
                else:
                    st.error(f"❌ [{timestamp[11:19]}] Derrota")
            elif event_type == 'level_up':
                st.success(f"⭐ [{timestamp[11:19]}] {data.get('pokemon', 'Pokémon')} subió a nivel {data.get('level', '?')}!")
            elif event_type == 'map_start':
                st.info(f"🗺️ [{timestamp[11:19]}] Iniciando Mapa {data.get('map', 0) + 1}")
            elif event_type == 'map_complete':
                st.success(f"🏆 [{timestamp[11:19]}] Mapa {data.get('map', 0) + 1} completado ({data.get('time', 0):.1f}s)")

# ========== ANÁLISIS COMPARATIVO ==========
elif page == "📊 Análisis Comparativo":
    st.header("📊 Análisis Comparativo de Partidas")
    
    if len(filtered_games) < 2:
        st.warning("Se necesitan al menos 2 partidas para comparar")
        st.stop()
    
    # Comparación de métricas clave
    st.subheader("📈 Tendencias Generales")
    
    # Preparar datos
    comparison_data = []
    for i, game in enumerate(filtered_games):
        comparison_data.append({
            'Partida': f"Game {i+1}",
            'Win Rate (%)': game['win_rate'] * 100,
            'Batallas': game['battles_won'] + game['battles_lost'],
            'Daño Infligido': game['total_damage_dealt'],
            'Daño Recibido': game['total_damage_received'],
            'Críticos': game['critical_hits'],
            'Duración (min)': game['duration_seconds'] / 60,
            'XP Ganada': game['total_exp_gained']
        })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Gráfica de radar
    st.subheader("🎯 Comparación Multi-dimensional")
    
    if len(filtered_games) <= 5:
        # Normalizar datos para el radar
        metrics = ['Win Rate (%)', 'Batallas', 'Daño Infligido', 'Críticos', 'XP Ganada']
        
        fig = go.Figure()
        
        for i, game in enumerate(filtered_games[:5]):
            values = [
                game['win_rate'] * 100,
                (game['battles_won'] + game['battles_lost']) / max(1, comp_df['Batallas'].max()) * 100,
                game['total_damage_dealt'] / max(1, comp_df['Daño Infligido'].max()) * 100,
                game['critical_hits'] / max(1, comp_df['Críticos'].max()) * 100,
                game['total_exp_gained'] / max(1, comp_df['XP Ganada'].max()) * 100
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Cerrar el polígono
                theta=metrics + [metrics[0]],
                name=f"Game {i+1}",
                fill='toself'
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            template="plotly_dark",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabla comparativa
    st.subheader("📋 Tabla Comparativa Completa")
    st.dataframe(comp_df, use_container_width=True)

# ========== RANKINGS ==========
elif page == "🏆 Rankings":
    st.header("🏆 Rankings y Records")
    
    if not filtered_games:
        st.warning("No hay partidas en el rango seleccionado")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🥇 Mejor Win Rate")
        best_wr = max(filtered_games, key=lambda x: x['win_rate'])
        st.markdown(f"""
        <div class='stat-card'>
            <h3>🎮 {best_wr['game_id']}</h3>
            <h2 style='color: #00FF00;'>{best_wr['win_rate']*100:.1f}%</h2>
            <p>{best_wr['battles_won']}W / {best_wr['battles_lost']}L</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("💪 Mayor Daño Infligido")
        best_dmg = max(filtered_games, key=lambda x: x['total_damage_dealt'])
        st.markdown(f"""
        <div class='stat-card'>
            <h3>🎮 {best_dmg['game_id']}</h3>
            <h2 style='color: #FFA500;'>{best_dmg['total_damage_dealt']:,.0f}</h2>
            <p>Daño total</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("⚡ Más Golpes Críticos")
        best_crit = max(filtered_games, key=lambda x: x['critical_hits'])
        st.markdown(f"""
        <div class='stat-card'>
            <h3>🎮 {best_crit['game_id']}</h3>
            <h2 style='color: #FFD700;'>{best_crit['critical_hits']}</h2>
            <p>Golpes críticos</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("⏱️ Partida Más Rápida")
        fastest = min(filtered_games, key=lambda x: x['duration_seconds'])
        st.markdown(f"""
        <div class='stat-card'>
            <h3>🎮 {fastest['game_id']}</h3>
            <h2 style='color: #4169E1;'>{fastest['duration_seconds']/60:.1f} min</h2>
            <p>Duración</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📈 Más Experiencia Ganada")
        best_xp = max(filtered_games, key=lambda x: x['total_exp_gained'])
        st.markdown(f"""
        <div class='stat-card'>
            <h3>🎮 {best_xp['game_id']}</h3>
            <h2 style='color: #9370DB;'>{best_xp['total_exp_gained']:,.0f} XP</h2>
            <p>Experiencia total</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🌟 Más Subidas de Nivel")
        best_lvl = max(filtered_games, key=lambda x: x['level_ups'])
        st.markdown(f"""
        <div class='stat-card'>
            <h3>🎮 {best_lvl['game_id']}</h3>
            <h2 style='color: #32CD32;'>{best_lvl['level_ups']}</h2>
            <p>Subidas de nivel</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background: rgba(255,215,0,0.1); padding: 15px; border-radius: 10px; border: 2px solid #FFD700;'>
    <h4 style='color: #FFD700; text-align: center;'>📊 Dashboard v1.0</h4>
    <p style='font-size: 12px; text-align: center;'>
        Sistema de análisis de estadísticas<br>
        para Pokemoncito RL
    </p>
</div>
""", unsafe_allow_html=True)
