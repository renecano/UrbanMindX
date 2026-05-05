# ============================================================
# graficas.py — Funciones de visualización con Plotly
# ============================================================

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd


def grafica_comparativa_escenarios(df: pd.DataFrame):
    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(df, x="nombre", y="espera_promedio",
                     title="⏱ Tiempo de Espera Promedio (segundos)",
                     color="nombre", text_auto=True)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(df, x="nombre", y="throughput",
                     title="🚗 Throughput (autos que cruzaron)",
                     color="nombre", text_auto=True)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.bar(df, x="nombre", y="frenadas_promedio",
                     title="🛑 Frenadas Promedio",
                     color="nombre", text_auto=True)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.bar(df, x="nombre", y="velocidad_promedio",
                     title="💨 Velocidad Promedio (m/s)",
                     color="nombre", text_auto=True)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def grafica_curva_aprendizaje(df: pd.DataFrame, titulo: str, emoji: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestep"],
        y=df["recompensa_promedio"],
        mode="lines",
        name="Promedio",
        line=dict(width=2),
    ))
    if "recompensa_min" in df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df["timestep"], df["timestep"][::-1]]),
            y=pd.concat([df["recompensa_max"], df["recompensa_min"][::-1]]),
            fill="toself",
            fillcolor="rgba(100,100,200,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Rango",
        ))
    fig.update_layout(
        title=f"{emoji} Curva de Aprendizaje — {titulo}",
        xaxis_title="Timesteps",
        yaxis_title="Recompensa Promedio",
    )
    st.plotly_chart(fig, use_container_width=True)


def grafica_metricas_tiempo_real(metricas: list):
    """Para uso futuro en modo tiempo real."""
    df = pd.DataFrame(metricas)
    fig = px.line(df, x="tiempo_simulacion", y="espera_total",
                  title="Espera Total en Tiempo Real")
    st.plotly_chart(fig, use_container_width=True)
