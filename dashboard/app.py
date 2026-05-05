# ============================================================
# app.py — Dashboard Streamlit
# Ejecutar con: streamlit run dashboard/app.py
# ============================================================

import os
import pandas as pd
import streamlit as st
from dashboard.graficas import (
    grafica_curva_aprendizaje,
    grafica_comparativa_escenarios,
    grafica_metricas_tiempo_real,
)

st.set_page_config(
    page_title="UrbanMind X",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 UrbanMind X — Sistema Inteligente de Tráfico")
st.caption("Aprendizaje por Refuerzo aplicado a semáforos y vehículos autónomos")

# ---- Tabs principales ----
tab1, tab2, tab3 = st.tabs([
    "📊 Comparativa de Escenarios",
    "📈 Curvas de Aprendizaje",
    "📋 Tabla de Resultados",
])

# ============================================================
# Tab 1 — Comparativa
# ============================================================
with tab1:
    st.header("Comparativa entre Escenarios")
    archivo = "resultados/comparativa.csv"

    if os.path.exists(archivo):
        df = pd.read_csv(archivo)
        grafica_comparativa_escenarios(df)
    else:
        st.warning("Aún no hay resultados. Ejecuta primero: python main.py --modo evaluar")
        # Mostrar datos de ejemplo del documento
        df_ejemplo = pd.DataFrame({
            "nombre":           ["Tradicional", "Semáforo IA", "UrbanMind X Completo"],
            "espera_promedio":  [58, 40, 30],
            "throughput":       [250, 310, 340],
            "frenadas_promedio":[45, 28, 12],
            "velocidad_promedio":[5.2, 7.1, 8.8],
        })
        st.info("Mostrando datos de ejemplo del documento técnico")
        grafica_comparativa_escenarios(df_ejemplo)

# ============================================================
# Tab 2 — Curvas de Aprendizaje
# ============================================================
with tab2:
    st.header("Curvas de Aprendizaje")

    col1, col2 = st.columns(2)
    with col1:
        archivo_s = "resultados/entrenamiento_semaforo.csv"
        if os.path.exists(archivo_s):
            df_s = pd.read_csv(archivo_s)
            grafica_curva_aprendizaje(df_s, "Semáforo Inteligente", "🟢")
        else:
            st.info("Semáforo: aún no entrenado")

    with col2:
        archivo_v = "resultados/entrenamiento_vehiculo.csv"
        if os.path.exists(archivo_v):
            df_v = pd.read_csv(archivo_v)
            grafica_curva_aprendizaje(df_v, "Vehículos Autónomos", "🔵")
        else:
            st.info("Vehículos: aún no entrenados")

# ============================================================
# Tab 3 — Tabla
# ============================================================
with tab3:
    st.header("Tabla Comparativa Final")
    archivo = "resultados/comparativa.csv"

    if os.path.exists(archivo):
        df = pd.read_csv(archivo)

        # Calcular mejora porcentual vs sistema tradicional
        base = df[df["escenario"] == "tradicional"]["espera_promedio"].values
        if len(base) > 0:
            df["mejora_%"] = ((base[0] - df["espera_promedio"]) / base[0] * 100).round(1)

        st.dataframe(
            df[["nombre", "espera_promedio", "throughput",
                "velocidad_promedio", "frenadas_promedio", "mejora_%"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Ejecuta la evaluación primero para ver resultados reales.")
