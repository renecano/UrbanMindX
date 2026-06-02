# ============================================================
# entrenamiento/config.py — UrbanMind X
# Todos los hiperparámetros del proyecto en un solo lugar.
# ============================================================

# ------------------------------------------------------------
# Hiperparámetros PPO — Semáforo
# ------------------------------------------------------------
PPO_SEMAFORO = {
    "learning_rate": 0.0003,
    "gamma":         0.99,
    "n_steps":       2048,
    "batch_size":    64,
    "ent_coef":      0.01,
    "n_epochs":      10,
    "clip_range":    0.2,
    "verbose":       1,
}

# ------------------------------------------------------------
# Hiperparámetros PPO — Vehículo
# ------------------------------------------------------------
PPO_VEHICULO = {
    "learning_rate": 0.0003,
    "gamma":         0.99,
    "n_steps":       2048,
    "batch_size":    64,
    "ent_coef":      0.005,
    "n_epochs":      10,
    "clip_range":    0.2,
    "verbose":       1,
}

# ------------------------------------------------------------
# Duración del entrenamiento
# ------------------------------------------------------------
TIMESTEPS_SEMAFORO = 500_000
TIMESTEPS_VEHICULO = 500_000
TIMESTEPS_CONJUNTO = 200_000

# ------------------------------------------------------------
# Rutas de archivos SUMO (relativas a la raíz del proyecto)
# ------------------------------------------------------------
SUMO_CONFIG            = "sumo/config/simulacion.sumocfg"      # perfil normal (6h)
SUMO_CONFIG_HORA_PICO  = "sumo/config/hora_pico.sumocfg"       # hora pico (2h)
SUMO_CONFIG_DESBAL     = "sumo/config/desbalanceado.sumocfg"   # desbalanceado (6h)

MODELO_SEMAFORO  = "modelos/semaforo"
MODELO_VEHICULO  = "modelos/vehiculo"
RESULTADOS_DIR   = "resultados/"

# ------------------------------------------------------------
# Parámetros de la simulación
# ------------------------------------------------------------
PASO_SIMULACION  = 1.0       # segundos por step (no cambiar — alineado con step-length en .sumocfg)
MAX_PASOS        = 21600     # 6 horas × 3600 s — alineado con simulacion.sumocfg y desbalanceado.sumocfg
MAX_PASOS_PICO   = 7200      # 2 horas × 3600 s — alineado con hora_pico.sumocfg
PUERTO_SUMO      = 8813

# ------------------------------------------------------------
# Pesos de la función de recompensa — Semáforo
# ------------------------------------------------------------
REWARD_SEMAFORO = {
    "peso_espera":          -0.5,
    "peso_filas":           -0.3,
    "penalizacion_cambio":  -10.0,
    "min_tiempo_fase":       15,    # segundos mínimos antes de permitir cambio
}

# ------------------------------------------------------------
# Pesos de la función de recompensa — Vehículo
# ------------------------------------------------------------
REWARD_VEHICULO = {
    "peso_avance":          +1.0,
    "peso_tiempo_detenido": -0.5,
    "peso_frenada_brusca":  -2.0,
    "peso_distancia_corta": -3.0,
    "penalizacion_colision":-10.0,
    "penalizacion_rojo":    -8.0,
    "distancia_segura":      5.0,   # metros mínimos al auto de enfrente
}
