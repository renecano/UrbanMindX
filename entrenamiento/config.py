# ============================================================
# config.py — Persona 3
# Todos los hiperparámetros del proyecto en un solo lugar.
# Cambiar valores aquí afecta todo el sistema.
# ============================================================

# ------------------------------------------------------------
# Hiperparámetros PPO — Semáforo
# ------------------------------------------------------------
PPO_SEMAFORO = {
    "learning_rate": 0.0003,   # qué tan rápido aprende la red
    "gamma":         0.99,     # descuento de recompensas futuras (0-1)
    "n_steps":       2048,     # pasos antes de cada actualización
    "batch_size":    64,       # muestras por mini-batch
    "ent_coef":      0.01,     # coeficiente de entropía (exploración)
    "n_epochs":      10,       # épocas por actualización
    "clip_range":    0.2,      # clip de PPO (no cambiar sin razón)
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
    "ent_coef":      0.005,    # menos exploración que el semáforo
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
# Rutas de archivos
# ------------------------------------------------------------
SUMO_CONFIG      = "sumo/config/simulacion.sumocfg"
MODELO_SEMAFORO  = "modelos/semaforo"
MODELO_VEHICULO  = "modelos/vehiculo"
RESULTADOS_DIR   = "resultados/"

# ------------------------------------------------------------
# Parámetros de la simulación
# ------------------------------------------------------------
PASO_SIMULACION  = 1.0    # segundos por step (no cambiar)
MAX_PASOS = 21600   # 6 horas × 3600 segundos
PUERTO_SUMO      = 8813   # puerto TCP de TraCI

# ------------------------------------------------------------
# Pesos de la función de recompensa — Semáforo
# ------------------------------------------------------------
REWARD_SEMAFORO = {
    "peso_espera":          -0.5,
    "peso_filas":           -0.3,
    "penalizacion_cambio":  -5.0,   # cambiar fase antes de N segundos
    "min_tiempo_fase":       10,    # segundos mínimos antes de cambiar
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
