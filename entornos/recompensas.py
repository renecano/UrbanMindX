# ============================================================
# recompensas.py — VERSION 4 (recompensa con fluidez)
#
# CAMBIO PRINCIPAL:
#   La recompensa del semaforo ahora PREMIA la fluidez
#   (autos en movimiento) ademas de penalizar la espera.
#
#   autos_en_movimiento = autos_total - filas_total
#
#   Esto le da al agente un objetivo POSITIVO a perseguir,
#   no solo cosas negativas que evitar. Es el cambio
#   conceptual que no probamos en los intentos anteriores.
# ============================================================

from entrenamiento.config import REWARD_SEMAFORO, REWARD_VEHICULO


# ============================================================
# SEMAFORO
# ============================================================

def recompensa_semaforo_simple(estado: dict) -> float:
    """Version basica: solo penaliza la espera total."""
    return -float(estado["espera_total"]) / 1000.0


def recompensa_semaforo_completa(estado: dict,
                                 cambio_prematuro: bool) -> float:
    """
    Version 4 — premia fluidez ademas de penalizar espera.

    Componentes:
        - penaliza espera total          (lo malo)
        - penaliza filas                 (lo malo)
        - PREMIA autos en movimiento     (lo bueno) ← NUEVO
        - penaliza cambios prematuros    (lo malo)
    """
    cfg = REWARD_SEMAFORO

    # Autos totales y en movimiento
    autos_total = (
        estado["autos_norte"] + estado["autos_sur"]
        + estado["autos_este"] + estado["autos_oeste"]
    )
    autos_en_movimiento = autos_total - estado["filas_total"]

    reward = (
          cfg["peso_espera"] * estado["espera_total"]
        + cfg["peso_filas"]  * estado["filas_total"]
        + 3.0 * autos_en_movimiento          # ← NUEVO: premia fluidez
        + cfg["penalizacion_cambio"] * float(cambio_prematuro)
    ) / 1000.0

    return float(reward)


def recompensa_semaforo_throughput(estado: dict,
                                   autos_cruzaron: int) -> float:
    """Variante que premia directamente el throughput."""
    return (float(autos_cruzaron) * 2.0 - 0.1 * estado["espera_total"]) / 1000.0


# ============================================================
# VEHICULO (sin cambios)
# ============================================================

def recompensa_vehiculo_simple(estado_v: dict) -> float:
    return (
          estado_v["velocidad"] * 0.5
        - estado_v["tiempo_detenido"] * 0.3
    )


def recompensa_vehiculo_completa(estado_v: dict,
                                 frenada_brusca: bool,
                                 colision: bool,
                                 cruzo_en_rojo: bool) -> float:
    cfg = REWARD_VEHICULO
    vel_norm = estado_v["velocidad"] / max(estado_v["velocidad_max"], 1.0)
    distancia_insegura = (
        estado_v["distancia_al_frente"] < cfg["distancia_segura"]
    )
    reward = (
          cfg["peso_avance"]           * vel_norm
        + cfg["peso_tiempo_detenido"]  * estado_v["tiempo_detenido"]
        + cfg["peso_frenada_brusca"]   * float(frenada_brusca)
        + cfg["peso_distancia_corta"]  * float(distancia_insegura)
        + cfg["penalizacion_colision"] * float(colision)
        + cfg["penalizacion_rojo"]     * float(cruzo_en_rojo)
    )
    return float(reward)