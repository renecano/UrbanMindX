# ============================================================
# recompensas.py — Persona 2
# Funciones de recompensa para ambos agentes.
# Separadas del entorno para facilitar experimentación.
# Cambiar pesos aquí sin tocar el entorno.
# ============================================================

from entrenamiento.config import REWARD_SEMAFORO, REWARD_VEHICULO


# ------------------------------------------------------------
# Recompensas del Semáforo
# ------------------------------------------------------------

def recompensa_semaforo_simple(estado: dict) -> float:
    """
    Versión básica: solo penaliza autos esperando.
    Usar al inicio del entrenamiento para verificar que aprende algo.
    """
    return -float(estado["espera_total"])


def recompensa_semaforo_completa(estado: dict, cambio_prematuro: bool) -> float:
    """
    Versión completa con todos los componentes.
    Usar cuando la versión simple ya converge.

    Args:
        estado:           dict de get_estado_interseccion()
        cambio_prematuro: True si el agente cambió fase antes del mínimo
    """
    cfg = REWARD_SEMAFORO

    r = (
        cfg["peso_espera"] * estado["espera_total"]
      + cfg["peso_filas"]  * estado["filas_total"]
      + cfg["penalizacion_cambio"] * int(cambio_prematuro)
    )
    return float(r)


def recompensa_semaforo_throughput(estado: dict, autos_cruzaron: int) -> float:
    """
    Variante que premia directamente el throughput.
    Experimentar si las versiones anteriores no convergen bien.
    """
    return float(autos_cruzaron) - 0.1 * estado["espera_total"]


# ------------------------------------------------------------
# Recompensas del Vehículo
# ------------------------------------------------------------

def recompensa_vehiculo_simple(estado_v: dict) -> float:
    """
    Versión básica: premia velocidad, penaliza estar detenido.
    """
    return estado_v["velocidad"] - estado_v["tiempo_detenido"] * 0.5


def recompensa_vehiculo_completa(estado_v: dict,
                                  frenada_brusca: bool,
                                  colision: bool,
                                  cruzo_en_rojo: bool) -> float:
    """
    Versión completa para el vehículo autónomo.

    Args:
        estado_v:      dict de get_estado_vehiculo()
        frenada_brusca: True si la aceleración fue < umbral negativo
        colision:       True si distancia_al_frente < distancia_segura
        cruzo_en_rojo:  True si avanzó con semáforo en rojo
    """
    cfg = REWARD_VEHICULO

    distancia_insegura = (
        estado_v["distancia_al_frente"] < cfg["distancia_segura"]
    )

    r = (
        cfg["peso_avance"]          * (estado_v["velocidad"] / estado_v["velocidad_max"])
      + cfg["peso_tiempo_detenido"] * estado_v["tiempo_detenido"]
      + cfg["peso_frenada_brusca"]  * int(frenada_brusca)
      + cfg["peso_distancia_corta"] * int(distancia_insegura)
      + cfg["penalizacion_colision"]* int(colision)
      + cfg["penalizacion_rojo"]    * int(cruzo_en_rojo)
    )
    return float(r)
