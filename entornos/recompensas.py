# ============================================================
# recompensas.py — Persona 2
# Funciones de recompensa para ambos agentes.
# Separadas del entorno para facilitar experimentación.
#
# CÓMO EXPERIMENTAR:
#   1. Cambia los pesos en entrenamiento/config.py
#   2. Corre: python main.py --modo entrenar
#   3. Observa si la curva de recompensa sube más rápido
#   4. Documenta qué combinación funcionó mejor
# ============================================================

from entrenamiento.config import REWARD_SEMAFORO, REWARD_VEHICULO


# ============================================================
# SEMÁFORO
# ============================================================

def recompensa_semaforo_simple(estado: dict) -> float:
    """
    Versión básica: penaliza directamente la espera total.

    Usar primero para verificar que el agente aprende algo.
    Si después de 30k pasos la recompensa no sube, hay un
    problema en el entorno, no en la función de recompensa.

    Args:
        estado: dict de sim.get_estado_interseccion()

    Returns:
        float negativo — más cercano a 0 es mejor
    """
    return -float(estado["espera_total"]) / 1000.0


def recompensa_semaforo_completa(estado: dict,
                                  cambio_prematuro: bool) -> float:
    """
    Versión completa con 3 componentes:
        1. Penaliza tiempo de espera acumulado
        2. Penaliza longitud de filas
        3. Penaliza cambios antes del tiempo mínimo

    Usar cuando la versión simple ya converge.

    Args:
        estado:           dict de sim.get_estado_interseccion()
        cambio_prematuro: True si el agente intentó cambiar
                          antes de REWARD_SEMAFORO["min_tiempo_fase"]
    """
    cfg = REWARD_SEMAFORO

    reward = (
          cfg["peso_espera"] * estado["espera_total"]
        + cfg["peso_filas"]  * estado["filas_total"]
        + cfg["penalizacion_cambio"] * float(cambio_prematuro)
    )
    return float(reward) / 1000.0


def recompensa_semaforo_throughput(estado: dict,
                                    autos_cruzaron: int) -> float:
    """
    Variante alternativa: premia directamente el throughput.

    Usar si las versiones anteriores no convergen bien.
    El agente aprende a dejar pasar autos en lugar de
    solo evitar esperas.

    Args:
        estado:         dict de sim.get_estado_interseccion()
        autos_cruzaron: cuántos autos llegaron este paso
                        (sim.get_metricas_globales()["autos_llegaron"])
    """
    return float(autos_cruzaron) * 2.0 - 0.1 * estado["espera_total"]


# ============================================================
# VEHÍCULO
# ============================================================

def recompensa_vehiculo_simple(estado_v: dict) -> float:
    """
    Versión básica: premia velocidad, penaliza estar detenido.

    Usar al inicio del entrenamiento.

    Args:
        estado_v: dict de sim.get_estado_vehiculo(id)
    """
    return (
          estado_v["velocidad"] * 0.5
        - estado_v["tiempo_detenido"] * 0.3
    )


def recompensa_vehiculo_completa(estado_v: dict,
                                  frenada_brusca: bool,
                                  colision: bool,
                                  cruzo_en_rojo: bool) -> float:
    """
    Versión completa con 6 componentes:
        1. Premia avance proporcional a velocidad máxima
        2. Penaliza tiempo detenido
        3. Penaliza frenadas bruscas (aceleración < -3 m/s²)
        4. Penaliza distancia insegura al auto de enfrente
        5. Penaliza colisión
        6. Penaliza cruzar semáforo en rojo o amarillo

    Args:
        estado_v:      dict de sim.get_estado_vehiculo(id)
        frenada_brusca: aceleracion < UMBRAL_FRENADA_BRUSCA
        colision:       distancia_al_frente < distancia_segura
        cruzo_en_rojo:  avanzó con semáforo no verde
    """
    cfg = REWARD_VEHICULO

    # Normalizar velocidad entre 0 y 1
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
