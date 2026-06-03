# ============================================================
# entorno_semaforo.py — VERSION 3 (observacion mejorada)
# Interseccion real: Venustiano Carranza y Blvr. Pino Suarez
#
# CAMBIO PRINCIPAL vs v2:
#   La observacion pasa de 6 a 9 valores. Se agregan 3 valores
#   procesados que le facilitan a la IA decidir:
#     - presion_NS:  autos esperando en Pino Suarez (N+S)
#     - presion_EO:  autos esperando en Carranza (E+O)
#     - diferencia:  presion_NS - presion_EO (que lado urge mas)
#
#   Esto le da a la IA la comparacion entre direcciones ya hecha,
#   en lugar de obligarla a calcularla internamente.
# ============================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from entornos.recompensas import (
    recompensa_semaforo_simple,
    recompensa_semaforo_completa,
)
from entrenamiento.config import REWARD_SEMAFORO, MAX_PASOS

FASE_NS = 0   # Pino Suarez en verde
FASE_EO = 2   # Venustiano Carranza en verde


class EntornoSemaforo(gym.Env):
    """
    Entorno RL para el agente semaforo (observacion mejorada).
    Interseccion: Venustiano Carranza y Blvr. Pino Suarez, Toluca.

    Observacion (9 valores float32):
        [autos_norte, autos_sur, autos_este, autos_oeste,
         tiempo_en_fase, fase_actual_encoded,
         presion_NS, presion_EO, diferencia_presion]

        Los primeros 6 son los originales.
        Los ultimos 3 son procesados para facilitar la decision:
            presion_NS  = autos_norte + autos_sur   (0 a 120)
            presion_EO  = autos_este + autos_oeste   (0 a 70)
            diferencia  = presion_NS - presion_EO    (-70 a 120)

    Acciones:
        0 = mantener fase actual
        1 = cambiar a la otra fase verde
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, sim, modo_recompensa: str = "simple"):
        super().__init__()
        self.sim             = sim
        self.modo_recompensa = modo_recompensa
        self._pasos          = 0
        self._tiempo_en_fase = 0
        self._fase_verde     = FASE_NS

        # ---- Espacio de observacion (9 valores) ----
        self.observation_space = spaces.Box(
            low=np.array(
                [0,  0,  0,  0,   0,  0.0,  0,   0,  -70],
                dtype=np.float32),
            high=np.array(
                [60, 60, 35, 35, 120, 1.0, 120,  70, 120],
                dtype=np.float32),
        )

        # ---- Espacio de accion ----
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.resetear()
        self._pasos          = 0
        self._tiempo_en_fase = 0
        self._fase_verde     = FASE_NS
        return self._get_obs(), {}

    def step(self, accion: int):
        cambio_prematuro = False

        if accion == 1:
            min_tiempo = REWARD_SEMAFORO["min_tiempo_fase"]
            if self._tiempo_en_fase < min_tiempo:
                cambio_prematuro = True
            else:
                fase_amarilla = self._fase_verde + 1
                self.sim.set_fase_semaforo(fase_amarilla)
                for _ in range(3):
                    self.sim.avanzar_paso()
                    self._pasos += 1
                self._fase_verde = FASE_EO if self._fase_verde == FASE_NS else FASE_NS
                self.sim.set_fase_semaforo(self._fase_verde)
                self._tiempo_en_fase = 0

        self.sim.avanzar_paso()
        self._pasos          += 1
        self._tiempo_en_fase += 1

        obs    = self._get_obs()
        reward = self._calcular_recompensa(cambio_prematuro)
        done   = self.sim.simulacion_terminada() or self._pasos >= MAX_PASOS
        info   = {
            "pasos":            self._pasos,
            "fase_verde":       "Pino Suarez" if self._fase_verde == FASE_NS else "Carranza",
            "cambio_prematuro": cambio_prematuro,
        }

        return obs, reward, done, False, info

    def render(self):
        e    = self.sim.get_estado_interseccion()
        fase = "Pino Suarez (NS)" if self._fase_verde == FASE_NS else "Carranza (EO)"
        presion_ns = e["autos_norte"] + e["autos_sur"]
        presion_eo = e["autos_este"] + e["autos_oeste"]
        print(
            f"[Paso {self._pasos:4d}] "
            f"Presion NS:{presion_ns:3d} EO:{presion_eo:3d} | "
            f"Verde:{fase} t={self._tiempo_en_fase:3d}s | "
            f"Espera:{e['espera_total']:.1f}s"
        )

    def _get_obs(self) -> np.ndarray:
        e            = self.sim.get_estado_interseccion()
        fase_encoded = 0.0 if self._fase_verde == FASE_NS else 1.0

        # Valores originales
        autos_n = e["autos_norte"]
        autos_s = e["autos_sur"]
        autos_e = e["autos_este"]
        autos_o = e["autos_oeste"]

        # Valores procesados nuevos
        presion_ns = autos_n + autos_s
        presion_eo = autos_e + autos_o
        diferencia = presion_ns - presion_eo

        return np.array([
            autos_n,
            autos_s,
            autos_e,
            autos_o,
            float(self._tiempo_en_fase),
            fase_encoded,
            presion_ns,
            presion_eo,
            diferencia,
        ], dtype=np.float32)

    def _calcular_recompensa(self, cambio_prematuro: bool) -> float:
        estado = self.sim.get_estado_interseccion()
        if self.modo_recompensa == "simple":
            return recompensa_semaforo_simple(estado)
        return recompensa_semaforo_completa(estado, cambio_prematuro)