# ============================================================
# entorno_semaforo.py — Persona 2
# Entorno Gymnasium para el agente semáforo.
# Alineado con la red real de Persona 1:
#   - SEMAFORO_ID = "centro"
#   - 4 fases: 0=NS verde, 1=NS amarillo, 2=EO verde, 3=EO amarillo
#   - Edges: norte_entrada, sur_entrada, este_entrada, oeste_entrada
# ============================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from entornos.recompensas import (
    recompensa_semaforo_simple,
    recompensa_semaforo_completa,
)
from entrenamiento.config import REWARD_SEMAFORO, MAX_PASOS

# Fases válidas que el agente puede elegir (excluye amarillos)
FASE_NS = 0   # Norte-Sur en verde
FASE_EO = 2   # Este-Oeste en verde

class EntornoSemaforo(gym.Env):
    """
    Entorno RL para el agente semáforo.

    El agente solo decide entre 2 acciones:
        0 = mantener fase actual (verde)
        1 = cambiar a la otra fase verde

    Las fases amarillas (1 y 3) se insertan automáticamente
    como transición — el agente no las controla directamente.

    Observación (6 valores float32):
        [autos_norte, autos_sur, autos_este, autos_oeste,
         tiempo_en_fase, fase_actual_encoded]

        fase_actual_encoded:
            0.0 = Norte-Sur en verde (fase 0)
            1.0 = Este-Oeste en verde (fase 2)

    Recompensa:
        Negativa proporcional a la espera total.
        Penalización extra por cambios prematuros.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, sim, modo_recompensa: str = "simple"):
        """
        Args:
            sim:             SimulacionTrafico o SimulacionMock
            modo_recompensa: "simple" o "completa"
        """
        super().__init__()
        self.sim             = sim
        self.modo_recompensa = modo_recompensa
        self._pasos          = 0
        self._tiempo_en_fase = 0
        self._fase_verde     = FASE_NS  # fase verde actual (0 o 2)

        # ---- Espacio de observación ----
        # [autos_N, autos_S, autos_E, autos_O, tiempo_fase, fase_encoded]
        self.observation_space = spaces.Box(
            low=np.array( [0,   0,   0,   0,   0,  0.0], dtype=np.float32),
            high=np.array([100, 100, 100, 100, 120, 1.0], dtype=np.float32),
        )

        # ---- Espacio de acción ----
        # 0 = mantener, 1 = cambiar
        self.action_space = spaces.Discrete(2)

    # ----------------------------------------------------------

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
                # Demasiado pronto — penalizar pero no cambiar
                cambio_prematuro = True
            else:
                # Cambiar: insertar fase amarilla primero
                fase_amarilla = self._fase_verde + 1  # 0→1 o 2→3
                self.sim.set_fase_semaforo(fase_amarilla)

                # Avanzar 3 pasos para la transición amarilla
                for _ in range(3):
                    self.sim.avanzar_paso()
                    self._pasos += 1

                # Ahora poner la nueva fase verde
                self._fase_verde = FASE_EO if self._fase_verde == FASE_NS else FASE_NS
                self.sim.set_fase_semaforo(self._fase_verde)
                self._tiempo_en_fase = 0

        # Avanzar 1 paso normal
        self.sim.avanzar_paso()
        self._pasos          += 1
        self._tiempo_en_fase += 1

        obs    = self._get_obs()
        reward = self._calcular_recompensa(cambio_prematuro)
        done   = self.sim.simulacion_terminada() or self._pasos >= MAX_PASOS
        info   = {
            "pasos":            self._pasos,
            "fase_verde":       self._fase_verde,
            "cambio_prematuro": cambio_prematuro,
        }

        return obs, reward, done, False, info

    def render(self):
        e     = self.sim.get_estado_interseccion()
        fase  = "NS" if self._fase_verde == FASE_NS else "EO"
        print(
            f"[Paso {self._pasos:4d}] "
            f"N:{e['autos_norte']:2d} S:{e['autos_sur']:2d} "
            f"E:{e['autos_este']:2d} O:{e['autos_oeste']:2d} | "
            f"Fase:{fase} t={self._tiempo_en_fase:3d}s | "
            f"Espera:{e['espera_total']:.1f}s Filas:{e['filas_total']}"
        )

    # ----------------------------------------------------------
    # Helpers privados
    # ----------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        e             = self.sim.get_estado_interseccion()
        fase_encoded  = 0.0 if self._fase_verde == FASE_NS else 1.0
        return np.array([
            e["autos_norte"],
            e["autos_sur"],
            e["autos_este"],
            e["autos_oeste"],
            float(self._tiempo_en_fase),
            fase_encoded,
        ], dtype=np.float32)

    def _calcular_recompensa(self, cambio_prematuro: bool) -> float:
        estado = self.sim.get_estado_interseccion()
        if self.modo_recompensa == "simple":
            return recompensa_semaforo_simple(estado)
        return recompensa_semaforo_completa(estado, cambio_prematuro)
