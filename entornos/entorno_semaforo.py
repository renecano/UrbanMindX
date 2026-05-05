# ============================================================
# entorno_semaforo.py — Persona 2
# Entorno Gymnasium para el agente semáforo.
# ============================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from entornos.recompensas import (
    recompensa_semaforo_simple,
    recompensa_semaforo_completa,
)
from entrenamiento.config import REWARD_SEMAFORO, MAX_PASOS


class EntornoSemaforo(gym.Env):
    """
    Entorno RL para el agente semáforo.

    Observación (6 valores):
        [autos_norte, autos_sur, autos_este, autos_oeste,
         tiempo_fase, fase_actual]

    Acciones:
        0 = mantener fase actual
        1 = cambiar a la otra fase

    Recompensa:
        Negativa cuando hay autos esperando.
        Penalización extra por cambios prematuros de fase.
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

        # ---- Espacio de observación ----
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0],     dtype=np.float32),
            high=np.array([100, 100, 100, 100, 120, 1], dtype=np.float32),
        )

        # ---- Espacio de acción ----
        self.action_space = spaces.Discrete(2)

    # ----------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.resetear()
        self._pasos          = 0
        self._tiempo_en_fase = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, accion: int):
        cambio_prematuro = False

        if accion == 1:
            min_tiempo = REWARD_SEMAFORO["min_tiempo_fase"]
            if self._tiempo_en_fase < min_tiempo:
                cambio_prematuro = True
            else:
                fase_nueva = 1 - self.sim.get_fase_semaforo()
                self.sim.set_fase_semaforo(fase_nueva)
                self._tiempo_en_fase = 0

        self.sim.avanzar_paso()
        self._pasos          += 1
        self._tiempo_en_fase += 1

        obs      = self._get_obs()
        reward   = self._calcular_recompensa(cambio_prematuro)
        done     = self.sim.simulacion_terminada() or self._pasos >= MAX_PASOS
        info     = {"pasos": self._pasos, "cambio_prematuro": cambio_prematuro}

        return obs, reward, done, False, info

    def render(self):
        estado = self.sim.get_estado_interseccion()
        print(f"[Paso {self._pasos:4d}] "
              f"N:{estado['autos_norte']:2d} S:{estado['autos_sur']:2d} "
              f"E:{estado['autos_este']:2d} O:{estado['autos_oeste']:2d} "
              f"Fase:{estado['fase_actual']} "
              f"Espera:{estado['espera_total']:.1f}s")

    # ----------------------------------------------------------
    # Helpers privados
    # ----------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        e = self.sim.get_estado_interseccion()
        return np.array([
            e["autos_norte"], e["autos_sur"],
            e["autos_este"],  e["autos_oeste"],
            e["tiempo_fase"], e["fase_actual"],
        ], dtype=np.float32)

    def _calcular_recompensa(self, cambio_prematuro: bool) -> float:
        estado = self.sim.get_estado_interseccion()
        if self.modo_recompensa == "simple":
            return recompensa_semaforo_simple(estado)
        return recompensa_semaforo_completa(estado, cambio_prematuro)
