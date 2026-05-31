# ============================================================
# entorno_semaforo.py — actualizado para interseccion real
# Venustiano Carranza y Blvr. Pino Suarez, Toluca
#
# Cambio respecto a version anterior:
#   observation_space actualizado para reflejar carriles reales:
#       norte (3 carriles): max 150 autos
#       sur   (2 carriles): max 100 autos
#       oeste (3 carriles): max 150 autos
#       este  (2 carriles): max 100 autos
# ============================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from entornos.recompensas import (
    recompensa_semaforo_simple,
    recompensa_semaforo_completa,
)
from entrenamiento.config import REWARD_SEMAFORO, MAX_PASOS

FASE_NS = 0
FASE_EO = 2


class EntornoSemaforo(gym.Env):
    """
    Entorno RL para el agente semaforo.
    Interseccion real: Venustiano Carranza y Blvr. Pino Suarez, Toluca.

    Observacion (6 valores float32):
        [autos_norte, autos_sur, autos_este, autos_oeste,
         tiempo_en_fase, fase_actual_encoded]

        Limites reales por direccion:
            norte (3 carriles) → 0 a 150 autos
            sur   (2 carriles) → 0 a 100 autos
            oeste (3 carriles) → 0 a 150 autos
            este  (2 carriles) → 0 a 100 autos

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

        # ---- Espacio de observacion ----
        # Limites ajustados a carriles reales de la interseccion
        self.observation_space = spaces.Box(
            low=np.array( [0,   0,   0,   0,   0,  0.0], dtype=np.float32),
            high=np.array([150, 100, 100, 150, 120, 1.0], dtype=np.float32),
            #               ↑         ↑         ↑
            #            norte(3)  este(2)  oeste(3)
            #                   ↑
            #                sur(2)
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
            "fase_verde":       self._fase_verde,
            "cambio_prematuro": cambio_prematuro,
        }

        return obs, reward, done, False, info

    def render(self):
        e    = self.sim.get_estado_interseccion()
        fase = "Pino Suarez (NS)" if self._fase_verde == FASE_NS else "Venustiano Carranza (EO)"
        print(
            f"[Paso {self._pasos:4d}] "
            f"Norte:{e['autos_norte']:3d} Sur:{e['autos_sur']:3d} "
            f"Este:{e['autos_este']:3d} Oeste:{e['autos_oeste']:3d} | "
            f"Verde:{fase} t={self._tiempo_en_fase:3d}s | "
            f"Espera:{e['espera_total']:.1f}s"
        )

    def _get_obs(self) -> np.ndarray:
        e            = self.sim.get_estado_interseccion()
        fase_encoded = 0.0 if self._fase_verde == FASE_NS else 1.0
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
