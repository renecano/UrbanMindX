# ============================================================
# entorno_semaforo.py — VERSION FINAL
# Interseccion real: Venustiano Carranza y Blvr. Pino Suarez
# Fuente datos: Quivera UAEM 2022, aforos mayo 2021
#
# Limites del observation_space ajustados a flujos reales:
#   Pino Suarez  (norte/sur): max 870 veh/h → hasta 60 autos en cola
#   Venustiano Carranza (e/o): max 470 veh/h → hasta 35 autos en cola
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
    Entorno RL para el agente semaforo.
    Interseccion: Venustiano Carranza y Blvr. Pino Suarez, Toluca.
    Fuente datos: Quivera UAEM 2022.

    Observacion (6 valores float32):
        [autos_norte, autos_sur, autos_este, autos_oeste,
         tiempo_en_fase, fase_actual_encoded]

        Limites reales por direccion y flujo:
            norte (Pino Suarez, 3 carr, 870 veh/h) → max 60 autos
            sur   (Pino Suarez, 2 carr, 870 veh/h) → max 60 autos
            este  (Carranza,    2 carr, 470 veh/h) → max 35 autos
            oeste (Carranza,    3 carr, 470 veh/h) → max 35 autos

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
        # Limites calculados con flujos reales de Toluca
        self.observation_space = spaces.Box(
            low=np.array( [0,  0,  0,  0,   0,  0.0], dtype=np.float32),
            high=np.array([60, 60, 35, 35, 120,  1.0], dtype=np.float32),
            #               ↑   ↑   ↑   ↑
            #             N   S   E   O
            #         (Pino) (Pino) (Carr) (Carr)
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
        print(
            f"[Paso {self._pasos:4d}] "
            f"N:{e['autos_norte']:2d} S:{e['autos_sur']:2d} "
            f"E:{e['autos_este']:2d} O:{e['autos_oeste']:2d} | "
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
