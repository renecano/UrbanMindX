# ============================================================
# entorno_vehiculo.py — Persona 2
# Entorno Gymnasium para el agente vehículo autónomo.
# Alineado con la red real de Persona 1.
# ============================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from entornos.recompensas import (
    recompensa_vehiculo_simple,
    recompensa_vehiculo_completa,
)
from entrenamiento.config import REWARD_VEHICULO, MAX_PASOS

# Cambio de velocidad por acción (m/s)
DELTA_VELOCIDAD = {
    0: -2.0,   # frenar
    1:  0.0,   # mantener
    2: +2.0,   # acelerar
}

# Umbral para detectar frenada brusca (m/s²)
UMBRAL_FRENADA_BRUSCA = -3.0


class EntornoVehiculo(gym.Env):
    """
    Entorno RL para vehículos autónomos.

    Un modelo entrenado aquí controla cualquier vehículo
    de la simulación. Durante evaluación se aplica a todos.

    Observación (6 valores float32):
        [velocidad, distancia_al_frente, distancia_semaforo,
         semaforo_verde (0/1), carril_actual, tiempo_detenido]

    Acciones:
        0 = frenar     (-2 m/s)
        1 = mantener   (0 m/s)
        2 = acelerar   (+2 m/s)

    Recompensa:
        Premia avance fluido y seguro.
        Penaliza frenadas bruscas, cruzar en rojo y colisiones.
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
        self._vehicle_id     = None
        self._vel_anterior   = 0.0

        # ---- Espacio de observación ----
        # [vel, dist_frente, dist_semaforo, verde, carril, detenido]
        self.observation_space = spaces.Box(
            low=np.array( [0,   0,    0,   0,  0,   0  ], dtype=np.float32),
            high=np.array([14,  100,  200, 1,  3,   120 ], dtype=np.float32),
        )

        # ---- Espacio de acción ----
        self.action_space = spaces.Discrete(3)

    # ----------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.resetear()
        self._pasos        = 0
        self._vel_anterior = 0.0
        self._vehicle_id   = self._elegir_vehiculo()
        return self._get_obs(), {}

    def step(self, accion: int):
        ids = self.sim.get_ids_vehiculos()

        # Cambiar al siguiente vehículo si el actual ya salió
        if self._vehicle_id not in ids:
            self._vehicle_id = self._elegir_vehiculo()

        if self._vehicle_id:
            estado_v  = self.sim.get_estado_vehiculo(self._vehicle_id)
            vel_nueva = estado_v["velocidad"] + DELTA_VELOCIDAD[accion]
            vel_nueva = float(np.clip(vel_nueva, 0.0, estado_v["velocidad_max"]))
            self.sim.set_velocidad_vehiculo(self._vehicle_id, vel_nueva)
            self._vel_anterior = estado_v["velocidad"]

        self.sim.avanzar_paso()
        self._pasos += 1

        obs    = self._get_obs()
        reward = self._calcular_recompensa(accion)
        done   = self.sim.simulacion_terminada() or self._pasos >= MAX_PASOS
        info   = {
            "pasos":      self._pasos,
            "vehicle_id": self._vehicle_id,
        }

        return obs, reward, done, False, info

    def render(self):
        if not self._vehicle_id:
            print(f"[Paso {self._pasos:4d}] Sin vehículo activo")
            return
        ids = self.sim.get_ids_vehiculos()
        if self._vehicle_id not in ids:
            return
        e = self.sim.get_estado_vehiculo(self._vehicle_id)
        semaforo = "VERDE" if e["semaforo_estado"] == "G" else "ROJO"
        print(
            f"[Paso {self._pasos:4d}] ID:{self._vehicle_id} | "
            f"vel:{e['velocidad']:.1f} m/s | "
            f"dist_frente:{e['distancia_al_frente']:.1f}m | "
            f"dist_semaforo:{e['distancia_semaforo']:.1f}m | "
            f"semaforo:{semaforo} | "
            f"detenido:{e['tiempo_detenido']:.1f}s"
        )

    # ----------------------------------------------------------
    # Helpers privados
    # ----------------------------------------------------------

    def _elegir_vehiculo(self):
        """Elige el primer vehículo activo disponible."""
        ids = self.sim.get_ids_vehiculos()
        return ids[0] if ids else None

    def _get_obs(self) -> np.ndarray:
        if not self._vehicle_id:
            return np.zeros(6, dtype=np.float32)
        ids = self.sim.get_ids_vehiculos()
        if self._vehicle_id not in ids:
            return np.zeros(6, dtype=np.float32)

        e            = self.sim.get_estado_vehiculo(self._vehicle_id)
        verde        = 1.0 if e["semaforo_estado"] == "G" else 0.0

        return np.array([
            e["velocidad"],
            e["distancia_al_frente"],
            e["distancia_semaforo"],
            verde,
            float(e["carril_actual"]),
            e["tiempo_detenido"],
        ], dtype=np.float32)

    def _calcular_recompensa(self, accion: int) -> float:
        if not self._vehicle_id:
            return 0.0
        ids = self.sim.get_ids_vehiculos()
        if self._vehicle_id not in ids:
            return 0.0

        e = self.sim.get_estado_vehiculo(self._vehicle_id)

        if self.modo_recompensa == "simple":
            return recompensa_vehiculo_simple(e)

        frenada_brusca = (e["aceleracion"] < UMBRAL_FRENADA_BRUSCA)
        colision       = (e["distancia_al_frente"] < REWARD_VEHICULO["distancia_segura"])
        cruzo_en_rojo  = (
            e["semaforo_estado"] != "G"
            and e["distancia_semaforo"] < 2.0
            and e["velocidad"] > 0.5
        )

        return recompensa_vehiculo_completa(
            e, frenada_brusca, colision, cruzo_en_rojo
        )
