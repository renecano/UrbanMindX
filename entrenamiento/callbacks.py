# ============================================================
# callbacks.py — Persona 3
# Callbacks personalizados para monitorear el entrenamiento.
# Se ejecutan automáticamente durante modelo.learn()
# ============================================================

import os
import csv
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from entrenamiento.config import RESULTADOS_DIR


class GuardarMetricasCallback(BaseCallback):
    """
    Guarda recompensa promedio cada N pasos en un CSV.
    Usado para graficar la curva de aprendizaje en el dashboard.
    """

    def __init__(self, agente: str, guardar_cada: int = 1000):
        """
        Args:
            agente:       "semaforo" o "vehiculo"
            guardar_cada: cada cuántos timesteps guardar
        """
        super().__init__()
        self.agente       = agente
        self.guardar_cada = guardar_cada
        self._recompensas = []
        self._archivo     = os.path.join(RESULTADOS_DIR,
                                         f"entrenamiento_{agente}.csv")
        os.makedirs(RESULTADOS_DIR, exist_ok=True)

        # Crear CSV con encabezado
        with open(self._archivo, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "recompensa_promedio",
                             "recompensa_min", "recompensa_max"])

    def _on_step(self) -> bool:
        # Recopilar recompensa de este paso
        reward = self.locals.get("rewards", [0])[0]
        self._recompensas.append(float(reward))

        # Guardar cada N pasos
        if self.num_timesteps % self.guardar_cada == 0 and self._recompensas:
            with open(self._archivo, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps,
                    round(np.mean(self._recompensas), 4),
                    round(np.min(self._recompensas),  4),
                    round(np.max(self._recompensas),  4),
                ])
            self._recompensas = []  # reset buffer
            print(f"[{self.agente}] Paso {self.num_timesteps:,} guardado")

        return True  # True = continuar entrenamiento


class GuardarMejorModeloCallback(BaseCallback):
    """
    Guarda el modelo automáticamente cuando alcanza
    la mejor recompensa promedio hasta el momento.
    """

    def __init__(self, agente: str, evaluar_cada: int = 5000):
        super().__init__()
        self.agente       = agente
        self.evaluar_cada = evaluar_cada
        self._mejor_reward = -np.inf
        self._recompensas  = []
        self._ruta_modelo  = f"modelos/{agente}_mejor"

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        self._recompensas.append(float(reward))

        if self.num_timesteps % self.evaluar_cada == 0 and self._recompensas:
            promedio = np.mean(self._recompensas)
            if promedio > self._mejor_reward:
                self._mejor_reward = promedio
                self.model.save(self._ruta_modelo)
                print(f"[{self.agente}] ✓ Nuevo mejor modelo guardado "
                      f"(reward={promedio:.2f}) en {self._ruta_modelo}")
            self._recompensas = []

        return True


class ImprimirProgresoCallback(BaseCallback):
    """
    Imprime progreso simple en consola.
    Útil cuando verbose=0 en el modelo.
    """

    def __init__(self, agente: str, cada: int = 10_000):
        super().__init__()
        self.agente = agente
        self.cada   = cada

    def _on_step(self) -> bool:
        if self.num_timesteps % self.cada == 0:
            print(f"[{self.agente}] {self.num_timesteps:,} / "
                  f"{self.locals.get('total_timesteps', '?'):,} pasos")
        return True
