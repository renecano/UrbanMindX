# ============================================================
# evaluador.py — Persona 3
# Corre los 5 escenarios y guarda métricas comparativas.
# ============================================================

import os
import csv
import numpy as np
from stable_baselines3 import PPO
from simulacion.escenarios import ESCENARIOS, get_escenario
from entrenamiento.config import RESULTADOS_DIR, MAX_PASOS


class Evaluador:
    """
    Corre cada escenario N episodios y guarda resultados en CSV.

    Uso:
        evaluador = Evaluador(sim, env_semaforo, env_vehiculo)
        evaluador.correr_todos()
    """

    def __init__(self, sim, env_semaforo, env_vehiculo, episodios: int = 5):
        self.sim          = sim
        self.env_semaforo = env_semaforo
        self.env_vehiculo = env_vehiculo
        self.episodios    = episodios
        os.makedirs(RESULTADOS_DIR, exist_ok=True)

    def correr_todos(self):
        """Evalúa todos los escenarios y genera el CSV comparativo."""
        resultados = []

        for nombre, escenario in ESCENARIOS.items():
            print(f"\nEvaluando: {escenario.nombre}...")
            metricas = self._correr_escenario(escenario)
            metricas["escenario"] = nombre
            metricas["nombre"]    = escenario.nombre
            resultados.append(metricas)
            print(f"  Espera promedio: {metricas['espera_promedio']:.1f}s")
            print(f"  Throughput:      {metricas['throughput']:.0f} autos/h")

        self._guardar_comparativa(resultados)
        print(f"\n✓ Resultados guardados en {RESULTADOS_DIR}comparativa.csv")
        return resultados

    def _correr_escenario(self, escenario) -> dict:
        """Corre N episodios de un escenario y promedia las métricas."""
        esperas, throughputs, velocidades, frenadas = [], [], [], []

        # Cargar modelos si el escenario los usa
        modelo_s = None
        modelo_v = None
        if escenario.semaforo_ia and escenario.modelo_semaforo:
            try:
                modelo_s = PPO.load(escenario.modelo_semaforo,
                                    env=self.env_semaforo)
            except Exception as e:
                print(f"  ⚠ No se pudo cargar modelo semáforo: {e}")

        if escenario.vehiculos_ia and escenario.modelo_vehiculo:
            try:
                modelo_v = PPO.load(escenario.modelo_vehiculo,
                                    env=self.env_vehiculo)
            except Exception as e:
                print(f"  ⚠ No se pudo cargar modelo vehículo: {e}")

        for ep in range(self.episodios):
            ep_espera, ep_throughput, ep_vel, ep_frenadas = \
                self._correr_episodio(escenario, modelo_s, modelo_v)
            esperas.append(ep_espera)
            throughputs.append(ep_throughput)
            velocidades.append(ep_vel)
            frenadas.append(ep_frenadas)

        return {
            "espera_promedio":    round(np.mean(esperas),      2),
            "espera_std":         round(np.std(esperas),       2),
            "throughput":         round(np.mean(throughputs),  2),
            "velocidad_promedio": round(np.mean(velocidades),  2),
            "frenadas_promedio":  round(np.mean(frenadas),     2),
        }

    def _correr_episodio(self, escenario, modelo_s, modelo_v):
        """Corre un episodio completo y regresa métricas."""
        obs_s, _ = self.env_semaforo.reset()
        obs_v, _ = self.env_vehiculo.reset()

        espera_total = 0
        throughput   = 0
        velocidades  = []
        frenadas     = 0
        vel_anterior = 0

        for paso in range(MAX_PASOS):
            # Acción del semáforo
            if modelo_s:
                accion_s, _ = modelo_s.predict(obs_s, deterministic=True)
            else:
                accion_s = 0  # semáforo fijo

            # Acción del vehículo
            if modelo_v:
                accion_v, _ = modelo_v.predict(obs_v, deterministic=True)
            else:
                accion_v = 1  # mantener velocidad (vehículo normal)

            obs_s, _, done_s, _, _ = self.env_semaforo.step(accion_s)
            obs_v, _, done_v, _, _ = self.env_vehiculo.step(accion_v)

            metricas = self.sim.get_metricas_globales()
            espera_total += metricas["espera_total"]
            throughput   += metricas["autos_llegaron"]
            vel_actual    = metricas["velocidad_promedio"]
            velocidades.append(vel_actual)

            if vel_anterior - vel_actual > 3.0:
                frenadas += 1
            vel_anterior = vel_actual

            if done_s or done_v:
                break

        n = max(paso + 1, 1)
        return (
            espera_total / n,
            throughput,
            np.mean(velocidades) if velocidades else 0,
            frenadas,
        )

    def _guardar_comparativa(self, resultados: list):
        archivo = os.path.join(RESULTADOS_DIR, "comparativa.csv")
        campos  = ["escenario", "nombre", "espera_promedio", "espera_std",
                   "throughput", "velocidad_promedio", "frenadas_promedio"]

        with open(archivo, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            for r in resultados:
                writer.writerow({k: r.get(k, "") for k in campos})
