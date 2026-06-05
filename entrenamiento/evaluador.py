# ============================================================
# evaluador.py — VERSION CORREGIDA
# Mide correctamente:
#   - espera_promedio: segundos promedio de espera POR PASO
#     (espera total de la interseccion / pasos del episodio)
#   - throughput: autos que cruzaron POR HORA real
#     (total de autos / horas simuladas)
# ============================================================

import os
import csv
import numpy as np
from stable_baselines3 import PPO
from simulacion.escenarios import ESCENARIOS, get_escenario
from entrenamiento.config import RESULTADOS_DIR, MAX_PASOS

# Duracion del episodio en horas simuladas (21,600 pasos / 3600 = 6 horas)
HORAS_SIMULADAS = MAX_PASOS / 3600.0


class Evaluador:
    """
    Corre cada escenario N episodios y guarda resultados en CSV.

    Metricas corregidas:
        espera_promedio: promedio de espera total de la interseccion
                         por paso (segundos)
        throughput:      autos que cruzaron por hora real
    """

    def __init__(self, sim, env_semaforo, env_vehiculo, episodios: int = 5):
        self.sim          = sim
        self.env_semaforo = env_semaforo
        self.env_vehiculo = env_vehiculo
        self.episodios    = episodios
        os.makedirs(RESULTADOS_DIR, exist_ok=True)

    def correr_todos(self):
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
        esperas, throughputs, velocidades, frenadas = [], [], [], []

        modelo_s = None
        modelo_v = None
        if escenario.semaforo_ia and escenario.modelo_semaforo:
            try:
                modelo_s = PPO.load(escenario.modelo_semaforo, env=self.env_semaforo)
            except Exception as e:
                print(f"  ⚠ No se pudo cargar modelo semáforo: {e}")
        if escenario.vehiculos_ia and escenario.modelo_vehiculo:
            try:
                modelo_v = PPO.load(escenario.modelo_vehiculo, env=self.env_vehiculo)
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
        obs_s, _ = self.env_semaforo.reset()
        obs_v, _ = self.env_vehiculo.reset()

        suma_espera   = 0.0   # acumula espera_total de cada paso
        throughput    = 0     # acumula autos que llegaron (total del episodio)
        velocidades   = []
        frenadas      = 0
        vel_anterior  = 0
        pasos_reales  = 0

        for paso in range(MAX_PASOS):
            if modelo_s:
                accion_s, _ = modelo_s.predict(obs_s, deterministic=True)
            else:
                accion_s = 0
            if modelo_v:
                accion_v, _ = modelo_v.predict(obs_v, deterministic=True)
            else:
                accion_v = 1

            obs_s, _, done_s, _, _ = self.env_semaforo.step(int(accion_s))
            obs_v, _, done_v, _, _ = self.env_vehiculo.step(int(accion_v))

            metricas = self.sim.get_metricas_globales()
            suma_espera  += metricas["espera_total"]
            throughput   += metricas["autos_llegaron"]
            vel_actual    = metricas["velocidad_promedio"]
            velocidades.append(vel_actual)

            if vel_anterior - vel_actual > 3.0:
                frenadas += 1
            vel_anterior = vel_actual
            pasos_reales += 1

            if done_s or done_v:
                break

        # --- ESPERA PROMEDIO POR PASO ---
        # suma_espera es la espera total de la interseccion sumada en cada paso.
        # Dividimos entre los pasos para obtener la espera promedio por paso.
        espera_promedio = suma_espera / max(pasos_reales, 1)

        # --- THROUGHPUT POR HORA ---
        # throughput es el total de autos que cruzaron en todo el episodio.
        # Lo dividimos entre las horas simuladas reales del episodio.
        horas_reales = max(pasos_reales / 3600.0, 0.01)
        throughput_por_hora = throughput / horas_reales

        return (
            espera_promedio,
            throughput_por_hora,
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