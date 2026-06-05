# ============================================================
# evaluador.py — VERSION CON ESCENARIOS DIFERENCIADOS
# Cada escenario ahora carga su propio config_sumo:
#   - tradicional/semaforo_ia/completo → simulacion.sumocfg
#   - hora_pico                        → hora_pico.sumocfg
#   - desbalanceado                    → desbalanceado.sumocfg
#
# Mide:
#   - espera_promedio: espera total de la interseccion por paso
#   - throughput: autos que cruzaron por hora real
# ============================================================

import os
import csv
import numpy as np
from stable_baselines3 import PPO
from simulacion.escenarios import ESCENARIOS, get_escenario
from entrenamiento.config import RESULTADOS_DIR, MAX_PASOS

HORAS_SIMULADAS = MAX_PASOS / 3600.0


class Evaluador:
    """
    Corre cada escenario N episodios y guarda resultados en CSV.
    Cada escenario usa su propio archivo de configuracion SUMO.
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
            print(f"  Config: {escenario.config_sumo}")
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
        # --- CLAVE: recargar SUMO con el config de ESTE escenario ---
        self.sim.resetear(escenario.config_sumo)

        obs_s = self.env_semaforo._get_obs()
        obs_v = self.env_vehiculo._get_obs()

        # Reiniciar contadores internos de los entornos sin re-resetear SUMO
        self.env_semaforo._pasos          = 0
        self.env_semaforo._tiempo_en_fase = 0
        self.env_vehiculo._pasos          = 0

        suma_espera   = 0.0
        throughput    = 0
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

        espera_promedio = suma_espera / max(pasos_reales, 1)
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