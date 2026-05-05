# ============================================================
# pipeline.py — Persona 3
# Orquesta el entrenamiento completo en 3 etapas.
# ============================================================

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from entrenamiento.config import (
    PPO_SEMAFORO, PPO_VEHICULO,
    TIMESTEPS_SEMAFORO, TIMESTEPS_VEHICULO, TIMESTEPS_CONJUNTO,
    MODELO_SEMAFORO, MODELO_VEHICULO,
)
from entrenamiento.callbacks import (
    GuardarMetricasCallback,
    GuardarMejorModeloCallback,
    ImprimirProgresoCallback,
)


class PipelineUrbanMind:
    """
    Ejecuta el entrenamiento en 3 etapas:
        1. Solo el semáforo (con vehículos normales)
        2. Solo los vehículos (con semáforo ya entrenado)
        3. Ambos juntos — fine-tuning conjunto

    Uso:
        from simulacion.mock_simulacion import SimulacionMock
        sim = SimulacionMock()  # o SimulacionTrafico() cuando SUMO esté listo

        from entornos.entorno_semaforo import EntornoSemaforo
        from entornos.entorno_vehiculo import EntornoVehiculo
        env_s = EntornoSemaforo(sim)
        env_v = EntornoVehiculo(sim)

        pipeline = PipelineUrbanMind()
        pipeline.etapa_1_semaforo(env_s)
        pipeline.etapa_2_vehiculos(env_v)
        pipeline.etapa_3_conjunto(env_s, env_v)
    """

    def __init__(self):
        os.makedirs("modelos",    exist_ok=True)
        os.makedirs("resultados", exist_ok=True)

    # ----------------------------------------------------------
    # Etapa 1
    # ----------------------------------------------------------

    def etapa_1_semaforo(self, env, validar: bool = True):
        """
        Entrena el agente semáforo con vehículos normales (no autónomos).
        """
        print("\n" + "="*50)
        print("ETAPA 1 — Entrenamiento del Semáforo")
        print("="*50)

        if validar:
            print("Validando entorno...")
            check_env(env, warn=True)
            print("✓ Entorno válido\n")

        modelo = PPO("MlpPolicy", env, **PPO_SEMAFORO)

        callbacks = [
            GuardarMetricasCallback("semaforo", guardar_cada=1000),
            GuardarMejorModeloCallback("semaforo", evaluar_cada=5000),
            ImprimirProgresoCallback("semaforo",  cada=10_000),
        ]

        modelo.learn(total_timesteps=TIMESTEPS_SEMAFORO,
                     callback=callbacks,
                     progress_bar=True)

        modelo.save(MODELO_SEMAFORO + "_etapa1")
        print(f"\n✓ Etapa 1 completada → {MODELO_SEMAFORO}_etapa1.zip")
        return modelo

    # ----------------------------------------------------------
    # Etapa 2
    # ----------------------------------------------------------

    def etapa_2_vehiculos(self, env, validar: bool = True):
        """
        Entrena vehículos autónomos.
        El semáforo ya debe estar entrenado y corriendo en el entorno.
        """
        print("\n" + "="*50)
        print("ETAPA 2 — Entrenamiento de Vehículos Autónomos")
        print("="*50)

        if validar:
            print("Validando entorno...")
            check_env(env, warn=True)
            print("✓ Entorno válido\n")

        modelo = PPO("MlpPolicy", env, **PPO_VEHICULO)

        callbacks = [
            GuardarMetricasCallback("vehiculo", guardar_cada=1000),
            GuardarMejorModeloCallback("vehiculo", evaluar_cada=5000),
            ImprimirProgresoCallback("vehiculo",  cada=10_000),
        ]

        modelo.learn(total_timesteps=TIMESTEPS_VEHICULO,
                     callback=callbacks,
                     progress_bar=True)

        modelo.save(MODELO_VEHICULO + "_etapa2")
        print(f"\n✓ Etapa 2 completada → {MODELO_VEHICULO}_etapa2.zip")
        return modelo

    # ----------------------------------------------------------
    # Etapa 3
    # ----------------------------------------------------------

    def etapa_3_conjunto(self, env_semaforo, env_vehiculo):
        """
        Fine-tuning con ambos agentes activos simultáneamente.
        Carga los modelos de etapas anteriores y continúa entrenando.
        """
        print("\n" + "="*50)
        print("ETAPA 3 — Entrenamiento Conjunto")
        print("="*50)

        semaforo = PPO.load(MODELO_SEMAFORO + "_etapa1", env=env_semaforo)
        vehiculo = PPO.load(MODELO_VEHICULO + "_etapa2", env=env_vehiculo)

        print("Entrenando semáforo en entorno conjunto...")
        semaforo.learn(total_timesteps=TIMESTEPS_CONJUNTO,
                       callback=GuardarMetricasCallback("semaforo_conjunto"),
                       progress_bar=True)

        print("Entrenando vehículos en entorno conjunto...")
        vehiculo.learn(total_timesteps=TIMESTEPS_CONJUNTO,
                       callback=GuardarMetricasCallback("vehiculo_conjunto"),
                       progress_bar=True)

        semaforo.save(MODELO_SEMAFORO + "_final")
        vehiculo.save(MODELO_VEHICULO + "_final")

        print(f"\n✓ Etapa 3 completada")
        print(f"  → {MODELO_SEMAFORO}_final.zip")
        print(f"  → {MODELO_VEHICULO}_final.zip")
        return semaforo, vehiculo
