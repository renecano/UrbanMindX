# ============================================================
# test_entrenamiento.py — Persona 3
# Verifica que el pipeline de entrenamiento no explota.
# ============================================================

import pytest
from simulacion.mock_simulacion import SimulacionMock
from entornos.entorno_semaforo  import EntornoSemaforo
from entornos.entorno_vehiculo  import EntornoVehiculo
from stable_baselines3          import PPO


@pytest.fixture
def entornos():
    sim   = SimulacionMock(semilla=0)
    env_s = EntornoSemaforo(sim)
    env_v = EntornoVehiculo(sim)
    return env_s, env_v


class TestEntrenamiento:

    def test_ppo_semaforo_corre_100_pasos(self, entornos):
        env_s, _ = entornos
        modelo = PPO("MlpPolicy", env_s, verbose=0)
        modelo.learn(total_timesteps=100)
        # Si llegamos aquí sin excepción, el test pasa

    def test_ppo_vehiculo_corre_100_pasos(self, entornos):
        _, env_v = entornos
        modelo = PPO("MlpPolicy", env_v, verbose=0)
        modelo.learn(total_timesteps=100)

    def test_modelo_predice_accion_valida_semaforo(self, entornos):
        env_s, _ = entornos
        modelo = PPO("MlpPolicy", env_s, verbose=0)
        modelo.learn(total_timesteps=100)
        obs, _ = env_s.reset()
        accion, _ = modelo.predict(obs)
        assert accion in [0, 1]

    def test_modelo_predice_accion_valida_vehiculo(self, entornos):
        _, env_v = entornos
        modelo = PPO("MlpPolicy", env_v, verbose=0)
        modelo.learn(total_timesteps=100)
        obs, _ = env_v.reset()
        accion, _ = modelo.predict(obs)
        assert accion in [0, 1, 2]
