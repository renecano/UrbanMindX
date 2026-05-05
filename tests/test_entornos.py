# ============================================================
# test_entornos.py — Persona 2
# Tests básicos para verificar que los entornos funcionan.
# Ejecutar con: pytest tests/
# ============================================================

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env
from simulacion.mock_simulacion import SimulacionMock
from entornos.entorno_semaforo  import EntornoSemaforo
from entornos.entorno_vehiculo  import EntornoVehiculo


@pytest.fixture
def sim():
    return SimulacionMock(semilla=0)

@pytest.fixture
def env_semaforo(sim):
    return EntornoSemaforo(sim)

@pytest.fixture
def env_vehiculo(sim):
    return EntornoVehiculo(sim)


class TestEntornoSemaforo:

    def test_reset_regresa_observacion_valida(self, env_semaforo):
        obs, info = env_semaforo.reset()
        assert obs.shape == (6,)
        assert obs.dtype == np.float32

    def test_step_con_accion_0(self, env_semaforo):
        env_semaforo.reset()
        obs, reward, done, truncated, info = env_semaforo.step(0)
        assert obs.shape == (6,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_step_con_accion_1(self, env_semaforo):
        env_semaforo.reset()
        obs, reward, done, truncated, info = env_semaforo.step(1)
        assert obs.shape == (6,)

    def test_gymnasium_check_env(self, env_semaforo):
        """Verifica que el entorno cumple el contrato de Gymnasium."""
        check_env(env_semaforo, warn=True)  # no debe lanzar excepciones

    def test_recompensa_es_negativa_con_trafico(self, env_semaforo):
        env_semaforo.reset()
        _, reward, _, _, _ = env_semaforo.step(0)
        # Con tráfico siempre hay espera, la recompensa debe ser <= 0
        assert reward <= 0


class TestEntornoVehiculo:

    def test_reset_regresa_observacion_valida(self, env_vehiculo):
        obs, info = env_vehiculo.reset()
        assert obs.shape == (6,)
        assert obs.dtype == np.float32

    def test_todas_las_acciones_son_validas(self, env_vehiculo):
        for accion in [0, 1, 2]:
            env_vehiculo.reset()
            obs, reward, done, _, _ = env_vehiculo.step(accion)
            assert obs.shape == (6,)

    def test_gymnasium_check_env(self, env_vehiculo):
        check_env(env_vehiculo, warn=True)

    def test_episodio_completo_sin_crash(self, env_vehiculo):
        obs, _ = env_vehiculo.reset()
        for _ in range(50):
            accion = env_vehiculo.action_space.sample()
            obs, reward, done, _, _ = env_vehiculo.step(accion)
            if done:
                break
        # Si llegamos aquí sin excepción, el test pasa
