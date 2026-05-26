# ============================================================
# test_entornos.py — Persona 2
# Tests alineados con la red real de Persona 1.
# Ejecutar con: pytest tests/test_entornos.py -v
# ============================================================

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env
from simulacion.mock_simulacion import SimulacionMock
from entornos.entorno_semaforo  import EntornoSemaforo, FASE_NS, FASE_EO
from entornos.entorno_vehiculo  import EntornoVehiculo


@pytest.fixture
def sim():
    return SimulacionMock(semilla=0)

@pytest.fixture
def env_semaforo(sim):
    return EntornoSemaforo(sim, modo_recompensa="simple")

@pytest.fixture
def env_semaforo_completo(sim):
    return EntornoSemaforo(sim, modo_recompensa="completa")

@pytest.fixture
def env_vehiculo(sim):
    return EntornoVehiculo(sim, modo_recompensa="simple")

@pytest.fixture
def env_vehiculo_completo(sim):
    return EntornoVehiculo(sim, modo_recompensa="completa")


# ============================================================
# Tests del semáforo
# ============================================================

class TestEntornoSemaforo:

    def test_reset_regresa_forma_correcta(self, env_semaforo):
        obs, info = env_semaforo.reset()
        assert obs.shape == (6,), f"Esperaba (6,), got {obs.shape}"
        assert obs.dtype == np.float32

    def test_obs_dentro_de_limites(self, env_semaforo):
        obs, _ = env_semaforo.reset()
        low  = env_semaforo.observation_space.low
        high = env_semaforo.observation_space.high
        assert np.all(obs >= low),  f"Obs por debajo del mínimo: {obs}"
        assert np.all(obs <= high), f"Obs por encima del máximo: {obs}"

    def test_step_mantener_no_cambia_fase(self, env_semaforo):
        env_semaforo.reset()
        fase_antes = env_semaforo._fase_verde
        env_semaforo.step(0)  # mantener
        assert env_semaforo._fase_verde == fase_antes

    def test_step_cambiar_alterna_fase(self, env_semaforo):
        env_semaforo.reset()
        # Forzar tiempo mínimo para que el cambio sea válido
        env_semaforo._tiempo_en_fase = 999
        fase_antes = env_semaforo._fase_verde
        env_semaforo.step(1)  # cambiar
        assert env_semaforo._fase_verde != fase_antes
        assert env_semaforo._fase_verde in [FASE_NS, FASE_EO]

    def test_cambio_prematuro_no_cambia_fase(self, env_semaforo):
        env_semaforo.reset()
        env_semaforo._tiempo_en_fase = 0  # muy temprano
        fase_antes = env_semaforo._fase_verde
        _, reward, _, _, info = env_semaforo.step(1)
        assert env_semaforo._fase_verde == fase_antes
        assert info["cambio_prematuro"] is True

    def test_recompensa_simple_es_negativa(self, env_semaforo):
        env_semaforo.reset()
        _, reward, _, _, _ = env_semaforo.step(0)
        assert reward <= 0, "Con tráfico la recompensa debe ser <= 0"

    def test_recompensa_completa_funciona(self, env_semaforo_completo):
        env_semaforo_completo.reset()
        _, reward, _, _, _ = env_semaforo_completo.step(0)
        assert isinstance(reward, float)

    def test_gymnasium_check_env_simple(self, sim):
        env = EntornoSemaforo(sim, modo_recompensa="simple")
        check_env(env, warn=True)

    def test_gymnasium_check_env_completo(self, sim):
        env = EntornoSemaforo(sim, modo_recompensa="completa")
        check_env(env, warn=True)

    def test_episodio_completo_sin_crash(self, env_semaforo):
        obs, _ = env_semaforo.reset()
        for _ in range(100):
            accion = env_semaforo.action_space.sample()
            obs, reward, done, _, info = env_semaforo.step(accion)
            assert obs.shape == (6,)
            assert isinstance(reward, float)
            if done:
                break

    def test_render_no_explota(self, env_semaforo):
        env_semaforo.reset()
        env_semaforo.step(0)
        env_semaforo.render()  # solo verifica que no lanza excepción


# ============================================================
# Tests del vehículo
# ============================================================

class TestEntornoVehiculo:

    def test_reset_regresa_forma_correcta(self, env_vehiculo):
        obs, _ = env_vehiculo.reset()
        assert obs.shape == (6,)
        assert obs.dtype == np.float32

    def test_obs_dentro_de_limites(self, env_vehiculo):
        obs, _ = env_vehiculo.reset()
        low  = env_vehiculo.observation_space.low
        high = env_vehiculo.observation_space.high
        assert np.all(obs >= low),  f"Obs por debajo del mínimo: {obs}"
        assert np.all(obs <= high), f"Obs por encima del máximo: {obs}"

    def test_todas_las_acciones_son_validas(self, env_vehiculo):
        for accion in [0, 1, 2]:
            env_vehiculo.reset()
            obs, reward, done, _, info = env_vehiculo.step(accion)
            assert obs.shape == (6,), f"Acción {accion} rompió la obs"
            assert isinstance(reward, float)

    def test_frenar_no_baja_de_cero(self, env_vehiculo):
        """La velocidad nunca debe ser negativa."""
        env_vehiculo.reset()
        for _ in range(20):
            obs, _, done, _, _ = env_vehiculo.step(0)  # siempre frenar
            assert obs[0] >= 0.0, "Velocidad negativa detectada"
            if done:
                break

    def test_recompensa_simple_es_float(self, env_vehiculo):
        env_vehiculo.reset()
        _, reward, _, _, _ = env_vehiculo.step(1)
        assert isinstance(reward, float)

    def test_recompensa_completa_funciona(self, env_vehiculo_completo):
        env_vehiculo_completo.reset()
        _, reward, _, _, _ = env_vehiculo_completo.step(1)
        assert isinstance(reward, float)

    def test_gymnasium_check_env_simple(self, sim):
        env = EntornoVehiculo(sim, modo_recompensa="simple")
        check_env(env, warn=True)

    def test_gymnasium_check_env_completo(self, sim):
        env = EntornoVehiculo(sim, modo_recompensa="completa")
        check_env(env, warn=True)

    def test_episodio_completo_sin_crash(self, env_vehiculo):
        obs, _ = env_vehiculo.reset()
        for _ in range(100):
            accion = env_vehiculo.action_space.sample()
            obs, reward, done, _, _ = env_vehiculo.step(accion)
            assert obs.shape == (6,)
            if done:
                break

    def test_render_no_explota(self, env_vehiculo):
        env_vehiculo.reset()
        env_vehiculo.step(1)
        env_vehiculo.render()


# ============================================================
# Test de integración rápida con PPO
# ============================================================

class TestIntegracionPPO:

    def test_ppo_puede_entrenar_semaforo(self, sim):
        from stable_baselines3 import PPO
        env   = EntornoSemaforo(sim)
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=200)
        obs, _ = env.reset()
        accion, _ = model.predict(obs)
        assert int(accion) in [0, 1]

    def test_ppo_puede_entrenar_vehiculo(self, sim):
        from stable_baselines3 import PPO
        env   = EntornoVehiculo(sim)
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=200)
        obs, _ = env.reset()
        accion, _ = model.predict(obs)
        assert int(accion) in [0, 1, 2]
