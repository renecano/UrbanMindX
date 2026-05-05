# ============================================================
# test_simulacion.py — Persona 1
# Tests para SimulacionMock (y luego para SimulacionTrafico).
# ============================================================

import pytest
from simulacion.mock_simulacion import SimulacionMock


@pytest.fixture
def sim():
    s = SimulacionMock(semilla=42)
    s.iniciar()
    return s


class TestSimulacionMock:

    def test_estado_interseccion_tiene_todas_las_claves(self, sim):
        claves_esperadas = [
            "autos_norte", "autos_sur", "autos_este", "autos_oeste",
            "espera_total", "filas_total", "fase_actual", "tiempo_fase"
        ]
        estado = sim.get_estado_interseccion()
        for clave in claves_esperadas:
            assert clave in estado, f"Falta clave: {clave}"

    def test_autos_son_enteros_no_negativos(self, sim):
        estado = sim.get_estado_interseccion()
        for dir in ["norte", "sur", "este", "oeste"]:
            assert estado[f"autos_{dir}"] >= 0

    def test_set_fase_cambia_fase(self, sim):
        sim.set_fase_semaforo(0)
        assert sim.get_fase_semaforo() == 0
        sim.set_fase_semaforo(1)
        assert sim.get_fase_semaforo() == 1

    def test_avanzar_paso_incrementa_tiempo(self, sim):
        sim.resetear()
        sim.avanzar_paso()
        sim.avanzar_paso()
        assert sim._paso_actual == 2

    def test_resetear_vuelve_a_cero(self, sim):
        for _ in range(10):
            sim.avanzar_paso()
        sim.resetear()
        assert sim._paso_actual == 0

    def test_estado_vehiculo_tiene_velocidad(self, sim):
        ids = sim.get_ids_vehiculos()
        assert len(ids) > 0
        estado = sim.get_estado_vehiculo(ids[0])
        assert "velocidad" in estado
        assert 0 <= estado["velocidad"] <= 14

    def test_simulacion_termina(self, sim):
        # Avanzar hasta el máximo
        for _ in range(sim._max_pasos):
            sim.avanzar_paso()
        assert sim.simulacion_terminada() is True
