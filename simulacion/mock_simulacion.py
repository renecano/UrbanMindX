# ============================================================
# mock_simulacion.py — Persona 1
# Simulación falsa para que Persona 2 y 3 puedan trabajar
# sin tener SUMO instalado.
# Simula el mismo contrato de SimulacionTrafico con datos aleatorios.
# ============================================================

import numpy as np


class SimulacionMock:
    """
    Reemplaza SimulacionTrafico durante el desarrollo.

    Persona 2 usa esto para probar los entornos sin SUMO.
    Persona 3 usa esto para probar el pipeline de entrenamiento.

    Cuando SUMO esté listo, solo se cambia:
        sim = SimulacionMock()
    por:
        sim = SimulacionTrafico()
    Y todo lo demás sigue igual.
    """

    def __init__(self, semilla: int = 42):
        self.rng           = np.random.default_rng(semilla)
        self._paso_actual  = 0
        self._fase_actual  = 0
        self._tiempo_fase  = 0
        self._max_pasos    = 500

    # ----------------------------------------------------------
    # Control de la simulación
    # ----------------------------------------------------------

    def iniciar(self):
        self._paso_actual = 0

    def resetear(self, config_path=None):
        self._paso_actual = 0
        self._fase_actual = 0
        self._tiempo_fase = 0

    def avanzar_paso(self):
        self._paso_actual  += 1
        self._tiempo_fase  += 1

    def cerrar(self):
        pass

    def simulacion_terminada(self) -> bool:
        return self._paso_actual >= self._max_pasos

    # ----------------------------------------------------------
    # Estado de la intersección
    # ----------------------------------------------------------

    def get_estado_interseccion(self) -> dict:
        autos = self.rng.integers(0, 25, size=4).tolist()
        esperas = (np.array(autos) * self.rng.uniform(1, 3, size=4)).tolist()
        return {
            "autos_norte":   autos[0],
            "autos_sur":     autos[1],
            "autos_este":    autos[2],
            "autos_oeste":   autos[3],
            "espera_norte":  esperas[0],
            "espera_sur":    esperas[1],
            "espera_este":   esperas[2],
            "espera_oeste":  esperas[3],
            "filas_norte":   max(0, autos[0] - 3),
            "filas_sur":     max(0, autos[1] - 3),
            "filas_este":    max(0, autos[2] - 3),
            "filas_oeste":   max(0, autos[3] - 3),
            "fase_actual":   self._fase_actual,
            "tiempo_fase":   self._tiempo_fase,
            "espera_total":  sum(esperas),
            "filas_total":   sum(max(0, a - 3) for a in autos),
        }

    def get_fase_semaforo(self) -> int:
        return self._fase_actual

    def set_fase_semaforo(self, fase: int):
        self._fase_actual = fase
        self._tiempo_fase = 0

    # ----------------------------------------------------------
    # Estado de vehículos
    # ----------------------------------------------------------

    def get_ids_vehiculos(self) -> list:
        n = self.rng.integers(5, 20)
        return [f"veh_{i}" for i in range(n)]

    def get_estado_vehiculo(self, vehicle_id: str) -> dict:
        return {
            "velocidad":           float(self.rng.uniform(0, 14)),
            "velocidad_max":       13.9,
            "distancia_al_frente": float(self.rng.uniform(5, 100)),
            "distancia_semaforo":  float(self.rng.uniform(0, 100)),
            "semaforo_estado":     self.rng.choice(["G", "r"]),
            "carril_actual":       int(self.rng.integers(0, 2)),
            "tiempo_detenido":     float(self.rng.uniform(0, 30)),
            "aceleracion":         float(self.rng.uniform(-3, 3)),
            "posicion":            (float(self.rng.uniform(-50, 50)),
                                    float(self.rng.uniform(-50, 50))),
        }

    def set_velocidad_vehiculo(self, vehicle_id: str, velocidad: float):
        pass  # en mock no hay nada que hacer

    # ----------------------------------------------------------
    # Métricas globales
    # ----------------------------------------------------------

    def get_metricas_globales(self) -> dict:
        return {
            "tiempo_simulacion":  float(self._paso_actual),
            "autos_activos":      int(self.rng.integers(5, 20)),
            "velocidad_promedio": float(self.rng.uniform(4, 10)),
            "espera_total":       float(self.rng.uniform(0, 200)),
            "filas_total":        int(self.rng.integers(0, 30)),
            "autos_llegaron":     int(self._paso_actual // 10),
        }
