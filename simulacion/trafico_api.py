# ============================================================
# trafico_api.py — Persona 1
# Capa de abstracción sobre TraCI.
# Las otras personas usan esta clase, nunca TraCI directo.
# ============================================================

import os
import sys
import traci
import sumolib
from entrenamiento.config import SUMO_CONFIG, PUERTO_SUMO, PASO_SIMULACION

# SUMO_HOME debe estar definido como variable de entorno
if "SUMO_HOME" not in os.environ:
    raise EnvironmentError(
        "Variable SUMO_HOME no definida.\n"
        "Agrégala con: export SUMO_HOME='/usr/share/sumo'"
    )

sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))

# IDs de los elementos en SUMO (deben coincidir con los archivos .xml)
EDGES = {
    "norte": "norte_entrada",
    "sur":   "sur_entrada",
    "este":  "este_entrada",
    "oeste": "oeste_entrada",
}
SEMAFORO_ID = "centro"


class SimulacionTrafico:
    """
    Interfaz principal entre Python y SUMO.
    Persona 1 es responsable de mantener esta clase.

    Uso básico:
        sim = SimulacionTrafico()
        sim.iniciar()
        estado = sim.get_estado_interseccion()
        sim.set_fase_semaforo(1)
        sim.avanzar_paso()
        sim.cerrar()
    """

    def __init__(self, config_path: str = SUMO_CONFIG, gui: bool = False):
        self.config_path = config_path
        self.gui         = gui         # True para ver la simulación visualmente
        self._activa     = False

    # ----------------------------------------------------------
    # Control de la simulación
    # ----------------------------------------------------------

    def iniciar(self):
        """Arranca SUMO y abre la conexión TraCI."""
        binario = "sumo-gui" if self.gui else "sumo"
        cmd = [binario, "-c", self.config_path,
               "--no-warnings", "true",
               "--step-length", str(PASO_SIMULACION)]
        traci.start(cmd, port=PUERTO_SUMO)
        self._activa = True

    def resetear(self, config_path: str = None):
        """Reinicia la simulación sin cerrar la conexión."""
        path = config_path or self.config_path
        traci.load(["-c", path, "--no-warnings", "true"])

    def avanzar_paso(self):
        """Avanza la simulación 1 paso de tiempo."""
        traci.simulationStep()

    def cerrar(self):
        """Cierra la conexión con SUMO."""
        if self._activa:
            traci.close()
            self._activa = False

    def simulacion_terminada(self) -> bool:
        """True cuando no quedan vehículos por entrar o cruzar."""
        return traci.simulation.getMinExpectedNumber() == 0

    # ----------------------------------------------------------
    # Estado de la intersección
    # ----------------------------------------------------------

    def get_estado_interseccion(self) -> dict:
        """
        Regresa el estado completo de la intersección.
        Este es el dato principal que usa EntornoSemaforo.
        """
        return {
            "autos_norte":   traci.edge.getLastStepVehicleNumber(EDGES["norte"]),  # Python pregunta: ¿cuántos autos hay en norte_entrada?
            "autos_sur":     traci.edge.getLastStepVehicleNumber(EDGES["sur"]),
            "autos_este":    traci.edge.getLastStepVehicleNumber(EDGES["este"]),
            "autos_oeste":   traci.edge.getLastStepVehicleNumber(EDGES["oeste"]),
            "espera_norte":  traci.edge.getWaitingTime(EDGES["norte"]), # Python pregunta: ¿cuántos segundos llevan esperando?
            "espera_sur":    traci.edge.getWaitingTime(EDGES["sur"]),
            "espera_este":   traci.edge.getWaitingTime(EDGES["este"]),
            "espera_oeste":  traci.edge.getWaitingTime(EDGES["oeste"]),
            "filas_norte":   traci.edge.getLastStepHaltingNumber(EDGES["norte"]), # Python pregunta: ¿cuántos están detenidos?
            "filas_sur":     traci.edge.getLastStepHaltingNumber(EDGES["sur"]),
            "filas_este":    traci.edge.getLastStepHaltingNumber(EDGES["este"]),
            "filas_oeste":   traci.edge.getLastStepHaltingNumber(EDGES["oeste"]),
            "fase_actual":   traci.trafficlight.getPhase(SEMAFORO_ID), #Python pregunta: ¿en qué fase está el semáforo?
            "tiempo_fase":   traci.trafficlight.getPhaseDuration(SEMAFORO_ID), #Python pregunta: ¿cuánto lleva en esa fase?
            "espera_total":  self._calcular_espera_total(),
            "filas_total":   self._calcular_filas_total(),
        }

    def _calcular_espera_total(self) -> float:
        return sum(traci.edge.getWaitingTime(e) for e in EDGES.values())

    def _calcular_filas_total(self) -> int:
        return sum(traci.edge.getLastStepHaltingNumber(e) for e in EDGES.values())

    # ----------------------------------------------------------
    # Control del semáforo
    # ----------------------------------------------------------

    def get_fase_semaforo(self) -> int:
        return traci.trafficlight.getPhase(SEMAFORO_ID)

    def set_fase_semaforo(self, fase: int):
        """
        Cambia la fase del semáforo.
        Fase 0 = Norte-Sur en verde
        Fase 1 = Este-Oeste en verde
        """
        traci.trafficlight.setPhase(SEMAFORO_ID, fase)

    # ----------------------------------------------------------
    # Estado de vehículos individuales
    # ----------------------------------------------------------

    def get_ids_vehiculos(self) -> list:
        """Lista de IDs de todos los vehículos activos."""
        return traci.vehicle.getIDList()

    def get_estado_vehiculo(self, vehicle_id: str) -> dict:
        """
        Regresa el estado de un vehículo específico.
        Este es el dato principal que usa EntornoVehiculo.
        """
        lider = traci.vehicle.getLeader(vehicle_id, dist=100)
        tls   = traci.vehicle.getNextTLS(vehicle_id)

        return {
            "velocidad":           traci.vehicle.getSpeed(vehicle_id),  # Python pregunta: ¿a qué velocidad va este auto?
            "velocidad_max":       traci.vehicle.getMaxSpeed(vehicle_id),
            "distancia_al_frente": lider[1] if lider else 100.0, # Python pregunta: ¿qué tan lejos está el auto de enfrente?
            "distancia_semaforo":  tls[0][2] if tls else 100.0,   # Python pregunta: ¿qué tan lejos está el semáforo y de qué color?
            "semaforo_estado":     tls[0][3] if tls else "G",  # G=verde R=rojo
            "carril_actual":       traci.vehicle.getLaneIndex(vehicle_id),
            "tiempo_detenido":     traci.vehicle.getWaitingTime(vehicle_id), # Python pregunta: ¿cuánto lleva detenido?
            "aceleracion":         traci.vehicle.getAcceleration(vehicle_id), # Python pregunta: ¿cuál es su aceleración actual?
            "posicion":            traci.vehicle.getPosition(vehicle_id),
        }

    def set_velocidad_vehiculo(self, vehicle_id: str, velocidad: float):
        """Fuerza la velocidad de un vehículo (m/s)."""
        traci.vehicle.setSpeed(vehicle_id, max(0.0, velocidad))

    # ----------------------------------------------------------
    # Métricas globales para el dashboard
    # ----------------------------------------------------------

    def get_metricas_globales(self) -> dict:
        """Snapshot de métricas para guardar en CSV."""
        ids = self.get_ids_vehiculos()
        velocidades = [traci.vehicle.getSpeed(v) for v in ids] if ids else [0]

        return {
            "tiempo_simulacion":   traci.simulation.getTime(),
            "autos_activos":       len(ids),
            "velocidad_promedio":  sum(velocidades) / len(velocidades),
            "espera_total":        self._calcular_espera_total(),
            "filas_total":         self._calcular_filas_total(),
            "autos_llegaron":      traci.simulation.getArrivedNumber(),
        }
