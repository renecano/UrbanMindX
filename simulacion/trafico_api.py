# ============================================================
# simulacion/trafico_api.py — UrbanMind X
# API TraCI completa para la intersección real de Toluca.
# Venustiano Carranza × Blvr. Pino Suárez
#
# Persona 2 (RL) consume SOLO esta API — nunca llama TraCI directamente.
#
# Métodos públicos:
#   iniciar()                        — arranca SUMO + conecta TraCI
#   cerrar()                         — desconecta TraCI + mata proceso
#   resetear()                       — reinicia episodio sin relanzar SUMO
#   avanzar_paso()                   — avanza 1 segundo de simulación
#   simulacion_terminada() → bool    — True cuando llegamos a end o error
#   get_estado_interseccion() → dict — colas, esperas, filas para semáforo
#   get_estado_vehiculo(id) → dict   — estado de un vehículo específico
#   get_ids_vehiculos() → list[str]  — IDs de vehículos activos
#   set_fase_semaforo(fase: int)     — cambia fase del semáforo (0-3)
#   set_velocidad_vehiculo(id, v)    — fija velocidad de un vehículo
#   get_metricas_globales() → dict   — throughput y estadísticas globales
# ============================================================

import os
import sys
import subprocess
import time
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Constantes de la intersección
# ──────────────────────────────────────────────────────────────
TL_ID          = "interseccion"      # ID del semáforo en interseccion.net.xml
FASE_NS        = 0                   # Verde Norte-Sur
FASE_AMARILLO_NS = 1
FASE_EO        = 2                   # Verde Este-Oeste
FASE_AMARILLO_EO = 3

EDGES_ENTRADA  = ["norte_entrada", "sur_entrada", "oeste_entrada", "este_entrada"]
EDGES_SALIDA   = ["norte_salida",  "sur_salida",  "oeste_salida",  "este_salida"]

# Distancia de detección de cola (m) — cuánto mira SUMO hacia atrás del semáforo
DIST_DETECCION = 200.0

# Puerto TCP para TraCI (cambiar si hay conflicto)
PUERTO_TRACI   = 8813

# Ruta base del proyecto (dos niveles arriba de este archivo)
BASE_DIR = Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────────
# Clase principal
# ──────────────────────────────────────────────────────────────
class SimulacionTrafico:
    """
    Wrapper completo sobre SUMO/TraCI para UrbanMind X.

    Uso básico:
        sim = SimulacionTrafico(gui=False)
        sim.iniciar()
        while not sim.simulacion_terminada():
            sim.avanzar_paso()
            estado = sim.get_estado_interseccion()
        sim.cerrar()

    Para cambiar escenario:
        sim = SimulacionTrafico(gui=False, escenario="hora_pico")
    """

    ESCENARIOS = {
        "normal":        "sumo/config/simulacion.sumocfg",
        "hora_pico":     "sumo/config/hora_pico.sumocfg",
        "desbalanceado": "sumo/config/desbalanceado.sumocfg",
    }

    def __init__(self, gui: bool = False, escenario: str = "normal",
                 puerto: int = PUERTO_TRACI):
        self.gui       = gui
        self.escenario = escenario
        self.puerto    = puerto
        self._sumo_proc: Optional[subprocess.Popen] = None
        self._traci    = None          # módulo traci importado dinámicamente
        self._conectado = False
        self._paso_actual = 0
        self._paso_fin    = 0          # leído del .sumocfg en iniciar()
        self._autos_llegaron = 0       # contador acumulado para throughput

    # ──────────────────────────────────────────────────────────
    # CICLO DE VIDA
    # ──────────────────────────────────────────────────────────

    def iniciar(self) -> None:
        """Arranca SUMO y establece conexión TraCI."""
        if self._conectado:
            logger.warning("SimulacionTrafico.iniciar() llamado cuando ya está conectado.")
            return

        try:
            import traci
            self._traci = traci
        except ImportError as e:
            raise RuntimeError(
                "TraCI no encontrado. Instala SUMO y asegúrate de que "
                "SUMO_HOME esté configurado en las variables de entorno.\n"
                f"Error original: {e}"
            )

        config_relativa = self.ESCENARIOS.get(self.escenario)
        if not config_relativa:
            raise ValueError(
                f"Escenario '{self.escenario}' desconocido. "
                f"Opciones: {list(self.ESCENARIOS.keys())}"
            )

        config_path = BASE_DIR / config_relativa
        if not config_path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo de configuración SUMO:\n{config_path}\n"
                "Verifica que la ruta BASE_DIR sea correcta."
            )

        # Leer duración de simulación del .sumocfg
        self._paso_fin = self._leer_end_sumocfg(config_path)

        # Binario de SUMO
        sumo_bin = "sumo-gui" if self.gui else "sumo"
        sumo_home = os.environ.get("SUMO_HOME", "")
        if sumo_home:
            candidato = Path(sumo_home) / "bin" / sumo_bin
            if candidato.exists():
                sumo_bin = str(candidato)

        cmd = [
            sumo_bin,
            "--configuration-file", str(config_path),
            "--remote-port",        str(self.puerto),
            "--no-step-log",        "true",
            "--waiting-time-memory","1000",
            "--quit-on-end",        "true",
        ]

        logger.info(f"Arrancando SUMO: {' '.join(cmd)}")
        self._sumo_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Esperar a que SUMO levante el servidor TraCI
        time.sleep(1.5)

        try:
            self._traci.init(port=self.puerto, numRetries=10)
            self._conectado = True
            self._paso_actual = 0
            self._autos_llegaron = 0
            logger.info(
                f"TraCI conectado. Escenario: {self.escenario} | "
                f"Duración: {self._paso_fin}s | GUI: {self.gui}"
            )
        except Exception as e:
            self._matar_proceso()
            raise RuntimeError(
                f"No se pudo conectar a TraCI en puerto {self.puerto}.\n"
                f"Error: {e}"
            )

    def cerrar(self) -> None:
        """Desconecta TraCI y termina el proceso SUMO."""
        if self._conectado:
            try:
                self._traci.close()
            except Exception:
                pass
            self._conectado = False
        self._matar_proceso()
        logger.info("SimulacionTrafico cerrada.")

    def resetear(self) -> None:
        """
        Reinicia la simulación para un nuevo episodio.
        Cierra TraCI, mata SUMO y vuelve a iniciar.
        Más lento que un step, pero garantiza estado limpio.
        """
        self.cerrar()
        time.sleep(0.5)
        self.iniciar()

    # ──────────────────────────────────────────────────────────
    # CONTROL DE SIMULACIÓN
    # ──────────────────────────────────────────────────────────

    def avanzar_paso(self) -> None:
        """Avanza exactamente 1 step (1 segundo) en SUMO."""
        self._verificar_conexion()
        self._traci.simulationStep()
        self._paso_actual += 1
        # Acumular throughput
        llegados = self._traci.simulation.getArrivedNumber()
        self._autos_llegaron += llegados

    def simulacion_terminada(self) -> bool:
        """True si SUMO terminó o el paso actual superó el límite."""
        if not self._conectado:
            return True
        try:
            min_esperados = self._traci.simulation.getMinExpectedNumber()
            sim_time      = self._traci.simulation.getTime()
            return (
                sim_time >= self._paso_fin
                or (min_esperados == 0 and self._paso_actual > 10)
            )
        except Exception:
            return True

    # ──────────────────────────────────────────────────────────
    # SEMÁFORO
    # ──────────────────────────────────────────────────────────

    def set_fase_semaforo(self, fase: int) -> None:
        """
        Cambia la fase del semáforo.
        Fases válidas: 0 (Verde NS), 1 (Amarillo NS), 2 (Verde EO), 3 (Amarillo EO)
        """
        self._verificar_conexion()
        if fase not in (0, 1, 2, 3):
            raise ValueError(f"Fase inválida: {fase}. Usar 0, 1, 2 o 3.")
        self._traci.trafficlight.setPhase(TL_ID, fase)

    def get_fase_semaforo(self) -> int:
        """Devuelve la fase actual del semáforo."""
        self._verificar_conexion()
        return self._traci.trafficlight.getPhase(TL_ID)

    # ──────────────────────────────────────────────────────────
    # ESTADO — INTERSECCIÓN (para agente semáforo)
    # ──────────────────────────────────────────────────────────

    def get_estado_interseccion(self) -> dict:
        """
        Devuelve el estado actual de la intersección.

        Retorna:
            dict con:
                autos_norte   (int)   — vehículos esperando en norte_entrada
                autos_sur     (int)   — vehículos esperando en sur_entrada
                autos_este    (int)   — vehículos esperando en este_entrada
                autos_oeste   (int)   — vehículos esperando en oeste_entrada
                espera_total  (float) — suma de tiempos de espera (segundos)
                filas_total   (int)   — suma de longitud de colas (vehículos)
                fase_actual   (int)   — fase del semáforo (0-3)
                tiempo_sim    (float) — tiempo de simulación actual (s)
        """
        self._verificar_conexion()

        autos = {}
        espera_total = 0.0
        filas_total  = 0

        for edge in EDGES_ENTRADA:
            direccion = edge.replace("_entrada", "")  # "norte", "sur", etc.
            n_autos   = self._traci.edge.getLastStepHaltingNumber(edge)
            espera    = self._traci.edge.getWaitingTime(edge)
            autos[direccion] = n_autos
            espera_total += espera
            filas_total  += n_autos

        return {
            "autos_norte":  autos.get("norte",  0),
            "autos_sur":    autos.get("sur",    0),
            "autos_este":   autos.get("este",   0),
            "autos_oeste":  autos.get("oeste",  0),
            "espera_total": espera_total,
            "filas_total":  filas_total,
            "fase_actual":  self._traci.trafficlight.getPhase(TL_ID),
            "tiempo_sim":   self._traci.simulation.getTime(),
        }

    # ──────────────────────────────────────────────────────────
    # ESTADO — VEHÍCULO INDIVIDUAL (para agente vehículo)
    # ──────────────────────────────────────────────────────────

    def get_estado_vehiculo(self, vehicle_id: str) -> dict:
        """
        Devuelve el estado de un vehículo específico.

        Retorna:
            dict con:
                velocidad          (float) — m/s actual
                velocidad_max      (float) — m/s máximo permitido
                aceleracion        (float) — m/s² actual
                distancia_al_frente(float) — metros al vehículo de enfrente (100 si libre)
                distancia_semaforo (float) — metros al próximo semáforo
                semaforo_estado    (str)   — "G" verde, "y" amarillo, "r" rojo
                carril_actual      (int)   — índice de carril (0, 1, 2...)
                tiempo_detenido    (float) — segundos acumulados detenido
        """
        self._verificar_conexion()
        tc = self._traci.vehicle

        velocidad     = tc.getSpeed(vehicle_id)
        velocidad_max = tc.getMaxSpeed(vehicle_id)
        aceleracion   = tc.getAcceleration(vehicle_id)
        carril        = tc.getLaneIndex(vehicle_id)
        detenido      = tc.getWaitingTime(vehicle_id)

        # Distancia al frente (líder en el mismo carril)
        leader = tc.getLeader(vehicle_id, dist=100.0)
        dist_frente = leader[1] if leader and leader[1] >= 0 else 100.0

        # Semáforo más cercano por delante
        tls_info = tc.getNextTLS(vehicle_id)
        if tls_info:
            dist_semaforo   = tls_info[0][2]
            semaforo_estado = tls_info[0][3]   # 'G', 'y', 'r', etc.
        else:
            dist_semaforo   = 200.0
            semaforo_estado = "G"

        return {
            "velocidad":           velocidad,
            "velocidad_max":       velocidad_max,
            "aceleracion":         aceleracion,
            "distancia_al_frente": dist_frente,
            "distancia_semaforo":  dist_semaforo,
            "semaforo_estado":     semaforo_estado,
            "carril_actual":       carril,
            "tiempo_detenido":     detenido,
        }

    # ──────────────────────────────────────────────────────────
    # CONTROL DE VEHÍCULO
    # ──────────────────────────────────────────────────────────

    def set_velocidad_vehiculo(self, vehicle_id: str, velocidad: float) -> None:
        """
        Fija la velocidad de un vehículo (m/s).
        SUMO respetará esta velocidad durante el siguiente step.
        """
        self._verificar_conexion()
        velocidad = max(0.0, velocidad)
        self._traci.vehicle.setSpeed(vehicle_id, velocidad)

    # ──────────────────────────────────────────────────────────
    # IDs Y MÉTRICAS GLOBALES
    # ──────────────────────────────────────────────────────────

    def get_ids_vehiculos(self) -> list:
        """Devuelve lista de IDs de vehículos activos en la simulación."""
        self._verificar_conexion()
        return list(self._traci.vehicle.getIDList())

    def get_metricas_globales(self) -> dict:
        """
        Devuelve métricas globales de la simulación.

        Retorna:
            dict con:
                autos_llegaron  (int)   — total de vehículos que completaron su ruta
                autos_en_red    (int)   — vehículos activos en este momento
                tiempo_sim      (float) — tiempo actual de simulación (s)
                paso_actual     (int)   — número de step actual
        """
        self._verificar_conexion()
        return {
            "autos_llegaron": self._autos_llegaron,
            "autos_en_red":   self._traci.vehicle.getIDCount(),
            "tiempo_sim":     self._traci.simulation.getTime(),
            "paso_actual":    self._paso_actual,
        }

    # ──────────────────────────────────────────────────────────
    # HELPERS PRIVADOS
    # ──────────────────────────────────────────────────────────

    def _verificar_conexion(self) -> None:
        """Lanza RuntimeError si TraCI no está conectado."""
        if not self._conectado:
            raise RuntimeError(
                "SimulacionTrafico no está conectada. "
                "Llama sim.iniciar() antes de usar la API."
            )

    def _matar_proceso(self) -> None:
        """Termina el proceso SUMO si sigue corriendo."""
        if self._sumo_proc and self._sumo_proc.poll() is None:
            try:
                self._sumo_proc.terminate()
                self._sumo_proc.wait(timeout=5)
            except Exception:
                try:
                    self._sumo_proc.kill()
                except Exception:
                    pass
        self._sumo_proc = None

    @staticmethod
    def _leer_end_sumocfg(config_path: Path) -> float:
        """Lee el valor de <end> del .sumocfg para saber cuándo termina la sim."""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(config_path)
            end_elem = tree.find(".//end")
            if end_elem is not None:
                return float(end_elem.get("value", 21600))
        except Exception as e:
            logger.warning(f"No se pudo leer <end> de {config_path}: {e}")
        return 21600.0  # default: 6 horas
