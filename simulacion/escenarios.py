# ============================================================
# escenarios.py — Persona 1
# Define los 5 escenarios de evaluación del proyecto.
# Cada escenario apunta a un archivo .rou.xml diferente.
# ============================================================

from dataclasses import dataclass
from typing import Optional


@dataclass
class Escenario:
    nombre:       str
    descripcion:  str
    config_sumo:  str              # archivo .sumocfg a usar
    semaforo_ia:  bool = False     # ¿usa semáforo entrenado?
    vehiculos_ia: bool = False     # ¿usa vehículos autónomos entrenados?
    modelo_semaforo: Optional[str] = None
    modelo_vehiculo: Optional[str] = None


ESCENARIOS = {

    "tradicional": Escenario(
        nombre       = "Sistema Tradicional",
        descripcion  = "Semáforo fijo + vehículos normales. Línea base.",
        config_sumo  = "sumo/config/simulacion.sumocfg",
        semaforo_ia  = False,
        vehiculos_ia = False,
    ),

    "semaforo_ia": Escenario(
        nombre          = "Semáforo Inteligente",
        descripcion     = "Semáforo entrenado con RL + vehículos normales.",
        config_sumo     = "sumo/config/simulacion.sumocfg",
        semaforo_ia     = True,
        vehiculos_ia    = False,
        modelo_semaforo = "modelos/semaforo_final",
    ),

    "sistema_completo": Escenario(
        nombre          = "UrbanMind X Completo",
        descripcion     = "Semáforo IA + vehículos autónomos IA.",
        config_sumo     = "sumo/config/simulacion.sumocfg",
        semaforo_ia     = True,
        vehiculos_ia    = True,
        modelo_semaforo = "modelos/semaforo_final",
        modelo_vehiculo = "modelos/vehiculo_final",
    ),

    "hora_pico": Escenario(
        nombre          = "Hora Pico",
        descripcion     = "Sistema completo bajo alta densidad de tráfico.",
        config_sumo     = "sumo/config/hora_pico.sumocfg",
        semaforo_ia     = True,
        vehiculos_ia    = True,
        modelo_semaforo = "modelos/semaforo_final",
        modelo_vehiculo = "modelos/vehiculo_final",
    ),

    "desbalanceado": Escenario(
        nombre          = "Tráfico Desbalanceado",
        descripcion     = "Una avenida con mucho más tráfico que la otra.",
        config_sumo     = "sumo/config/desbalanceado.sumocfg",
        semaforo_ia     = True,
        vehiculos_ia    = True,
        modelo_semaforo = "modelos/semaforo_final",
        modelo_vehiculo = "modelos/vehiculo_final",
    ),
}


def get_escenario(nombre: str) -> Escenario:
    if nombre not in ESCENARIOS:
        raise ValueError(f"Escenario '{nombre}' no existe. "
                         f"Opciones: {list(ESCENARIOS.keys())}")
    return ESCENARIOS[nombre]
