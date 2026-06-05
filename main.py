# ============================================================
# main.py — Punto de entrada del proyecto
# Ejecuta entrenamiento, evaluacion, demo o dashboard.
# ============================================================
#
# Uso:
#   python main.py --modo evaluar          (evalua los escenarios)
#   python main.py --modo demo             (SUMO visual)
#   python main.py --modo dashboard        (dashboard streamlit)
#   python main.py --modo entrenar-semaforo   (reentrena SOLO semaforo)
#   python main.py --modo entrenar-vehiculo   (reentrena SOLO vehiculo)
#
# NOTA: el entrenamiento pide confirmacion antes de sobrescribir
#       un modelo existente, para no perder los modelos buenos.
# ============================================================

import argparse
import os
import subprocess


def _confirmar_sobrescritura(ruta_modelo: str, nombre: str) -> bool:
    """Pide confirmacion si el modelo ya existe."""
    if os.path.exists(ruta_modelo):
        print(f"\n⚠  ATENCION: ya existe un modelo de {nombre} en:")
        print(f"   {ruta_modelo}")
        print("   Reentrenar lo SOBRESCRIBIRA y perderas el modelo actual.")
        resp = input("   ¿Seguro que quieres reentrenar? (escribe 'si' para continuar): ")
        if resp.strip().lower() != "si":
            print("   Entrenamiento cancelado. Modelo actual conservado.")
            return False
    return True


def entrenar_semaforo():
    if not _confirmar_sobrescritura("modelos/semaforo_final.zip", "SEMAFORO"):
        return

    print("Entrenando SEMAFORO...")
    from simulacion.trafico_api import SimulacionTrafico
    from entornos.entorno_semaforo import EntornoSemaforo
    from entrenamiento.pipeline import PipelineUrbanMind

    sim = SimulacionTrafico(gui=False)
    sim.iniciar()
    env_s = EntornoSemaforo(sim, modo_recompensa="completa")
    pipeline = PipelineUrbanMind()
    pipeline.etapa_1_semaforo(env_s)
    sim.cerrar()
    print("\n✓ Semaforo entrenado.")


def entrenar_vehiculo():
    if not _confirmar_sobrescritura("modelos/vehiculo_final.zip", "VEHICULO"):
        return

    print("Entrenando VEHICULO...")
    from simulacion.trafico_api import SimulacionTrafico
    from entornos.entorno_vehiculo import EntornoVehiculo
    from entrenamiento.pipeline import PipelineUrbanMind

    sim = SimulacionTrafico(gui=False)
    sim.iniciar()
    env_v = EntornoVehiculo(sim, modo_recompensa="simple")
    pipeline = PipelineUrbanMind()
    pipeline.etapa_2_vehiculos(env_v)
    sim.cerrar()
    print("\n✓ Vehiculo entrenado.")


def evaluar():
    print("Evaluando escenarios...")
    from simulacion.trafico_api import SimulacionTrafico
    from entornos.entorno_semaforo import EntornoSemaforo
    from entornos.entorno_vehiculo import EntornoVehiculo
    from entrenamiento.evaluador import Evaluador

    sim = SimulacionTrafico(gui=False)
    sim.iniciar()
    env_s = EntornoSemaforo(sim)
    env_v = EntornoVehiculo(sim)
    evaluador = Evaluador(sim, env_s, env_v, episodios=5)
    evaluador.correr_todos()
    sim.cerrar()


def dashboard():
    print("Abriendo dashboard...")
    subprocess.run(["streamlit", "run", "dashboard/app.py"])


def demo():
    """Demo con SUMO visual: semaforo IA + vehiculos IA si existe el modelo."""
    from simulacion.trafico_api import SimulacionTrafico
    from entornos.entorno_semaforo import EntornoSemaforo
    from entornos.entorno_vehiculo import EntornoVehiculo
    from stable_baselines3 import PPO

    sim = SimulacionTrafico(gui=True)
    sim.iniciar()
    env_s = EntornoSemaforo(sim)
    env_v = EntornoVehiculo(sim)

    modelo_s = PPO.load("modelos/semaforo_final", env=env_s)

    # Cargar vehiculo solo si existe
    modelo_v = None
    if os.path.exists("modelos/vehiculo_final.zip"):
        modelo_v = PPO.load("modelos/vehiculo_final", env=env_v)
        print("Demo: semaforo IA + vehiculos IA")
    else:
        print("Demo: solo semaforo IA (no se encontro modelo de vehiculo)")

    obs_s, _ = env_s.reset()
    obs_v, _ = env_v.reset()

    print("(cierra la ventana de SUMO para terminar)")
    done = False
    pasos = 0
    while not done:
        accion_s, _ = modelo_s.predict(obs_s, deterministic=True)
        obs_s, _, done_s, _, info = env_s.step(int(accion_s))

        if modelo_v:
            accion_v, _ = modelo_v.predict(obs_v, deterministic=True)
            obs_v, _, done_v, _, _ = env_v.step(int(accion_v))
        else:
            done_v = False

        done = done_s or done_v
        pasos += 1
        if pasos % 500 == 0:
            print(f"Paso {pasos} | Fase verde: {info['fase_verde']}")

    sim.cerrar()
    print(f"\nDemo terminado tras {pasos} pasos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UrbanMind X")
    parser.add_argument(
        "--modo",
        choices=["evaluar", "demo", "dashboard",
                 "entrenar-semaforo", "entrenar-vehiculo"],
        default="evaluar",   # default seguro: evaluar, NO entrenar
        help="Que ejecutar",
    )
    args = parser.parse_args()

    if args.modo == "evaluar":
        evaluar()
    elif args.modo == "demo":
        demo()
    elif args.modo == "dashboard":
        dashboard()
    elif args.modo == "entrenar-semaforo":
        entrenar_semaforo()
    elif args.modo == "entrenar-vehiculo":
        entrenar_vehiculo()