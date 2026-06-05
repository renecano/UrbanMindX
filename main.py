# ============================================================
# main.py — Punto de entrada del proyecto
# Ejecuta entrenamiento, evaluación o dashboard según argumento.
# ============================================================
#
# Uso:
#   python main.py --modo entrenar
#   python main.py --modo evaluar
#   python main.py --modo dashboard
#   python main.py --modo demo        (con SUMO visual)
#
# ============================================================

import argparse
import subprocess


def entrenar():
    print("Iniciando entrenamiento UrbanMind X...")

    from simulacion.trafico_api import SimulacionTrafico
    from entornos.entorno_semaforo import EntornoSemaforo
    from entornos.entorno_vehiculo import EntornoVehiculo
    from entrenamiento.pipeline import PipelineUrbanMind

    sim = SimulacionTrafico(gui=False)
    sim.iniciar()

    env_s = EntornoSemaforo(sim, modo_recompensa="completa")
    env_v = EntornoVehiculo(sim, modo_recompensa="simple")

    pipeline = PipelineUrbanMind()
    pipeline.etapa_1_semaforo(env_s)
    #pipeline.etapa_2_vehiculos(env_v)
    #pipeline.etapa_3_conjunto(env_s, env_v)

    sim.cerrar()
    print("\n✓ Etapa 1 completada. Modelo semaforo guardado en /modelos")


def evaluar():
    print("Evaluando escenarios...")

    # Cambiado: usar SUMO real (TraCI) en vez de mock
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
    """Demo con SUMO visual — solo semaforo (vehiculos aun no entrenados)."""
    from simulacion.trafico_api import SimulacionTrafico
    from entornos.entorno_semaforo import EntornoSemaforo
    from stable_baselines3 import PPO

    sim = SimulacionTrafico(gui=True)
    sim.iniciar()
    env_s = EntornoSemaforo(sim)

    modelo_s = PPO.load("modelos/semaforo_final", env=env_s)

    obs_s, _ = env_s.reset()

    print("Corriendo demo visual del semaforo IA...")
    print("(cierra la ventana de SUMO para terminar)")
    done = False
    pasos = 0
    while not done:
        accion_s, _ = modelo_s.predict(obs_s, deterministic=True)
        obs_s, _, done, _, info = env_s.step(int(accion_s))
        pasos += 1
        if pasos % 500 == 0:
            print(f"Paso {pasos} | Fase verde: {info['fase_verde']}")

    sim.cerrar()
    print(f"\nDemo terminado tras {pasos} pasos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UrbanMind X")
    parser.add_argument(
        "--modo",
        choices=["entrenar", "evaluar", "dashboard", "demo"],
        default="entrenar",
        help="Qué ejecutar",
    )
    args = parser.parse_args()

    if args.modo == "entrenar":
        entrenar()
    elif args.modo == "evaluar":
        evaluar()
    elif args.modo == "dashboard":
        dashboard()
    elif args.modo == "demo":
        demo()
