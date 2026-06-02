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
    """Demo con SUMO visual — requiere SUMO instalado."""
    from simulacion.trafico_api import SimulacionTrafico
    from entornos.entorno_semaforo import EntornoSemaforo
    from entornos.entorno_vehiculo import EntornoVehiculo
    from stable_baselines3 import PPO

    sim = SimulacionTrafico(gui=True)
    sim.iniciar()
    env_s = EntornoSemaforo(sim)
    env_v = EntornoVehiculo(sim)

    modelo_s = PPO.load("modelos/semaforo_final", env=env_s)
    modelo_v = PPO.load("modelos/vehiculo_final", env=env_v)

    obs_s, _ = env_s.reset()
    obs_v, _ = env_v.reset()

    print("Corriendo demo visual... (cierra la ventana de SUMO para terminar)")
    done = False
    while not done:
        accion_s, _ = modelo_s.predict(obs_s, deterministic=True)
        accion_v, _ = modelo_v.predict(obs_v, deterministic=True)
        obs_s, _, done_s, _, _ = env_s.step(accion_s)
        obs_v, _, done_v, _, _ = env_v.step(accion_v)
        done = done_s or done_v

    sim.cerrar()


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
