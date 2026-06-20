# 🚦 UrbanMind X

**Intelligent traffic control system powered by Reinforcement Learning.**
Two agents —an adaptive traffic light and autonomous vehicles— learn to coordinate and reduce congestion at a real intersection in Toluca, Mexico.

> 🏆 **First place — Expo Ingenierías, Tecnológico de Monterrey, Toluca Campus.**

---

## 📋 Overview

Fixed-time traffic lights don't adapt to real traffic: they keep the same cycle even when one avenue is saturated and the other is empty. UrbanMind X tackles this by training two agents with the **PPO (Proximal Policy Optimization)** algorithm to observe traffic and act in real time:

- 🚦 **Smart traffic light** — learns *when* to switch phases based on each avenue's demand.
- 🚗 **Autonomous vehicles** — learn to regulate their speed to keep traffic flowing.

The system was modeled on the real intersection of **Venustiano Carranza and Blvd. Pino Suárez**, using vehicle flow data from the Quivera study (UAEM, 2022), and trained in the **SUMO** simulator.

---

## 📊 Results

Measured by average waiting time at the intersection, compared against a traditional fixed-time traffic light:

| Scenario | Waiting time | Improvement |
|----------|--------------|-------------|
| Traditional traffic light (baseline) | 303.3 s | — |
| Traffic light AI only | 192.6 s | **36.5%** |
| Vehicles AI only | 161.9 s | **46.6%** |
| **Full system** | **66.2 s** | **78.2%** |
| Rush hour | 82.1 s | 72.9% |
| Unbalanced traffic | 78.4 s | 74.1% |

**Key finding:** there is a **synergy** between the two agents. Working together they achieve an improvement (78.2%) greater than their individual contributions would suggest — intelligent infrastructure and intelligent vehicles reinforce each other. The system holds 72–74% improvement under adverse conditions (rush hour and unbalanced traffic), confirming its robustness.

---

## 🎥 Deliverables

| Resource | Description |
|----------|-------------|
| 📊 [Interactive dashboard](https://urbanmindx.netlify.app/) | Results and learning curves visualization |

---

## 🛠️ Technical architecture

```
SUMO ←→ TraCI ←→ TrafficSimulation
                        ↓
        TrafficLightEnv / VehicleEnv      (Gymnasium)
                        ↓
              PPO  (Stable-Baselines3 · PyTorch)
                        ↓
        models/traffic_light_final.zip · vehicle_final.zip
                        ↓
              results/comparison.csv
                        ↓
              dashboard/  (visualization)
```

| Component | Role |
|-----------|------|
| **SUMO** | Traffic simulator (the virtual world) |
| **TraCI** | Bridge between Python and SUMO (read state / send commands) |
| **Gymnasium** | RL environment standard (`reset`, `step`, observation, reward) |
| **Stable-Baselines3** | PPO algorithm implementation |
| **PyTorch** | Neural network computation engine |

### Agent design

**Traffic light** — Observation: vehicles per direction (N, S, E, W), time in current phase, and current phase. Actions: keep or switch phase (with a yellow transition phase and a minimum phase time). The reward favors vehicle flow and penalizes waiting, queues, and premature phase changes.

**Vehicles** — Observation: speed, distance to the vehicle ahead, distance to the traffic light, traffic light state, and time stopped. Actions: brake, maintain, or accelerate.

---

## 📁 Project structure

```
UrbanMindX/
├── simulacion/              TraCI interface and scenario definitions
│   ├── trafico_api.py
│   ├── mock_simulacion.py   Mock simulation (runs without SUMO)
│   └── escenarios.py
├── entornos/                Gymnasium environments and reward functions
│   ├── entorno_semaforo.py
│   ├── entorno_vehiculo.py
│   └── recompensas.py
├── entrenamiento/           Training and evaluation pipeline
│   ├── config.py            Hyperparameters
│   ├── pipeline.py          3-stage training
│   ├── callbacks.py         Metrics logging
│   └── evaluador.py         Scenario evaluation
├── sumo/                    Intersection network, routes and configs
│   ├── red/
│   ├── rutas/
│   └── config/
├── dashboard/               Results dashboard (interactive HTML)
├── deliverables/            Poster, paper and Expo material
├── modelos/                 Trained models (.zip)
├── resultados/              Metrics in CSV
├── tests/                   Unit tests
├── main.py                  Entry point
└── requirements.txt
```

---

## 🚀 Installation & usage

### 1. Install SUMO

```bash
# Ubuntu/Debian
sudo apt-get install sumo sumo-tools sumo-doc
# macOS
brew install sumo
# Windows: installer at https://sumo.dlr.de/docs/Downloads.php
```

### 2. Set the environment variable

```bash
# Linux/macOS (add to ~/.bashrc or ~/.zshrc)
export SUMO_HOME="/usr/share/sumo"
# Windows
set SUMO_HOME="C:\Program Files (x86)\Eclipse\Sumo"
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run

```bash
python main.py --modo evaluar              # Evaluate scenarios and save metrics
python main.py --modo demo                 # Visual demo with SUMO
python main.py --modo entrenar-semaforo    # Retrain the traffic light (optional)
python main.py --modo entrenar-vehiculo    # Retrain the vehicle (optional)
```

> ℹ️ The default mode is `evaluar` (evaluate). Training modes ask for confirmation before overwriting existing models.

### Tests

```bash
pytest tests/ -v
```

---

## 👥 Team

Developed by students at Tecnológico de Monterrey, Toluca Campus.

| | Member | Responsibility |
|---|--------|----------------|
| **Person 1** | Ileana Tapia | SUMO simulation: intersection network, traffic routes, and TraCI interface (`simulacion/`, `sumo/`) |
| **Person 2** | René Cano | Learning environments and reward design: traffic light agent (`entornos/`, `recompensas.py`) |
| **Person 3** | Sebastián Plata | Training pipeline and autonomous vehicle agent (`entrenamiento/`, vehicle environment) |

---

## 📚 References

- Rivera-Sánchez, et al. (2022). *Field traffic counts in the Toluca Ecozone.* Quivera, 24(2). UAEMéx.
- Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Raffin, A., et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* JMLR, 22(268).
- Lopez, P. A., et al. (2018). *Microscopic Traffic Simulation using SUMO.* IEEE ITSC.
- Towers, M., et al. (2023). *Gymnasium.* Farama Foundation.

---

## 📄 License

Academic project developed for educational purposes.
