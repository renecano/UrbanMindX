# 🚦 UrbanMind X

Sistema Inteligente de Semáforos y Vehículos Autónomos basado en Aprendizaje por Refuerzo.

---

## Estructura del proyecto

```
UrbanMindX/
├── simulacion/           ← PERSONA 1
│   ├── trafico_api.py    │  Interfaz sobre TraCI (requiere SUMO)
│   ├── mock_simulacion.py│  Simulación falsa para desarrollo sin SUMO
│   └── escenarios.py     │  Definición de los 5 escenarios
│
├── entornos/             ← PERSONA 2
│   ├── entorno_semaforo.py  │  Entorno Gymnasium del semáforo
│   ├── entorno_vehiculo.py  │  Entorno Gymnasium del vehículo
│   └── recompensas.py       │  Funciones de recompensa (ajustar aquí)
│
├── entrenamiento/        ← PERSONA 3
│   ├── config.py         │  TODOS los hiperparámetros del proyecto
│   ├── pipeline.py       │  Entrenamiento en 3 etapas
│   ├── callbacks.py      │  Guardar métricas durante entrenamiento
│   └── evaluador.py      │  Correr los 5 escenarios y comparar
│
├── sumo/                 ← PERSONA 1
│   ├── red/              │  Geometría de la intersección
│   ├── rutas/            │  Flujos de tráfico por escenario
│   └── config/           │  Archivos .sumocfg
│
├── dashboard/            ← TODOS al final
│   ├── app.py            │  App Streamlit principal
│   └── graficas.py       │  Funciones de visualización Plotly
│
├── tests/                ← TODOS (cada quien sus tests)
│   ├── test_simulacion.py
│   ├── test_entornos.py
│   └── test_entrenamiento.py
│
├── modelos/              ← Modelos entrenados (.zip) — generados
├── resultados/           ← CSVs de métricas — generados
├── main.py               ← Punto de entrada
└── requirements.txt
```

---

## Instalación

### 1. Instalar SUMO
```bash
# Ubuntu/Debian
sudo apt-get install sumo sumo-tools sumo-doc

# Mac
brew install sumo

# Windows: descargar instalador en https://sumo.dlr.de/docs/Downloads.php
```

### 2. Variable de entorno
```bash
# Linux/Mac — agregar a ~/.bashrc o ~/.zshrc
export SUMO_HOME="/usr/share/sumo"

# Windows
set SUMO_HOME="C:\Program Files (x86)\Eclipse\Sumo"
```

### 3. Instalar dependencias Python
```bash
pip install -r requirements.txt
```

---

## Uso

```bash
# Entrenar los modelos (etapas 1, 2 y 3)
python main.py --modo entrenar

# Evaluar los 5 escenarios y guardar métricas
python main.py --modo evaluar

# Abrir el dashboard
python main.py --modo dashboard

# Demo visual con SUMO (requiere SUMO instalado)
python main.py --modo demo
```

## Tests

```bash
pytest tests/ -v
```

---

## División del equipo

| Persona | Archivos principales | Prioridad |
|---------|---------------------|-----------|
| Persona 1 | `simulacion/trafico_api.py`, `sumo/` | Alta — todos dependen de esto |
| Persona 2 | `entornos/`, `entornos/recompensas.py` | Alta — bloquea a Persona 3 |
| Persona 3 | `entrenamiento/pipeline.py`, `entrenamiento/evaluador.py` | Media — usa mock mientras espera |

---

## Flujo de datos

```
SUMO ←→ TraCI ←→ SimulacionTrafico
                        ↓
              EntornoSemaforo / EntornoVehiculo  (Gymnasium)
                        ↓
                   PPO (Stable-Baselines3)
                        ↓
              modelos/semaforo_final.zip
              modelos/vehiculo_final.zip
                        ↓
              resultados/comparativa.csv
                        ↓
              dashboard/app.py (Streamlit)
```
