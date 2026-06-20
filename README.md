# 🚦 UrbanMind X

**Sistema inteligente de control de tráfico basado en Aprendizaje por Refuerzo.**
Dos agentes —un semáforo adaptativo y vehículos autónomos— aprenden a coordinarse para reducir la congestión en una intersección real de Toluca, México.

> 🏆 **Primer lugar — Expo Ingenierías, Tecnológico de Monterrey Campus Toluca.**

---

## 📋 Resumen

Los semáforos de tiempo fijo no se adaptan al tráfico real: mantienen el mismo ciclo aunque una avenida esté saturada y la otra vacía. UrbanMind X ataca ese problema entrenando, mediante el algoritmo **PPO (Proximal Policy Optimization)**, dos agentes que observan el tráfico y actúan en tiempo real:

- 🚦 **Semáforo inteligente** — aprende *cuándo* cambiar de fase según la demanda de cada avenida.
- 🚗 **Vehículos autónomos** — aprenden a regular su velocidad para circular con fluidez.

El sistema se modeló sobre la intersección real de **Venustiano Carranza y Blvr. Pino Suárez**, usando flujos vehiculares del estudio Quivera (UAEM, 2022), y se entrenó en el simulador **SUMO**.

---

## 📊 Resultados

Evaluado sobre el tiempo de espera promedio en la intersección, comparado contra un semáforo tradicional de tiempo fijo:

| Escenario | Tiempo de espera | Mejora |
|-----------|------------------|--------|
| Semáforo tradicional (baseline) | 303.3 s | — |
| Solo semáforo IA | 192.6 s | **36.5 %** |
| Solo vehículos IA | 161.9 s | **46.6 %** |
| **Sistema completo** | **66.2 s** | **78.2 %** |
| Hora pico | 82.1 s | 72.9 % |
| Tráfico desbalanceado | 78.4 s | 74.1 % |

**Hallazgo principal:** existe una **sinergia** entre ambos agentes. Operando juntos logran una mejora (78.2 %) mayor a lo que sus aportes individuales harían esperar — la infraestructura y los vehículos inteligentes se potencian entre sí. El sistema mantiene 72–74 % de mejora bajo condiciones adversas (hora pico y tráfico desbalanceado), lo que confirma su robustez.

---

## 🎥 Entregables

| Recurso | Descripción |
|---------|-------------|
| 📊 [Dashboard interactivo](https://renecano.github.io/UrbanMindX/) | Visualización de resultados y curvas de aprendizaje *(GitHub Pages)* |
| 📄 [Póster](entregables/UrbanMindX_Poster.pptx) | Póster científico presentado en la Expo |
| 📝 [Artículo](entregables/) | Documento en formato APA con la metodología completa |

---

## 🛠️ Arquitectura técnica

```
SUMO ←→ TraCI ←→ SimulacionTrafico
                        ↓
        EntornoSemaforo / EntornoVehiculo   (Gymnasium)
                        ↓
              PPO  (Stable-Baselines3 · PyTorch)
                        ↓
        modelos/semaforo_final.zip · vehiculo_final.zip
                        ↓
              resultados/comparativa.csv
                        ↓
              dashboard/  (visualización)
```

| Componente | Rol |
|------------|-----|
| **SUMO** | Simulador del tráfico (el mundo virtual) |
| **TraCI** | Puente entre Python y SUMO (leer estado / enviar órdenes) |
| **Gymnasium** | Estándar de entornos RL (`reset`, `step`, observación, recompensa) |
| **Stable-Baselines3** | Implementación del algoritmo PPO |
| **PyTorch** | Motor de cálculo de las redes neuronales |

### Diseño de los agentes

**Semáforo** — Observación: autos por dirección (N, S, E, O), tiempo en fase y fase actual. Acciones: mantener o cambiar de fase (con fase amarilla de transición y tiempo mínimo de fase). La recompensa premia el flujo de vehículos y penaliza la espera, las colas y los cambios prematuros.

**Vehículos** — Observación: velocidad, distancias al frente y al semáforo, estado del semáforo y tiempo detenido. Acciones: frenar, mantener o acelerar.

---

## 📁 Estructura del proyecto

```
UrbanMindX/
├── simulacion/              Interfaz sobre TraCI y definición de escenarios
│   ├── trafico_api.py
│   ├── mock_simulacion.py   Simulación de prueba sin SUMO
│   └── escenarios.py
├── entornos/                Entornos Gymnasium y funciones de recompensa
│   ├── entorno_semaforo.py
│   ├── entorno_vehiculo.py
│   └── recompensas.py
├── entrenamiento/           Pipeline de entrenamiento y evaluación
│   ├── config.py            Hiperparámetros
│   ├── pipeline.py          Entrenamiento en 3 etapas
│   ├── callbacks.py         Registro de métricas
│   └── evaluador.py         Evaluación de escenarios
├── sumo/                    Red de la intersección, rutas y configs
│   ├── red/
│   ├── rutas/
│   └── config/
├── dashboard/               Dashboard de resultados (HTML interactivo)
├── entregables/             Póster, artículo y material de la Expo
├── modelos/                 Modelos entrenados (.zip)
├── resultados/              Métricas en CSV
├── tests/                   Pruebas unitarias
├── main.py                  Punto de entrada
└── requirements.txt
```

---

## 🚀 Instalación y uso

### 1. Instalar SUMO

```bash
# Ubuntu/Debian
sudo apt-get install sumo sumo-tools sumo-doc
# macOS
brew install sumo
# Windows: instalador en https://sumo.dlr.de/docs/Downloads.php
```

### 2. Configurar la variable de entorno

```bash
# Linux/macOS (agregar a ~/.bashrc o ~/.zshrc)
export SUMO_HOME="/usr/share/sumo"
# Windows
set SUMO_HOME="C:\Program Files (x86)\Eclipse\Sumo"
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar

```bash
python main.py --modo evaluar      # Evalúa los escenarios y guarda métricas
python main.py --modo demo         # Demo visual con SUMO
python main.py --modo entrenar-semaforo    # Reentrenar el semáforo (opcional)
python main.py --modo entrenar-vehiculo    # Reentrenar el vehículo (opcional)
```

> ℹ️ El modo por defecto es `evaluar`. Los modos de entrenamiento piden confirmación antes de sobrescribir modelos existentes.

### Tests

```bash
pytest tests/ -v
```

---

## 👥 Equipo

Proyecto desarrollado por estudiantes del Tecnológico de Monterrey Campus Toluca.

| | Integrante | Responsabilidad |
|---|------------|-----------------|
| **Persona 1** | Ileana Tapia | Simulación SUMO: red de la intersección, rutas de tráfico e interfaz TraCI (`simulacion/`, `sumo/`) |
| **Persona 2** | René Cano | Entornos de aprendizaje y diseño de recompensas: agente semáforo (`entornos/`, `recompensas.py`) |
| **Persona 3** | Sebastián Plata | Pipeline de entrenamiento y agente de vehículos autónomos (`entrenamiento/`, entorno del vehículo) |

---

## 📚 Referencias

- Rivera-Sánchez, et al. (2022). *Aforos de campo en la Ecozona de Toluca.* Quivera, 24(2). UAEMéx.
- Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Raffin, A., et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* JMLR, 22(268).
- Lopez, P. A., et al. (2018). *Microscopic Traffic Simulation using SUMO.* IEEE ITSC.
- Towers, M., et al. (2023). *Gymnasium.* Farama Foundation.

---

## 📄 Licencia

Proyecto académico desarrollado con fines educativos.
