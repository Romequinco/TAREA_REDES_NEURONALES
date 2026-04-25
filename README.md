# Taller B3-T4/T5/T6 — Redes Neuronales para Forecasting Financiero

Taller universitario de Máster. Predice el precio de cierre promedio de 23 activos del SP500 usando redes neuronales, evaluado por MAE sobre 16 combinaciones de ventanas temporales.

- **Entrega**: 21 de Mayo de 2026 · 18:00 · Aula Virtual
- **Entregables**: Repositorio GitHub público + presentación PDF (5 min)
- **Evaluación**: 30 % GitHub · 70 % presentación

---

## Estructura del repositorio

```
TAREA_REDES_NEURONALES/
├── src/
│   └── utils.py                  # Funciones y constantes compartidas
├── notebooks/
│   ├── 00_datos.ipynb            # Carga de datos, log-retornos, exploración
│   ├── 01_baselines.ipynb        # Naive forecast + regresión lineal
│   ├── 02_mlp.ipynb              # Redes densas (MLP)
│   ├── 03_recurrentes.ipynb      # LSTM y GRU
│   ├── 04_convolucionales.ipynb  # Conv1D
│   ├── 05_mixtos.ipynb           # Conv1D + LSTM (híbrido)
│   ├── 06_resultados.ipynb       # Comparación global, matrices y gráficas
│   ├── 07_investigacion.ipynb    # Preprocesado avanzado (StandardScaler, FFD)
│   └── 08_carteras.ipynb         # Carteras 2025: Buy&Hold vs NN
├── models/                        # Modelos .keras guardados (mejor por tipo)
├── results/                       # CSVs opcionales de resultados
├── docs/
│   ├── *.ipynb                   # Notebooks de referencia del profesor
│   └── resumenes/                # Material teórico del curso
├── architecture.md               # Arquitectura del sistema y flujo de datos
├── decisionsmade.md              # Registro de decisiones de diseño
└── .claude/CLAUDE.md             # Contexto para Claude Code
```

---

## Instalación

```bash
pip install yfinance keras tensorflow numpy pandas matplotlib seaborn scikit-learn
```

Python 3.12 · Keras 3.x (backend TensorFlow) · Compatible con Google Colab y VS Code.

---

## Uso rápido

Ejecutar los notebooks en orden. Para una prueba rápida (CPU, ~1 hora):

```python
# Al inicio de cada notebook 02–05:
QUICK_MODE = True   # reduce EPOCHS de 300 a 50
```

Flujo completo (~8-12 horas en CPU con EPOCHS=300):

```
00_datos → 01_baselines → 02_mlp → 03_recurrentes → 04_convolucionales → 05_mixtos
         → 06_resultados (pegar dicts results de cada notebook anterior)
         → 07_investigacion (usar modelo ganador de 06)
         → 08_carteras      (usar modelo ganador de 06)
```

---

## El problema

**Regresión multivariante**: predecir `y ∈ ℝ^23` (promedio de precios de cierre futuros de 23 activos) a partir de `X ∈ ℝ^(V_in × 23)` (retornos logarítmicos pasados).

### Activos (23 del SP500)
`AEP BA CAT CNP CVX DIS DTE ED GD GE HON HPQ IBM IP JNJ KO KR MMM MO MRK MSI PG XOM`

### Combinaciones de ventanas (16 experimentos por modelo)
| | V_out=1 | V_out=5 | V_out=30 | V_out=90 |
|--|---------|---------|----------|----------|
| **V_in=5** | ✓ | ✓ | ✓ | ✓ |
| **V_in=10** | ✓ | ✓ | ✓ | ✓ |
| **V_in=30** | ✓ | ✓ | ✓ | ✓ |
| **V_in=90** | ✓ | ✓ | ✓ | ✓ |

### MAE de referencia (baseline lineal a superar)
| | V_out=1 | V_out=5 | V_out=30 | V_out=90 |
|--|---------|---------|----------|----------|
| **V_in=5** | 0.0124 | 0.0056 | 0.0023 | 0.0013 |
| **V_in=10** | 0.0125 | 0.0057 | 0.0024 | 0.0013 |
| **V_in=30** | 0.0129 | 0.0059 | 0.0024 | 0.0013 |
| **V_in=90** | 0.0140 | 0.0063 | 0.0026 | 0.0015 |

---

## Modelos incluidos (mínimo 80 entrenamientos)

| Notebook | Modelos activos | Entrenamientos |
|----------|----------------|----------------|
| 01_baselines | naive, lineal | 32 |
| 02_mlp | mlp_s | 16 |
| 03_recurrentes | lstm_s, gru_s | 32 |
| 04_convolucionales | conv_s | 16 |
| 05_mixtos | conv_lstm | 16 |
| **Total competición** | **5 modelos NN** | **80** |

Cada notebook incluye líneas comentadas `# [EXTENDER]` para añadir más arquitecturas sin cambiar la estructura del bucle de entrenamiento.

---

## Resultados de carteras 2025

El notebook `08_carteras.ipynb` construye dos carteras con pesos fijos calculados a partir de predicciones del mejor modelo (V_out=90):

| Métrica | Buy & Hold | Cartera NN |
|---------|------------|------------|
| Retorno total (%) | — | — |
| Retorno anual (%) | — | — |
| Volatilidad anual (%) | — | — |
| Sharpe ratio | — | — |
| Sortino ratio | — | — |
| Max Drawdown (%) | — | — |

*Tabla se rellena tras ejecutar 08_carteras.ipynb.*
