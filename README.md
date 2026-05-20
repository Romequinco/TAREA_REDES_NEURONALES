# Taller B3-T4/T5/T6 — Redes Neuronales para Forecasting Financiero

Taller universitario de Máster. Predice el retorno promedio de 23 activos del SP500 usando redes neuronales, evaluado por MAE sobre 16 combinaciones de ventanas temporales.

**Autores**: Fernando Dapena Tauste · Oscar Romero Quincoces · Daniel García López

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
│   ├── 03_recurrentes.ipynb      # SimpleRNN, GRU, LSTM y variantes (6 modelos)
│   ├── 04_convolucionales.ipynb  # Conv1D
│   ├── 05_mixtos.ipynb           # Híbridos Conv1D + LSTM/GRU (6 modelos)
│   ├── 06_resultados.ipynb       # Comparación global: 256 entrenamientos
│   ├── 07_investigacion.ipynb    # Preprocesado avanzado (StandardScaler, FFD, Features)
│   └── 08_carteras.ipynb         # Carteras 2025: Buy&Hold vs NN
├── architecture.md               # Arquitectura del sistema y flujo de datos
├── decisionsmade.md              # Registro de decisiones de diseño (29 decisiones)
└── requirements.notebooks.txt    # Dependencias
```

---

## Instalación

```bash
pip install -r requirements.notebooks.txt
# o directamente:
pip install yfinance keras tensorflow numpy pandas matplotlib seaborn scikit-learn
```

Python 3.12 · Keras 3.x (backend TensorFlow) · Compatible con Google Colab y VS Code.

---

## Uso rápido

Ejecutar los notebooks en orden. Para una prueba rápida (CPU, ~1 hora):

```python
# Al inicio de cada notebook 02–05:
QUICK_MODE = True   # reduce EPOCHS a 50
```

Flujo completo (varios días en CPU sin GPU):

```
00_datos → 01_baselines → 02_mlp → 03_recurrentes → 04_convolucionales → 05_mixtos
         → 06_resultados (los dicts results están pre-cargados con los valores obtenidos)
         → 07_investigacion
         → 08_carteras
```

---

## El problema

**Regresión multivariante**: predecir `y ∈ ℝ^23` (promedio de retornos logarítmicos futuros de 23 activos) a partir de `X ∈ ℝ^(V_in × 23)` (retornos logarítmicos pasados).

### Activos (23 del SP500)
`AEP BA CAT CNP CVX DIS DTE ED GD GE HON HPQ IBM IP JNJ KO KR MMM MO MRK MSI PG XOM`

### Combinaciones de ventanas (16 experimentos por modelo)
| | V_out=1 | V_out=5 | V_out=30 | V_out=90 |
|--|---------|---------|----------|----------|
| **V_in=5** | ✓ | ✓ | ✓ | ✓ |
| **V_in=10** | ✓ | ✓ | ✓ | ✓ |
| **V_in=30** | ✓ | ✓ | ✓ | ✓ |
| **V_in=90** | ✓ | ✓ | ✓ | ✓ |

### MAE de referencia (regresión lineal)
| | V_out=1 | V_out=5 | V_out=30 | V_out=90 |
|--|---------|---------|----------|----------|
| **V_in=5** | 0.0124 | 0.0056 | 0.0023 | 0.0013 |
| **V_in=10** | 0.0126 | 0.0057 | 0.0024 | 0.0013 |
| **V_in=30** | 0.0130 | 0.0059 | 0.0024 | 0.0014 |
| **V_in=90** | 0.0143 | 0.0065 | 0.0027 | 0.0015 |

---

## Modelos incluidos (256 entrenamientos)

| Notebook | Modelos activos | Params (V_in=10) | Entrenamientos |
|----------|----------------|-----------------|----------------|
| 01_baselines | naive, lineal | 0 / 5.313 | 32 |
| 02_mlp | mlp_s | 16.279 | 16 |
| 03_recurrentes | simple_rnn, gru, lstm, lstm_stack, bi_gru, lstm_drop | 2.551–27.143 | 96 |
| 04_convolucionales | conv_s | 51.287 | 16 |
| 05_mixtos | conv_lstm_ln, conv_gru_bottleneck, conv_bilstm, conv2_lstm, lstm_dense, conv_dense | 10.135–36.983 | 96 |
| **Total** | **16 modelos** | | **256** |

---

## Hallazgo principal: colapso al predictor de la media

Todos los modelos (MLP, LSTM, GRU, Conv1D, híbridos) convergen al mismo MAE test en todas las combinaciones de ventanas, equivalente al de la regresión lineal:

| V_out | MAE todas las NN | Mejora vs Naive | Mejora vs Lineal |
|-------|-----------------|----------------|-----------------|
| 1d | ≈ 0.0123 | −31% | ≈ 0% |
| 5d | ≈ 0.0056 | −59% | ≈ 0% |
| 30d | ≈ 0.0023 | −82% | ≈ 0% |
| 90d | ≈ 0.0013 | −89% | ≈ 0% |

**Por qué ocurre**: los log-retornos son ruido blanco (Efficient Market Hypothesis, forma débil). El estimador que minimiza MAE es la media. Sin señal en el input, predecir la media es la respuesta óptima — no es un error de implementación.

**Única mejora real**: FFD(d=0.2) en `07_investigacion.ipynb` reduce el MAE un 8.9% para V_out=1 día.

---

## Resultados de carteras 2025

El notebook `08_carteras.ipynb` construye dos carteras con pesos fijos (calculados con datos hasta 2024) y las evalúa sobre los datos reales de 2025:

| Métrica | Buy & Hold | Cartera NN |
|---------|------------|------------|
| Retorno total (%) | 21.94 | 22.11 |
| Retorno anual (%) | 15.64 | 15.76 |
| Volatilidad anual (%) | 13.19 | 13.12 |
| Sharpe ratio | 1.186 | 1.201 |
| Sortino ratio | 1.519 | 1.559 |
| Max Drawdown (%) | −11.33 | −11.12 |

La diferencia es marginal (+0.17% retorno total), consistente con el hallazgo del colapso: el modelo predice retornos positivos para todos los activos (la media histórica es positiva), por lo que la cartera NN es efectivamente long-only con pesos ligeramente distintos al uniforme.
