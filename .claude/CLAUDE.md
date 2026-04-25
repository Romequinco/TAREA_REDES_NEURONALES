# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Contexto del proyecto

Taller universitario de Máster: **Redes Neuronales para Forecasting financiero (B3-T4, T5, T6)**.

- **Entrega**: 21 de Mayo de 2026 a las 18:00 via aula virtual
- **Entregables**: Repositorio GitHub público + presentación PDF
- **Evaluación**: 30% GitHub, 70% presentación (exposición de 5 minutos)
- **Equipo**: Grupos de 3 estudiantes

---

## El problema

Predecir el **promedio del precio de cierre de 23 activos del SP500 a futuro** a partir de retornos logarítmicos históricos. Es un problema de **regresión multivariante**, métrica objetivo exclusiva: **MAE (Mean Absolute Error)**.

### Datos
- 23 activos del SP500: `['AEP','BA','CAT','CNP','CVX','DIS','DTE','ED','GD','GE','HON','HPQ','IBM','IP','JNJ','KO','KR','MMM','MO','MRK','MSI','PG','XOM']`
- ~16.000 días de histórico desde 1945 (descarga en vivo con `yfinance`)
- Preprocesado base: **log-retornos** → `np.log(precios_close).diff().dropna()`

### Formato de tensores (no cambiar)
- **Entrada X**: `(N, V_in, 23)` — N muestras, V días de ventana de entrada, 23 activos
- **Salida y**: `(N, 23)` — promedio de precios de cierre de los días de la ventana de salida futura

### Partición fija (NUNCA modificar semilla ni orden)
```python
RANDOM_SEED = 42
# Paso 1: 90/10 — shuffle=False obligatorio para respetar orden cronológico
X_tr_full, X_ts, y_tr_full, y_ts = train_test_split(
    X, y, test_size=0.10, shuffle=False, random_state=RANDOM_SEED)
# Paso 2: 20% del train_full para validación (~18% del total)
X_tr, X_v, y_tr, y_v = train_test_split(
    X_tr_full, y_tr_full, test_size=0.20, shuffle=False, random_state=RANDOM_SEED)
```

### Ventanas a estudiar (16 combinaciones = 16 experimentos por modelo)
| | Out=1 | Out=5 | Out=30 | Out=90 |
|--|-------|-------|--------|--------|
| **In=5** | ✓ | ✓ | ✓ | ✓ |
| **In=10** | ✓ | ✓ | ✓ | ✓ |
| **In=30** | ✓ | ✓ | ✓ | ✓ |
| **In=90** | ✓ | ✓ | ✓ | ✓ |

---

## Estado actual de la implementación

Todos los archivos han sido creados. El repositorio está listo para ejecutarse.

| Archivo | Estado | Notas |
|---------|--------|-------|
| `src/utils.py` | ✅ Completo | Todas las funciones compartidas |
| `notebooks/00_datos.ipynb` | ✅ Completo | Exploración de datos y ventanas |
| `notebooks/01_baselines.ipynb` | ✅ Completo | Naive + lineal |
| `notebooks/02_mlp.ipynb` | ✅ Completo | mlp_s activo; mlp_m/mlp_l en [EXTENDER] |
| `notebooks/03_recurrentes.ipynb` | ✅ Completo | lstm_s + gru_s; lstm_d/gru_d en [EXTENDER] |
| `notebooks/04_convolucionales.ipynb` | ✅ Completo | conv_s activo; variantes en [EXTENDER] |
| `notebooks/05_mixtos.ipynb` | ✅ Completo | conv_lstm activo; otras en [EXTENDER] |
| `notebooks/06_resultados.ipynb` | ✅ Completo | Pegar dicts tras ejecutar 01-05 |
| `notebooks/07_investigacion.ipynb` | ✅ Completo | StandardScaler; FFD en [EXTENDER] |
| `notebooks/08_carteras.ipynb` | ✅ Completo | Carteras 2025 con métricas |

**Próximo paso**: ejecutar los notebooks 00–05, copiar los dicts `results` en `06_resultados.ipynb`, e identificar el modelo ganador para actualizar `build_best()` en notebooks 07 y 08.

---

## Tareas requeridas

### Tarea 1 — Competición
Para **cada una de las 16 combinaciones de ventanas**, entrenar y comparar (mínimo **64 modelos** en total):

| Tipo | Input shape | Preprocesado necesario |
|------|-------------|----------------------|
| **MLP/Dense** | `(N, V*23)` | Flatten: `X.reshape(N, -1)` |
| **LSTM/GRU** | `(N, V, 23)` | Directo |
| **Conv1D** | `(N, V, 23)` | Directo |
| **Mixtos** (Conv1D+LSTM, etc.) | `(N, V, 23)` | Directo |
| **Baselines** (naive, lineal) | — | — |

Reportar por modelo: MAE en train, val y test + número de parámetros.

### Tarea 2 — Investigación
- Repetir la competición con **preprocesado financiero avanzado** (StandardScaler, FFD)
- Con el mejor modelo para ventana de salida de 90 días:
  - Implementar **cartera sin predicciones** (Buy & Hold ponderado)
  - Implementar **cartera con predicciones** del modelo
  - Comparar rendimientos para el año **2025**

---

## Entregables del GitHub (checklist)

- [ ] Tabla MAE (train/val/test) + nº parámetros por modelo y combinación de ventanas
- [ ] 16 gráficas: comparación de modelos por combinación de ventanas
- [ ] 4 gráficas resumen (una por tamaño de ventana de salida)
- [ ] Curvas de entrenamiento por modelo (deben mostrar convergencia)
- [ ] Matriz 4×4: mejor MAE en test por combinación de ventanas
- [ ] Código que genere todas las gráficas y tablas automáticamente
- [ ] Resultados de carteras para 2025

---

## Notebooks de referencia del profesor

Los tres notebooks en `docs/` son el punto de partida. **No modificar la función `create_time_series_data` ni la semilla de partición.**

### MAE de referencia mínimo a superar (baseline lineal)

| | Out=1 | Out=5 | Out=30 | Out=90 |
|--|-------|-------|--------|--------|
| **In=5** | 0.0124 | 0.0056 | 0.0023 | 0.0013 |
| **In=10** | 0.0125 | 0.0057 | 0.0024 | 0.0013 |
| **In=30** | 0.0129 | 0.0059 | 0.0024 | 0.0013 |
| **In=90** | 0.0140 | 0.0063 | 0.0026 | 0.0015 |

> Patrón: ventanas de salida largas → MAE más bajo (el promedio suaviza la volatilidad). Ventanas de entrada cortas → mejor generalización en el baseline lineal.

---

## Configuración estándar de modelos Keras

```python
# SIEMPRE MAE como función de pérdida (no MSE)
model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate=3e-4))

# Callbacks (encapsulados en get_callbacks() de utils.py)
# Sin EarlyStopping — el modelo entrena todas las épocas para ver la curva completa
ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
ModelCheckpoint(tmp_path, monitor='val_loss', save_best_only=True)
# Tras model.fit() — obligatorio en todos los notebooks:
restore_best_weights(model)
```

---

## Preprocesado financiero avanzado (Tarea 2)

- **Datos NO i.i.d.**: no hacer shuffle en ningún split
- **StandardScaler**: fit solo en train, transform en val/test — no data leakage
- **Outliers financieros**: las crisis son información relevante, no eliminar
- **FFD (Diferenciación Fraccional)**: implementada como `[EXTENDER]` en `07_investigacion.ipynb`
- **PROHIBIDO**: backward fill en imputación, usar estadísticas del test en normalización

---

## Implementación de carteras (Tarea 2)

```python
pesos_bh = np.ones(23) / 23                          # Buy & Hold: pesos iguales
pesos_nn = y_pred / np.sum(np.abs(y_pred))            # NN: long/short normalizado
ret_diario = returns_2025.values @ pesos              # retorno diario de cartera
cum_return = np.exp(np.cumsum(ret_diario)) - 1        # retorno acumulado
```

Pesos **fijos** durante todo 2025. Rebalanceo mensual marcado con `# [EXTENDER]`.

---

## Dependencias

```bash
pip install yfinance keras tensorflow numpy pandas matplotlib seaborn scikit-learn
```

Python 3.12 · Keras 3.x (backend TensorFlow) · Google Colab o VS Code.

---

## Documentación teórica disponible en `docs/resumenes/`

| Archivo | Contenido clave |
|---------|----------------|
| `training-nn-2026.md` | Estrategia dos extremos, diagnóstico curvas, callbacks, optimizadores |
| `intro-deep-2026.md` | Funciones de coste, comparativa optimizadores, regularización |
| `b3_s1_mapa.md` | ML financiero: overfitting en test, Walk Forward, datos no-IID |
| `b3_s2_mapa.md` | Preprocesado: FFD, barras por actividad, data leakage temporal |
| `b3_s3_mapa.md` | Ensemble, PCA, XAI, Granger Causality |
| `b3_s4_mapa.md` | Mapa de dependencias: MLP, ReLU, regularización, CNN, transfer learning |
| `preprocesado-datos.md` | Z-Score, estandarización, outliers, train/test split correcto |
| `calculo-optimizacion.md` | Descenso por gradiente, backpropagation, MAE vs MSE |
| `redes-neuronales-fundamentos.md` | Glosario Keras: callbacks, regularizadores, inicialización Xavier |
