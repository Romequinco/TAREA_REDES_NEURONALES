# Decisiones de Diseño

Registro de todas las decisiones tomadas durante el diseño e implementación del taller. Cada entrada incluye la opción elegida y el razonamiento.

---

## D1 — Función de pérdida

**Decisión**: MAE (`mean_absolute_error`)

El enunciado del taller exige MAE como métrica de evaluación. Los notebooks del profesor usan MSE por simplicidad didáctica, pero el taller es explícito en requerir MAE. MAE es además más robusto a outliers, lo cual es relevante para retornos financieros con colas pesadas.

---

## D2 — Optimizador y learning rate

**Decisión**: `Adam(learning_rate=3e-4)`

Adam es la primera opción recomendada en el material teórico (`training-nn-2026.md`). `lr=3e-4` es el valor estándar para Adam en problemas de regresión con este rango de valores de target. Mismo valor que usan los notebooks del profesor.

---

## D3 — Número máximo de épocas

**Decisión**: `EPOCHS = 100` con EarlyStopping(patience=10)

100 épocas es un techo razonable que rara vez se alcanza gracias a EarlyStopping. En la práctica, los modelos convergen antes de la época 50. `QUICK_MODE = True` reduce a 20 épocas para pruebas rápidas (~1 hora CPU vs ~4-5 horas completo).

---

## D4 — Batch size

**Decisión**: `BATCH_SIZE = 64`

Balance entre velocidad de entrenamiento (batches grandes = menos actualizaciones por época) y calidad del gradiente (batches pequeños = más ruido, potencialmente mejor generalización). 64 es CPU-friendly y estándar para este tipo de problemas.

---

## D5 — Callbacks

**Decisión**: `EarlyStopping(patience=10, restore_best_weights=True)` + `ReduceLROnPlateau(patience=5, factor=0.5)`

- **EarlyStopping**: detiene el entrenamiento si `val_loss` no mejora en 10 épocas. `restore_best_weights=True` evita guardar checkpoints a disco durante el bucle (más eficiente en CPU).
- **ReduceLROnPlateau**: reduce el LR a la mitad si no hay mejora en 5 épocas, permitiendo salir de mínimos locales planos.

No se usa `ModelCheckpoint` en el bucle principal para evitar I/O innecesario.

---

## D6 — Guardado de modelos

**Decisión**: No guardar durante el bucle de entrenamiento; solo guardar manualmente el mejor global tras `06_resultados.ipynb`

`restore_best_weights=True` mantiene el mejor estado en RAM durante el entrenamiento. Guardar 80+ archivos `.keras` durante el bucle consumiría espacio y tiempo de I/O innecesarios. El usuario guarda el ganador final en `models/` si lo necesita.

---

## D7 — Partición de datos

**Decisión**: Partición exacta del profesor en dos pasos con `shuffle=False`

```python
# Paso 1: 90% train_full / 10% test
X_tr_full, X_ts, y_tr_full, y_ts = train_test_split(
    X, y, test_size=0.10, shuffle=False, random_state=42)
# Paso 2: 95% train / 5% val del train_full
X_tr, X_v, y_tr, y_v = train_test_split(
    X_tr_full, y_tr_full, test_size=0.05, shuffle=False, random_state=42)
```

`shuffle=False` es obligatorio para series temporales financieras: mezclar rompería el orden cronológico y produciría data leakage del futuro al pasado. Esta partición produce ~85.5% train / ~4.5% val / 10% test.

---

## D8 — Función `create_time_series_data`

**Decisión**: Usar exactamente la del profesor, sin modificaciones

La función genera `X:(N, V_in, N_assets)` e `y:(N, N_assets)` como promedio de los `V_out` pasos futuros. No se modifica para garantizar reproducibilidad y compatibilidad con los resultados del profesor.

---

## D9 — Baselines

**Decisión**: Naive (último valor conocido) + Regresión Lineal (sklearn)

- **Naive**: `y_pred = X[:, -1, :]` — predice el último retorno observado. Es el benchmark más difícil de superar en mercados eficientes.
- **Lineal**: `LinearRegression` de sklearn sobre X aplanado. Es el baseline explícito del profesor con MAEs de referencia ya documentados.

---

## D10 — Modelos activos en la competición (mínimo 64)

**Decisión**: 5 modelos NN × 16 combinaciones = 80 entrenamientos

| Modelo | Notebook | Param aprox |
|--------|----------|------------|
| mlp_s (Dense 64) | 02 | ~6K |
| lstm_s (LSTM 64) | 03 | ~22K |
| gru_s (GRU 64) | 03 | ~17K |
| conv_s (Conv1D 64 + GAP) | 04 | ~5K |
| conv_lstm (Conv1D + LSTM) | 05 | ~26K |

Supera el mínimo de 64 con 2 modelos recurrentes (32 entrenamientos cada uno cuenta). Modelos adicionales están disponibles descomentando `[EXTENDER]`.

---

## D11 — Input shape para MLP

**Decisión**: Flatten a `(N, V_in * 23)` con `X.reshape(N, -1)`

MLP no admite entrada 3D nativa. El flatten es la transformación estándar y la que usan los notebooks del profesor.

---

## D12 — Conv1D kernel size

**Decisión**: `kernel_size=3`

Kernel mínimo para capturar patrones locales de al menos 3 días. Compatible con `V_in=5` (el mínimo en `INPUT_WINDOWS`). Un kernel mayor (5, 7) requeriría `V_in` más grandes y reduciría la longitud de salida significativamente.

---

## D13 — Rebalanceo de carteras

**Decisión**: Pesos **fijos** durante todo 2025; rebalanceo mensual marcado como `[EXTENDER]`

Pesos fijos simplifican la comparación y son el comportamiento estándar descrito en el enunciado del taller. El rebalanceo mensual (~21 días de trading) está documentado en el código como extensión opcional sin implementación activa.

---

## D14 — Construcción de pesos de la cartera NN

**Decisión**: Long/Short proporcional a retorno predicho, normalizado por suma de valores absolutos

```python
pesos_nn = y_pred / np.sum(np.abs(y_pred))
```

- Activos con predicción positiva → posición larga (peso > 0)
- Activos con predicción negativa → posición corta (peso < 0)
- Normalización: presupuesto completo invertido (suma de |pesos| = 1)

---

## D15 — Preprocesado en la competición

**Decisión**: Solo log-retornos; sin normalización adicional en notebooks 01–05

El material teórico advierte contra normalizar con estadísticas del dataset completo. Los log-retornos ya tienen media ≈ 0 y están en la misma escala para todos los activos. La normalización avanzada se reserva para `07_investigacion.ipynb`.

---

## D16 — Preprocesado en la investigación

**Decisión**: `StandardScaler` (fit solo en train) como técnica activa; FFD como `[EXTENDER]`

- **StandardScaler**: técnica estándar, fácil de implementar correctamente sin leakage, impacto medible en MAE.
- **FFD**: técnica avanzada (diferenciación fraccional), requiere implementación propia de `ffd_weights()` y `apply_ffd()`. Incluida como código comentado listo para activarse.

---

## D17 — Visualizaciones

**Decisión**: `matplotlib` para curvas de convergencia; `seaborn` para heatmaps 4×4

`seaborn.heatmap` genera los heatmaps de MAE en una línea con anotaciones automáticas. `matplotlib` es suficiente para curvas de pérdida y barplots. No se introducen dependencias adicionales de visualización.

---

## D18 — Agregación de resultados en `06_resultados.ipynb`

**Decisión**: Pegar los dicts `results` manualmente; no hay ejecución automática entre notebooks

Cada notebook es autocontenido (descarga datos, entrena, evalúa). No existe un sistema de persistencia automática entre notebooks para mantener el código simple y compatible con Colab. El usuario copia los dicts o re-ejecuta los notebooks antes de `06_resultados.ipynb`.

---

## D19 — Número de activos

**Decisión**: 23 activos fijos (los que tienen datos completos desde 1945)

`precios.dropna(axis=1, inplace=True)` elimina automáticamente activos sin datos históricos completos. En la práctica, los 23 activos de `TICKERS` siempre tienen datos desde 1945, por lo que el número es estable.

---

## D20 — Período de evaluación de carteras

**Decisión**: Año 2025 completo (`start='2025-01-01'` sin `end`)

El enunciado especifica comparar rendimientos para 2025. Descargar sin fecha de fin captura todos los datos disponibles hasta la fecha de ejecución del notebook.

---

## D21 — Métricas de carteras

**Decisión**: Retorno total, retorno anual, volatilidad anual, Sharpe, Sortino, Max Drawdown

Conjunto estándar de métricas de gestión de carteras. Se usan `TRADING_DAYS = 252` para anualización. El Sortino usa solo los retornos negativos en el denominador, siendo más informativo que el Sharpe en distribuciones asimétricas (como los retornos financieros).

---

## D22 — Estructura de archivos

**Decisión**: `src/utils.py` compartido + 9 notebooks independientes

Centralizar funciones en `utils.py` evita duplicación de código y hace que cambiar un hiperparámetro global (e.g., `INPUT_WINDOWS`) afecte automáticamente a todos los notebooks. Cada notebook sigue siendo autocontenido para facilitar la ejecución independiente en Colab.

---

## D23 — Marcadores de extensión

**Decisión**: Usar `# [EXTENDER]` para código comentado ampliable

Permite al equipo ir más allá de 64 modelos sin cambiar la estructura. Cada línea comentada es una arquitectura completa lista para activarse con un solo comentario eliminado. El bucle de entrenamiento no cambia.

---

## D24 — `QUICK_MODE`

**Decisión**: Flag booleano al inicio de cada notebook, `EPOCHS = 20` cuando activo

Permite probar que todo el flujo funciona en ~1 hora antes del entrenamiento completo (~4-5 horas). Se activa localmente sin cambiar ningún otro parámetro.
