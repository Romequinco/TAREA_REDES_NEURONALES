# Arquitectura del Sistema

## Visión general

El taller implementa un pipeline de forecasting financiero multivariante. Los datos fluyen desde yfinance, se transforman en ventanas deslizantes, se usan para entrenar y comparar distintas familias de redes neuronales, y finalmente se aplican a la construcción de carteras para 2025.

```
yfinance (datos crudos)
    ↓ log-retornos
returns ∈ ℝ^(T × 23)
    ↓ create_time_series_data(V_in, V_out)
X ∈ ℝ^(N × V_in × 23)    y ∈ ℝ^(N × 23)
    ↓ make_splits()
train / val / test  (orden cronológico, sin shuffle)
    ↓
[01] Baselines  →  naive, lineal
[02] MLP        →  mlp_s  (flatten: N × V_in*23)
[03] Recurrentes → lstm_s, gru_s  (N × V_in × 23)
[04] Conv1D     →  conv_s          (N × V_in × 23)
[05] Mixtos     →  conv_lstm        (N × V_in × 23)
    ↓
[06] Comparación global → mejor modelo por (V_in, V_out)
    ↓
[07] Investigación → StandardScaler · FFD [EXTENDER]
[08] Carteras 2025 → Buy&Hold · Cartera NN
```

---

## Módulo compartido — `src/utils.py`

Importado por todos los notebooks con:
```python
sys.path.insert(0, os.path.join(os.getcwd(), '..', 'src'))
from utils import TICKERS, create_time_series_data, make_splits, ...
```

### Constantes globales
| Constante | Valor | Descripción |
|-----------|-------|-------------|
| `TICKERS` | lista 23 activos | Universo de activos SP500 |
| `INPUT_WINDOWS` | `[5, 10, 30, 90]` | Días de historia en X |
| `OUTPUT_WINDOWS` | `[1, 5, 30, 90]` | Días de horizonte en y |
| `RANDOM_SEED` | `42` | Semilla de partición (no cambiar) |
| `N_ASSETS` | `23` | Dimensión de features |

### Funciones principales

| Función | Firma | Descripción |
|---------|-------|-------------|
| `create_time_series_data` | `(data, V_in, V_out) → X, y` | Ventanas deslizantes; función del profesor, no modificar |
| `make_splits` | `(X, y, seed) → X_tr, X_v, X_ts, y_tr, y_v, y_ts` | Partición en dos pasos, shuffle=False |
| `eval_mae` | `(model, X, y) → float` | MAE medio sobre los 23 activos |
| `eval_mae_naive` | `(X, y) → float` | MAE del último valor conocido |
| `get_callbacks` | `(patience_lr) → list` | ReduceLROnPlateau + ModelCheckpoint |
| `restore_best_weights` | `(model)` | Restaura pesos del mejor epoch tras fit() |
| `compile_model` | `(model, lr) → model` | MAE loss + Adam; estándar para todos los modelos |
| `build_results_df` | `(results) → DataFrame` | Dict → MultiIndex (modelo, V_in, V_out) |
| `best_per_window` | `(df, metric) → DataFrame 4×4` | Mejor MAE por combinación de ventanas |
| `plot_history` | `(hist, title)` | Curva loss/val_loss por época |
| `plot_mae_matrix` | `(mat_df, title)` | Heatmap seaborn 4×4 |
| `plot_model_comparison` | `(df, V_in, V_out, metric)` | Barplot MAE por modelo |

---

## Formato de datos a través del pipeline

### Tensores
```
X : (N, V_in, 23)   — ventana de entrada (log-retornos)
y : (N, 23)         — promedio de cierres futuros (target)
```

### Partición cronológica
```
|──────────── train (~72%) ────────────|──── val (~18%) ────|── test (10%) ──|
                                        ↑ no shuffle en ningún paso
```

### Formato de resultados (dict estándar)
```python
results = {
    (nombre_modelo, V_in, V_out): {
        'train': float,   # MAE en train
        'val':   float,   # MAE en validación
        'test':  float,   # MAE en test
        'params': int     # nº de parámetros del modelo
    }, ...
}
```
Este dict se genera en cada notebook 01–05 y se agrega en `06_resultados.ipynb`.

---

## Familias de modelos

### Input shape por familia

| Familia | Input a Keras | Preprocesado |
|---------|--------------|--------------|
| Baselines (naive, lineal) | — / `(N, V_in*23)` | Flatten para lineal |
| MLP | `(N, V_in*23)` | `X.reshape(N, -1)` |
| LSTM / GRU | `(N, V_in, 23)` | Directo |
| Conv1D | `(N, V_in, 23)` | Directo; V_in ≥ kernel_size=3 |
| Conv1D + LSTM | `(N, V_in, 23)` | Directo |

### Arquitecturas activas (mínimo 80 entrenamientos)

```python
# 01_baselines — sin Keras
naive:     y_pred = X[:, -1, :]                            # 16 combinaciones
lineal:    LinearRegression().fit(X.reshape(N,-1), y)      # 16 combinaciones

# 02_mlp
mlp_s:     Input(V*23) → Dense(64, relu) → Dense(23)       # 16 combinaciones

# 03_recurrentes
lstm_s:    Input(V,23) → LSTM(64) → Dense(23)              # 16 combinaciones
gru_s:     Input(V,23) → GRU(64)  → Dense(23)              # 16 combinaciones

# 04_convolucionales
conv_s:    Input(V,23) → Conv1D(64,k=3,relu) → GAP → Dense(23)  # 16 combinaciones

# 05_mixtos
conv_lstm: Input(V,23) → Conv1D(64,k=3,relu) → LSTM(64) → Dense(23)  # 16 combins
```

### Extensiones disponibles (`[EXTENDER]`)
Descomentando líneas en el dict `MODELOS` de cada notebook:

| Notebook | Modelos adicionales |
|----------|-------------------|
| 02_mlp | mlp_m (2 capas, 128u), mlp_l (2 capas, 256u) |
| 03_recurrentes | lstm_d (2 capas LSTM), gru_d (2 capas GRU) |
| 04_convolucionales | conv_gmp (GlobalMaxPool), conv_d (doble Conv1D) |
| 05_mixtos | conv_gru, conv_dense, lstm_dense |

---

## Flujo de compilación y entrenamiento

```python
# Todos los modelos usan la misma configuración
model = compile_model(Sequential([...]))   # MAE + Adam(lr=3e-4)
hist  = model.fit(X_tr, y_tr,
                  validation_data=(X_v, y_v),
                  epochs=EPOCHS,           # 300 (50 en QUICK_MODE)
                  batch_size=64,
                  callbacks=get_callbacks(),   # ReduceLR + ModelCheckpoint
                  verbose=0)
restore_best_weights(model)   # recupera el mejor epoch
```

Sin EarlyStopping: el modelo entrena todas las épocas para ver la curva completa. `ModelCheckpoint` guarda el mejor estado en disco temporal; `restore_best_weights(model)` lo recupera al finalizar. El bucle de entrenamiento es idéntico en todos los notebooks 02–05.

---

## Notebooks de investigación y carteras

### `07_investigacion.ipynb` — Preprocesado avanzado

```
returns_raw → StandardScaler (fit solo en train) → X_tr_s, X_v_s, X_ts_s
           → entrenar build_best() → comparar MAE crudo vs scaled
           → [EXTENDER] FFD (diferenciación fraccional)
```

El scaler se ajusta sobre el reshape `(N*T, F)` y se aplica por separado a val y test para evitar data leakage.

### `08_carteras.ipynb` — Portfolio 2025

```
Datos históricos hasta 2024
    → entrenar mejor modelo (V_out=90) sobre todo el histórico
    → predecir retornos para los próximos 90 días (última ventana de 2024)
    → construir pesos fijos:
        pesos_bh = [1/23, ..., 1/23]
        pesos_nn = y_pred / sum(|y_pred|)   # long/short
    → descargar datos 2025 → calcular retornos de cartera
    → métricas: retorno total, anual, volatilidad, Sharpe, Sortino, MaxDD
```

---

## Convenciones de código

- Cada notebook es autocontenido: importa desde `src/utils.py` y descarga sus propios datos.
- Los marcadores `# [EXTENDER]` indican código comentado listo para activarse.
- `QUICK_MODE = True` reduce `EPOCHS` a 50 en todos los notebooks 02–05.
- Los dicts `results` de cada notebook se agregan manualmente en `06_resultados.ipynb`.
