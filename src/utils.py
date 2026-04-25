"""
utils.py — Funciones y constantes compartidas por todos los notebooks del taller.
Modificar las constantes aquí afecta a todos los notebooks.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

_CKPT_PATH = tempfile.mktemp(suffix='.keras')

# ── CONSTANTES GLOBALES ───────────────────────────────────────────────────────
TICKERS = ['AEP','BA','CAT','CNP','CVX','DIS','DTE','ED','GD','GE',
           'HON','HPQ','IBM','IP','JNJ','KO','KR','MMM','MO','MRK','MSI','PG','XOM']

INPUT_WINDOWS  = [5, 10, 30, 90]   # días de entrada
OUTPUT_WINDOWS = [1,  5, 30, 90]   # días de salida (ventana de predicción)
RANDOM_SEED    = 42
N_ASSETS       = 23


# ── DATOS ─────────────────────────────────────────────────────────────────────
def create_time_series_data(data, V_in, V_out):
    """
    Genera pares (X, y) de ventanas deslizantes sobre la serie temporal.
    X : (N, V_in, N_ASSETS)  — ventana de entrada (retornos pasados)
    y : (N, N_ASSETS)        — promedio de los V_out pasos futuros de cierre
    Función del profesor — no modificar.
    """
    X, y = [], []
    arr = data.values if isinstance(data, pd.DataFrame) else data

    for i in range(len(arr) - V_in - V_out + 1):
        X.append(arr[i : i + V_in])
        if V_out > 0:
            y.append(np.mean(arr[i + V_in : i + V_in + V_out], axis=0))
        else:
            y.append(arr[i + V_in - 1])

    return np.array(X), np.array(y)


def make_splits(X, y, seed=RANDOM_SEED):
    """
    Partición en dos pasos, shuffle=False (orden cronológico obligatorio):
      Paso 1 → 90 % train_full / 10 % test
      Paso 2 → 80 % train    / 20 % val  (del train_full)
    Resultado: ~72 % train / ~18 % val / 10 % test
    20% de val (vs. 5% original) da señal más robusta para ReduceLROnPlateau y ModelCheckpoint.
    """
    X_tr_full, X_ts, y_tr_full, y_ts = train_test_split(
        X, y, test_size=0.10, shuffle=False, random_state=seed)
    X_tr, X_v, y_tr, y_v = train_test_split(
        X_tr_full, y_tr_full, test_size=0.20, shuffle=False, random_state=seed)
    return X_tr, X_v, X_ts, y_tr, y_v, y_ts


# ── EVALUACIÓN ────────────────────────────────────────────────────────────────
def eval_mae(model, X, y):
    """MAE medio sobre todos los activos (escalar)."""
    return float(np.mean(np.abs(model.predict(X, verbose=0) - y)))


def eval_mae_naive(X, y):
    """MAE del naive forecast: predice el último retorno conocido."""
    y_pred = X[:, -1, :]   # último timestep de la ventana de entrada
    return float(np.mean(np.abs(y_pred - y)))


# ── ENTRENAMIENTO ─────────────────────────────────────────────────────────────
def get_callbacks(patience_lr=5):
    """
    ReduceLROnPlateau + ModelCheckpoint sobre val_loss.
    Sin EarlyStopping: entrena todas las épocas para ver la curva completa.
    Llamar restore_best_weights(model) tras model.fit() para recuperar el mejor estado.
    """
    return [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=patience_lr, min_lr=1e-6, verbose=0),
        ModelCheckpoint(_CKPT_PATH, monitor='val_loss',
                        save_best_only=True, verbose=0),
    ]


def restore_best_weights(model):
    """Restaura los pesos del mejor epoch guardado por ModelCheckpoint."""
    model.load_weights(_CKPT_PATH)


def compile_model(model, lr=3e-4):
    """Compilación estándar: MAE + Adam. Misma para todos los modelos."""
    from keras.optimizers import Adam
    model.compile(loss='mean_absolute_error', optimizer=Adam(learning_rate=lr))
    return model


# ── RESULTADOS ────────────────────────────────────────────────────────────────
def build_results_df(results):
    """
    results : dict  { (modelo, V_in, V_out) : {'train', 'val', 'test', 'params'} }
    Devuelve un DataFrame con MultiIndex (modelo, V_in, V_out).
    """
    rows = []
    for (nombre, V_in, V_out), m in results.items():
        rows.append({'modelo': nombre, 'V_in': V_in, 'V_out': V_out,
                     'train': m['train'], 'val': m['val'], 'test': m['test'],
                     'params': m.get('params', 0)})
    df = pd.DataFrame(rows).set_index(['modelo', 'V_in', 'V_out'])
    return df


def best_per_window(results_df, metric='test'):
    """Matriz 4×4: mejor MAE en `metric` por (V_in, V_out)."""
    mat = np.full((4, 4), np.nan)
    for i, V_in in enumerate(INPUT_WINDOWS):
        for j, V_out in enumerate(OUTPUT_WINDOWS):
            subset = results_df.xs((V_in, V_out), level=('V_in', 'V_out'),
                                   drop_level=False)[metric]
            if not subset.empty:
                mat[i, j] = subset.min()
    return pd.DataFrame(mat, index=INPUT_WINDOWS, columns=OUTPUT_WINDOWS)


# ── VISUALIZACIÓN ─────────────────────────────────────────────────────────────
def plot_history(hist, title=''):
    """Curva loss / val_loss por época."""
    plt.figure(figsize=(6, 3))
    plt.plot(hist.history['loss'],     label='train')
    plt.plot(hist.history['val_loss'], label='val')
    plt.xlabel('Época'); plt.ylabel('MAE'); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.show()


def plot_mae_matrix(mat_df, title='MAE en test'):
    """Heatmap seaborn 4×4 (filas=V_in, columnas=V_out)."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(mat_df.astype(float), annot=True, fmt='.4f',
                cmap='YlOrRd_r', linewidths=.5)
    plt.xlabel('Ventana salida (días)'); plt.ylabel('Ventana entrada (días)')
    plt.title(title); plt.tight_layout(); plt.show()


def plot_model_comparison(results_df, V_in, V_out, metric='test'):
    """Barplot comparando MAE de todos los modelos para una combinación de ventanas."""
    subset = results_df.xs((V_in, V_out), level=('V_in', 'V_out'),
                           drop_level=False)[metric].reset_index(level=[1, 2], drop=True)
    ax = subset.plot(kind='bar', figsize=(7, 3), color='steelblue', edgecolor='k')
    ax.set_title(f'MAE test — entrada={V_in}d, salida={V_out}d')
    ax.set_ylabel('MAE'); ax.set_xlabel('Modelo')
    plt.xticks(rotation=30, ha='right'); plt.tight_layout(); plt.show()
