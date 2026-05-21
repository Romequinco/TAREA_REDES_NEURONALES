# Guion de Presentación — Taller 4 Bloque 3: Redes Neuronales
**Fernando Dapena Tauste · Oscar Romero Quincoces · Daniel García López**

---

## Cómo usar este guion

Cada sección explica **qué aparece en la diapositiva**, **por qué se incluye**, **cómo argumentarlo**, **qué debilidades tiene** y **qué alternativas existen**. Los números citados son los reales de los notebooks, no estimaciones.

---

---

## § PORTADA

### Qué aparece
Título, autores, cuatro cajas resumen (secciones 01–04) y cinco estadísticas globales: 23 activos, 16 arquitecturas, 256 entrenamientos, matriz 4×4, 6 secciones.

### Por qué está aquí
La portada establece el **alcance real del trabajo** antes de que nadie pregunte. 256 entrenamientos no es un número casual: es 16 modelos × 16 combinaciones de ventanas (4 V_in × 4 V_out). Ponerlo explícito en portada fija el marco: esto no es probar una red, es un experimento sistemático con control estadístico.

### Cómo argumentarlo
> "Este trabajo no busca encontrar el mejor modelo para el SP500 — busca responder si existe diferencia entre arquitecturas cuando el input son log-retornos. Para eso necesitábamos un diseño factorial completo: las mismas 16 ventanas para cada uno de los 16 modelos, con los mismos hiperparámetros base. De ahí los 256 entrenamientos."

### Puntos fuertes
- La escala del experimento (256 entrenamientos) da legitimidad estadística a la conclusión central.
- Citar 23 activos del SP500 desde 1945 (16.194 días) es un dataset serio, no un ejemplo de juguete.

### Puntos débiles
- No se hace búsqueda de hiperparámetros por modelo — todos comparten Adam(3×10⁻⁴), 100 épocas (NB02–04), batch 32. Un revisor puede señalar que un LSTM bien afinado podría romper el techo. La respuesta: la investigación (NB07) muestra que cambiar LR, batch, regularización no mueve el MAE más de 0.0001, validando el diseño común.

---

---

## § 00 — Pipeline de Entrenamiento (utils.py)

### Qué aparece
Tres tablas: (0a) hiperparámetros globales, (0b) callbacks, (0c) partición cronológica. Nota sobre el naive forecast.

### Por qué está aquí
Esta sección existe porque **todos los 256 modelos usan exactamente el mismo pipeline**. Sin documentarlo, cualquier diferencia de MAE entre modelos podría achacarse a diferencias de entrenamiento. Al mostrar que todo está centralizado en `src/utils.py`, se garantiza que las comparaciones son justas.

---

### Tabla 0a — Hiperparámetros globales

| Componente | Valor | Por qué este valor |
|---|---|---|
| Optimizador | Adam · lr = 3×10⁻⁴ | Default material teórico del taller. La evidencia D25 muestra que bajar a 1×10⁻⁴ solo desplaza el best_epoch de 3–28 a 8–300 sin cambiar el MAE final (Δval_min = 0.00009, por debajo del umbral de 0.0005). |
| Función de pérdida | MAE | Obligatorio por enunciado. Ventaja: directamente interpretable en escala de retornos (0.0123 = 1.23% de error medio). Menos sensible a outliers que MSE — relevante en retornos financieros con colas gordas. |
| Épocas | 100 (NB02–04) / 500 (NB05) | Sin EarlyStopping intencionado para observar la curva completa. ModelCheckpoint restaura el mejor estado. |
| Batch size | 32 | La evidencia D26/Ev.8 compara [16, 64, 256, 512]: MAE invariante. El colapso es estructural (problema de datos), no resultado del optimizador. |
| Semilla | 42 | Reproducibilidad de particiones y pesos iniciales. Fijada desde NB00. |

**Cómo argumentarlo:**
> "Usar los mismos hiperparámetros en los 256 entrenamientos es una decisión metodológica, no una limitación. Si cada modelo tuviera su propio tuning, no podríamos comparar arquitecturas — estaríamos comparando el resultado del tuning. La evidencia del NB02 muestra que ni el LR ni el batch size mueven el MAE, con lo cual el coste de no hacer tuning es cero."

---

### Tabla 0b — Callbacks

**ReduceLROnPlateau**: si val_loss no mejora en 15 épocas → lr ← lr × 0.9 (hasta mínimo 1×10⁻⁵). Permite que el optimizador se adapte sin EarlyStopping.

**ModelCheckpoint**: guarda los pesos del epoch con menor val_loss. `restore_best_weights()` los recarga tras `model.fit()`. Esto garantiza que el MAE test se mide sobre el mejor modelo, no el del último epoch.

**EarlyStopping omitido deliberadamente**: se quería ver la curva completa de aprendizaje para diagnosticar si el modelo converge, diverge o colapsa. Fue una decisión pedagógica y diagnóstica.

**Puntos débiles:**
- Sin EarlyStopping, algunos modelos entrenan epochs innecesarios (NB05 usa 500 epochs). Con EarlyStopping se ahorraría tiempo de cómputo.
- ReduceLROnPlateau con patience=15 puede reaccionar lento en series ruidosas.

---

### Tabla 0c — Partición cronológica

| Split | % total | Detalle |
|---|---|---|
| Train | 67.5% | 75% de train_full, ventanas temporalmente anteriores |
| Validación | 22.5% | 25% de train_full, señal para ReduceLROnPlateau y ModelCheckpoint |
| Test | 10% | Separado antes del split train/val, nunca visto |

**Por qué 25% de validación (no el típico 5%):**
Con un val del 5% sobre series de retornos diarios, la señal de val_loss es muy ruidosa. Un val pequeño dispara reducciones de LR espurias. El 25% da a ReduceLROnPlateau una señal más estable (D7).

**Shuffle=False siempre**: las ventanas temporales no pueden mezclarse. Usar shuffle=True en series temporales es filtración de información futura al entrenamiento.

**Gap val → test documentado**: el MAE sube un 15–22% de val a test en todos los modelos. No es overfitting — es el comportamiento estructural de una serie cronológica donde el test está en el futuro más lejano.

**Cómo argumentarlo:**
> "El gap entre validación y test no es una señal de que los modelos sobreajustan. Es el efecto del desplazamiento temporal: val cubre un período más cercano al train, test es el más lejano. Este patrón aparece en todos los modelos, incluido el naive, lo que confirma que es una propiedad del dataset, no de las redes."

---

### Nota — Naive Forecast

El naive predice `X[:, −1, :]`: el último retorno conocido de la ventana de entrada.

**MAE test:**
- V_out=1d: 0.0178
- V_out=5d: 0.0137
- V_out=30d: 0.0125
- V_out=90d: 0.0122

**Por qué es un baseline asimétrico:** es bueno en V_out=1d porque la autocorrelación de primer orden del SP500 es ligeramente positiva. A partir de V_out=5 ya no sirve — predecir el retorno de ayer para los próximos 30 días es peor que predecir cero. Por eso las redes mejoran al naive hasta −89% en V_out=90d simplemente prediciendo la media.

---

---

## § 01 — Resultado de la Competición

### Qué aparece
Matriz heatmap 4×4 (V_in × V_out) de mejor MAE test para redes neuronales, referencias de lineal y naive, gráfico comparativo por horizonte, tabla de resumen y nota sobre EMH.

### El hallazgo central: las 4 filas de la matriz son idénticas

**Dato concreto (NB06, todos los modelos NN, V_in=10):**
| V_out | MAE NN | MAE Naive | MAE Lineal | NN vs Naive | NN vs Lineal |
|---|---|---|---|---|---|
| 1d | 0.0123 | 0.0178 | 0.0130 | −31% | −5.4% |
| 5d | 0.0056 | 0.0137 | 0.0059 | −59% | −5.1% |
| 30d | 0.0023 | 0.0125 | 0.0024 | −82% | −4.2% |
| 90d | 0.0013 | 0.0122 | 0.0014 | −89% | −7.1% |

**El MAE es idéntico independientemente de V_in.** Cambiar la ventana de entrada de 5 a 90 días no mueve el MAE test. Las 4 filas de la matriz son iguales.

**Por qué:** Los log-retornos del SP500 son ruido blanco (autocorrelación ≈ 0, test Ljung-Box no significativo). Si el input no contiene señal predictiva, añadir más días de input solo añade más ruido. El modelo aprende a ignorarlo.

---

### Por qué sale la Fig.1 (heatmap) y no solo una tabla

El heatmap visualiza de golpe las 16 celdas y permite ver la invarianza de filas sin leer 16 números. Es la prueba visual de que V_in es irrelevante.

---

### Nota EMH — por qué es la respuesta correcta

> "Lo que todos los modelos aprenden no es un fallo de implementación. Bajo MAE como función de pérdida, el estimador óptimo de una variable aleatoria con media μ es predecir siempre μ. Si los log-retornos son ruido blanco (que es exactamente lo que dice la Hipótesis de Mercados Eficientes en forma débil), entonces predecir la media incondicional ≈ 0 es la respuesta estadísticamente correcta. La red no está fallando — está siendo óptima dado el problema que se le plantea."

**Puntos fuertes:**
- El resultado es consistente con la literatura financiera (Fama, 1970; Malkiel, 2003).
- La diferencia entre NN y lineal (−5%) es marginal y en la práctica no tiene valor económico.
- El resultado es reproducible en los 256 entrenamientos.

**Puntos débiles:**
- Solo se prueba con retornos univariantes de precio. Datos exógenos (VIX, macro) podrían romper la hipótesis.
- El período 1945–2026 incluye regímenes muy distintos. Un modelo específico por régimen podría tener MAE diferente.
- No se prueba con retornos de alta frecuencia donde la autocorrelación sí existe.

**Alternativas que se podrían haber probado:**
- Clasificación de dirección (↑/↓) con cross-entropy en lugar de regresión con MAE.
- Predicción de volatilidad (GARCH como baseline) — que sí tiene estructura temporal.
- Variables exógenas (spreads de crédito, VIX, datos macroeconómicos).

---

---

## § 02 — Reflexión sobre Modelos — 16 Arquitecturas

### Qué aparece
Gráfico de barras con MAE por arquitectura (V_out=1d, V_in=30d), Tabla 2 comparativa de todos los modelos, Tabla 3 de mejor arquitectura por criterio, nota de conclusión de diseño.

### El hallazgo: Δ MAE < 0.0001 entre todas las NN en 15 de 16 combinaciones

**Fig.3 — MAE test por arquitectura (V_out=1d, V_in=30d):**

Todos los modelos NN se agrupan en una banda estrecha alrededor de 0.0123. La diferencia máxima entre cualquier par de redes neuronales es 0.0001 (1 punto base en retornos diarios = 0.01%). Esto es estadísticamente irrelevante.

**Tabla comparativa completa (V_in=30, MAE 1d):**

| Modelo | Familia | MAE 1d | MAE 5d | MAE 30d | MAE 90d |
|---|---|---|---|---|---|
| naive | Baseline | 0.0178 | 0.0137 | 0.0125 | 0.0122 |
| lineal (V_in=5) | Baseline | 0.0124 | 0.0056 | 0.0023 | 0.0013 |
| lineal (V_in=30) | Baseline | 0.0130 | 0.0059 | 0.0024 | 0.0014 |
| mlp_s | MLP | 0.0123 | 0.0056 | 0.0023 | 0.0013 |
| simple_rnn | RNN | 0.0123 | 0.0056 | 0.0024 | 0.0013 |
| gru | RNN | 0.0123 | 0.0056 | 0.0024 | 0.0013 |
| lstm | RNN | 0.0123 | 0.0056 | 0.0024 | 0.0013 |
| lstm_stack | RNN | 0.0123 | 0.0056 | 0.0024 | 0.0013 |
| bi_gru | RNN | 0.0123 | 0.0056 | 0.0024 | 0.0013 |
| lstm_drop | RNN | 0.0123 | 0.0056 | 0.0024 | 0.0013 |
| conv_s | Conv1D | 0.0123 | 0.0056 | 0.0023 | 0.0013 |
| conv_lstm_ln | Mixto | 0.0123 | 0.0056 | 0.0024 | 0.0013 |
| conv_gru_bottleneck | Mixto | 0.0123 | 0.0056 | 0.0024 | 0.0013 |
| conv_bilstm | Mixto | 0.0123 | 0.0056 | 0.0024 | 0.0013 |
| conv2_lstm | Mixto | 0.0123 | 0.0056 | 0.0024 | 0.0013 |
| lstm_dense | Mixto | 0.0123 | 0.0056 | 0.0024 | 0.0013 |
| conv_dense | Mixto | 0.0123 | 0.0056 | 0.0023 | 0.0013 |

**La regresión lineal con V_in=5 (2.668 parámetros) es idéntica a cualquier NN en 3 de 4 horizontes.** Solo en V_out=1d el lineal queda un 5.4% por detrás.

---

### Por qué MLP pierde estructura temporal y aun así obtiene el mismo resultado

El MLP aplana la entrada: `(N, V_in, 23) → (N, V_in×23)`. Pierde completamente el orden temporal. Sin embargo el MAE es idéntico al LSTM que sí mantiene la dimensión temporal. **Esto confirma que no hay estructura temporal que explotar** — si la hubiera, el LSTM debería mejorar al MLP.

---

### Por qué el LSTM bidireccional no mejora

`bi_gru` tiene +0.0001 respecto a `gru` simple. La bidireccionalidad usa información futura del input (dentro de la ventana), lo que debería ayudar si hay patrones en la secuencia. El resultado nulo confirma que no los hay.

---

### Tabla 3 — Mejor arquitectura por criterio

| Criterio | Modelo | Params |
|---|---|---|
| Mínimos params | simple_rnn | 2.551 |
| Mejor ratio eficiencia | conv_dense | 10.135 |
| Con regularización explícita | lstm_drop | 24.023 |
| Referencia sin NN | lineal (V_in=5) | 2.668 |

**Cómo argumentar la elección:**
> "Si el MAE es idéntico en todos los modelos, el criterio de selección cambia: se elige el más simple. simple_rnn con 2.551 parámetros obtiene exactamente el mismo MAE que conv_bilstm con 30.807 parámetros. Desde el punto de vista de parsimonia estadística, simple_rnn es la arquitectura correcta — y la regresión lineal, con prácticamente los mismos parámetros, es indistinguible."

---

### Nota de conclusión de diseño — por qué profundidad y regularización no ayudan

La evidencia acumulada de NB02 (Ev.1 a Ev.8) muestra:
- 9 variantes de MLP (11K–108K params): Δ_max = 0.0001
- LR=1e-4 vs 3e-4: Δval_min = 0.00009 (por debajo del umbral 0.0005)
- L2=1e-4: mejora en 15/16 combinaciones de val, pero el MAE test no cambia
- Batch [16, 64, 256, 512]: MAE invariante
- NB03 (6 recurrentes): todos convergen al mismo MAE independientemente de profundidad, dropout o bidireccionalidad
- NB04 (convolucionales): idéntico al lineal en todos los horizontes
- NB05 (mixtos): 4 iteraciones de regularización refinada → las curvas mejoran visualmente pero el MAE test no se mueve

**Conclusión ineludible:** el cuello de botella no es la arquitectura. No hay señal en los retornos pasados bajo EMH. Añadir capas, gates, convoluciones o regularización es maquillaje sobre un problema de datos.

---

---

## § 03 — Preprocesado Avanzado — Diferenciación Fraccional (FFD)

### Qué aparece
Definición matemática de FFD, gráfico comparativo de las 5 técnicas de preprocesado, tabla de resultados, barrido de d óptimo, resultados por horizonte.

### Qué es FFD (M. López de Prado, *Advances in Financial ML*, cap. 5)

La diferenciación fraccional aplica un operador de diferenciación de orden **d fraccionario** (0 < d < 1) sobre el log-precio:

```
w_k = (-1)^k · Γ(d+1) / (k! · Γ(d-k+1))
```

Los pesos decaen según k (memoria larga) y se truncan cuando |w_k| < 1×10⁻⁵.

**El tradeoff fundamental:**
- d = 0 → log-precio puro: no estacionario, no se puede entrenar
- d = 1 → log-retornos puros: estacionario, pero sin memoria (información destruida)
- d ≈ 0.2 → **estacionario Y con memoria a largo plazo preservada** ← hipótesis de trabajo

---

### Las 5 técnicas y sus resultados (LSTM(64), V_in=30)

| Técnica | MAE V_out=1d | Δ vs crudo | MAE V_out=30d | Δ vs crudo |
|---|---|---|---|---|
| Crudo (log-retornos) | 0.0123 | — | 0.0024 | — |
| StandardScaler | 0.0128 | +4.1% ✗ | 0.0026 | +8.3% ✗ |
| Rolling Z-score | 0.0126 | +2.4% ✗ | 0.0026 | +8.3% ✗ |
| **FFD (d=0.2)** | **0.0112** | **−8.9% ✓** | **0.0035** | **+45.8% ✗** |
| Feature Engineering | 0.0125 | +1.6% ✗ | 0.0026 | +8.3% ✗ |

**Por qué StandardScaler empeora:** la estandarización global elimina información sobre el régimen de volatilidad. El modelo pierde la capacidad de distinguir un período de alta volatilidad (2008, 2020) de uno de baja. Bajo MAE esta información marginal de régimen tiene valor, aunque sea pequeño.

**Por qué Rolling Z-score empeora:** la normalización local introduce artefactos en los bordes de ventana. Además, en retornos ya estacionarios, normalizar localmente distorsiona la escala relativa entre activos.

**Por qué Feature Engineering no ayuda:** las 5 features derivadas (vol_5d, momentum_10d, vol_ratio, corr_cross) son transformaciones no lineales de los retornos. Si los retornos son ruido blanco, sus transformaciones también lo son. La arquitectura multi-rama (NB07 Ext.C) confirma: MAE V_out=1d = 0.0124 (+0.8%).

---

### Barrido de d óptimo (V_in=30, V_out=1d)

| d | MAE | Δ vs crudo |
|---|---|---|
| 0.1 | 0.0122 | −0.1% |
| **0.2** | **0.0112** | **−8.9% ✓** |
| 0.3 | 0.0113 | −8.1% |
| 0.4 | 0.0114 | −7.3% |
| 0.5 | 0.0116 | −5.7% |
| 0.6 | 0.0124 | +0.8% |
| 0.8 | 0.0124 | +0.8% |
| 1.0 | 0.0124 | +0.8% (≡ crudo) |

**d=0.2 es el óptimo.** A partir de d=0.6 la serie FFD converge a los retornos crudos (el operador ya es casi diferenciación entera) y el MAE es idéntico.

**Por qué d=0.2 ayuda solo en V_out=1d:**
- V_out=1 captura tendencia inercial de muy corto plazo: el log-precio fraccional retiene la correlación serial de nivel que los retornos destruyen.
- V_out=30 y V_out=90 requieren estructura de largo plazo que FFD no puede aportar — de hecho empeora (+45.8%) porque añade una componente de nivel que confunde la predicción a largo plazo.

**Cómo argumentarlo:**
> "FFD de López de Prado es una herramienta sofisticada que en nuestro contexto solo funciona en el horizonte de 1 día. La razón es teóricamente coherente: conservar memoria fraccional del log-precio ayuda a capturar la inercia de muy corto plazo, pero a 30 días esa información ya no es predictiva y se convierte en ruido adicional. El resultado es una mejora selectiva, no general."

---

### Puntos fuertes de la sección FFD
- Es la única mejora documentada del experimento (−8.9% en V_out=1d).
- El barrido sistemático de d valida que d=0.2 es el óptimo y no un valor arbitrario.
- La combinación FFD + Feature Engineering (Ext.D) consigue −6.5% en V_out=1d y ≈0% en V_out=30d, mostrando que se puede combinar sin empeorar el largo plazo.

### Puntos débiles
- La mejora absoluta es de 0.0011 en MAE (0.11% de error medio). En términos económicos es muy pequeña.
- No se probó FFD con d óptimo diferente por horizonte (d=0.2 para 1d, d=0.05 para 30d).
- Se usó un único modelo de referencia (LSTM(64)); la mejora podría ser específica del modelo.

---

---

## § 04 — Carteras 2025 — Resultados Reales

### Qué aparece
Gráfico de retorno acumulado Ene–Dic 2025 (249 días), tabla de métricas de portfolio, nota de hallazgo sobre la naturaleza long-only y el colapso de predicciones, metodología con rebalanceo ejecutada.

### El contexto: por qué MLP Dense(64) y V_out=90

Del NB06, todos los modelos NN son equivalentes en MAE. Para la cartera se eligió:
- **MLP Dense(64)**: arquitectura más simple que obtiene el mismo resultado, más interpretable.
- **V_out=90**: predicción de retorno promedio para 90 días futuros → determina ponderación de largo plazo.
- **V_in=10**: NB06 confirma que V_in es irrelevante; se elige 10 como valor central.

**Entrenamiento:** datos 1960–2024-12-31 (15.857 días). El modelo se entrena una sola vez y se usa para inferencia periódica durante 2025.

---

### Estrategia de rebalanceo (novedad respecto al diseño inicial)

La cartera NN implementa **dos niveles de rebalanceo**:

1. **Rebalanceo del modelo cada 90 días** (3 veces en 2025): en las fechas 2025-01-03, 2025-05-15 y 2025-09-24 se genera una nueva predicción usando la ventana de V_in=10 días más reciente. Los pesos objetivo de la cartera se actualizan con la nueva predicción.

2. **Rebalanceo correctivo cada 21 días bursátiles** (≈ mensual): dentro de cada período de 90 días, los pesos derivan naturalmente por el comportamiento de los activos. Cada 21 días se vuelve a los pesos objetivo del modelo para corregir esa deriva.

La cartera Buy & Hold **no rebalancea**: se compra en la fecha inicial y los pesos derivan libremente.

---

### El hallazgo clave: las predicciones son idénticas en los 3 rebalanceos

El modelo genera **exactamente los mismos valores** en los tres rebalanceos de 2025, confirmando el colapso al predictor de la media:

| Ticker | Predicción | Peso NN |
|---|---|---|
| KR | +0.00077 | 6.84% |
| MO | +0.00071 | 6.35% |
| MSI | +0.00058 | 5.18% |
| JNJ | +0.00055 | 4.87% |
| GD | +0.00055 | 4.86% |
| … | … | … |
| IP | +0.00029 | 2.54% (mínimo) |

Todas las predicciones son positivas → **cartera long-only** (no long/short como se esperaba). El modelo no distingue el contexto de mercado entre enero, mayo y septiembre de 2025. La ponderación es estática en la práctica.

---

### Resultados 2025 (249 días de trading, Ene–Dic 2025)

| Métrica | Cartera NN | Buy & Hold | Δ |
|---|---|---|---|
| Retorno total | **+16.96%** | +16.82% | +0.14 pp |
| Retorno anual | **+17.18%** | +17.04% | +0.14 pp |
| Volatilidad anual | 13.26% | **13.22%** | +0.04 pp |
| Sharpe ratio | **1.296** | 1.289 | +0.007 |
| Sortino ratio | **1.588** | 1.566 | +0.022 |
| Max Drawdown | **−10.75%** | −10.85% | +0.10 pp (mejora) |

**La cartera NN gana en retorno, Sharpe, Sortino y MaxDD**, aunque con márgenes mínimos. La volatilidad es ligeramente superior (+0.04 pp) al concentrar más en ciertos activos.

---

### Cómo argumentar el resultado

**Argumento principal:**
> "La diferencia de +0.14 pp en retorno total durante 249 días de trading es marginal. El modelo predice retornos positivos idénticos en los tres rebalanceos — el contexto de mercado no cambia las predicciones. La ventaja de la cartera NN no proviene de predicción sino de **sobre-ponderación de activos defensivos con dividendos** (KR, MO, JNJ) que históricamente tienen menor beta y mayor Sharpe. El modelo aprende implícitamente esta exposición al entrenarse sobre 65 años de datos. Es gestión pasiva de factor, no predicción activa."

**Evidencia adicional del colapso:**
El hecho de que las predicciones sean **literalmente idénticas** en enero, mayo y septiembre de 2025 — tres contextos de mercado completamente distintos (bull market, corrección de abril, recuperación) — es la prueba más contundente del colapso. Un modelo con capacidad predictiva real variaría sus estimaciones según las condiciones de mercado.

**Puntos fuertes:**
- Los resultados son reales (datos de mercado 2025, no simulación).
- El Sharpe de 1.296 y Sortino de 1.588 son excelentes para cualquier cartera pasiva.
- El MaxDD de −10.75% (vs −10.85%) confirma que los pesos NN ofrecen una ligera protección.
- El rebalanceo correctivo cada 21 días es una práctica institucional estándar.

**Puntos débiles:**
- 249 días no es suficiente para extraer conclusiones estadísticamente significativas (se necesitan al menos 3–5 años).
- Las predicciones idénticas en los 3 rebalanceos convierten la estrategia en prácticamente estática.
- El año 2025 fue positivo para el SP500. Un año bajista podría revelar si los pesos NN realmente ofrecen protección o si la correlación con BH es total.
- La mayor volatilidad de NN (+0.04 pp) frente a BH es una pequeña penalización por concentración.

**Alternativa que se podría implementar:**
- Re-entrenar el modelo en cada rebalanceo de 90 días (actualmente solo se usa para inferencia). Incorporar datos de 2025 podría mover las predicciones — si aun así permanecen iguales, confirmaría el colapso definitivamente.
- Cartera con predicciones de clasificación (↑/↓) generaría posiciones cortas reales y diferenciaría más del BH.

---

### Metodología ejecutada paso a paso

1. Modelo: MLP Dense(64, L2=1×10⁻⁴) → Dense(23). V_in=10, V_out=90, lr=1×10⁻⁴, 300 épocas, batch=64. Entrenado sobre 1960–2024 (15.857 días).
2. **3 rebalanceos del modelo** (cada 90 días hábiles): 2025-01-03, 2025-05-15, 2025-09-24. En cada fecha se genera nueva predicción con los últimos 10 días disponibles.
3. Cartera NN: `w_i = y_pred_i / Σ|y_pred_i|` — todos positivos → long-only. **Rebalanceo correctivo cada 21 días** dentro de cada período de 90 días.
4. Cartera BH: `w_i = 1/23` inicial, sin rebalanceo posterior. Pesos derivan libremente.
5. Período de evaluación: 1 ene 2025 – 31 dic 2025 (249 días de trading).

---

---

## § 05 — Conclusiones

### Qué aparece
Tres hallazgos principales, tabla de "qué haríamos diferente", nota central del taller.

### Hallazgo 1 — El colapso a la media es la respuesta correcta

**Respaldo cuantitativo:**
- 256 entrenamientos, 16 arquitecturas, 16 combinaciones de ventanas.
- MAE test: 0.0123 (V_out=1d), 0.0056 (V_out=5d), 0.0023 (V_out=30d), 0.0013 (V_out=90d).
- Coeficiente de variación entre modelos: <0.5% para V_out=1d.
- std(pred)/std(y) ≈ 0.06–0.10 → el modelo predice prácticamente un valor constante.
- Test Ljung-Box no significativo → los log-retornos son ruido blanco.

**Cómo argumentarlo:**
> "Cuando la función de pérdida es MAE y el objetivo es ruido blanco, el estimador de Bayes — el que minimiza el error esperado — es la media incondicional. Las redes lo aprenden empíricamente. Si los retornos del SP500 fueran predecibles a partir de su historia, habría una diferencia entre LSTM y regresión lineal. No la hay."

---

### Hallazgo 2 — La regresión lineal iguala a cualquier red neuronal

**Regresión lineal V_in=5 (2.668 params):**
- MAE 1d: 0.0124 vs NN: 0.0123 → diferencia: 0.8%
- MAE 5d: 0.0056 vs NN: 0.0056 → diferencia: 0%
- MAE 30d: 0.0023 vs NN: 0.0023 → diferencia: 0%
- MAE 90d: 0.0013 vs NN: 0.0013 → diferencia: 0%

Un modelo con 2.668 parámetros y sin activaciones no lineales es estadísticamente indistinguible de un LSTM con 27.143 parámetros, gates de memoria, y 300 epochs de entrenamiento.

**Cómo argumentarlo:**
> "Este resultado no desacredita las redes neuronales — las valida en un sentido más profundo: si el problema tiene estructura no lineal, el LSTM la encontrará y superará al lineal. El hecho de que no lo haga confirma que el problema no tiene esa estructura. El experimento funciona como test de detección de no-linealidad, y el resultado es negativo."

---

### Hallazgo 3 — FFD(d=0.2) rompe el techo en V_out=1d (−8.9%) pero empeora en V_out≥30d (+45.8%)

**Por qué esto es importante:**
- Demuestra que el preprocesado puede abrir ventanas de mejora específicas.
- Pero también que una mejora en un horizonte puede ser destructiva en otro.
- La señal útil (inercia de muy corto plazo en el log-precio) existe solo en el horizonte de 1 día.

**Implicación práctica:**
> "Para predicción intradiaria o de 1 día, FFD(d=0.2) sobre el log-precio es la transformación recomendada. Para horizontes de 30+ días, los retornos crudos siguen siendo la mejor entrada."

---

### Tabla — Qué haríamos diferente

| Limitación actual | Alternativa y por qué |
|---|---|
| Regresión del nivel (MAE) | **Clasificación ↑/↓** (cross-entropy). Bajo EMH débil, el signo del retorno puede tener estructura aunque el nivel no la tenga. Cross-entropy no colapsa a la media — colapsa al 50/50, que ya tiene valor en trading. |
| Solo log-retornos | **VIX + datos macro**. El VIX tiene autocorrelación alta y predice la volatilidad futura. Las variables macro (ISM, empleo) tienen rezagos conocidos. Features con autocorrelación real. |
| Ventana fija V_in | **Transformers / Atención**. Un mecanismo de atención aprendería qué días del pasado son relevantes sin que nosotros lo fijemos. Potencialmente útil si hay patrones estacionales. |
| Retornos crudos para V_out=1d | **FFD(d=0.2) selectivo**. La única mejora real documentada. Aplicarlo solo a V_out=1 evita el deterioro en horizontes largos. |

---

### La lección central

> **"El cuello de botella es siempre la señal — nunca la arquitectura."**

Los 256 experimentos lo demuestran empíricamente. Ni profundidad, ni bidireccionalidad, ni convoluciones, ni regularización L2, ni dropout pueden crear información predictiva donde no existe.

La única mejora real (FFD, −8.9%) no viene de cambiar la arquitectura sino de cambiar **qué información entra al modelo**: preservar memoria fraccional del log-precio en lugar de usar solo los retornos.

---

---

## Preguntas frecuentes que puede hacer el tribunal

### "¿Por qué no hicisteis búsqueda de hiperparámetros por modelo?"

La evidencia de NB02 (Evidencias 1 a 8) muestra que LR, batch size, tamaño de capa, L2, dropout no mueven el MAE en más de 0.0001. El colapso es anterior al optimizador: es una propiedad del problema. El tuning por modelo habría multiplicado el tiempo de cómputo sin cambiar la conclusión.

### "¿El resultado no podría deberse a un bug en el código?"

Tres validaciones independientes lo descartan: (1) el naive forecast da el MAE esperado (0.0178 para V_out=1d) y se degrada correctamente al aumentar el horizonte; (2) la regresión lineal da valores coherentes con la literatura; (3) el gap val→test es consistente en todos los modelos, incluyendo el naive, lo que confirma que es estructural al dataset.

### "¿Por qué no usasteis datos de mayor frecuencia (intradiario)?"

Con datos de mayor frecuencia, la autocorrelación a corto plazo (microestructura de mercado) sí es estadísticamente significativa. Los resultados probablemente serían diferentes. Pero el enunciado del taller especifica datos diarios del SP500, y en esa escala la forma débil de la EMH es bien establecida en la literatura.

### "¿La cartera NN realmente usa redes neuronales para tomar decisiones?"

En sentido estricto, sí — los pesos provienen de las predicciones del MLP. Pero dado que todas las predicciones son positivas (colapso a la media positiva histórica), la "decisión" de la red es efectivamente la misma para todos los activos: mantener largo. La diferencia con Buy & Hold viene solo de la magnitud relativa de las predicciones, que sobre-pondera ligeramente activos defensivos.

### "¿Qué añadiríais si tuvierais más tiempo?"

Por orden de impacto esperado:
1. Cambiar a clasificación de dirección (cross-entropy) — cambia fundamentalmente el problema.
2. Incorporar VIX y variables macro como features — introduce autocorrelación real.
3. Modelos específicos por régimen (alta/baja volatilidad).
4. Rebalanceo mensual de la cartera.
5. Transformer / Atención sobre las 23 series conjuntamente (modelo multivariante real).

---

---

*Fin del guion — todos los datos provienen de los notebooks 00–08 del proyecto.*
