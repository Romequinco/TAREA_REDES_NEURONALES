# Preprocesado de Datos

## Sub-conceptos

### Estandarización
**Definición:** Preprocesamiento que resta la media y divide por desviación estándar para que todas las variables tengan escala comparable.
*Ref adicional:* Transformación que centra los datos en media 0 y desviación estándar 1: (x - μ) / σ

### One-hot encoding
**Definición:** Conversión de etiquetas categóricas a vectores binarios (ej: 3 → [0,0,0,1,0,0,0,0,0,0])
*Ref adicional:* Transformación de etiquetas categóricas en vectores binarios donde solo un elemento es 1 y el resto 0.
*Ref adicional:* Transformación de etiquetas categóricas en vectores binarios (to_categorical), necesaria para clasificación multiclase.
*Ref adicional:* Técnica que convierte categorías en vectores binarios para evitar que el modelo interprete relaciones ordinales entre categorías nominales
*Ref adicional:* Codificación de variables categóricas nominales creando columnas binarias (0/1) por categoría, haciendo cada categoría equidistante sin imponer orden artificial.

### Categorical encoding
**Definición:** Transformación de etiquetas numéricas (0-9) a vectores one-hot de 10 dimensiones mediante `to_categorical()`

### Data Augmentation
**Definición:** Técnica para aumentar artificialmente el conjunto de datos mediante transformaciones que preservan las características relevantes.

### Normalización local
**Definición:** Técnica inspirada en neurociencia que ajusta valores según su contexto local, similar a cómo el cerebro adapta la percepción visual al contraste

### Train/Test split
**Definición:** División obligatoria de datos en conjuntos de entrenamiento y prueba, manteniendo test sin usar durante entrenamiento.
*Ref adicional:* División temporal de datos donde el periodo de entrenamiento se usa para estimar parámetros (μ, Σ) y el periodo de test para evaluar el rendimiento real de la estrategia sin reoptimizar.
*Ref adicional:* División disjunta del dataset en subconjuntos de entrenamiento y prueba para evaluar la capacidad de generalización del modelo a datos no vistos.

### Outlier
**Definición:** Valor atípico que se desvía significativamente del patrón general de los datos.

### Valores faltantes (missing values)
**Definición:** Datos ausentes en el dataset que requieren identificación y tratamiento

### Outliers
**Definición:** Valores anómalos que se desvían significativamente del patrón general de los datos
*Ref adicional:* Valores atípicos que se pueden detectar visualmente en boxplots como puntos fuera de los bigotes

### Normalización de nombres
**Definición:** Proceso de estandarización de nombres corporativos para detectar duplicados por fusiones/adquisiciones (ej: "Corporation" → "Corp")

### Z-Score (normalización)
**Definición:** Estandarización estadística que transforma retornos a unidades de desviación estándar: $Z = (R - \mu) / \sigma$, permitiendo comparabilidad entre periodos de distinta volatilidad.

### Normalización de pesos
**Definición:** Proceso de dividir cada peso aleatorio por la suma total para garantizar que Σwi = 1 (restricción de presupuesto completo).

### Normalización de Factores
**Definición:** Proceso de estandarización: (valor_factor - media) / desviación_típica. Permite comparabilidad entre factores y facilita interpretación (>0 = por encima media).

### Outliers en mercados financieros
**Definición:** Eventos extremos que ocurren con frecuencia muy superior a la predicha por distribuciones normales. Caídas del 10-15% deberían verse "en 1000 años" estadísticamente, pero son recurrentes

### Normalización
**Definición:** Transformación de variables a un rango común (típicamente [0,1]) dividiendo por el máximo o usando (x-min)/(max-min)

### Normalización de datos
**Definición:** División de valores de píxeles por 255 para escalar de [0,255] a [0,1], mejorando la convergencia del modelo.

### Train-Test Split
**Definición:** División del dataset en conjuntos de entrenamiento (70%) y prueba (30%)

### train_test_split
**Definición:** Función que divide el dataset en conjuntos de entrenamiento y prueba para validar la capacidad de generalización del modelo

### Detección de anomalías no supervisada
**Definición:** Estrategia de entrenar solo con datos normales para que el modelo identifique patrones anómalos por su mayor error.

### Outlier/Ruido
**Definición:** En DBSCAN, puntos etiquetados como -1 que no pertenecen a ningún cluster por no cumplir criterios de densidad.

### SMOTE (Synthetic Minority Over-sampling Technique)
**Definición:** Técnica de sobremuestreo que genera muestras sintéticas de la clase minoritaria interpolando entre vecinos cercanos, reduciendo el riesgo de overfitting comparado con simple duplicación.

### Class Imbalance
**Definición:** Desbalance significativo entre clases (ej: 99% vs 1%) que requiere técnicas como SMOTE, ADASYN, undersampling o métricas apropiadas (F1, AUC-ROC) en lugar de accuracy.

## Apariciones consolidadas
| Sub-concepto | Bloque | Sección | Archivo |
|-------------|--------|---------|---------|
| Estandarización | B3_IA_Basica | S3_ML_Supervisado | transcripcion-ml-supervisado-y-no-supervisado-4.md |
| One-hot encoding | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-minimo-funcional.md |
| One-Hot Encoding | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-regularization-batchnormalization-bn.md |
| Categorical encoding | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-regularization-dropout.md |
| One-hot encoding | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-regularization-regularizador-capa.md |
| One-Hot Encoding | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-visualizaciones-simples.md |
| Data Augmentation | B3_IA_Basica | S4_Redes_Neuronales | training-nn-2026.md |
| One-hot encoding | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| Normalización local | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| Train/Test split | B1_Python_IA | S1_Python_Herramientas | 4-bme-diapositivas-pablo-dia1-p-2.md |
| Outlier | B1_Python_IA | S3_Matematicas_IA | 2-probabilidad-y-estadistica.md |
| Valores faltantes (missing values) | B1_Python_IA | S4_Material_Adicional | bme-4-descriptores-ejercicio.md |
| Outliers | B1_Python_IA | S4_Material_Adicional | bme-4-descriptores-ejercicio.md |
| Outliers | B1_Python_IA | S4_Material_Adicional | bme-7-descriptores-graficos-ejercicio.md |
| Normalización de nombres | B2_Finanzas | S10_Backtesting_Avanzado | 01-carga-datos.md |
| Z-Score (normalización) | B2_Finanzas | S10_Backtesting_Avanzado | 03-estrategia.md |
| Train/Test Split | B2_Finanzas | S7_Gestion_Carteras | 22-comparacion-carteras-ipynb-colab.md |
| Normalización de pesos | B2_Finanzas | S7_Gestion_Carteras | 4-carteras-cuatro-activos-solucion-ipynb-colab.md |
| Normalización de Factores | B2_Finanzas | S7_Gestion_Carteras | transcripcion-gestion-carteras-3.md |
| Outliers en mercados financieros | B2_Finanzas | S9_Algoritmos_Inversion | transcripcion-algoritmos-inversion-2.md |
| Normalización | B3_IA_Basica | S2_Tipos_Aprendizaje | data-preproc-y-seleccion-feats.md |
| Estandarización | B3_IA_Basica | S2_Tipos_Aprendizaje | data-preproc-y-seleccion-feats.md |
| Normalización de datos | B3_IA_Basica | S2_Tipos_Aprendizaje | models-ann-cifar100.md |
| One-hot encoding | B3_IA_Basica | S2_Tipos_Aprendizaje | models-ann-cifar100.md |
| Train-Test Split | B3_IA_Basica | S2_Tipos_Aprendizaje | models-ml-my-linear-regresion-by-hand.md |
| train_test_split | B3_IA_Basica | S2_Tipos_Aprendizaje | models-ml-my-linear-regresion.md |
| Detección de anomalías no supervisada | B3_IA_Basica | S3_ML_Supervisado | 5-dimensionality-reduction-anomalias.md |
| Outlier/Ruido | B3_IA_Basica | S3_ML_Supervisado | 9-clustering-dbscan-meanshift.md |
| Data Augmentation | B3_IA_Basica | S3_ML_Supervisado | intro-ml-2025.md |
| SMOTE (Synthetic Minority Over-sampling Technique) | B3_IA_Basica | S3_ML_Supervisado | miax-ml-01-data-preprocessing.md |
| One-Hot Encoding | B3_IA_Basica | S3_ML_Supervisado | miax-ml-01-data-preprocessing.md |
| Class Imbalance | B3_IA_Basica | S3_ML_Supervisado | miax-ml-01-data-preprocessing.md |
| Train/Test Split | B3_IA_Basica | S3_ML_Supervisado | miax-ml-02-evaluation-metrics.md |
