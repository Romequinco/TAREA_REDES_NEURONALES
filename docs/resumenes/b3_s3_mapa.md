# Mapa de conocimiento — B3 — ML Supervisado y No Supervisado

## Conceptos centrales del bloque

- **Reducción de dimensionalidad**: Tema transversal más recurrente del bloque, apareciendo en contextos de optimización computacional, visualización, detección de patrones y preprocesamiento. Abarca tanto selección (filter/wrapper/embedded) como extracción (PCA, ICA, LDA, t-SNE, UMAP, autoencoders). Es fundamental para combatir la maldición de la dimensionalidad y habilita tanto aprendizaje supervisado como no supervisado.

- **PCA (Principal Component Analysis)**: Método más referenciado del bloque completo. Técnica base de extracción de características utilizada como método principal, herramienta de visualización, inicialización de t-SNE, preprocesamiento para clustering y detección de anomalías. Requiere comprensión de álgebra lineal (autovalores, autovectores, matriz de covarianza).

- **Clustering**: Concepto vertebrador del aprendizaje no supervisado con múltiples variantes (K-means, jerárquico, DBSCAN, Mean Shift, Spectral). Representa el problema fundamental de descubrir patrones en datos sin etiquetas, con aplicaciones críticas en segmentación de clientes, fondos de inversión, detección de regímenes de mercado y sistemas de recomendación.

- **Ensemble Learning**: Paradigma dominante en ML supervisado moderno que combina múltiples modelos débiles para crear predictores robustos. Incluye estrategias paralelas (Bagging, Random Forest) y secuenciales (AdaBoost, Gradient Boosting, XGBoost). Fundamental para reducir varianza y sesgo simultáneamente.

- **Explicabilidad de modelos (XAI)**: Requisito crítico que atraviesa todo el bloque, especialmente en finanzas reguladas. Incluye LIME, SHAP (valores de Shapley), contrafactuales, permutation importance. Representa la tensión fundamental entre interpretabilidad y rendimiento predictivo.

- **Redes Bayesianas**: Framework conceptual que unifica teoría de grafos, probabilidad condicional e inferencia causal mediante DAGs. Permite descomponer probabilidades complejas, realizar inferencia probabilística y descubrir relaciones causales automáticamente (structure learning).

- **Validación temporal**: Principio crítico que diferencia ML académico de ML aplicado en finanzas. Incluye Time Series Split, Purged K-Fold, prohibición de backward fill, test set sagrado. Evita data leakage temporal y reconoce que los modelos deben recalibrarse con cambios de régimen.

- **Normalización/Estandarización**: Prerequisito crítico mencionado en prácticamente todos los documentos técnicos. Condición necesaria para PCA, K-means, comparación de coeficientes en modelos lineales, y métricas de distancia. StandardScaler es la implementación más utilizada.

- **Regularización (L1/L2/ElasticNet)**: Técnica omnipresente para controlar el trade-off bias-variance. Aparece en regresión lineal/logística, replicación de índices, ensembles (max_depth, learning_rate), y Deep Learning (dropout, batch normalization). Los hiperparámetros actúan como regularizadores implícitos.

- **Métricas de evaluación**: Concepto diferenciador entre supervisado (accuracy, precision, recall, F1, ROC-AUC, MSE, MAE, R²) y no supervisado (Silhouette, Calinski-Harabasz, Davies-Bouldin, método del codo). Crítico para selección de modelos y ajuste de threshold en clases desbalanceadas.

- **Feature Engineering**: Arte que conecta conocimiento del dominio con ML. En finanzas incluye log-retornos vs precios, factores Fama-French, momentum/volatilidad, betas ajustadas. Consume 15-75% del tiempo del proyecto según la fuente.

- **Embeddings**: Representaciones vectoriales densas que unifican el tratamiento de datos heterogéneos (texto, imágenes, audio). Conecta técnicas clásicas (Word2Vec) con modernas (CLIP, Gemini) y es aplicable a NLP, visión por computador y detección de anomalías.

- **Weak Supervision**: Concepto fundamental que aborda el problema real de etiquetas imperfectas mediante matrices de mezcla y pérdidas propias débiles, reconociendo que las etiquetas perfectas son la excepción, no la norma.

- **Causalidad vs Correlación**: Distinción fundamental enfatizada repetidamente. Correlación no implica causalidad, pero causalidad sí implica correlación. Confounders, colliders y Factor Mirage complican la inferencia causal. Granger Causality es el método estándar en finanzas (90% de aplicaciones).

- **Descenso por Gradiente**: Método fundamental de optimización que conecta todos los temas. Aparece explícitamente en Gradient Boosting, subyace en entrenamiento de redes neuronales, y revela que muchos algoritmos aparentemente diferentes son casos particulares de un framework de optimización más general.

## Grafo de dependencias

- **Preprocesamiento de datos**
  - requiere → Auditoría de tipos de datos, Detección de valores faltantes/outliers (IQR), Encoding de categóricas (LabelEncoder)
  - permite → Feature Selection, Feature Extraction, Clustering, Entrenamiento de modelos supervisados
  - relacionado con → Data Leakage (evitar normalización con test), Imbalance de clases, Survivorship bias
  - implementado por → StandardScaler, MinMaxScaler, Imputación, Winsorización

- **Normalización/Estandarización**
  - requiere → Datos numéricos limpios
  - permite → PCA, K-means, Comparación de coeficientes, Métricas de distancia
  - relacionado con → Data Leakage (fit solo en train), Escalado de features
  - implementado por → StandardScaler (más común), MinMaxScaler (casos específicos)

- **Feature Selection**
  - requiere → Preprocesamiento de datos, Métricas de importancia
  - permite → Reducción de variables redundantes, Mejora de interpretabilidad, Entrenamiento eficiente
  - relacionado con → Feature Extraction (alternativa), Interpretabilidad
  - **Filter methods**:
    - requiere → Métricas estadísticas (varianza, correlación de Pearson/Spearman, información mutua)
    - permite → Ranking rápido de variables, Independencia del modelo
    - implementado por → SelectKBest, VarianceThreshold
  - **Wrapper methods**:
    - requiere → Modelo supervisado específico, Validación train/test, Búsqueda exhaustiva/forward/backward
    - permite → Optimización específica del modelo
    - relacionado con → Cross-validation, Coste computacional alto
  - **Embedded methods**:
    - requiere → Modelos interpretables (regresión lineal, árboles)
    - permite → Selección durante entrenamiento
    - implementado por → Coeficientes de regresión, Feature importance, Lasso (L1)

- **Feature Extraction (métodos lineales)**
  - requiere → Normalización obligatoria, Álgebra lineal (autovalores, autovectores)
  - permite → Reducción dimensional, Captura de relaciones lineales, Descorrelación
  - **PCA**:
    - requiere → Matriz de covarianza, Diagonalización
    - permite → Componentes ortogonales, Maximización de varianza, Visualización, Compresión, Denoising
    - permite → Inicialización de t-SNE, Preprocesamiento para clustering, Detección de anomalías
    - relacionado con → Varianza explicada (80-95%), Método del codo, Eigenfaces
    - implementado por → sklearn.decomposition.PCA
  - **ICA (Independent Component Analysis)**:
    - requiere → PCA (como comparación)
    - permite → Separación de fuentes independientes, Blind source separation
    - relacionado con → Independencia estadística vs descorrelación
  - **LDA (Linear Discriminant Analysis)**:
    - requiere → Etiquetas de clase (supervisado)
    - permite → Maximización de separabilidad entre clases, Reducción supervisada
    - relacionado con → Clasificación, Proyección discriminante
  - **MDS (Multidimensional Scaling)**:
    - requiere → Matriz de distancias
    - permite → Preservación de distancias euclidianas, Visualización

- **Feature Extraction (métodos no lineales)**
  - requiere → Normalización, Hiperparámetros específicos, Estructura de manifold
  - permite → Capturar estructuras complejas, Formas no lineales
  - **Isomap**:
    - requiere → Grafo de vecinos (n_neighbors)
    - permite → Preservación de distancias geodésicas, Manifold learning
  - **t-SNE**:
    - requiere → Perplexity (hiperparámetro crítico), Inicialización (PCA recomendada)
    - permite → Visualización de estructura local, Separación de clusters
    - relacionado con → Divergencia KL, No determinista
  - **UMAP**:
    - requiere → n_neighbors, min_dist
    - permite → Preservación de estructura local y global, Más rápido que t-SNE
    - relacionado con → Teoría de grafos, Topología algebraica

- **Autoencoders**
  - requiere → Redes neuronales, Datos de entrenamiento (comportamiento normal)
  - permite → Reducción no lineal, Detección de anomalías, Compresión, Denoising, Embeddings
  - relacionado con → Error de reconstrucción, Arquitectura encoder-decoder, PCA (versión lineal)
  - aplicaciones → Detección de crisis de mercado, Sistemas de alerta temprana

- **Embeddings**
  - requiere → Preprocesamiento de texto (tokenización, stop words), Modelos preentrenados
  - permite → Representaciones vectoriales densas, Clustering de datos no estructurados, Búsqueda por similitud
  - relacionado con → Similitud coseno, Espacio semántico continuo, Reducción dimensional (t-SNE, UMAP) para visualización
  - **Word2Vec**:
    - requiere → Corpus de entrenamiento
    - permite → Representaciones contextuales, Analogías semánticas
  - **CLIP**:
    - requiere → Modelo preentrenado
    - permite → Embeddings multimodales (texto-imagen), Zero-shot classification
  - **Gemini**:
    - permite → Embeddings de última generación, Multimodalidad avanzada

- **Medidas de dependencia**
  - requiere → Teoría de la información (entropía, información mutua), Métodos de kernel
  - permite → Feature selection, Inferencia causal, Construcción de grafos de dependencia
  - relacionado con → Correlación de Pearson/Spearman, HSIC, Causalidad de Granger
  - progresión → Correlación lineal → Dependencia no lineal → Dependencia condicional → Causalidad direccional

- **Clustering**
  - requiere → Métrica de Similaridad/Distancia, Normalización obligatoria, Definición de cluster
  - permite → Segmentación de clientes/fondos, Detección de anomalías, Sistemas de recomendación, Identificación de regímenes de mercado
  - relacionado con → Reducción de Dimensionalidad (visualización), Evaluación (métricas internas)
  - **K-means**:
    - requiere → Número de clusters (k), Distancia euclidiana, Inicialización de centroides
    - permite → Clustering rápido de datos esféricos, Diagrama de Voronoi
    - relacionado con → Inercia (WCSS), Método del codo, K-medoids (robusta), K-median (L1)
    - limitaciones → Formas no esféricas, Sensibilidad a outliers, Clusters de tamaño similar
  - **Clustering Jerárquico**:
    - requiere → Método de Linkage (single/complete/average/ward), Matriz de distancias
    - permite → Dendrograma, Clustering sin k previo, Estructura anidada, Visualización jerárquica
    - relacionado con → Coeficiente de correlación cofenética, Altura de corte
    - variantes → Aglomerativo (bottom-up), Divisivo (top-down)
  - **DBSCAN**:
    - requiere → Epsilon (ε), Min_samples, Distancia euclidiana
    - permite → Formas arbitrarias, Detección automática de outliers, Clustering sin k previo
    - relacionado con → Mean Shift (también basado en densidad), Detección de anomalías
    - conceptos clave → Core Points, Border Points, Noise
  - **Mean Shift**:
    - requiere → Bandwidth (hiperparámetro crítico), KDE (Kernel Density Estimation)
    - permite → Clustering basado en densidad, Detección de modos
    - relacionado con → Gradiente de densidad, DBSCAN
  - **Spectral Clustering**:
    - requiere → Matriz Laplaciana, Autovectores, Similitud entre puntos
    - permite → Captura de estructuras entrelazadas, Formas no convexas
    - relacionado con → PCA (diagonalización), Teoría de grafos

- **Dynamic Time Warping (DTW)**:
  - requiere → Series temporales, Sakoe-Chiba radius
  - permite → Comparación de series con desfases temporales, Clustering de activos con lag
  - relacionado con → Distancia entre series, Estrategias de arbitraje estadístico
  - aplicaciones → Clustering de series financieras, Detección de patrones temporales

- **Métricas de evaluación de clustering**
  - requiere → Clusters generados, Datos originales
  - permite → Selección de k óptimo, Comparación de algoritmos, Evaluación de calidad
  - **Silhouette Coefficient**:
    - permite → Evaluación de cohesión y separación
    - relacionado con → Sesgo hacia convexidad, Rango [-1, 1]
  - **Calinski-Harabasz**:
    - permite → Evaluación de ratio varianza inter/intra
    - relacionado con → Mayor es mejor
  - **Davies-Bouldin**:
    - permite → Evaluación de similitud entre clusters
    - relacionado con → Menor es mejor
  - **Método del codo (Elbow)**:
    - permite → Selección heurística de k
    - relacionado con → Inercia, Punto de inflexión

- **Validación de modelos**
  - requiere → Train/Test split correcto, Estratificación (clasificación), Respeto de temporalidad (series)
  - permite → Selección de hiperparámetros, Comparación de modelos, Estimación no sesgada de rendimiento
  - relacionado con → Data Leakage (evitar), Overfitting
  - **Cross-Validation**:
    - requiere → Particiones múltiples, Agregación de resultados
    - permite → Reducción de sensibilidad a particiones específicas
    - variantes → K-Fold, Stratified K-Fold, Time Series Split, Purged K-Fold
  - **Validación Temporal**:
    - requiere → Time Series Cross-Validation, Purga temporal, Forward fill (permitido), Backward fill (PROHIBIDO)
    - permite → Backtesting riguroso, Prevención de data leakage temporal
    - relacionado con → Test set sagrado (una sola evaluación), Cambios de régimen
  - **Out-of-Bag (OOB) Validation**:
    - requiere → Bootstrap sampling (~37% no seleccionadas)
    - permite → Validación sin conjunto separado, Optimización de uso de datos
    - relacionado con → Bagging, Random Forest

- **Regresión Lineal/Logística**
  - requiere → Preprocesamiento (escalado, encoding), Validación correcta, Métricas apropiadas
  - permite → Modelos interpretables, CAPM/APT, Stock picking, Predicción de bancarrota
  - relacionado con → Regularización (Ridge/Lasso/ElasticNet), OLS, Cross-entropy loss, Odds ratios
  - **Regresión multifactorial (Fama-French)**:
    - requiere → Factores de mercado (Market-RF, SMB, HML, MOM), Retornos logarítmicos, Alineación temporal
    - permite → Cálculo de Alpha, Beta, R², P-values, T-values
    - relacionado con → Ajuste de betas por significancia, Exceso de rendimiento
    - regla crítica → Siempre incluir factor mercado para evitar estimaciones sesgadas

- **Árboles de Decisión**
  - requiere → Impureza (Gini/Entropía de Shannon), Algoritmo Greedy
  - permite → Random Forest, Gradient Boosting, Interpretabilidad, Visualización de decisiones
  - relacionado con → Overfitting, Desbalanceo de clases (class_weight), Feature importance
  - limitaciones → Greedy (no óptimo global), Inestabilidad, Sesgo hacia variables con muchas categorías

- **K-Nearest Neighbors (KNN)**
  - requiere → Distancia Euclidiana, Espacio Métrico, Normalización
  - permite → Clasificación No Paramétrica, Comparación de patrones temporales completos
  - relacionado con → Maldición de la Dimensionalidad, Dynamic Time Warping
  - ventaja → Superior en series temporales vs árboles (enfoque de características independientes)

- **Bootstrap Sampling**
  - requiere → Muestreo con Reemplazo
  - permite → Bagging, Random Forest, Out-of-Bag Validation, Estimación de incertidumbre
  - relacionado con → Reducción de Varianza, Cross-Validation

- **Bagging (Bootstrap Aggregating)**
  - requiere → Bootstrap Sampling, Agregación de Predicciones (voting/mean)
  - permite → Random Forest, Reducción de varianza, Paralelización
  - relacionado con → Ensemble Learning, Overfitting Prevention
  - mejor con → Predictores inestables (árboles, NNs)

- **Random Forest**
  - requiere → Bagging, Árboles de Decisión, Bootstrap Sampling, Selección aleatoria de features por split
  - permite → Feature Importance, OOB Error, Robustez a outliers, Predicciones robustas
  - relacionado con → ExtraTrees (thresholds aleatorios), Gradient Boosting (contraste paralelo vs secuencial)
  - hiperparámetros críticos → max_features, min_samples_leaf, n_estimators, max_depth

- **Boosting**
  - requiere → Entrenamiento secuencial, Re-ponderación de muestras, Weak Learners
  - permite → Reducción de sesgo y varianza, Enfoque en muestras difíciles
  - relacionado con → Función de pérdida, Optimización, Learning rate
  - **AdaBoost**:
    - requiere → Pesos exponenciales, Clasificadores débiles
    - permite → Combinación ponderada de modelos
    - relacionado con → Error ponderado, Actualización de pesos
  - **Gradient Boosting**:
    - requiere → Función de pérdida diferenciable, Cálculo de gradientes, Descenso por gradiente
    - permite → Funciones de pérdida personalizables, Robustez a outliers (con loss apropiada)
    - relacionado con → XGBoost (optimizado), LightGBM (rápido), CatBoost (categóricas)
    - conceptos → Residuos, Gradiente negativo, Shrinkage
    - conexión profunda → Entrenar sobre residuos = descenso por gradiente en espacio de funciones

- **Stacking**
  - requiere → KFold Cross-Validation, Out-of-Fold Predictions, Múltiples Modelos Base
  - permite → Agregación Ponderada, Meta-Modelos, Combinación de tipos diferentes
  - relacionado con → Bagging, Boosting, Ensemble Learning
  - extensión → Usar redes bayesianas como meta-modelo para modelar dependencias

- **Descenso por Gradiente**
  - requiere → Función diferenciable, Learning rate, Backpropagation (en NNs)
  - permite → Optimización iterativa, Minimización de función de coste
  - variantes → Batch, Mini-batch, SGD, ADAM (adaptativo)
  - relacionado con → Regularización, Hiperparámetros, Gradient Boosting
  - conexión → Unifica muchos algoritmos aparentemente diferentes

- **Regularización**
  - requiere → Función de coste, Término de penalización
  - permite → Control bias-variance, Prevención de overfitting
  - técnicas → L1/Lasso (sparsity), L2/Ridge (shrinkage), ElasticNet (combinación)
  - técnicas DL → Dropout, Batch Normalization, Early Stopping
  - relacionado con → Hiperparámetros (λ), Validación, max_depth, learning_rate (regularizadores implícitos)

- **Desbalanceo de Clases**
  - requiere → Class Weight Balancing, Métricas Apropiadas (F1, Recall > Accuracy)
  - permite → Aprendizaje de clases minoritarias, Ajuste de threshold
  - relacionado con → SMOTE (generación sintética), Matriz de Confusión, Estratificación
  - soluciones específicas → class_weight='balanced' (árboles), threshold adjustment (ROC), SMOTE previo (KNN)
  - aplicaciones → Predicción de bancarrota, fraude, eventos raros

- **Métricas de evaluación supervisada**
  - requiere → Confusion Matrix (clasificación), Valores predichos vs reales (regresión)
  - permite → Selección de modelos, Ajuste de threshold, Evaluación de trade-offs
  - **Clasificación**:
    - Precision/Recall/F1 → Clases desbalanceadas
    - ROC-AUC → Threshold-independent
    - Accuracy → Solo si clases balanceadas
  - **Regresión**:
    - MAE → Robusta a outliers
    - RMSE → Penaliza errores grandes
    - R² → Varianza explicada
  - relacionado con → Precision vs Recall Trade-off (coste de FP vs FN)

- **Explicabilidad (XAI)**
  - requiere → Modelo entrenado, Datos de background (SHAP), Perturbaciones locales (LIME)
  - permite → Interpretación de predicciones individuales, Debugging de modelos, Cumplimiento regulatorio
  - relacionado con → Tensión interpretabilidad-rendimiento
  - **LIME (Local Interpretable Model-agnostic Explanations)**:
    - requiere → Perturbaciones locales, Modelo lineal local
    - permite → Explicaciones locales, Independencia del modelo
  - **SHAP (SHapley Additive exPlanations)**:
    - requiere → Valores de Shapley (teoría de juegos), Coaliciones de features
    - permite → Asignación justa de contribuciones, Garantías de consistencia
    - variantes → Kernel SHAP, Tree SHAP
  - **Contrafactuales (FACE)**:
    - requiere → Distribución de datos, Modelo de clasificación
    - permite → "Qué cambiar para cambiar predicción"
  - **Permutation Importance**:
    - requiere → Validación set, Permutación de features
    - permite → Importancia global de features

- **Weak Supervision**
  - requiere → Matrices de mezcla, Pérdidas propias débiles, Modelado probabilístico
  - permite → Aprendizaje con etiquetas ruidosas/parciales, Reducción de costes de etiquetado
  - relacionado con → Label proportions, Class-conditional noise, Semi-supervised learning
  - reconocimiento → Etiquetas perfectas son la excepción, no la norma

- **Redes Bayesianas**
  - requiere → DAG, Teorema de Bayes, Independencia condicional, Teoría de Grafos
  - permite → Inferencia causal, Simulación de datos, Modelado probabilístico, Descomposición de probabilidades
  - relacionado con → Grafos probabilísticos, Modelos gráficos, Propiedad de Markov
  - ventaja → Inherentemente más interpretables que redes neuronales (XAI)

- **DAG (Directed Acyclic Graph)**
  - requiere → Teoría de grafos, Orden topológico, Ausencia de ciclos
  - permite → Cálculo de probabilidad conjunta, Simulación, Inferencia, Causalidad
  - relacionado con → Matriz de adyacencia, Matriz Laplaciana, Estructuras básicas (Collider/Fork/Chain)
  - limitación crítica → Bidireccionalidad (A→B y B→A) incompatible con DAG

- **Estructuras básicas de grafos**
  - requiere → DAG
  - permite → Determinar independencias, Razonamiento causal
  - **Collider (A→C←B)**:
    - permite → Explain Away, Dependencia inducida al observar C
    - relacionado con → Mirage regressions, Factor Mirage
  - **Fork (A←C→B)**:
    - permite → Independencia condicional dado C
    - relacionado con → Confounders
  - **Chain (A→C→B)**:
    - permite → Independencia condicional dado C
    - relacionado con → Mediación

- **Independencia Condicional**
  - requiere → Probabilidad condicional
  - permite → Simplificación de redes bayesianas, Reducción exponencial de parámetros
  - relacionado con → Markov Blanket, d-separation, Tests de independencia

- **Structure Learning**
  - requiere → Tests de independencia, Datos observacionales, Scoring functions
  - permite → Descubrimiento automático de relaciones causales, Aprendizaje de estructura
  - relacionado con → Algoritmo PC, Hill Climbing, Máxima verosimilitud
  - implementado por → pgmpy, causalnex, Bayes Server
  - desafío → Más difícil que estimar parámetros, sensible a hiperparámetros, no determinista
  - **Algoritmo PC (Constraint-based)**:
    - requiere → Tests de independencia condicional (Pillai), Nivel de significancia, max_cond_vars
    - permite → CPDAG (clase de equivalencia de DAGs)
    - relacionado con → Búsqueda basada en restricciones
  - **Hill Climbing (Score-based)**:
    - requiere → Función de scoring (BIC-d), Operaciones sobre grafos (add/remove/reverse edge)
    - permite → DAG único optimizado
    - relacionado con → Búsqueda heurística, Optimización local

- **Causalidad**
  - requiere → Datos temporales o experimentales, Control de confounders, Direccionalidad
  - permite → Predicción de intervenciones, Modelado causal, Decisiones accionables
  - relacionado con → Correlación (necesaria pero no suficiente)
  - **Granger Causality**:
    - requiere → Series temporales, Lags, Tests estadísticos
    - permite → Causalidad direccional en finanzas (90% de aplicaciones)
    - relacionado con → Directed information, Predicción mejorada
  - **Factor Mirage**:
    - problema → Variables con causalidad invertida mejoran métricas en entrenamiento pero fallan en producción
    - ejemplo → Ganancias no causan ventas, pero correlacionan
    - solución → Validación con conocimiento del dominio, variables lag
  - **Confounders**:
    - problema → Variable oculta causa ambas variables observadas
    - solución → Control estadístico, diseño experimental
  - **Double Machine Learning**:
    - permite → Estimación causal robusta con ML
    - relacionado con → Inferencia causal moderna

- **Teorema de Bayes**
  - requiere → Probabilidad condicional, Probabilidad conjunta
  - permite → Redes Bayesianas, Inferencia probabilística, Clasificación bayesiana, Actualización de creencias
  - relacionado con → Prior, Posterior, Likelihood, Evidencia
  - conexión → Fundamento matemático de inferencia bayesiana

- **Entropía de Shannon**
  - requiere → Teoría de probabilidad
  - permite → Medición de información como "sorpresa", Divergencia KL, Información mutua
  - relacionado con → Cross-entropy (función de pérdida), Árboles de decisión (criterio de split)
  - conexión → Eventos raros son muy informativos

- **Máxima Verosimilitud**
  - requiere → Función de densidad, Datos observados
  - permite → Estimación de parámetros, Aprendizaje de distribuciones
  - relacionado con → Cross-entropy (equivalencia), KDE (Kernel Density Estimation)
  - conexión profunda → "Contar y normalizar" en casos discretos = minimizar cross-entropy en NNs

- **Detección de anomalías**
  - requiere → Reducción dimensional (PCA, Autoencoders), Error de reconstrucción, Entrenamiento solo con datos normales
  - permite → Identificación de patrones anómalos, Sistemas de alerta temprana, Detección de crisis
  - relacionado con → DBSCAN (puntos noise), Outliers, Threshold de error
  - aplicaciones → Fraude, crisis de mercado, fallos de sistemas

- **Transfer Learning**
  - requiere → Modelo preentrenado, Domain Adaptation, Fine-tuning
  - permite → Reutilización de conocimiento, Adaptación a nuevas distribuciones, Reducción de datos necesarios
  - relacionado con → Distribution Shift, Catastrophic Forgetting, Embeddings preentrenados
  - aplicaciones → CLIP, Gemini, modelos de lenguaje

## Flujo de aprendizaje recomendado

### Fase 1: Fundamentos Matemáticos y Estadísticos (Prerequisitos)
1. **Probabilidad Bayesiana y Teorema de Bayes**
   - *Justificación*: Base conceptual para redes bayesianas, clasificación bayesiana e inferencia probabilística. Sin entender probabilidad condicional, el resto del bloque es inaccesible.
   - *Contenido*: Prior, posterior, likelihood, evidencia, actualización de creencias.

2. **Teoría de la Información (Entropía de Shannon)**
   - *Justificación*: Proporciona intuición sobre "información" y "sorpresa". Conecta con divergencia KL, cross-entropy, información mutua y criterios de split en árboles.
   - *Contenido*: Entropía, información mutua, divergencia KL.

3. **Álgebra Lineal Aplicada**
   - *