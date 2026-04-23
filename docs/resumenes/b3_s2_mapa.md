# Mapa de conocimiento — B3 — Tipos de Aprendizaje

## Conceptos centrales del bloque

- **Modelo como compresión de datos**: Principio filosófico fundamental que establece que ningún modelo puede aprender más de lo que contienen los datos originales. Atraviesa todo el bloque justificando la importancia crítica del preprocesamiento, la calidad de datos y las decisiones sobre transformaciones.

- **Gradient Descent y optimización**: Algoritmo de optimización fundamental que sustenta todo el aprendizaje automático. Aparece desde implementación matemática básica hasta aplicaciones en redes neuronales profundas, siendo el mecanismo central de ajuste de parámetros.

- **Preprocesamiento de datos financieros**: Concepto transversal más importante del bloque. Incluye normalización/estandarización, tratamiento de outliers, imputación y técnicas especializadas. Crítico porque toda transformación tiene coste informativo y determina qué puede aprender el modelo.

- **Datos NO i.i.d. en finanzas**: Principio específico del dominio financiero que invalida asunciones clásicas de ML. Fundamenta la necesidad de técnicas especializadas (dollar bars, diferenciación fraccional, validación temporal) y previene errores críticos de data leakage.

- **Regularización para generalización**: Concepto unificador que aparece en múltiples formas (L1/L2, Dropout, BatchNorm, Bagging, Fair Learning). Representa el principio de "dificultar el entrenamiento" para mejorar generalización y controlar el tradeoff bias-variance.

- **Ensemble Methods**: Paradigma fundamental que combina múltiples modelos para mejorar predicciones. Implementado en múltiples variantes (Bagging, Random Forest, Stacking) y representa una de las técnicas más efectivas para reducir varianza.

- **Técnicas de López de Prado**: Conjunto especializado de métodos estado del arte en ML financiero: barras basadas en actividad (Dollar/Volume/Tick), diferenciación fraccional (FFD), triple barrier labeling, limpieza de matrices de covarianza mediante Random Matrix Theory. Integra múltiples conceptos del bloque.

- **Reducción dimensional**: Familia de técnicas (PCA, ICA, Wavelets, Autoencoders, t-SNE) que aparece tanto en preprocesamiento como en visualización. Central para manejar alta dimensionalidad y extraer características relevantes.

- **Partición temporal y prevención de data leakage**: Concepto crítico para validación en series temporales. Incluye técnicas de purging, embargo y walk-forward validation. Probablemente el error más peligroso en ML financiero.

- **Función de coste como definición de objetivo**: Establece que el modelo optimiza exactamente lo que se define en la función de coste. Central para alinear modelos con objetivos de negocio (Sharpe, drawdown) y entender limitaciones.

- **Convolutional Neural Networks (CNN)**: Arquitectura fundamental para visión por computador, implementada completamente con técnicas modernas de regularización. Demuestra aplicación práctica de gradient descent, normalización y regularización.

- **Validación Out-of-Bag (OOB)**: Técnica específica de ensembles que permite evaluar modelos eficientemente sin sacrificar datos de entrenamiento, usando muestras no seleccionadas en bootstrap.

## Grafo de dependencias

- **Fundamentos matemáticos de optimización**
  - requiere → Derivadas/Gradientes, Álgebra lineal (matrices, inversión), Estadística (media, varianza, covarianza)
  - permite → Gradient Descent, Mínimos cuadrados, Backpropagation
  - relacionado con → Diferenciación automática (JAX), Funciones de pérdida

- **Gradient Descent**
  - requiere → Derivadas/Gradientes, Función de pérdida, Learning rate, Datos normalizados
  - permite → Entrenamiento de modelos ML/DL, Optimización de parámetros, Convergencia
  - relacionado con → Backpropagation, Diferenciación automática (JAX), Inicialización de pesos

- **Diferenciación automática (JAX)**
  - requiere → Conocimiento de derivadas, Funciones diferenciables
  - permite → Gradient Descent eficiente, Implementación de modelos desde cero
  - relacionado con → Optimización numérica, Cálculo diferencial, Frameworks modernos

- **Visualización obligatoria de datos**
  - requiere → Datos crudos, Herramientas de plotting
  - permite → Detección de outliers, Comprensión de distribuciones, Decisiones de preprocesamiento informadas
  - relacionado con → Análisis exploratorio, Detección de problemas, Comprensión del dominio

- **Detección y tratamiento de outliers**
  - requiere → Visualización, Comprensión de distribuciones, Contexto del dominio
  - permite → Decisiones de escalado apropiadas, Modelos más robustos
  - relacionado con → Robust Scaler, Z-score, Distancia de Cook, IQR

- **Tratamiento de datos faltantes**
  - requiere → Comprensión del mecanismo de falta (MCAR, MAR, MNAR)
  - permite → Datasets completos para modelado
  - relacionado con → Imputación (media, mediana, KNN), Eliminación, Prohibición de backward fill en series temporales

- **Normalización/Estandarización**
  - requiere → Datos con outliers tratados, Estadísticas (media, desviación, min/max)
  - permite → Entrenamiento estable de modelos, Convergencia más rápida, Inicialización correcta de redes neuronales
  - relacionado con → Preprocesamiento, Feature engineering, Preservación de información como features adicionales

- **Escalado con preservación de información**
  - requiere → Comprensión de coste informativo, Normalización básica
  - permite → Que el modelo decida relevancia de escala original
  - relacionado con → Pasar media/desviación como features separadas, Filosofía de preservación de información

- **Datos NO i.i.d. en finanzas**
  - requiere → Comprensión de autocorrelación, Microestructura de mercados
  - permite → Justificación de técnicas especializadas, Prevención de errores conceptuales
  - relacionado con → Series temporales, Estacionariedad, Eventos raros con alta información

- **Partición temporal y prevención de data leakage**
  - requiere → Comprensión de datos NO i.i.d., Autocorrelación
  - permite → Validación sin look-ahead bias, Estimación realista de performance
  - relacionado con → Purging, Embargo, K-Fold temporal, Walk-forward validation, Prohibición de usar estadísticas futuras

- **Barras basadas en actividad (Dollar/Volume/Tick)**
  - requiere → Datos tick-by-tick, Comprensión de microestructura, Datos NO i.i.d.
  - permite → Series más estacionarias, Mejor captura de información de mercado, Reducción de ruido
  - relacionado con → Muestreo alternativo vs time bars, Sincronización por actividad económica

- **Diferenciación Fraccional (FFD)**
  - requiere → Series temporales, Test de estacionariedad (ADF), Comprensión de memoria vs estacionariedad
  - permite → Balance estacionariedad-memoria, Features para ML financiero sin destruir información histórica
  - relacionado con → Preprocesamiento financiero, Parámetro d continuo, Extensión de diferenciación clásica

- **Triple Barrier Labeling**
  - requiere → Series temporales financieras, Definición de take profit/stop loss/timeout
  - permite → Etiquetado de eventos con consideración temporal, Gestión de riesgo integrada
  - relacionado con → Clasificación financiera, Meta-labeling, Purga y embargo

- **Random Matrix Theory (RMT) y limpieza de covarianza**
  - requiere → Matrices de covarianza, Ratio q=T/N, PCA (autovalores/autovectores)
  - permite → Denoising de covarianza, Optimización de carteras robusta, Separación señal-ruido
  - relacionado con → Distribución de Marchenko-Pastur, Física estadística, Detoned covariance

- **Selección de características**
  - requiere → Datos preprocesados, Métricas estadísticas, Comprensión del dominio
  - permite → Reducción de dimensionalidad, Modelos más interpretables, Reducción de overfitting
  - relacionado con → Filter methods, Wrapper methods, LASSO (L1), Información mutua, Correlación

- **PCA (Principal Component Analysis)**
  - requiere → Matriz de covarianza, Autovalores/autovectores, Centrado de datos, Datos escalados
  - permite → Reducción de dimensionalidad, Compresión, Decorrelación, Visualización
  - relacionado con → Implementación manual vs sklearn, Varianza explicada, Limpieza de covarianza

- **Wavelets**
  - requiere → Concepto de frecuencia, Descomposición multinivel
  - permite → Análisis multiescala, Denoising, Compresión adaptativa, Localización tiempo-frecuencia
  - relacionado con → Alternativa a PCA, Procesamiento de señales, Transformadas adaptativas

- **Reducción dimensional avanzada**
  - requiere → PCA como base, Comprensión de no-linealidad
  - permite → Visualización de alta dimensionalidad, Extracción de características complejas
  - relacionado con → ICA, Autoencoders, t-SNE, UMAP

- **Regresión Lineal (fundamento)**
  - requiere → Álgebra lineal (matrices, inversión), Estadística (media, covarianza)
  - permite → Comprensión de mínimos cuadrados, Modelos lineales generalizados, Base para modelos complejos
  - relacionado con → MSE como métrica, Train-test split, Implementación manual vs sklearn

- **Árboles de Decisión**
  - requiere → Criterios de split (MSE para regresión, Gini/Entropy para clasificación), Concepto de profundidad
  - permite → Random Forest, Bagging, Boosting, Modelos interpretables
  - relacionado con → Overfitting (control mediante max_depth), Modelo base para ensembles

- **Bagging (Bootstrap Aggregating)**
  - requiere → Bootstrap sampling, Modelo base (ej. DecisionTree), Agregación
  - permite → Random Forest, Reducción de varianza, Predicciones más robustas
  - relacionado con → Validación OOB, Diversidad de datos, Ensemble learning

- **Random Forest**
  - requiere → Bagging, Árboles de decisión, Aleatoriedad en features (max_features)
  - permite → Predicciones robustas, Importancia de características, Reducción de varianza
  - relacionado con → OOB error, Ensemble averaging, Implementación manual vs sklearn

- **Stacking (Agregación de Modelos)**
  - requiere → Múltiples modelos base diversos, KFold para predicciones out-of-fold, Estrategia de agregación
  - permite → Combinación de modelos diversos, Meta-learners (nivel 2), Mejora sobre modelos individuales
  - relacionado con → Media ponderada, Blending, AutoML, Ensemble heterogéneo

- **Convolutional Neural Networks (CNN)**
  - requiere → Datos normalizados, Gradient descent, Backpropagation, Diferenciación automática
  - permite → Clasificación de imágenes, Extracción de características espaciales, Reconocimiento de patrones
  - relacionado con → Deep Learning, Computer Vision, Arquitecturas modernas (CIFAR-100)

- **Regularización (concepto unificador)**
  - requiere → Función de pérdida, Arquitectura de modelo, Comprensión del bias-variance tradeoff
  - permite → Prevención de overfitting, Mejor generalización, Inyección de conocimiento del dominio
  - relacionado con → L1 (LASSO), L2 (Ridge), Dropout, BatchNormalization, Bagging, Fair Learning

- **Función de coste y alineación con objetivos**
  - requiere → Definición clara de objetivo de negocio, Comprensión de métricas
  - permite → Optimización alineada con objetivos, Evaluación apropiada, Modelos útiles en producción
  - relacionado con → MSE, MAE, Huber Loss, Cross-entropy, Métricas financieras (Sharpe, drawdown)

- **Tipos de aprendizaje (taxonomía)**
  - requiere → Disponibilidad de etiquetas, Naturaleza del problema, Recursos computacionales
  - permite → Selección de algoritmo apropiado, Estrategia de entrenamiento, Decisiones arquitecturales
  - relacionado con → Supervisado, No supervisado, Semi-supervisado, Online, Active, Transfer, Fair Learning

- **Whisper (Speech-to-Text)**
  - requiere → Transformers, Pipeline de Hugging Face, GPU (CUDA), Modelos pre-entrenados
  - permite → Transcripción automática, ASR, Aplicación de transfer learning
  - relacionado con → Modelos pre-entrenados, Transfer learning, Aplicación práctica de DL

## Flujo de aprendizaje recomendado

### Fase 1: Fundamentos matemáticos y filosóficos (Conceptual)

1. **Principios fundamentales: Modelo como compresión**
   - Documentos: Introducción de transcripción tipos de aprendizaje
   - Justificación: Establece el marco conceptual que fundamenta todas las decisiones posteriores. Previene aplicación mecánica de técnicas sin comprensión del coste informativo.

2. **Teoría de la información de Shannon**
   - Documentos: Conceptos teóricos de transcripción
   - Justificación: Proporciona base matemática para entender por qué eventos raros contienen más información y justifica decisiones sobre preprocesamiento.

3. **Derivadas y Gradient Descent (fundamentos matemáticos)**
   - Documentos: `LEARNING_Gradient_descent.ipynb`
   - Justificación: Fundamento matemático de todo el aprendizaje automático. Comprender cómo los modelos optimizan parámetros es esencial antes de aplicar técnicas avanzadas.

4. **Diferenciación automática con JAX**
   - Documentos: `LEARNING_Derivadas_en_JAX.ipynb`
   - Justificación: Implementación práctica de gradient descent. Permite entender cómo frameworks modernos calculan derivadas automáticamente y facilita implementaciones desde cero.

### Fase 2: Preprocesamiento y calidad de datos (Práctico fundamental)

5. **Visualización obligatoria de datos**
   - Documentos: Sección de visualización en transcripción
   - Justificación: Primer paso práctico antes de cualquier transformación. Detecta problemas, outliers, desbalanceos que guiarán todas las decisiones posteriores.

6. **Detección y tratamiento de datos faltantes**
   - Documentos: Tratamiento de datos faltantes en transcripción
   - Justificación: Problema frecuente que debe resolverse antes de otras transformaciones. Diferentes métodos (imputación, eliminación) tienen distintos costes informativos.

7. **Detección y tratamiento de outliers**
   - Documentos: Detección y tratamiento de outliers en transcripción
   - Justificación: Afecta decisiones de escalado y modelado. En finanzas, outliers pueden ser información valiosa (crisis) no ruido. Debe preceder al escalado.

8. **Normalización/Estandarización y escalado**
   - Documentos: `DATA_Preproc_y_seleccion_Feats.ipynb` (Parte 1), sección de escalado en transcripción
   - Justificación: Técnicas universales aplicables a cualquier problema de ML. Necesarias para convergencia de algoritmos. Incluye técnica avanzada de preservación de información como features adicionales.

9. **Selección de características (métodos filter y wrapper)**
   - Documentos: `DATA_Preproc_y_seleccion_Feats.ipynb` (Parte 2)
   - Justificación: Métodos para reducir dimensionalidad y mejorar interpretabilidad. Filter methods son independientes del modelo, wrapper methods requieren modelo definido.

10. **Carga de datos desde repositorios remotos**
    - Documentos: `DATA_Huggingface.ipynb`, `DATA_Load_github_and_HF_DDBB_example.ipynb`
    - Justificación: Habilidades prácticas para acceder a datasets. Puede estudiarse en paralelo con otros temas pero es necesario para ejercicios prácticos.

### Fase 3: Especialización financiera (Dominio específico)

11. **Datos NO i.i.d. en finanzas**
    - Documentos: Contexto financiero en transcripción
    - Justificación: Concepto crítico que invalida asunciones clásicas de ML. Fundamenta la necesidad de todas las técnicas especializadas posteriores.

12. **Partición temporal y prevención de data leakage**
    - Documentos: Partición temporal en transcripción, parte 3 de preprocesado financiero
    - Justificación: Crítico para validación correcta en series temporales. Debe establecerse antes de cualquier entrenamiento. Incluye purging, embargo, walk-forward.

13. **Barras basadas en actividad (Dollar/Volume/Tick Bars)**
    - Documentos: `B3_T1_Preprocesado_Datos_Financieros` - Técnica 1, ejercicio 1 en transcripción
    - Justificación: Primera técnica específica de finanzas. Alternativa superior a time bars que respeta la naturaleza NO i.i.d. de datos financieros.

14. **Diferenciación Fraccional (FFD)**
    - Documentos: `B3_T1_Preprocesado_Datos_Financieros` - Técnica 2, ejercicio 2 en transcripción
    - Justificación: Técnica avanzada que resuelve el dilema estacionariedad-memoria. Requiere comprender estacionariedad y series temporales. Permite aplicar ML sin destruir información histórica.

15. **Triple Barrier Labeling**
    - Documentos: Parte 3 de transcripción preprocesado financiero
    - Justificación: Método avanzado de etiquetado para clasificación financiera. Integra consideración temporal y gestión de riesgo. Requiere comprensión de partición temporal.

### Fase 4: Reducción dimensional y extracción de características (Técnicas avanzadas)

16. **PCA (implementación manual)**
    - Documentos: `models-pca.md` (sección 1.1)
    - Justificación: Fundamento matemático de reducción de dimensionalidad. La implementación paso a paso refuerza álgebra lineal y comprensión de autovalores/autovectores.

17. **PCA con sklearn y aplicaciones**
    - Documentos: `models-pca.md` (secciones 1.2 y aplicadas)
    - Justificación: Uso práctico tras comprender la teoría. Aplicaciones en compresión, denoising y visualización consolidan el concepto.

18. **Random Matrix Theory y limpieza de covarianza**
    - Documentos: `B3_T1_Preprocesado_Datos_Financieros` - Técnica 3, ejercicio 3 en transcripción
    - Justificación: Técnica matemáticamente sofisticada que combina PCA con teoría de matrices aleatorias. Crítica para optimización de carteras robusta. Requiere dominio previo de PCA.

19. **Wavelets como alternativa a PCA**
    - Documentos: `models-pca.md` (secciones wavelets)
    - Justificación: Alternativa a PCA con localización tiempo-frecuencia. Se entiende mejor por contraste tras dominar PCA. Aplicaciones en análisis multiescala.

20. **Técnicas avanzadas de reducción dimensional**
    - Documentos: Sección de reducción dimensional en transcripción
    - Justificación: ICA, autoencoders, t-SNE, UMAP como extensiones no lineales. Requieren comprensión sólida de PCA como base.

### Fase 5: Modelos supervisados y ensembles (Modelado)

21. **Regresión Lineal (fundamentos e implementación manual)**
    - Documentos: `models-ml-my-linear-regresion-by-hand.md`, `models-ml-my-linear-regresion.md`
    - Justificación: Establece la base matemática (mínimos cuadrados, MSE) necesaria para comprender modelos más complejos. La implementación manual refuerza conceptos.

22. **Función de coste y alineación con objetivos**
    - Documentos: Sección de optimización y funciones de coste en transcripción
    - Justificación: Define qué optimiza el modelo. Debe alinearse con métricas de negocio (Sharpe, drawdown). Crítico antes de entrenar cualquier modelo.

23. **Árboles de Decisión (baseline)**
    - Documentos: `models-ensembles-rf-regresion.md` (sección baseline)
    - Justificación: Modelo base simple que permite entender el concepto de predictor antes de ensembles. Introduce criterios de split y control de overfitting.

24. **Bagging y Bootstrap**
    - Documentos: `models-ensembles-rf-regresion.md`, `MODELS_ENSEMBLES_bagging_regresion.ipynb`
    - Justificación: Introduce el concepto de ensemble mediante muestreo. Demuestra cómo la diversidad de datos reduce varianza. Prerequisito directo de Random Forest.

25. **Random Forest**
    - Documentos: `models-ensembles-rf-regresion.md`
    - Justificación: Integra bagging con aleatoriedad en features. Modelo completo que demuestra reducción de varianza mediante ensemble. Incluye importancia de características.

26. **Validación Out-of-Bag (OOB)**
    - Documentos: `models-ensembles-rf-regresion.md`
    - Justificación: Técnica específica de ensembles que optimiza uso de datos. Se entiende mejor tras conocer bagging. Permite validación sin datos adicionales.

27. **Stacking y agregación de modelos diversos**
    - Documentos: `models-ensembles-basic-aggregating.md`
    - Justificación: Generaliza el concepto de ensemble a múltiples tipos de modelos. Introduce estrategias de agregación (media, ponderada, meta-learners).

### Fase 6: Deep Learning y aplicaciones avanzadas (Especialización)

28. **Regularización como concepto unificador**
    - Documentos: Sección de regularización en transcripción, técnicas en CNN
    - Justificación: Técnicas transversales para generalización. Aparece en múltiples formas (L1/L2, Dropout, BatchNorm, Bagging, Fair Learning). Crítico antes de entrenar redes profundas.

29. **Convolutional Neural Networks (CNN)**
    - Documentos: `MODELS_ANN_cifar100.ipynb`
    - Justificación: Aplicación completa de gradient descent, normalización y regularización en arquitectura moderna. Integra múltiples conceptos del bloque en un caso práctico completo.

30. **Transfer Learning y modelos pre-entrenados (Whisper)**
    - Documentos: `models-s2t-whisper-hf.md`
    - Justificación: Aplicación práctica de modelos pre-entrenados. Demuestra transfer learning y uso de frameworks modernos (Hugging Face). Independiente del resto pero consolida conceptos de DL.

### Fase 7: Consideraciones éticas y avanzadas (Integración)

31. **Taxonomía completa de tipos de aprendizaje**
    - Documentos: Tipos de datos y aprendizaje en transcripción
    - Justificación: Visión completa del espacio de posibilidades (supervisado, no supervisado, semi-supervisado, online, active, transfer, fair). Permite seleccionar estrategia apropiada según disponibilidad de datos.

32. **Fair Learning y consideraciones éticas**
    - Documentos: Fair Learning en transcripción
    - Justificación: Regularización especializada para mitigar sesgos. Tema avanzado que integra conceptos de regularización y objetivos de negocio. Crítico para aplicaciones en producción.

## Patrones y observaciones

### 1. Tensión fundamental: Preprocesamiento vs Preservación de Información

**Patrón recurrente**: Existe un tradeoff constante entre transformar datos para facilitar el aprendizaje y preservar información valiosa. Aparece en:
- **Normalización**: Elimina escala pero el material propone pasar media/desviación como features adicionales
- **Diferenciación**: Elimina tendencia pero FFD preserva memoria parcial mediante parámetro continuo d
- **Tratamiento de outliers**: En finanzas, eventos extremos contienen información valiosa (crisis) no ruido
- **Autocorrelación**: Datos financieros NO son i.i.d., eliminar autocorrelación destruye información temporal

**Observación crítica**: El material propone consistentemente soluciones que **preservan información** (ej: "dejar que el modelo decida" qué información es relevante) en lugar de eliminarla. Esto refleja una filosofía de preprocesamiento conservadora que reconoce el coste informativo de cada transformación.

**Implicación práctica**: Antes de aplicar cualquier transformación, debe evaluarse si la información eliminada es ruido o señal en el contexto específico del problema.

### 2. Dualidad teoría-práctica en optimización

**Patrón pedagógico**: El bloque presenta conceptos primero desde fundamentos matemáticos y luego desde implementación práctica:
- **Gradient descent**: Derivadas matemáticas → Implementación en JAX
- **Regresión lineal**: Mínimos cuadrados manual → sklearn
- **PCA**: Cálculo de autovectores manual → sklearn.decomposition.PCA
- **Random Forest**: Bagging desde cero → sklearn.ensemble

**Observación**: Esta estructura refuerza la comprensión conceptual antes de usar abstracciones. Permite entender qué hacen las librerías "por debajo" y detectar problemas cuando fallan.

**Implicación práctica**: La implementación manual no es solo ejercicio académico, sino herramienta de debugging y comprensión profunda necesaria para aplicaciones avanzadas.

### 3. Especialización financiera como extensión, no reemplazo

**Patrón arquitectural**: Las técnicas de López de Prado **extienden** técnicas clásicas reconociendo características específicas de datos financieros:
- **Dollar bars** extienden time bars considerando impacto económico
- **Diferenciación fraccional** extiende diferenciación clásica con parámetro continuo d
- **Triple barrier labeling** extiende etiquetado binario con consideración temporal y gestión de riesgo
- **Limpieza de covarianza** extiende PCA con Random Matrix Theory

**Observación**: Esto sugiere que dominar técnicas clásicas es **prerequisito** para aplicar técnicas financieras avanzadas, no un camino alternativo.

**Implicación práctica**: No se puede "saltar" directamente a técnicas financieras avanzadas sin comprender los fundamentos que extienden. La secuencia de aprendizaje debe ser: clásico → comprensión de limitaciones en finanzas → extensión especializada.

### 4. Validación temporal como tema transversal crítico

**Patrón de riesgo**: La prevención de data leakage mediante partición temporal correcta aparece en **múltiples contextos**:
- **Purging**: Eliminar datos intermedios entre train/test
- **Embargo**: Buffer temporal adicional
- **Prohibición de backward fill** en imputación
- **Cuidado con normalización** usando estadísticas de todo el dataset
- **K-Fold temporal** vs K-Fold aleatorio
- **Walk-forward validation** como gold standard

**Observación crítica**: Este es probablemente el **error más común y peligroso** en ML financiero. El material lo enfatiza repetidamente en diferentes contextos, sugiriendo que es una laguna frecuente en implementaciones prácticas.

**Implicación práctica**: Cualquier uso de información futura (incluso indirecto, como normalizar con estadísticas del dataset completo) invalida completamente el modelo. Debe ser el primer check en cualquier pipeline de ML financiero.

### 5. Regularización como concepto unificador

**Patrón conceptual**: La regularización aparece en **múltiples formas y contextos**, sugiriendo que es un principio más general que técnicas específicas:
- **LASSO (L1)** para selección de características (parsimonia)
- **Ridge (L2)** para estabilidad de parámetros
- **Dropout y Batch Normalization** en redes neuronales
- **Bagging** como regularización mediante diversidad de datos
- **Fair Learning** como regularización ética
- **"Dificultar el entrenamiento"** como filosofía general

**Observación**: El material presenta regularización no solo como técnica anti-sobreajuste, sino como **forma de inyectar conocimiento del dominio** (ej: parsimonia, fairness, estabilidad) en el proceso de optimización.

**Implicación práctica**: La elección del tipo de regularización debe estar alineada con los objetivos del problema (interpretabilidad → L1, estabilidad → L2, fairness → Fair Learning).

### 6. Ensemble como paradigma dominante

**Patrón de efectividad**: Tres documentos completos dedicados a ensembles con diferentes enfoques:
- **Bagging** (Random Forest): Diversidad mediante muestreo de datos
- **Stacking**: Diversidad mediante modelos heterogéneos
- **Implícito en limpieza de covarianza**: Combinación de autovectores

**Observación**: Los ensembles representan una de las técnicas más efectivas para mejorar rendimiento en ML. El bloque dedica recursos significativos a múltiples variantes.

**Implicación práctica**: En aplicaciones reales, un ensemble bien diseñado frecuentemente supera a un modelo individual complejo. La diversidad (de datos, modelos o características) es clave para reducir varianza.

### 7. Posible laguna: Integración de pipeline completo

**Observación crítica**: Los documentos tratan cada técnica de forma **relativamente aislada**. No se detecta un documento que integre explícitamente:
- Cómo usar PCA como preprocesado ANTES de ensembles
- Cómo combinar barras alternativas → FFD → limpieza de covarianza → modelo → validación temporal
- Pipeline completo: preprocesado financiero → reducción dimensionalidad → ensemble → evaluación con métricas de negocio

**Implicación**: Los estudiantes deben **inferir** cómo conectar las piezas. Un caso de uso integrado (ej: predicción de retorn