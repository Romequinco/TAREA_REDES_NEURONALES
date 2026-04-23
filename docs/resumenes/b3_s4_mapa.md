# Mapa de conocimiento — B3 — Redes Neuronales

## Conceptos centrales del bloque

- **Arquitectura MLP (Perceptrón Multicapa)**: Estructura fundamental de redes feedforward con capas densas totalmente conectadas. La arquitectura estándar 784→64→64→10 aparece recurrentemente como base para comparar técnicas. Representa el punto de partida para entender redes neuronales antes de especializaciones (CNN, RNN).

- **ReLU (Rectified Linear Unit)**: Función de activación que revolucionó el deep learning moderno (2012, AlexNet). Central porque resuelve el gradiente desvaneciente (derivada 0 o 1), permite entrenar redes profundas eficientemente, e introduce sparsity natural. Es el estándar en capas ocultas en todos los ejemplos prácticos.

- **Regularización**: Conjunto de técnicas transversales críticas para prevenir sobreajuste: Dropout (modificación estructural durante entrenamiento), Batch Normalization (estabilización y aceleración), L1/L2 (penalización matemática de pesos), Early Stopping (detención automática). Aparece en 8+ documentos como estrategia esencial para generalización.

- **Backpropagation**: Algoritmo central de entrenamiento mediante propagación del error hacia atrás usando la regla de la cadena. Técnica matemática con >200 años de antigüedad aplicada a redes neuronales. Fundamental para ajuste de pesos en todas las arquitecturas.

- **Funciones de pérdida (Loss functions)**: Objetivo que guía el entrenamiento, debe elegirse según el tipo de problema. El material muestra tensión entre teoría (categorical_crossentropy para clasificación) y práctica en notebooks (MSE usado incorrectamente), con múltiples variantes especializadas (focal loss, quantile loss, MAE).

- **Callbacks de Keras**: Mecanismo de control avanzado del entrenamiento que permite automatización de decisiones: Early Stopping (detección de overfitting), ModelCheckpoint (preservación del mejor modelo), ReduceLROnPlateau (ajuste dinámico del learning rate). Representa la transición de entrenamiento manual a automatizado.

- **Visualización e interpretabilidad**: Concepto transversal que abarca activaciones, filtros convolucionales, feature maps, Grad-CAM y heatmaps. Fundamental para debugging, explicabilidad y validación de modelos. Priorizado sobre optimización ciega de métricas (4 de 7 documentos en S4 dedicados exclusivamente a esto).

- **Redes Convolucionales (CNN)**: Arquitectura especializada en procesamiento de imágenes mediante operaciones de convolución y pooling. Permite extracción jerárquica de características espaciales. VGG16/VGG19 usadas consistentemente como arquitecturas de referencia para enseñanza.

- **Transfer Learning**: Técnica avanzada que permite reutilizar conocimiento pre-entrenado mediante congelamiento de capas, equivalente a feature engineering automático. Conecta con eficiencia computacional y aplicaciones con datos limitados.

- **Optimizadores**: SGD con momentum Nesterov es el optimizador predominante en ejemplos, aunque se menciona Adam como alternativa moderna. El learning rate es identificado como el hiperparámetro más importante, con impacto cuantificable en convergencia.

- **No-linealidad**: Principio matemático crítico que justifica la existencia de redes profundas. Sin funciones no lineales, múltiples capas se colapsan en una transformación lineal única, eliminando la ventaja de la profundidad.

- **Particiones del espacio**: Interpretación geométrica fundamental de cómo funcionan las redes neuronales. Explica tanto su capacidad expresiva (dividir el espacio en regiones de decisión) como sus vulnerabilidades (ataques adversariales que explotan fronteras).

- **Bias-Variance Tradeoff**: Principio rector del diseño de arquitecturas que guía decisiones sobre complejidad del modelo y regularización. Detectado mediante curvas de aprendizaje (error_test vs error_train).

- **Normalización de datos**: Práctica crítica con impacto cuantificable (puede hacer entrenamiento 255× más lento si se omite). Incluye normalización de entrada (dividir por 255) y normalización interna (Batch Normalization).

- **Descenso por gradiente**: Algoritmo de optimización fundamental que aparece desde implementaciones manuales (JAX, regresión lineal) hasta entrenamiento de redes complejas. Base matemática de todo el aprendizaje supervisado.

## Grafo de dependencias

- **Fundamentos matemáticos**
  - requiere → Regla de la cadena, Cálculo de variaciones, Álgebra lineal, Teoría de la probabilidad
  - permite → Backpropagation, Descenso por gradiente, Optimización
  - relacionado con → Diferenciación automática (JAX), Gradientes

- **Neurona artificial (perceptrón)**
  - requiere → Suma ponderada, Bias, Inicialización de pesos (Xavier)
  - permite → Construcción de capas, Arquitecturas multicapa
  - relacionado con → Conteo de parámetros, Backpropagation

- **No-linealidad (funciones de activación)**
  - requiere → Neurona artificial básica, Comprensión del problema de linealidad
  - permite → Redes profundas, Particiones del espacio complejas
  - relacionado con → ReLU, Sigmoid, Tanh, Softmax, Gradiente desvaneciente

- **ReLU**
  - requiere → Concepto de no-linealidad
  - permite → Redes profundas (>10 capas), Sparsity natural, Entrenamiento eficiente
  - relacionado con → Gradiente desvaneciente (lo resuelve), Deep learning moderno (2012+)

- **Particiones del espacio**
  - requiere → Funciones no lineales, Múltiples neuronas
  - permite → Clasificación compleja, Regresión no lineal, Comprensión de capacidad expresiva
  - relacionado con → Ataques adversariales, Fronteras de decisión, Árboles de decisión (analogía)

- **Backpropagation**
  - requiere → Funciones derivables, Regla de la cadena, Descenso por gradiente
  - permite → Entrenamiento de redes profundas, Ajuste de pesos
  - relacionado con → Optimizadores (SGD, Adam), Learning rate, Batch size

- **Descenso por gradiente**
  - requiere → Cálculo de gradientes (JAX, backpropagation), Función de pérdida
  - permite → Entrenamiento de redes neuronales, Optimización de parámetros
  - relacionado con → Learning rate, Momentum, Convergencia

- **Funciones de pérdida**
  - requiere → Definición clara del problema (regresión/clasificación), Predicciones del modelo
  - permite → Entrenamiento dirigido, Evaluación de modelos, Cálculo de gradientes
  - relacionado con → MSE, MAE, Cross-entropy (binary/categorical/sparse), Focal loss, Quantile loss

- **Cross-entropy**
  - requiere → Softmax o Sigmoid en salida, Problemas de clasificación
  - permite → Optimización de clasificadores, Interpretación probabilística
  - relacionado con → Binary cross-entropy, Categorical cross-entropy, Sparse categorical cross-entropy

- **Optimizadores**
  - requiere → Funciones de coste derivables, Backpropagation
  - permite → Convergencia del modelo, Ajuste de hiperparámetros
  - relacionado con → SGD, Adam, Learning rate, Momentum Nesterov

- **Arquitectura MLP básica**
  - requiere → Capas Dense, Funciones de activación (ReLU, Softmax), Preprocesamiento de datos
  - permite → Clasificación multiclase, Extracción de características, Regresión
  - relacionado con → One-hot encoding, Reshape (28×28→784), Dataset MNIST

- **Redes Convolucionales (CNN)**
  - requiere → Comprensión de MLPs, Operaciones de convolución, Pooling
  - permite → Procesamiento de imágenes, Extracción de features espaciales jerárquicas
  - relacionado con → Feature maps, Filtros convolucionales, Arquitecturas profundas (VGG, ResNet)

- **Feature Maps (Mapas de características)**
  - requiere → Capas convolucionales, Activaciones intermedias
  - permite → Comprensión de representaciones aprendidas, Visualización de procesamiento
  - relacionado con → Extracción de capas intermedias, Transfer learning

- **Normalización de datos**
  - requiere → Preprocesamiento antes del entrenamiento
  - permite → Convergencia rápida (255× más rápido), Estabilidad numérica
  - relacionado con → Learning rate, Inicialización Xavier, Batch Normalization

- **Regularización**
  - requiere → Arquitectura base definida, Datos de validación, Diagnóstico de overfitting
  - permite → Prevención de overfitting, Mejor generalización, Control de complejidad
  - relacionado con → Dropout, Batch Normalization, L1/L2, Early Stopping, Weight decay

- **Dropout**
  - requiere → Arquitectura definida, Fase de entrenamiento vs inferencia
  - permite → Regularización estructural, Ensemble implícito
  - relacionado con → Sparsity, Overfitting, Tasa de dropout (típicamente 0.2-0.5)

- **Batch Normalization**
  - requiere → Capas de red, Mini-batches
  - permite → Estabilización del entrenamiento, Aceleración de convergencia, Regularización implícita
  - relacionado con → Normalización interna, Covariate shift

- **Sobreajuste (Overfitting)**
  - requiere → Conjunto de entrenamiento y validación separados, Curvas de aprendizaje
  - permite → Detección de generalización pobre, Decisiones de regularización
  - relacionado con → Bias-Variance Tradeoff, Early stopping, Complejidad del modelo

- **Bias-Variance Tradeoff**
  - requiere → Conjuntos de train/validación/test, Métricas de error
  - permite → Decisiones sobre complejidad del modelo, Estrategias de regularización
  - relacionado con → Overfitting, Underfitting, Curvas de aprendizaje

- **Callbacks**
  - requiere → Modelo compilado, Métricas de monitorización, Historial de entrenamiento
  - permite → Control automático del entrenamiento, Guardado de modelos, Ajuste dinámico
  - relacionado con → Early Stopping, ModelCheckpoint, ReduceLROnPlateau

- **Early Stopping**
  - requiere → Datos de validación, Monitorización de métricas, Paciencia (patience)
  - permite → Detención automática ante overfitting, Ahorro computacional
  - relacionado con → Overfitting, Callbacks, Mejor modelo

- **ModelCheckpoint**
  - requiere → Métrica de monitorización, Ruta de guardado
  - permite → Preservación del mejor modelo durante entrenamiento
  - relacionado con → Early Stopping, Callbacks, Validación

- **ReduceLROnPlateau**
  - requiere → Monitorización de métrica, Detección de estancamiento
  - permite → Ajuste dinámico del learning rate, Refinamiento de convergencia
  - relacionado con → Learning rate, Optimización, Callbacks

- **Curvas de aprendizaje**
  - requiere → Historial de entrenamiento, Métricas en train y validación
  - permite → Diagnóstico de problemas, Decisiones de arquitectura, Detección de overfitting/underfitting
  - relacionado con → Monitorización, Early stopping, Bias-Variance Tradeoff

- **Modelos pre-entrenados (VGG16/VGG19)**
  - requiere → Entrenamiento previo en ImageNet, Arquitectura CNN establecida
  - permite → Transfer learning, Extracción de características sin entrenamiento, Visualización de patrones
  - relacionado con → include_top=False, preprocess_input, Fine-tuning, Congelamiento de capas

- **Transfer Learning**
  - requiere → Modelo pre-entrenado, Congelamiento de capas (trainable=False)
  - permite → Reutilización de características, Entrenamiento eficiente con pocos datos, Feature engineering automático
  - relacionado con → Fine-tuning, Extracción de features, PCA sobre datos naturales

- **Visualización de activaciones**
  - requiere → Modelos pre-entrenados, Extracción de capas intermedias, Feature maps
  - permite → Interpretabilidad, Debugging de modelos, Selección de capas para transfer learning
  - relacionado con → Grad-CAM, Heatmaps, Visualización de filtros

- **Visualización de filtros convolucionales**
  - requiere → Capas convolucionales, Extracción de pesos (get_weights)
  - permite → Comprensión de patrones aprendidos, Comparación pre/post-entrenamiento
  - relacionado con → Feature maps, Interpretabilidad, Debugging

- **Grad-CAM (Gradient-weighted Class Activation Mapping)**
  - requiere → GradientTape, Cálculo de gradientes, Activaciones de capas convolucionales
  - permite → Localización visual de regiones relevantes, Explicabilidad de predicciones
  - relacionado con → Heatmaps, Interpretabilidad, Validación de modelos

- **Extracción de pesos (get_weights/set_weights)**
  - requiere → Modelo entrenado o inicializado, API de Keras
  - permite → Inspección de parámetros, Modificación manual, Análisis de aprendizaje
  - relacionado con → Visualización de filtros, Transfer learning, Inicialización

- **Desbalanceo de clases**
  - requiere → Detección de distribución de clases, Análisis de predicciones
  - permite → Identificación de problema de predicción vaga (siempre clase mayoritaria)
  - relacionado con → Ponderación de clases, Focal loss, Mínimo local, Métricas balanceadas

- **Ataques adversariales**
  - requiere → Particiones del espacio, Espacios de alta dimensión, Fronteras de decisión
  - permite → Comprensión de vulnerabilidades, Defensa mediante entrenamiento adversarial
  - relacionado con → Robustez del modelo, Perturbaciones imperceptibles

## Flujo de aprendizaje recomendado

### Fase 1: Fundamentos teóricos y matemáticos

1. **Fundamentos teóricos de Deep Learning** (*intro-deep-2026.md*)
   - Justificación: Establece la base conceptual completa: teoría de la decisión, riesgo empírico, contexto histórico y fundamentos matemáticos necesarios para todo lo demás. Proporciona el "por qué" antes del "cómo".

2. **Neurona artificial y el problema de la linealidad** (Material teórico: neurona artificial, problema crítico de linealidad)
   - Justificación: Base matemática fundamental (suma ponderada + bias) y concepto crítico que justifica por qué necesitamos funciones no lineales. Sin entender esto, no se comprende la necesidad de activaciones ni la ventaja de redes profundas.

3. **Funciones de activación: evolución histórica y ReLU** (Material teórico: funciones de activación)
   - Justificación: Una vez entendida la necesidad de no-linealidad, estudiar las soluciones históricas (signo→sigmoid→ReLU) y por qué ReLU domina actualmente. Incluye contexto del gradiente desvaneciente.

4. **Interpretación geométrica: particiones del espacio** (Material teórico: interpretación geométrica)
   - Justificación: Proporciona intuición visual de cómo funcionan las redes, conectando la matemática con la geometría. Fundamental para entender tanto capacidades como limitaciones (ataques adversariales).

### Fase 2: Implementación básica y optimización

5. **Fundamentos de gradientes y optimización** (*learning-derivadas-en-jax.md*, *learning-ml-my-linear-regresion-by-hand.md*)
   - Justificación: Antes de entrenar redes complejas, es esencial entender cómo funcionan los gradientes y el descenso por gradiente en contextos simples. JAX introduce el cálculo automático de derivadas, y la regresión lineal manual muestra la implementación completa del algoritmo.

6. **Implementación MLP mínima** (*keras-example-mlp-minimo-1.md*)
   - Justificación: Primer contacto práctico con el flujo básico (datos→modelo→compilación→entrenamiento→predicción) usando API Sequential, la más simple. Aplica los conceptos teóricos en código real.

7. **API Funcional de Keras** (*keras-example-mlp-minimo-funcional.md*)
   - Justificación: Introduce paradigma alternativo más explícito y flexible, necesario para arquitecturas complejas posteriores. Muestra la misma red con sintaxis diferente.

8. **Arquitectura y conteo de parámetros** (Material teórico: arquitectura y conteo de parámetros)
   - Justificación: Aspecto práctico de diseño de redes. Requiere entender neuronas, capas y funciones de activación previamente. Fundamental para dimensionar modelos.

9. **Funciones de pérdida según tipo de problema** (Material teórico: funciones de pérdida + activaciones en capa de salida)
   - Justificación: Decisión crítica que debe alinearse con el problema (regresión vs clasificación vs multi-etiqueta). Requiere entender arquitectura completa.

### Fase 3: Control del entrenamiento y regularización

10. **Normalización de datos y técnicas de entrenamiento** (Material teórico: normalización, learning rate, batch size)
    - Justificación: Aspectos prácticos que determinan si el entrenamiento converge eficientemente. Impacto cuantificable (255× diferencia). Prerequisito para entrenamientos exitosos.

11. **Optimización del entrenamiento: Early Stopping** (*keras-example-mlp-callback-early-stopping.md*)
    - Justificación: Primera técnica de regularización automática, introduce el concepto de callbacks y validación. Previene overfitting sin modificar arquitectura.

12. **Gestión de modelos: ModelCheckpoint** (*keras-example-mlp-callback-guardar-mejor-modelo.md*)
    - Justificación: Complementa Early Stopping, enseña a preservar el mejor modelo durante entrenamiento. Práctica esencial en proyectos reales.

13. **Ajuste dinámico: ReduceLROnPlateau** (*keras-example-mlp-callback-leaning-rate-reduceonplateau.md*)
    - Justificación: Técnica avanzada de optimización que ajusta automáticamente el hiperparámetro más crítico (learning rate). Refina la convergencia.

14. **Regularización estructural: Dropout** (*keras-example-mlp-regularization-dropout.md*)
    - Justificación: Técnica de regularización más popular, modifica la arquitectura durante entrenamiento. Introduce concepto de ensemble implícito.

15. **Regularización por normalización: Batch Normalization** (*keras-example-mlp-regularization-batchnormalization-bn.md*)
    - Justificación: Técnica complementaria a Dropout, estabiliza y acelera el entrenamiento. Aborda el problema del covariate shift.

16. **Regularización de pesos: L1/L2** (*keras-example-mlp-regularization-regularizador-capa.md*)
    - Justificación: Enfoque matemático clásico de regularización, completa el panorama de técnicas anti-overfitting. Penalización directa de pesos.

17. **Problemas comunes: sobreajuste y desbalanceo** (Material teórico: overfitting, desbalanceo de clases)
    - Justificación: Problemas prácticos recurrentes que requieren entender todo lo anterior para diagnosticar y solucionar correctamente. Incluye soluciones específicas (focal loss, ponderación).

### Fase 4: Arquitecturas especializadas y visualización

18. **Introducción a CNNs** (*copia-de-keras-example-mlp-minimo.md*)
    - Justificación: Transición natural desde MLPs a arquitecturas especializadas en imágenes, introduce convolución y pooling. Mantiene la misma estructura de entrenamiento.

19. **Arquitectura y visualización básica de redes** (*keras-example-mlp-visualizaciones-simples.md*)
    - Justificación: Una vez comprendida la optimización y arquitecturas, se necesita entender cómo inspeccionar y visualizar modelos. Introduce las APIs de Keras para manipular modelos.

20. **Visualización de filtros convolucionales** (*keras-example-mlp-visualizaciones-filtros-conv.md*, *keras-example-mlp-visualizaciones-filtros.md*)
    - Justificación: Con conocimiento de CNNs, se puede explorar qué aprenden las capas convolucionales. La comparación pre/post-entrenamiento revela el proceso de aprendizaje visualmente.

21. **Extracción de características con modelos pre-entrenados** (*keras-example-mlp-visualizaciones-cnn-activations.md*)
    - Justificación: Introduce el concepto de transfer learning y muestra cómo extraer representaciones intermedias de redes pre-entrenadas (VGG), fundamental para aplicaciones prácticas.

22. **Técnicas avanzadas de interpretabilidad: Grad-CAM** (*keras-example-mlp-visualizaciones-cnn-heatmap.md*)
    - Justificación: Combina conocimientos de gradientes (fase 2) y extracción de activaciones para generar visualizaciones que explican decisiones del modelo. Crítico para validación y explicabilidad.

### Fase 5: Integración y estrategias avanzadas

23. **Entrenamiento práctico y estrategias de diseño** (*training-nn-2026.md*)
    - Justificación: Integra todos los conceptos anteriores en un marco práctico completo. Requiere comprensión de gradientes, arquitecturas, visualización y regularización para aplicar la estrategia de dos extremos y diagnosticar problemas mediante curvas de aprendizaje.

24. **Búsqueda de arquitecturas óptimas** (Material teórico: búsqueda de arquitecturas)
    - Justificación: Aspecto meta-aprendizaje que requiere experiencia con todos los conceptos anteriores. Se menciona explícitamente que "no existe fórmula matemática" y requiere experimentación (AutoKeras como ejemplo).

25. **Técnicas avanzadas: Transfer Learning y ataques adversariales** (Material teórico: transfer learning, ataques adversariales)
    - Justificación: Aplicaciones y vulnerabilidades avanzadas que requieren comprensión profunda de particiones del espacio, representaciones aprendidas y arquitecturas completas. Cierra el ciclo conectando teoría con aplicaciones reales y limitaciones.

## Patrones y observaciones

### 1. Tensión sistemática entre teoría y práctica implementada

**Patrón detectado**: Existe una desconexión recurrente entre las recomendaciones teóricas y los ejemplos prácticos:

- **Funciones de coste**: 7 de 9 notebooks prácticos usan `mean_squared_error` para clasificación multiclase, cuando el material teórico (*intro-deep-2026.md*) enfatiza explícitamente que debe usarse `categorical_crossentropy`. Esta inconsistencia sugiere que los notebooks son ejemplos didácticos simplificados que priorizan la demostración de técnicas específicas (callbacks, regularización) sobre la configuración óptima completa.

- **Estrategias de entrenamiento**: El documento *training-nn-2026.md* presenta estrategias sofisticadas (búsqueda de arquitectura por dos extremos, diagnóstico por curvas de aprendizaje, callbacks avanzados), pero los notebooks prácticos muestran entrenamientos más simples sin aplicar sistemáticamente estas técnicas.

- **Normalización de datos**: Ningún notebook implementa la normalización de píxeles (dividir por 255), a pesar de ser una práctica estándar y tener impacto cuantificable (255× más lento sin ella según el material teórico). El material teórico menciona la importancia de la normalización (GDN, Batch Normalization), pero no se aplica al preprocesamiento básico de entrada.

**Implicación pedagógica**: Esta tensión sugiere que el máster adopta un enfoque de "complejidad incremental", donde los ejemplos iniciales sacrifican optimización por claridad conceptual, esperando que el estudiante integre las mejores prácticas gradualmente.

### 2. Arquitectura estándar como baseline universal

**Patrón detectado**: La arquitectura 784→64→64→10 aparece en 6+ documentos con variaciones mínimas. Esta estandarización:
- Facilita la comparación directa del efecto de diferentes técnicas (Dropout vs Batch Normalization vs L1/L2)
- Permite aislar el impacto de cada modificación manteniendo constante la estructura base
- Limita la exploración de arquitecturas alternativas (más profundas, más anchas, con skip connections)

**Observación complementaria**: VGG16/VGG19 cumplen el mismo rol en CNNs: arquitecturas antiguas (2014) pero conceptualmente simples (bloques repetitivos de Conv+Pool) que sirven como referencia pedagógica. **Laguna detectada**: No se mencionan arquitecturas modernas (ResNet, EfficientNet, Vision Transformers) que dominan el estado del arte actual.

### 3. Énfasis en interpretabilidad sobre rendimiento puro

**Patrón detectado**: El bloque dedica 4 de 7 documentos de S4 exclusivamente a visualización e interpretabilidad (activaciones, filtros, heatmaps, arquitecturas). Esto contrasta con un enfoque puramente orientado a métricas, sugiriendo que el máster prioriza:
- Comprensión profunda del funcionamiento interno de las redes
- Capacidad de debugging y diagnóstico
- Explicabilidad de decisiones (crítico para aplicaciones reguladas)
- Validación cualitativa complementaria a métricas cuantitativas

**Conexión con otros patrones**: Esta prioridad se alinea con la progresión pedagógica de "simple a complejo" y con la tensión teoría-práctica (primero entender, luego optimizar).

### 4. Progresión pedagógica clara: de implementación manual a frameworks

**Patrón detectado**: Se observa una transición pedagógica deliberada:
1. **Implementaciones desde cero** (gradientes en JAX, regresión lineal manual) para construir intuición matemática
2. **APIs simples** (Sequential) para primeros contactos prácticos
3. **APIs flexibles** (Functional) para arquitecturas complejas
4. **Modelos pre-entrenados** (VGG) para aplicaciones reales
5. **Automatización** (AutoKeras) para búsqueda de arquitecturas

Esta progresión refuerza la comprensión conceptual antes de la aplicación práctica, evitando el uso de "cajas negras" sin fundamento.

### 5. Evolución histórica como narrativa pedagógica

**Patrón detectado**: El contenido usa consistentemente la evolución histórica para explicar conceptos:
- **Funciones de activación**: función signo (1940s) → sigmoid/tanh (1980s) → ReLU (2012)
- **Causalidad**: Granger con tiempo → Holler sin tiempo
- **Backpropagation**: "técnica con >200 años de antigüedad matemática" aplicada a redes neuronales

Esta narrativa no es meramente histórica sino que explica *por qué* cada solución surgió (gradiente desvaneciente) y *por qué* fue superada. Proporciona contexto que ayuda a entender las limitaciones de cada técnica y anticipa futuras evoluciones.

### 6. Dualidad entre capacidad expresiva y vulnerabilidad

**Patrón detectado**: Las particiones del espacio aparecen como concepto unificador que explica tanto las capacidades (clasificación compleja mediante regiones) como las vulnerabilidades (ataques adversariales explotando fronteras). Esta dualidad sugiere que las fortalezas y debilidades de las redes neuronales emergen del mismo mecanismo fundamental.

**Solución propuesta**: Entrenar con ejemplos adversariales refleja un patrón más amplio del campo: los problemas se resuelven mediante más datos/entrenamiento, no mediante cambios arquitectónicos fundamentales. Esto conecta con la filosofía del deep learning de "escalar" sobre "diseñar".

### 7. Conexiones interdisciplinarias recurrentes

**Patrón detectado**: El material establece conexiones explícitas con múltiples disciplinas:
- **Neurociencia**: normalización local, sparsity cerebral, neuronas biológicas
- **Finanzas**: quantile loss para capturar incertidumbre, feature engineering con promedios móviles
- **Procesamiento de señales**: transformada de Fourier, espectrogramas
- **Estadística**: PCA extrayendo funciones de Fourier

**Observación notable**: PCA sobre datos naturales "revela representaciones óptimas implementadas en el cerebro", sugiriendo principios universales de representación eficiente que trascienden disciplinas. Esto refuerza la idea de que el deep learning redescubre principios fundamentales de procesamiento de información.

### 8. Tensión entre teoría prescriptiva y heurísticas experimentales

**Patrón detectado**: Mientras hay fundamentos matemáticos sólidos (backpropagation, regla de la cadena, optimización), muchas decisiones críticas carecen de teoría prescriptiva:
- "No existe fórmula matemática para determinar el número óptimo de neuronas/capas"
- AutoKeras funciona "como un becario probando cosas"
- Búsqueda de arquitecturas requiere experimentación sistemática

Esto sugiere que el campo combina rigor matemático en los mecanismos con heurísticas experimentales en el diseño. El máster prepara para ambos aspectos: comprensión teórica profunda + experimentación práctica disciplinada.

### 9. Laguna detectada: Falta de notebook integrador completo

**Observación crítica**: El documento *training-nn-2026.md* presenta un flujo de entrenamiento profesional completo (estrategia de dos extremos, diagnóstico por curvas, callbacks coordinados), pero no existe un notebook que implemente este flujo de principio a fin. Los notebooks existentes demuestran técnicas aisladas pero no su integración sistemática.

**Impacto**: Los estudiantes pueden dominar técnicas individuales sin saber cómo orquestarlas en un proyecto real. Esta laguna podría llenarse con un notebook "capstone" que aplique todas las mejores prácticas en un problema completo.

### 10. Dataset MNIST como caso de estudio universal

**Patrón detectado**: MNIST aparece en 8+ documentos como problema estándar. Esto tiene ventajas (comparabilidad, rapidez