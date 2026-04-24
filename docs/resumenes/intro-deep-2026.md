# Introducción a Deep Learning

## Resumen
Curso introductorio sobre Deep Learning impartido por Valero Laparra que cubre los fundamentos del aprendizaje profundo: desde la teoría de la decisión y minimización de riesgo, pasando por arquitecturas de redes neuronales inspiradas en el córtex visual, funciones de coste para diferentes tareas, algoritmos de optimización (SGD, Adam), técnicas de regularización, hasta el contexto histórico y aplicaciones avanzadas como normalización divisiva (GDN) en finanzas.

## Ideas clave
- El aprendizaje se basa en minimizar el riesgo empírico: combinación de función de coste, datos y modelo
- Las redes neuronales se inspiran en el córtex visual (trabajos de Ramón y Cajal, Hubel & Wiesel)
- La elección de la función de coste define el objetivo del modelo y debe adaptarse al problema específico
- El descenso por gradiente con backpropagation es la técnica fundamental de entrenamiento
- La regularización (dropout, batch normalization, early stopping) previene el sobreajuste
- El learning rate es el hiperparámetro más importante del entrenamiento
- La revolución del deep learning (~2012) se debe a mayor capacidad computacional (GPUs) y datos

## Conceptos detectados

- **Riesgo de Bayes**: Minimización teórica de Probabilidad × Pérdidas para tomar decisiones óptimas
- **Riesgo Empírico**: Aproximación práctica del riesgo de Bayes usando datos disponibles {x,y}, función de coste L y modelo f
- **Backpropagation**: Algoritmo basado en la regla de la cadena para calcular gradientes y entrenar redes neuronales multicapa
- **GDN (Generalized Divisive Normalization)**: Función de activación no lineal inspirada en normalización por contraste del córtex visual
- **Mini-batch**: Técnica de entrenamiento que divide los datos en bloques para optimizar el balance entre velocidad y convergencia
- **Dropout**: Técnica de regularización que desactiva aleatoriamente neuronas durante el entrenamiento para prevenir sobreajuste
- **Batch Normalization**: Normalización de activaciones por lotes para estabilizar y acelerar el entrenamiento
- **Early Stopping**: Técnica de regularización que detiene el entrenamiento cuando la validación deja de mejorar
- **Cross-Entropy**: Función de coste estándar para problemas de clasificación (binaria, categórica, sparse)
- **Adam**: Algoritmo de optimización adaptativo que combina momentum y learning rate adaptativo
- **ROC (Receiver Operating Characteristic)**: Curva para evaluar clasificadores binarios considerando desbalanceo de clases

## Estructura del contenido

### 1. Fundamentos Teóricos
**Teoría de la Decisión**
- Minimización de Probabilidad × Pérdidas
- Riesgo de Bayes vs Riesgo Empírico
- Componentes: Función de coste (L), Datos ({x,y}), Modelo (f)

### 2. Arquitecturas de Modelos
**Inspiración Biológica**
- Neurona de Ramón y Cajal (1852-1934)
- Córtex visual: células simples y complejas (Hubel & Wiesel, Premio Nobel 1981)

**Tipos de Capas**
- **Lineales**: Fully connected, Convolucional
- **No lineales**: Sign, Sigmoid, tanh, ReLU, GDN, Subsampling/Pooling

### 3. Funciones de Coste

| Tipo de Problema | Funciones de Coste |
|------------------|-------------------|
| **Regresión** | MAE, MSE |
| **Clasificación Binaria** | Binary Cross-Entropy, Weighted Cross-Entropy, Focal Loss |
| **Clasificación Multi-clase** | Categorical Cross-Entropy, Sparse Categorical Cross-Entropy |
| **Segmentación** | Dice, Jaccard (IoU) |
| **Series Temporales** | DTW, Soft-DTW, Frechet, DILATE, CTC |
| **Probabilidades** | Negative Log-Likelihood, KLD, Wasserstein, Quantile Loss |
| **Perceptual** | Funciones basadas en percepción humana |

**Consideraciones Importantes**
- Definir bien la función de coste es crítico: "Es el objetivo de tu modelo"
- Cuidado con outliers y clases desbalanceadas
- Diferencia entre "loss" (derivable, para entrenar) y "metric" (para evaluar)

### 4. Algoritmos de Aprendizaje

**Descenso por Gradiente**
- Regla de la cadena (backpropagation)
- Variantes: Batch, Mini-batch, SGD (Stochastic)

**Optimizadores Avanzados**

| Optimizador | Características |
|-------------|-----------------|
| **SGD** | Básico, requiere ajuste manual de learning rate |
| **Momentum** | Sigue la inercia de gradientes previos |
| **RMSprop** | Learning rate adaptativo, se fija en el signo |
| **Adam** | Combina momentum y learning rate adaptativo (Kingma & Ba, 2015) |
| **Newton's Method** | Basado en expansión de Taylor |
| **Conjugate Gradient** | Óptimo para funciones cuadráticas |

### 5. Regularización

**Técnicas de Regularización de Pesos**
- L1, L2, Max norm

**Técnicas de Regularización en Entrenamiento**
- **Dropout**: Desactivación aleatoria de neuronas
- **Batch Normalization**: Normalización por lotes
- **Early Stopping**: Detención cuando validación no mejora

### 6. Hiperparámetros de Entrenamiento

| Hiperparámetro | Descripción | Recomendación |
|----------------|-------------|---------------|
| **Épocas** | Veces que el modelo ve todos los datos | Hasta convergencia clara |
| **Batch Size** | Tamaño de bloques de datos | Tan grande como permita la memoria |
| **Learning Rate** | Longitud del paso en gradiente | **Parámetro más importante** |
| **Regularización** | Métodos anti-sobreajuste | Hasta que train y validación estén "pegaditas" |

### 7. Infraestructura Computacional

**Hardware**
- GPUs: Miles de cores para procesamiento paralelo
- TPUs: Procesadores especializados
- CLAVE: Memoria disponible

**Software**
- Lenguajes: Python (dominante)
- Frameworks: Keras, TensorFlow, JAX, PyTorch
- Plataformas: GitHub, HuggingFace, Google Colab

### 8. Contexto Histórico

**Pre-Historia (hasta 1943)**
- Siglo III a.C.: Griegos hacían derivadas
- 1676: Regla de la cadena (Leibniz)
- 1847: Descenso por gradiente (Cauchy)

**Historia Moderna**
- 1943: Primeras arquitecturas NN (McCulloch y Pitts)
- 1959: Córtex visual (Wiesel y Hubel)
- 1970: Diferenciación automática (Linnainmaa)
- 1974-80: Primer "Invierno de la IA"
- 1979: Primer deep learning - Neocognitron (Fukushima)
- 1986: Término "Deep learning" (Rina Dechter)
- 1987-93: Segundo "Invierno de la IA"
- **~2012: Revolución del Deep Learning**
  - Victoria en ImageNet (Krizhevsky)
  - Grandes empresas se vuelcan: Google, Facebook, Amazon

### 9. Aplicaciones Avanzadas

**GDN (Generalized Divisive Normalization)**
- Inspirado en Contrast Gain Control del cerebro
- Aplicaciones:
  - Rendering perceptualmente optimizado
  - Compresión de imágenes (Ballé et al., 2017)
  - PerceptNet
  - **GDN 1D y 2D en Bolsa** (aplicación financiera)

## Tablas

### Comparación de Optimizadores
| Optimizador | Ventajas | Desventajas | Uso Recomendado |
|-------------|----------|-------------|-----------------|
| SGD | Simple, bien entendido | Requiere ajuste manual | Baseline |
| Momentum | Acelera convergencia | Puede sobrepasar mínimos | Problemas con valles |
| RMSprop | Adaptativo, robusto | No publicado formalmente | Después de SGD y Adam |
| Adam | Adaptativo, eficiente | Puede no converger en algunos casos | Primera opción general |

### Funciones de Coste en Keras
| Problema | Loss Function | Metric |
|----------|---------------|--------|
| Regresión | MAE, MSE | MAE, MSE |
| Clasificación Binaria | Binary Crossentropy | Accuracy, AUC, Precision, Recall |
| Clasificación Multi-clase | Categorical/Sparse Crossentropy | Accuracy, AUC |
| Segmentación | IoU, Dice | IoU |

## Conexiones detectadas

- **Relacionado con**: Neurociencia computacional (Ramón y Cajal, Hubel & Wiesel), Teoría de la decisión, Cálculo de variaciones, Optimización matemática
- **Prerequisito de**: Arquitecturas avanzadas (CNNs, RNNs, Transformers), Transfer Learning, Aplicaciones específicas (visión por computador, NLP, series temporales financieras)
- **Aplicaciones en finanzas**: GDN 1D/2D para análisis de series temporales bursátiles, predicción de precios, detección de patrones
- **Fundamentos matemáticos**: Regla de la cadena, Cálculo de variaciones, Teoría de la probabilidad, Optimización convexa

---
*Fuente: Intro_deep_2026.pdf | Bloque: B3_IA_Basica | Sección: S4_Redes_Neuronales*