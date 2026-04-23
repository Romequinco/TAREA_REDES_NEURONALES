# Fundamentos de Redes Neuronales

## Sub-conceptos
### Conv2D
**Definición:** capa convolucional 2D que aplica filtros para extraer características espaciales de imágenes

### Flatten
**Definición:** capa que convierte matrices multidimensionales en vectores unidimensionales para capas densas

### Grad-CAM
**Definición:** Técnica que usa gradientes de la clase objetivo respecto a las activaciones convolucionales para generar mapas de localización visual

### GradientTape
**Definición:** Contexto de TensorFlow que registra operaciones para calcular gradientes automáticamente mediante diferenciación automática

### Pooled gradients
**Definición:** Promedio de los gradientes a lo largo de las dimensiones espaciales, usado para ponderar la importancia de cada canal

### CIFAR-100
**Definición:** Dataset de 60,000 imágenes en color (32x32 píxeles, 3 canales RGB) distribuidas en 100 clases diferentes, dividido en 50,000 imágenes de entrenamiento y 10,000 de test.


### Sequential API
**Definición:** modelo de Keras que permite apilar capas secuencialmente, ideal para arquitecturas lineales sin ramificaciones

### Dense con softmax
**Definición:** capa totalmente conectada con activación softmax para clasificación multiclase

### to_categorical
**Definición:** convierte etiquetas numéricas en vectores one-hot encoding

### CIFAR-10
**Definición:** dataset de 60,000 imágenes color 32x32 en 10 categorías (aviones, coches, pájaros, etc.)

### Backpropagation
**Definición:** Algoritmo basado en la regla de la cadena para calcular gradientes y entrenar redes neuronales multicapa
*Ref adicional:* Proceso de propagación del error hacia atrás en la red para ajustar pesos mediante la regla de la cadena (técnica con >200 años de antigüedad matemática)

### Patience
**Definición:** Número de épocas sin mejora que el modelo tolera antes de detener el entrenamiento (en este caso: 5 épocas)
*Ref adicional:* Número de épocas sin mejora en la métrica monitoreada antes de que se active la acción del callback (en este caso, 2 épocas).

### SGD con Nesterov
**Definición:** Optimizador de descenso de gradiente estocástico con momentum de Nesterov, que anticipa la dirección del gradiente

### Softmax
**Definición:** Función de activación que convierte logits en probabilidades que suman 1, ideal para clasificación multiclase
*Ref adicional:* Función de activación en la capa de salida que convierte logits en distribución de probabilidad sobre 10 clases
*Ref adicional:* Función que convierte un vector de valores en una distribución de probabilidad (suma=1), exagerando diferencias mediante exponenciales. Usada en clasificación multiclase
*Ref adicional:* Función que convierte scores en probabilidades normalizadas que suman 1, usada para combinar outputs de clasificadores OvR.

### save_best_only
**Definición:** Parámetro que indica guardar solo cuando la métrica monitoreada mejora respecto a épocas anteriores

### monitor='val_loss'
**Definición:** Especifica que la métrica a vigilar es la pérdida en el conjunto de validación

### mode='min'
**Definición:** Indica que se busca minimizar la métrica (alternativa: 'max' para accuracy)

### ReduceLROnPlateau
**Definición:** Callback de Keras que reduce la tasa de aprendizaje cuando una métrica deja de mejorar durante un número específico de épocas (patience). Ayuda a refinar el entrenamiento en fases avanzadas.

### Factor
**Definición:** Multiplicador aplicado al learning rate actual cuando se reduce (0.1 = reducción al 10% del valor anterior).

### Sequential
**Definición:** modelo de Keras que permite apilar capas de forma lineal, una tras otra, ideal para arquitecturas feedforward simples

### Dense (capa densa)
**Definición:** capa totalmente conectada donde cada neurona recibe input de todas las neuronas de la capa anterior

### ReLU (Rectified Linear Unit)
**Definición:** función de activación que devuelve max(0, x), introduce no-linealidad y es computacionalmente eficiente
*Ref adicional:* f(x) = max(0,x). Función de activación estándar actual que resuelve el gradiente desvaneciente. Su derivada es 0 (x<0) o 1 (x>0), permitiendo entrenar redes profundas eficientemente

### MNIST
**Definición:** Dataset estándar de dígitos manuscritos (0-9) con imágenes de 28x28 píxeles en escala de grises

### Dense Layer
**Definición:** Capa totalmente conectada donde cada neurona se conecta con todas las de la capa anterior

### Activación ReLU
**Definición:** Función de activación Rectified Linear Unit (f(x) = max(0,x))

### Activación Softmax
**Definición:** Función que convierte valores en probabilidades que suman 1, típica para clasificación multiclase

### activity_regularizer
**Definición:** Regularizador aplicado a las salidas (activaciones) de una capa neuronal

### VGG16/VGG19
**Definición:** Arquitecturas CNN clásicas pre-entrenadas en ImageNet, útiles para transfer learning y visualización de características

### include_top=False
**Definición:** Parámetro que carga la red sin las capas densas finales, manteniendo solo las capas convolucionales para extracción de características

### preprocess_input
**Definición:** Función de preprocesamiento específica de cada arquitectura que normaliza imágenes según los estándares de entrenamiento original

### Model extraction
**Definición:** Técnica para crear modelos parciales que extraen salidas de capas intermedias específicas

### VGG16
**Definición:** Arquitectura de red neuronal convolucional pre-entrenada con 16 capas, ampliamente usada para clasificación de imágenes

### get_weights()
**Definición:** Método de Keras que retorna una lista con todos los parámetros entrenables del modelo (pesos y sesgos) de cada capa.

### Filtros/Pesos
**Definición:** Parámetros de la red que se ajustan durante el entrenamiento y determinan las transformaciones aplicadas a los datos

### Ventana temporal
**Definición:** Técnica de preparación de datos donde se agrupan secuencias consecutivas (aquí, 10 precios de apertura) para predecir un valor futuro

### Bias
**Definición:** Término independiente en cada neurona que permite desplazar la función de activación

### MLP (Multilayer Perceptron)
**Definición:** Red neuronal feedforward con múltiples capas densas, utilizada para tareas de clasificación y regresión.

### Sequential Model
**Definición:** API de Keras para construir modelos lineales capa por capa, donde cada capa tiene exactamente un tensor de entrada y uno de salida.

### model.summary()
**Definición:** Método que muestra la arquitectura del modelo, número de parámetros por capa y total.

### model.get_weights() / set_weights()
**Definición:** Métodos para extraer y establecer los pesos de todas las capas del modelo.

### plot_model
**Definición:** Función de Keras para generar representaciones gráficas de la arquitectura en formato PNG o SVG.

### ANN Visualizer
**Definición:** Librería externa para crear visualizaciones atractivas de arquitecturas de redes neuronales.

### Design Matrix (X_b)
**Definición:** matriz aumentada que incluye columna de unos para incorporar el término de sesgo

### Empirical Risk Minimization
**Definición:** Estrategia de entrenamiento que minimiza el error basándose en la distribución empírica de los datos de entrenamiento.

### XAI (Explainable AI)
**Definición:** Conjunto de técnicas para visualizar y comprender el funcionamiento interno de las redes neuronales.

### Neurona artificial
**Definición:** Unidad computacional que realiza suma ponderada de entradas más bias: y = W₁x₁ + W₂x₂ + ... + Wₙxₙ + b

### Particiones del espacio
**Definición:** Las redes neuronales dividen el espacio de entrada en regiones mediante hiperplanos, asignando un valor de salida a cada región

### Inicialización Xavier
**Definición:** Método de inicialización de pesos aleatorios con varianza dependiente del número de entradas, para mantener salidas en rango controlado [0,1] o [-1,1] ### Funciones de activación

### Tangente hiperbólica (tanh)
**Definición:** Similar a sigmoide pero con salidas entre -1 y 1. Mejor centrada que sigmoide pero con problemas similares de gradiente

### Sigmoid en salida
**Definición:** Para problemas multi-etiqueta donde cada salida es independiente y puede estar entre 0 y 1 ### Funciones de pérdida (Loss functions)

### Focal loss
**Definición:** Enfoca en muestras difíciles, útil para desbalanceo de clases

### Pérdida perceptual
**Definición:** Función de coste que penaliza según cómo un humano percibe el error, no solo distancia matemática

### DTW (Dynamic Time Warping)
**Definición:** Función de pérdida para series temporales que permite comparar patrones con desplazamientos o compresiones temporales

### Wasserstein distance
**Definición:** Métrica de "mínimo movimiento de tierras" entre distribuciones, útil en modelos generativos ### Técnicas de entrenamiento y regularización

### Época
**Definición:** Una pasada completa por todos los datos de entrenamiento

### Mínimo local
**Definición:** Punto donde el modelo se estanca sin alcanzar la solución óptima global (ej: predecir siempre la media) ### Problemas específicos

### Ataques adversariales
**Definición:** Modificaciones mínimas e imperceptibles en los datos de entrada que hacen que la red cambie completamente su predicción, explotando las fronteras de decisión en espacios de alta dimensión

### AutoKeras
**Definición:** Sistema de búsqueda automática de arquitecturas de redes neuronales mediante prueba y error sistemático

### Sparsity (Dispersión)
**Definición:** Propiedad de tener pocos valores activos (no-cero) y muchos inactivos. Deseable por eficiencia computacional y energética, similar al cerebro ### Conceptos auxiliares

### Espectrograma
**Definición:** Representación visual de audio mediante transformada de Fourier, descomponiendo la señal en frecuencias a lo largo del tiempo

### Transformada de Fourier
**Definición:** Descomposición de señales en componentes de frecuencia, base de compresión MP3 y JPEG

### PCA sobre datos naturales
**Definición:** Análisis de componentes principales que, aplicado a imágenes o audio, extrae direcciones principales equivalentes a funciones de Fourier

### Similitud de coseno
**Definición:** Métrica que mide el ángulo entre vectores, útil en modelos multimodales (imagen-texto)

### Granger Causality
**Definición:** Método para determinar causalidad con series temporales. Compara el error de predicción de Y usando solo valores pasados de Y versus usar valores pasados de Y y X

### Causalidad sin tiempo (Holler)
**Definición:** Cuando no hay componente temporal, se testea la normalidad de los residuos en ambas direcciones (X→Y y Y→X). La dirección cuyos residuos son más gaussianos indica la causalidad correcta

### TensorFlow
**Definición:** Librería de Google para diferenciación automática y deep learning, con soporte para TPU

### Keras
**Definición:** API de alto nivel para redes neuronales, funciona sobre TensorFlow/Theano, muy fácil de usar

### PyTorch
**Definición:** Framework de Facebook/Meta para deep learning, similar a NumPy pero con soporte GPU

### Pesos constantes
**Definición:** Los pesos calculados en el periodo train se mantienen fijos durante todo el periodo de evaluación, sin rebalanceo ni reoptimización.

### Perceptrón (Feed-forward NN)
**Definición:** Red neuronal básica sin ciclos; cada neurona aplica F(X) = Σ W_i × X_i con función de activación.

### Conv2D
**Definición:** capa convolucional 2D que aplica filtros para extraer características espaciales de imágenes

### Flatten
**Definición:** capa que convierte matrices multidimensionales en vectores unidimensionales para capas densas

### Grad-CAM
**Definición:** Técnica que usa gradientes de la clase objetivo respecto a las activaciones convolucionales para generar mapas de localización visual

### GradientTape
**Definición:** Contexto de TensorFlow que registra operaciones para calcular gradientes automáticamente mediante diferenciación automática

### Pooled gradients
**Definición:** Promedio de los gradientes a lo largo de las dimensiones espaciales, usado para ponderar la importancia de cada canal

### grad()
**Definición:** Función de JAX que calcula la derivada de la salida respecto a la entrada especificada. Retorna una nueva función que computa el gradiente.

### jit (Just-In-Time compilation)
**Definición:** Compilador que optimiza funciones JAX para ejecución más rápida, especialmente útil en bucles de entrenamiento.

### argnums
**Definición:** Parámetro de `grad()` que especifica el índice del argumento respecto al cual calcular la derivada (0=primer argumento, 1=segundo, etc.).

### MSE (Mean Squared Error)
**Definición:** Penaliza errores grandes cuadráticamente, sensible a outliers, tiene solución cerrada en regresión lineal

### MAE (Mean Absolute Error)
**Definición:** Penalización lineal del error, robusto a outliers, da resultados en unidades físicas del problema

### Feature Engineering
**Definición:** Extracción manual de características relevantes antes de entrenar (ej: promedios móviles, varianza local en finanzas)

### Perplexity
**Definición:** Otra denominación de cross-entropy en NLP (misma fórmula matemática) ### Métodos de causalidad (contexto inicial)

## Apariciones consolidadas
| Sub-concepto | Bloque | Sección | Archivo |
|-------------|--------|---------|---------|
| Sequential API | B3_IA_Basica | S4_Redes_Neuronales | copia-de-keras-example-mlp-minimo.md |
| Dense con softmax | B3_IA_Basica | S4_Redes_Neuronales | copia-de-keras-example-mlp-minimo.md |
| to_categorical | B3_IA_Basica | S4_Redes_Neuronales | copia-de-keras-example-mlp-minimo.md |
| CIFAR-10 | B3_IA_Basica | S4_Redes_Neuronales | copia-de-keras-example-mlp-minimo.md |
| Backpropagation | B3_IA_Basica | S4_Redes_Neuronales | intro-deep-2026.md |
| Patience | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-callback-early-stopping.md |
| SGD con Nesterov | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-callback-early-stopping.md |
| Softmax | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-callback-early-stopping.md |
| save_best_only | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-callback-guardar-mejor-modelo.md |
| monitor='val_loss' | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-callback-guardar-mejor-modelo.md |
| mode='min' | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-callback-guardar-mejor-modelo.md |
| ReduceLROnPlateau | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-callback-leaning-rate-reduceonplateau.md |
| Patience | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-callback-leaning-rate-reduceonplateau.md |
| Factor | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-callback-leaning-rate-reduceonplateau.md |
| SGD con Nesterov | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-callback-leaning-rate-reduceonplateau.md |
| Sequential | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-minimo-1.md |
| Dense (capa densa) | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-minimo-1.md |
| ReLU (Rectified Linear Unit) | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-minimo-1.md |
| Softmax | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-minimo-1.md |
| to_categorical | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-minimo-1.md |
| MNIST | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-minimo-funcional.md |
| Dense Layer | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-minimo-funcional.md |
| Activación ReLU | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-minimo-funcional.md |
| Activación Softmax | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-minimo-funcional.md |
| SGD con Nesterov | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-regularization-dropout.md |
| Softmax | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-regularization-dropout.md |
| activity_regularizer | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-regularization-regularizador-capa.md |
| SGD con Nesterov | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-regularization-regularizador-capa.md |
| VGG16/VGG19 | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-visualizaciones-cnn-activations.md |
| include_top=False | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-visualizaciones-cnn-activations.md |
| preprocess_input | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-visualizaciones-cnn-activations.md |
| Model extraction | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-visualizaciones-cnn-activations.md |
| VGG16 | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-visualizaciones-cnn-heatmap.md |
| get_weights() | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-visualizaciones-filtros-conv.md |
| Filtros/Pesos | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-visualizaciones-filtros.md |
| Ventana temporal | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-visualizaciones-filtros.md |
| Bias | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-visualizaciones-filtros.md |
| MLP (Multilayer Perceptron) | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-visualizaciones-simples.md |
| Sequential Model | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-visualizaciones-simples.md |
| Dense Layer | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-visualizaciones-simples.md |
| model.summary() | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-visualizaciones-simples.md |
| model.get_weights() / set_weights() | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-visualizaciones-simples.md |
| plot_model | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-visualizaciones-simples.md |
| ANN Visualizer | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-visualizaciones-simples.md |
| Design Matrix (X_b) | B3_IA_Basica | S4_Redes_Neuronales | learning-ml-my-linear-regresion-by-hand.md |
| Empirical Risk Minimization | B3_IA_Basica | S4_Redes_Neuronales | training-nn-2026.md |
| XAI (Explainable AI) | B3_IA_Basica | S4_Redes_Neuronales | training-nn-2026.md |
| Neurona artificial | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| Backpropagation | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| Particiones del espacio | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| Inicialización Xavier | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| ReLU (Rectified Linear Unit) | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| Tangente hiperbólica (tanh) | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| Softmax | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| Sigmoid en salida | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| Focal loss | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| Pérdida perceptual | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| DTW (Dynamic Time Warping) | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| Wasserstein distance | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| Época | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| Mínimo local | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| Ataques adversariales | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| AutoKeras | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| Sparsity (Dispersión) | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| Espectrograma | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| Transformada de Fourier | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| PCA sobre datos naturales | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| Similitud de coseno | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| Granger Causality | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| Causalidad sin tiempo (Holler) | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| TensorFlow | B1_Python_IA | S3_Matematicas_IA | 3-calculo.md |
| Keras | B1_Python_IA | S3_Matematicas_IA | 3-calculo.md |
| PyTorch | B1_Python_IA | S3_Matematicas_IA | 3-calculo.md |
| Pesos constantes | B2_Finanzas | S7_Gestion_Carteras | 22-comparacion-carteras-ipynb-colab.md |
| Perceptrón (Feed-forward NN) | B2_Finanzas | S8_Gestion_Riesgos | 1-bme-riesgo-financieros-p-1.md |
| Backpropagation | B3_IA_Basica | S1_Intro_ML | intro-ml-2026-p.md |
| Backpropagation | B3_IA_Basica | S3_ML_Supervisado | intro-ml-2025.md |
| Softmax | B3_IA_Basica | S3_ML_Supervisado | miax-ml-02-evaluation-metrics.md |
| Conv2D | B3_IA_Basica | S4_Redes_Neuronales | copia-de-keras-example-mlp-minimo.md |
| Flatten | B3_IA_Basica | S4_Redes_Neuronales | copia-de-keras-example-mlp-minimo.md |
| Grad-CAM | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-visualizaciones-cnn-heatmap.md |
| GradientTape | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-visualizaciones-cnn-heatmap.md |
| Pooled gradients | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-visualizaciones-cnn-heatmap.md |
| Flatten | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-visualizaciones-filtros-conv.md |
| grad() | B3_IA_Basica | S4_Redes_Neuronales | learning-derivadas-en-jax.md |
| jit (Just-In-Time compilation) | B3_IA_Basica | S4_Redes_Neuronales | learning-derivadas-en-jax.md |
| argnums | B3_IA_Basica | S4_Redes_Neuronales | learning-derivadas-en-jax.md |
| MSE (Mean Squared Error) | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| MAE (Mean Absolute Error) | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| Feature Engineering | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
| Perplexity | B3_IA_Basica | S4_Redes_Neuronales | transcripcion-redes-neuronales-1.md |
