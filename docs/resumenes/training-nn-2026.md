# Entrenamiento de Redes Neuronales en la Práctica

## Resumen
Guía práctica sobre el entrenamiento de redes neuronales, enfatizando que es un proceso iterativo que requiere monitorización constante. Cubre desde la preparación de datos hasta el ajuste fino del modelo, pasando por la búsqueda de arquitectura óptima mediante el balance entre bias y variance. Se destaca que entrenar NN no es completamente automatizable y requiere comprensión profunda del proceso.

## Ideas clave
- Entrenar redes neuronales es un arte que requiere monitorización exhaustiva de todos los componentes
- La estrategia óptima es partir de modelos simples y complejos, convergiendo hacia la arquitectura necesaria
- El balance entre bias (error en train) y variance (error en validación) guía las decisiones de complejidad del modelo
- La visualización y el uso de curvas de aprendizaje son fundamentales para diagnosticar problemas
- No existe entrenamiento totalmente automático; la comprensión del proceso es esencial

## Conceptos detectados

- **Empirical Risk Minimization**: Estrategia de entrenamiento que minimiza el error basándose en la distribución empírica de los datos de entrenamiento.
- **Bias-Variance Tradeoff**: Balance entre el error por sesgo (underfitting en train) y error por varianza (overfitting en validación).
- **Data Augmentation**: Técnica para aumentar artificialmente el conjunto de datos mediante transformaciones que preservan las características relevantes.
- **Baseline**: Modelo de referencia simple que establece el rendimiento mínimo aceptable para comparar mejoras.
- **Early Stopping**: Técnica de regularización que detiene el entrenamiento cuando el error de validación deja de mejorar.
- **Learning Rate Schedule**: Estrategia de ajuste dinámico de la tasa de aprendizaje durante el entrenamiento.
- **Callbacks**: Funciones que se ejecutan en momentos específicos del entrenamiento para monitorizar o modificar el proceso.
- **XAI (Explainable AI)**: Conjunto de técnicas para visualizar y comprender el funcionamiento interno de las redes neuronales.

## Estructura del contenido

### 1. Preparación: Datos
**Principio fundamental**: "La red es una versión comprimida de tus datos"

- Inspección exhaustiva: outliers y clases desbalanceadas
- Preprocesado cuidadoso (normalización a [0,1] o [-1,1])
- Estrategias de aumento: data augmentation, modelos generativos
- Partición correcta: train/validación/test/development
- Advertencias: mismatching train/test, mezcla de calidades
- Preferencia por aprendizaje supervisado

### 2. Preparación: Función de Coste
**Principio fundamental**: "Es el objetivo de tu red"

- Definición precisa del objetivo
- Atención a outliers y desbalanceo
- Basada en Empirical Risk Minimization
- Precaución con funciones especiales

### 3. Establecimiento de Baselines
Múltiples referencias necesarias:
- Rendimiento deseable
- Estado del arte en problemas similares
- Rendimiento humano (gold standard)
- Valor de pérdida inicial

### 4. Búsqueda de Arquitectura: Estrategia de Dos Extremos

#### Modelo Simple → Complejo
**Objetivo**: Minimizar bias (mejorar train)

Progresión incremental:
1. Modelo más simple posible (incluso lineal)
2. Usar semilla para reproducibilidad
3. Inicialización apropiada de última capa
4. Pocas épocas para exploración
5. Aumentar complejidad gradualmente

**Jerarquía de complejidad**:
- Número de parámetros
- Para aumentar: añadir neuronas → capas → cambiar tipo
- Tipos por complejidad: Densas < Recurrentes < Convolucionales

#### Modelo Complejo → Simple
**Objetivo**: Minimizar variance (mejorar validación)

Estrategia:
1. Modelo grande que pueda sobreentrenar
2. Usar arquitecturas probadas o preentrenadas
3. Regla: parámetros << datos (al menos 10x menos)
4. Reducir complejidad gradualmente

**Jerarquía de reducción**:
1. Quitar neuronas
2. Añadir poolings
3. Quitar capas
4. Orden: Densas → Recurrentes → Convolucionales

### 5. Regularización
**Objetivo**: Curvas de train y validación "pegaditas"

**Técnicas principales**:
- Dropout
- Weight decay
- Batch normalization
- Early stopping

**Estrategias**:
- Reducir dimensiones, modelo y batch size
- Paradoja: a veces modelos más grandes regularizan mejor

### 6. Entrenamiento Final y Optimización

**Fine-tuning**:
- Entrenar más tiempo
- Búsqueda aleatoria en grid (Talos, Keras-tuner)
- Ensembles de modelos

**Optimización de datos**:
- Seleccionar datos más informativos
- Decorrelacionar datos
- Aleatorización

**Inicialización**:
- Identidad
- Pesos preentrenados
- Glorot aleatoria

**Gestión computacional**:
- Memoria: backpropagation es el cuello de botella
- Ajustar mini-batch size con precaución
- Reentrenamiento de última capa por separado

**Callbacks recomendados**:
- Early stopping
- Guardar mejor modelo
- Learning rate scheduling (no usar default al inicio)

**Optimizadores**:
- Recomendado: Adam con LR=3e-4
- SGD ayuda a regularizar
- Avanzado: gradientes naturales, LR adaptativo por peso

## Tablas

### Progresión Arquitectura Simple (mal train → mejor train)

| Paso | Arquitectura | Objetivo |
|------|--------------|----------|
| 1 | Conv2D(100) + GAP | Baseline mínimo |
| 2 | Conv2D(128) + GAP + Densa(100) | Añadir capacidad |
| 3 | Conv2D(3,32) + Pool(2) + Conv2D(3,128) + GAP + Densa(100) | Jerarquía espacial |
| 4 | Conv2D(3,64) + Pool(2) + Conv2D(3,128) + GAP + Densa(100) | Más filtros |
| 5 | Conv2D(3,64) + Pool(2) + Conv2D(3,128) + Pool(2) + FL + Densa(100) | Más reducción |
| 6 | 3x(Conv2D(3) + Pool(2)) [32,64,128] + FL + Densa(100) | Arquitectura objetivo |

### Progresión Arquitectura Compleja (buen train → regularizar)

| Paso | Arquitectura | Cambio |
|------|--------------|--------|
| 1 | 4x(Conv+Pool)[64,128,256,512] + FL + Densa(1000) + Densa(100) | Modelo muy complejo |
| 2 | 4x(Conv+Pool)[64,128,256,512] + FL + Densa(200) + Densa(100) | Reducir última densa |
| 3 | 4x(Conv+Pool)[64,128,256,512] + FL + Densa(100) | Eliminar capa densa |
| 4 | 4x(Conv+Pool)[64,128,256,256] + FL + Densa(100) | Reducir filtros finales |
| 5 | 3x(Conv+Pool)[64,128,256] + FL + Densa(100) | Eliminar bloque |
| 6 | 3x(Conv+Pool)[64,128,128] + FL + Densa(100) | Reducir filtros |
| 7 | 3x(Conv+Pool)[64,64,128] + FL + Densa(100) | Reducir más |
| 8 | 3x(Conv+Pool)[32,64,128] + FL + Densa(100) | **Arquitectura objetivo** |

### Diagnóstico por Curvas de Aprendizaje

| Patrón | Train | Validación | Diagnóstico | Solución |
|--------|-------|------------|-------------|----------|
| Buen fitting | Baja y estable | Baja y cercana a train | Modelo óptimo | Mantener |
| Overfitting | Muy baja | Alta y divergente | Exceso complejidad | Regularizar |
| Underfitting | Alta | Alta y cercana | Falta complejidad | Aumentar modelo |
| Sobrerregularizado | Moderada | Mejor que train | Exceso regularización | Reducir regularización |
| Datos no representativos | Baja | Errática | Problema en datos | Revisar partición |
| LR incorrecto | Errática/plana | Errática/plana | Problema optimización | Ajustar learning rate |

## Monitorización y XAI

### Elementos a visualizar
- Datos de entrada (post-preprocesado)
- Pérdidas por época y por muestra
- Salidas del modelo
- Pesos de las capas
- Gradientes
- Activaciones intermedias

### Herramientas recomendadas
- **TensorBoard**: Monitorización estándar
- **Weights & Biases (W&B)**: Tracking avanzado
- **Visualizadores CNN**: CNN Explainer, Microscope, Lucid

### Recursos de modelos preentrenados
- TensorFlow Hub
- TensorFlow.js Models
- TensorFlow Detection Model Zoo
- Hugging Face

## Conexiones detectadas

- **Relacionado con**: Optimización (algoritmos Adam, SGD), Regularización (Dropout, Batch Normalization), Arquitecturas CNN
- **Prerequisito de**: Transfer Learning, Fine-tuning de modelos, AutoML
- **Requiere conocimiento de**: Funciones de pérdida, Backpropagation, Gradiente descendente
- **Aplicable a**: Visión por computador, NLP, Series temporales

## Advertencias críticas del autor

1. **"Entrenar NN es magia negra"**: No es completamente automatizable
2. **"Hay errores visibles e invisibles"**: El overfitting puede estar oculto
3. **"La red es una versión comprimida de tus datos"**: La calidad del modelo depende de los datos
4. **"No te fíes del unsupervised"**: Preferir aprendizaje supervisado cuando sea posible
5. **"Monitoriza TODO"**: La visualización es fundamental para detectar problemas

---
*Fuente: Training_NN_2026.pdf | Bloque: B3_IA_Basica | Sección: S4_Redes_Neuronales*