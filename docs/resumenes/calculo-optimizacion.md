# Cálculo y Optimización

## Sub-conceptos

### SGD (Stochastic Gradient Descent)
**Definición:** Optimizador que actualiza pesos usando gradientes calculados en mini-batches, con opciones de momentum y Nesterov.
*Ref adicional:* Actualización por dato individual.

### Descenso por gradiente
**Definición:** Algoritmo de optimización que ajusta parámetros iterativamente moviéndolos en dirección opuesta al gradiente de la función de pérdida.
*Ref adicional:* Algoritmo iterativo que ajusta parámetros en dirección opuesta al gradiente para minimizar error.
*Ref adicional:* Algoritmo iterativo que encuentra mínimos de funciones moviéndose en dirección opuesta al gradiente

### Función de pérdida (loss)
**Definición:** Métrica que cuantifica el error entre predicciones y valores reales; aquí se usa el error absoluto medio (MAE).

### Gradientes
**Definición:** derivadas parciales de la función de coste respecto a w y b, indican la dirección de mayor crecimiento

### Derivada
**Definición:** Mide cómo cambia una función al variar sus inputs; fundamental para optimización.
*Ref adicional:* Medida de cómo cambia una función respecto a una variable. Fundamental para encontrar máximos y mínimos
*Ref adicional:* Medida de la tasa de cambio de una función respecto a su variable. Definida como $f'(x) = \lim_{h \rightarrow 0} \frac{f(x + h) - f(x)}{h}$. Representa la pendiente de la recta tangente.

### Gradiente
**Definición:** Vector de derivadas parciales para funciones multivariables; indica dirección de máximo crecimiento.
*Ref adicional:* Vector de derivadas parciales que indica la dirección de máxima variación de una función multivariable
*Ref adicional:* Vector de derivadas parciales de una función multidimensional $\nabla f = (\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n})$. Apunta en la dirección de máximo crecimiento.

### Regla de la cadena
**Definición:** Método para derivar funciones compuestas, esencial en backpropagation de redes neuronales

### Jacobiano
**Definición:** Matriz de derivadas parciales de una función vectorial respecto a variables vectoriales

### Integral
**Definición:** Operación opuesta a la derivada, calcula áreas bajo curvas o acumulación de cantidades

### Programación Convexa (CVXPY)
**Definición:** Framework para resolver problemas de optimización convexa con restricciones lineales y cuadráticas.
*Ref adicional:* Framework de optimización que permite formular y resolver problemas convexos con restricciones, usando solvers especializados como CLARABEL.

### CVXPY
**Definición:** Librería Python para optimización convexa que permite formular problemas cuadráticos con restricciones de forma declarativa.
*Ref adicional:* Librería Python para optimización convexa que integra solvers estándar usando sintaxis de alto nivel y verifica automáticamente la convexidad del problema.

### Enfoque de Mínimos Cuadrados
**Definición:** Minimiza la diferencia cuadrada entre contribuciones al riesgo de cada activo y un objetivo predefinido.

### PIT (Probability Integral Transform)
**Definición:** Transformación U = F(X) que convierte una variable aleatoria a uniforme [0,1], permitiendo estudiar dependencia sin contaminación marginal.

### Optimización convexa
**Definición:** Subcampo de optimización donde la función objetivo es convexa sobre un conjunto convexo, garantizando que cualquier mínimo local es global.

### Función convexa
**Definición:** Función cuadrática (como la varianza) que tiene forma de "cuenco", sin múltiples mínimos locales.
*Ref adicional:* Función donde cualquier mínimo local es también mínimo global, garantizando convergencia del descenso de gradiente.

### Batch Gradient Descent
**Definición:** Calcula el gradiente usando todo el dataset en cada iteración. Preciso pero costoso computacionalmente.

### Stochastic Gradient Descent (SGD)
**Definición:** Estima el gradiente usando una muestra individual (o pequeña) en cada paso. Más rápido pero más ruidoso.

### Mini-batch Gradient Descent
**Definición:** Compromiso entre batch y SGD, actualizando parámetros con subconjuntos de n ejemplos.

### Mínimos Cuadrados (LS)
**Definición:** Método de optimización que minimiza la suma de errores cuadráticos entre predicciones y valores reales

## Apariciones consolidadas
| Sub-concepto | Bloque | Sección | Archivo |
|-------------|--------|---------|---------|
| SGD (Stochastic Gradient Descent) | B3_IA_Basica | S4_Redes_Neuronales | keras-example-mlp-visualizaciones-simples.md |
| Descenso por gradiente | B3_IA_Basica | S4_Redes_Neuronales | learning-derivadas-en-jax.md |
| Función de pérdida (loss) | B3_IA_Basica | S4_Redes_Neuronales | learning-derivadas-en-jax.md |
| Descenso por Gradiente | B3_IA_Basica | S4_Redes_Neuronales | learning-ml-my-linear-regresion-by-hand.md |
| Gradientes | B3_IA_Basica | S4_Redes_Neuronales | learning-ml-my-linear-regresion-by-hand.md |
| Derivada | B1_Python_IA | S1_Python_Herramientas | 4-bme-diapositivas-pablo-dia1-p-2.md |
| Gradiente | B1_Python_IA | S1_Python_Herramientas | 4-bme-diapositivas-pablo-dia1-p-2.md |
| Descenso por gradiente | B1_Python_IA | S1_Python_Herramientas | 4-bme-diapositivas-pablo-dia1-p-2.md |
| SGD (Stochastic Gradient Descent) | B1_Python_IA | S1_Python_Herramientas | 4-bme-diapositivas-pablo-dia1-p-2.md |
| Derivada | B1_Python_IA | S3_Matematicas_IA | 3-calculo.md |
| Regla de la cadena | B1_Python_IA | S3_Matematicas_IA | 3-calculo.md |
| Gradiente | B1_Python_IA | S3_Matematicas_IA | 3-calculo.md |
| Jacobiano | B1_Python_IA | S3_Matematicas_IA | 3-calculo.md |
| Integral | B1_Python_IA | S3_Matematicas_IA | 3-calculo.md |
| Descenso por gradiente | B1_Python_IA | S3_Matematicas_IA | 3-calculo.md |
| Programación Convexa (CVXPY) | B2_Finanzas | S7_Gestion_Carteras | 10-rv-optimizaciones-carteras-soluciones-ipynb-colab.md |
| CVXPY | B2_Finanzas | S7_Gestion_Carteras | 20-apt-cartera-ipynb-colab.md |
| Enfoque de Mínimos Cuadrados | B2_Finanzas | S7_Gestion_Carteras | 21-tisk-parity-ipynb-colab.md |
| Programación Convexa (CVXPY) | B2_Finanzas | S7_Gestion_Carteras | 24-evolucion-carteras-ipynb-colab.md |
| PIT (Probability Integral Transform) | B2_Finanzas | S7_Gestion_Carteras | 30-generacion-sinteticos-copulas-solucion-ipynb-colab.md |
| Optimización convexa | B2_Finanzas | S7_Gestion_Carteras | 8-optimizacion-intro-ipynb-colab.md |
| Función convexa | B2_Finanzas | S7_Gestion_Carteras | 8-optimizacion-intro-ipynb-colab.md |
| CVXPY | B2_Finanzas | S7_Gestion_Carteras | 9-cvxpy-intro-solucion-ipynb-colab.md |
| Descenso por gradiente | B3_IA_Basica | S2_Tipos_Aprendizaje | learning-derivadas-en-jax.md |
| Función de pérdida (loss) | B3_IA_Basica | S2_Tipos_Aprendizaje | learning-derivadas-en-jax.md |
| Derivada | B3_IA_Basica | S2_Tipos_Aprendizaje | learning-gradient-descent.md |
| Gradiente | B3_IA_Basica | S2_Tipos_Aprendizaje | learning-gradient-descent.md |
| Función convexa | B3_IA_Basica | S2_Tipos_Aprendizaje | learning-gradient-descent.md |
| Batch Gradient Descent | B3_IA_Basica | S2_Tipos_Aprendizaje | learning-gradient-descent.md |
| Stochastic Gradient Descent (SGD) | B3_IA_Basica | S2_Tipos_Aprendizaje | learning-gradient-descent.md |
| Mini-batch Gradient Descent | B3_IA_Basica | S2_Tipos_Aprendizaje | learning-gradient-descent.md |
| Mínimos Cuadrados (LS) | B3_IA_Basica | S2_Tipos_Aprendizaje | models-ml-my-linear-regresion-by-hand.md |
