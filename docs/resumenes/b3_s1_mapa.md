# Mapa de conocimiento — B3 — Introducción a Machine Learning

## Conceptos centrales del bloque

- **Los 4 componentes fundamentales (Datos, Modelo, Función de coste, Optimización)**: Marco unificador que desmitifica toda IA y estructura el bloque completo. Aparece como fundamento teórico que se aplica desde casos básicos hasta sistemas complejos (ChatGPT, agentes, modelos financieros). Es la base conceptual que permite entender cualquier sistema de ML como combinación de estos elementos. En finanzas, cada componente requiere adaptaciones específicas por la naturaleza no-IID de los datos.

- **Riesgo Empírico vs Riesgo de Bayes**: Distinción fundamental que define la esencia del ML. El riesgo de Bayes representa el objetivo teórico inalcanzable (decisión óptima con distribución real), mientras el riesgo empírico (Σ L(f(X), Y)) es la aproximación práctica con datos disponibles que se minimiza mediante optimización. Esta tensión subyace a todos los problemas de generalización y conecta teoría de decisión con implementación práctica.

- **Overfitting**: Enemigo fundamental en ML financiero que aparece en dos formas críticas que requieren soluciones distintas:
  - **Overfitting en conjunto de entrenamiento**: Solucionable con técnicas estándar (CV, regularización L1/L2, dropout, early stopping, ensemble methods)
  - **Overfitting en conjunto de prueba**: Específico de finanzas por iteración de backtests; requiere control estadístico de múltiples tests (Deflated Sharpe Ratio, FWER, corrección de Šidàk)
  
  López de Prado lo identifica como causa principal del fracaso en proyectos de ML financiero.

- **Estructuración de Datos Financieros**: Los datos financieros requieren tratamiento especial por su naturaleza no-IID (overlapping outcomes, autocorrelación temporal, heteroscedasticidad). Las técnicas específicas (barras basadas en información TIBs/VIBs/DIBs, ETF trick, CUSUM filter) sincronizan el muestreo con la llegada de información al mercado, no con el tiempo cronológico. Esto contrasta radicalmente con el tratamiento estándar de datos en ML general.

- **Feature Engineering/Importance**: Proceso crítico que transforma datos brutos en señales informativas. López de Prado dedica estaciones completas (Feature Analysts) a esto y desarrolla métodos robustos (MDI, MDA, SFI, CFI) que superan limitaciones de p-valores estadísticos tradicionales. Es el puente entre datos y modelos, y en finanzas debe incorporar conocimiento del dominio (microestructura, retornos logarítmicos, look-ahead bias).

- **Validación Temporal (Walk Forward)**: Paradigma de validación específico para finanzas que respeta la estructura secuencial de datos. Contrasta radicalmente con K-fold CV estándar que asume IID. Requiere técnicas adicionales (purging, embargoing) para evitar leakage por solapamiento de labels y correlación serial. Es fundamental para estimación honesta de generalización en series temporales.

- **Descenso por Gradiente**: Método de optimización universal que conecta la función de coste con el aprendizaje efectivo. Aparece como técnica principal para ajustar parámetros en prácticamente todos los modelos (desde regresión lineal hasta redes profundas). Requiere funciones de coste diferenciables y se beneficia críticamente de hardware especializado (GPUs) para cálculo paralelo.

- **Regularización**: Conjunto de técnicas (L1, L2, dropout, early stopping, batch normalization) que controlan el trade-off bias-variance. Aparece como solución transversal al overfitting en entrenamiento, complementando la validación cruzada y los métodos ensemble. En finanzas, debe combinarse con técnicas específicas para datos no-IID.

- **Funciones de Coste (Loss Functions)**: Definen el objetivo del modelo y determinan qué aprende. Su correcta especificación es crítica; López de Prado enfatiza: "es el objetivo de tu modelo". Incluyen losses derivables para optimización (MSE, cross-entropy) y métricas no derivables para evaluación (accuracy, F1, AUC). Pueden incorporar asimetrías según el problema (coste diferencial de falsos positivos vs negativos en medicina/finanzas).

- **Machine Learning vs Deep Learning**: Distinción arquitectural fundamental. ML clásico separa extracción de características (manual/PCA) y modelado; DL los integra end-to-end mediante capas apiladas optimizadas conjuntamente. Esta diferencia determina cuándo usar cada paradigma: DL requiere más datos y hardware pero puede descubrir características no obvias.

- **Aprendizaje supervisado, no supervisado y por refuerzo**: Los tres paradigmas que estructuran todo el campo:
  - **Supervisado**: Requiere datos etiquetados (X,y), función de pérdida, optimización. Más común y directo.
  - **No supervisado**: Solo datos X, aprende estructura latente (clustering, reducción dimensionalidad, autoencoders).
  - **Por refuerzo**: Aprende por interacción con entorno mediante recompensas. Prometedor teóricamente pero "extremadamente difícil en producción" por catastrophic forgetting y dependencia de simuladores.

- **Denoising de Matrices de Covarianza**: Aplicación de Random Matrix Theory (teorema Marcenko-Pastur) para separar señal de ruido en matrices de correlación. Reduce RMSE 60-95% vs matrices sin tratar. Fundamental para optimización de portafolios (resuelve maldición de Markowitz) y análisis de dependencias. Incluye técnicas: Constant Residual Eigenvalue, Targeted Shrinkage, Detoning.

- **Meta-Estrategia y Organización Industrial**: Paradigma organizacional propuesto por López de Prado que estructura el ML financiero como cadena de producción con estaciones especializadas (Data Curators → Feature Analysts → Strategists → Backtesters → Deployment → Portfolio Oversight). Contrasta con el "paradigma de Sísifo" fallido donde un solo investigador intenta todo. Incluye Cursus Honorum para desarrollo profesional.

- **Modelos Generativos y Temperatura**: Concepto transversal que aparece en ChatGPT, datos sintéticos y limitaciones prácticas. Los modelos son deterministas (mismo input → mismo output con temperatura=0); la temperatura como parámetro de control desmitifica la "creatividad" aparente. Crítico entender que son autorregresivos (predicen siguiente token) y su capacidad generativa tiene límites: datos sintéticos mejoran hasta ~2,000 ejemplos, luego degradan.

- **Hardware (GPU/TPU) y Memoria**: Factor limitante real en implementación. Se enfatiza que la **memoria RAM es el cuello de botella**, no la velocidad de procesamiento. Aparece en contexto de ChatGPT (servidores €300,000), entrenamiento de redes profundas, y consideraciones prácticas. Conecta directamente capacidades algorítmicas con viabilidad económica.

- **Explicabilidad y Causalidad**: Tensión fundamental entre poder predictivo y comprensión. López de Prado (2020) enfatiza: "Backtesting is not a research tool. Feature importance is." El ML debe usarse para **descubrir teorías** (identificar variables ocultas, relaciones complejas), no como oráculo predictivo. Las técnicas más sofisticadas (denoising, CFI, NCO) buscan revelar estructura subyacente, no maximizar métricas de backtest.

## Grafo de dependencias

- **Fundamentos matemáticos (Riesgo de Bayes, Teoría de decisión)**
  - requiere → Probabilidad, Funciones de pérdida
  - permite → Riesgo empírico, Optimización, Toma de decisiones bajo incertidumbre
  - relacionado con → Consideraciones éticas (valoración de costes), Trade-off bias-variance

- **Los 4 componentes (Datos, Modelo, Función de coste, Optimización)**
  - requiere → Fundamentos matemáticos
  - permite → Todos los paradigmas de aprendizaje, Desmitificación de IA
  - relacionado con → Baseline obligatorio (modelos lineales), Estructura universal de sistemas ML

- **Riesgo Empírico**
  - requiere → Datos {X,y}, Función de Coste, Modelo f
  - permite → Optimización (Gradient Descent), Evaluación de Performance
  - relacionado con → Overfitting, Generalización, Aproximación del Riesgo de Bayes

- **Descenso por Gradiente**
  - requiere → Función de coste diferenciable, Riesgo empírico, Learning rate
  - permite → Entrenamiento de redes neuronales, Deep Learning, Backpropagation
  - relacionado con → Hardware (GPUs para cálculo paralelo), Optimizadores avanzados (ADAM, RMSprop)

- **Preprocesado de Datos**
  - requiere → Datos brutos, Conocimiento del dominio
  - permite → Feature Engineering, Estructuración de Datos Financieros, Imputación, Normalización
  - relacionado con → Calidad de resultados, Reducción de ruido, Detección de outliers

- **Estructuración de Datos Financieros**
  - requiere → Preprocesado, Comprensión de microestructura de mercado
  - permite → Barras basadas en información (TIBs, VIBs, DIBs, TRBs), ETF Trick, CUSUM Filter
  - relacionado con → Sincronización con llegada de información, Reducción de autocorrelación, Naturaleza no-IID

- **Feature Engineering**
  - requiere → Datos estructurados, Domain knowledge (finanzas), Retornos logarítmicos
  - permite → Extracción de señales, Reducción de dimensionalidad (PCA, t-SNE, Autoencoders)
  - relacionado con → Feature Importance, Poder predictivo, Prevención de look-ahead bias

- **Feature Importance**
  - requiere → Modelo entrenado, Datos de validación
  - permite → MDI, MDA, SFI, Clustered Feature Importance (CFI)
  - relacionado con → Interpretabilidad, Selección de features, Reducción de substitution effects, Descubrimiento de teorías

- **Labeling (Etiquetado)**
  - requiere → Datos estructurados, Definición de objetivo, Coste de etiquetado
  - permite → Triple Barrera, Meta-Labeling, Manejo de overlapping outcomes
  - relacionado con → Naturaleza no-IID de datos financieros, Average Uniqueness

- **Validación**
  - requiere → Datos particionados, Función de coste
  - permite → Walk Forward, PurgedKFold, Sequential Bootstrap, K-fold (solo para datos IID)
  - relacionado con → Control de overfitting, Estimación de generalización, Purging y Embargoing

- **Overfitting (conjunto de entrenamiento)**
  - requiere → Modelo flexible, Datos limitados
  - se combate con → Regularización (L1, L2, Dropout, Batch Norm), Cross-Validation, Ensemble Methods, Early Stopping
  - relacionado con → Bias-Variance Trade-off, Complejidad del modelo

- **Overfitting (conjunto de prueba)**
  - requiere → Múltiples tests sobre mismos datos, Iteración de backtests
  - se combate con → Deflated Sharpe Ratio, Corrección de Šidàk, FWER control, CPCV
  - relacionado con → Selection Bias, False Discovery Rate, False Strategy Theorem

- **Regularización**
  - requiere → Función de coste, Modelo parametrizado
  - permite → Control de complejidad, Mejora de generalización
  - relacionado con → Penalización de parámetros (L1, L2), Técnicas durante entrenamiento (Dropout, Batch Norm, Early Stopping)

- **Diferenciación Fraccionaria (FFD)**
  - requiere → Series temporales, Parámetro d
  - permite → Balance estacionariedad-memoria, Reducción de autocorrelación
  - relacionado con → Dilema fundamental de series financieras, Fixed-Width Window

- **Aprendizaje Supervisado**
  - requiere → Datos etiquetados {X,y}, Función de pérdida, Optimización
  - permite → Clasificación, Regresión, Predicción
  - relacionado con → Coste de etiquetado, Calidad de datos, Baseline obligatorio

- **Aprendizaje No Supervisado (Autoencoders, Clustering)**
  - requiere → Solo datos X, Arquitectura con cuello de botella (autoencoders)
  - permite → Reducción de dimensionalidad, Embeddings, Extracción de características, Clustering
  - relacionado con → PCA (versión lineal), Representaciones latentes, ONC (Optimal Number of Clusters)

- **Aprendizaje por Refuerzo**
  - requiere → Simulador/entorno, Función de recompensa
  - permite → Aprendizaje por interacción (sin datos previos), Políticas de decisión
  - relacionado con → Catastrophic forgetting, RLHF, FinRL, Dificultad en producción, Aleatoriedad

- **Deep Learning**
  - requiere → Descenso por gradiente, Hardware (GPUs/TPUs), Datos abundantes
  - permite → Filtros convolucionales, Modelos end-to-end, Transfer learning, Backpropagation
  - relacionado con → Inspiración biológica (Hubel y Wiesel), Integración de feature extraction y modelado

- **Denoising de Matrices de Covarianza**
  - requiere → Matriz de covarianza empírica, Random Matrix Theory, Teorema Marcenko-Pastur
  - permite → Constant Residual Eigenvalue Method, Targeted Shrinkage, Detoning
  - relacionado con → Optimización de portafolios, Clustering, Reducción RMSE 60-95%

- **Métricas de Distancia basadas en Información**
  - requiere → Teoría de la información, Discretización óptima
  - permite → Entropía de Shannon, Información Mutua, Variación de Información
  - relacionado con → Detección de dependencias no lineales, Clustering

- **Clustering Óptimo**
  - requiere → Matriz de distancias/proximidad, Criterio de optimalidad
  - permite → ONC (Optimal Number of Clusters), Clustering jerárquico vs particional
  - relacionado con → CFI, NCO, Identificación de estructuras latentes

- **Ensemble Methods**
  - requiere → Múltiples modelos base, Estrategia de agregación
  - permite → Bagging, Random Forest, Reducción de varianza
  - relacionado con → Sequential Bootstrap, Class weights, Observation redundancy, Average Uniqueness

- **Meta-Labeling**
  - requiere → Modelo primario (predicción de dirección), Modelo secundario (tamaño de apuesta)
  - permite → Filtrado de falsos positivos, Mejora de F1-score, Bet sizing
  - relacionado con → Construcción sobre modelos white-box, Fondos quantamental

- **Construcción de Portafolios**
  - requiere → Matriz de covarianza denoised, Retornos esperados
  - permite → NCO (Nested Clustered Optimization), Optimización jerárquica
  - relacionado con → Maldición de Markowitz, Número de condición, Clustering

- **Modelos Generativos (ChatGPT, LLMs)**
  - requiere → Deep Learning, Modelos autorregresivos, Tokens, Temperatura, Hardware masivo
  - permite → Generación de texto, RAG, Agentes de IA, Datos sintéticos
  - relacionado con → Determinismo (temperatura=0), RLHF, Prompt del sistema, Limitaciones de contexto

- **RAG (Retrieval Augmented Generation)**
  - requiere → Modelos generativos, Embeddings, Fragmentación de documentos
  - permite → Consulta eficiente de documentos largos, Superación de límite de contexto
  - relacionado con → Pérdida de información por fragmentación, Calidad de embeddings

- **Agentes de IA**
  - requiere → Múltiples modelos especializados, MCP (protocolo de comunicación)
  - permite → Resolución de tareas complejas, Orquestación de modelos
  - relacionado con → N8N, OpenClaude, Nivel de abstracción superior

- **Hardware Especializado (GPU/TPU)**
  - requiere → Operaciones paralelas masivas, Memoria RAM suficiente
  - permite → Entrenamiento de Deep Learning, Inferencia rápida
  - relacionado con → Memoria como factor limitante (no velocidad), CUDA, Edge computing, Coste económico

- **Datos Sintéticos**
  - requiere → Modelos generativos, Datos originales de calidad
  - permite → Aumento de dataset, Exploración de escenarios
  - relacionado con → Limitación crítica (mejora hasta ~2,000, degrada con 20,000), Nunca igualan originales

## Flujo de aprendizaje recomendado

1. **Fundamentos matemáticos: Riesgo de Bayes y Teoría de decisión**
   - *Justificación*: Establece el marco conceptual matemático que sustenta todo el ML. Entender probabilidad × coste es esencial antes de cualquier implementación. Conecta decisiones óptimas teóricas con aproximaciones prácticas.
   - *Contenido*: Riesgo de Bayes, teoría de decisión, funciones de pérdida, trade-offs fundamentales.

2. **Los 4 componentes universales: Datos, Modelo, Función de coste, Optimización**
   - *Justificación*: Marco unificador que desmitifica la IA. Permite ver cualquier sistema (desde regresión lineal hasta ChatGPT) como combinación de estos elementos. Es la base conceptual del bloque completo.
   - *Contenido*: Estructura universal de sistemas ML, baseline obligatorio (modelos lineales), conexión entre componentes.

3. **Riesgo empírico y función de pérdida**
   - *Justificación*: Conexión entre teoría (Riesgo de Bayes) y práctica (datos reales). Introduce la métrica concreta que se optimizará. Fundamental para entender qué significa "aprender".
   - *Contenido*: Riesgo empírico (Σ L(f(X), Y)), tipos de losses (MSE, cross-entropy), métricas no derivables (accuracy, F1, AUC).

4. **Descenso por gradiente**
   - *Justificación*: Método universal de optimización. Necesario antes de entender cualquier paradigma de aprendizaje, ya que todos lo utilizan. Conecta función de coste con aprendizaje efectivo.
   - *Contenido*: Gradient descent (batch, mini-batch, SGD), learning rate, optimizadores avanzados (ADAM, RMSprop), backpropagation.

5. **Aprendizaje supervisado: fundamentos**
   - *Justificación*: Paradigma más directo y común. Requiere entender los 4 componentes y optimización. Base para contrastar con otros paradigmas.
   - *Contenido*: Clasificación, regresión, datos etiquetados, coste de etiquetado, baseline obligatorio.

6. **Preprocesado general de datos**
   - *Justificación*: Antes de técnicas específicas de finanzas, dominar tratamiento general de datos faltantes, escalado, outliers. La calidad de datos determina calidad del modelo.
   - *Contenido*: Imputación, normalización/estandarización, detección de outliers, partición básica.

7. **Especificidades de datos financieros**
   - *Justificación*: Introduce las diferencias críticas: retornos logarítmicos, look-ahead bias, autocorrelación temporal, naturaleza no-IID. Fundamental antes de técnicas avanzadas.
   - *Contenido*: Retornos logarítmicos, fuentes de datos financieros, feature engineering financiero, advertencias sobre leakage.

8. **Estructuración avanzada de datos financieros**
   - *Justificación*: Técnicas específicas que sincronizan muestreo con información de mercado, no con tiempo cronológico. Resuelve problemas fundamentales de datos financieros.
   - *Contenido*: Barras basadas en información (TIBs, VIBs, DIBs, TRBs), ETF Trick, CUSUM Filter.

9. **Labeling y manejo de datos no-IID**
   - *Justificación*: Los datos financieros tienen overlapping outcomes; requiere técnicas especiales de etiquetado y ponderación. Crítico para validación posterior.
   - *Contenido*: Triple Barrera, Meta-Labeling, Average Uniqueness, Sequential Bootstrap, Return Attribution Weighting.

10. **Diferenciación Fraccionaria (FFD)**
    - *Justificación*: Resuelve el dilema estacionariedad vs memoria, fundamental para series temporales financieras. Permite aplicar técnicas ML estándar a datos financieros.
    - *Contenido*: Fixed-Width Window Fracdiff (FFD), determinación de d óptimo, aplicación a futuros.

11. **Validación temporal y control de leakage**
    - *Justificación*: K-fold estándar falla en finanzas; requiere Walk Forward, purging y embargoing. Esencial para estimación honesta de generalización.
    - *Contenido*: PurgedKFold, embargoing, Walk Forward, prevención de leakage por correlación serial.

12. **Aprendizaje no supervisado y Autoencoders**
    - *Justificación*: Introduce el concepto de aprender sin etiquetas. Los autoencoders ilustran cómo la arquitectura (cuello de botella) puede forzar aprendizaje de representaciones útiles.
    - *Contenido*: Clustering, reducción de dimensionalidad, autoencoders, embeddings, representaciones latentes.

13. **Inspiración biológica: Hubel y Wiesel → Filtros convolucionales**
    - *Justificación*: Contexto histórico que conecta neurociencia con arquitecturas de Deep Learning. Prepara para entender diseño de redes especializadas.
    - *Contenido*: Experimentos de Hubel y Wiesel, detección de bordes, filtros convolucionales, jerarquía de características.

14. **Deep Learning: integración end-to-end**
    - *Justificación*: Diferencia arquitectural clave respecto a ML clásico. Requiere entender supervisado/no supervisado y descenso por gradiente para apreciar la optimización conjunta de capas.
    - *Contenido*: ML vs DL, optimización end-to-end, transfer learning, cuándo usar cada paradigma.

15. **Hardware: GPUs, TPUs y limitación de memoria**
    - *Justificación*: Consideraciones prácticas críticas para implementación. Explica por qué ciertos modelos son viables y otros no. Conecta capacidades algorítmicas con viabilidad económica.
    - *Contenido*: GPUs vs TPUs, memoria como cuello de botella, CUDA, edge computing, coste de infraestructura.

16. **Regularización y control de overfitting en entrenamiento**
    - *Justificación*: Técnicas para entrenar modelos controlando overfitting en conjunto de entrenamiento. Complementa validación cruzada.
    - *Contenido*: L1/L2 regularization, dropout, batch normalization, early stopping, bias-variance trade-off.

17. **Ensemble Methods adaptados a finanzas**
    - *Justificación*: Bagging y Random Forest requieren ajustes por observation redundancy en datos financieros. Reducen varianza y mejoran generalización.
    - *Contenido*: Sequential bootstrapping, ajuste de max_samples por uniqueness, class weights, bagging, random forest.

18. **Feature Importance robusta**
    - *Justificación*: Supera limitaciones de p-valores; esencial para interpretabilidad y selección de features. López de Prado: "Feature importance is a research tool, backtesting is not."
    - *Contenido*: MDI, MDA, SFI, Clustered Feature Importance (CFI), control de substitution effects.

19. **Denoising de matrices de covarianza**
    - *Justificación*: Random Matrix Theory permite separar señal de ruido; reduce RMSE 60-95%. Fundamental para optimización de portafolios.
    - *Contenido*: Teorema Marcenko-Pastur, Constant Residual Eigenvalue, Targeted Shrinkage, Detoning.

20. **Métricas de distancia basadas en información**
    - *Justificación*: Correlación solo captura dependencia lineal; entropía detecta relaciones no lineales. Base para clustering óptimo.
    - *Contenido*: Entropía de Shannon, Información Mutua, Variación de Información, discretización óptima.

21. **Clustering óptimo**
    - *Justificación*: Identifica estructuras latentes en datos; base para CFI y NCO. Requiere métricas de distancia robustas.
    - *Contenido*: ONC (Optimal Number of Clusters), clustering jerárquico vs particional, matrices de proximidad.

22. **Construcción robusta de portafolios (NCO)**
    - *Justificación*: NCO resuelve la maldición de Markowitz mediante optimización jerárquica. Integra denoising, clustering y optimización.
    - *Contenido*: Número de condición, NCO (Nested Clustered Optimization), optimización intra/inter-cluster.

23. **Meta-Labeling y Bet Sizing**
    - *Justificación*: Traduce predicciones de ML en tamaños de posición concretos. Filtra falsos positivos y mejora F1-score.
    - *Contenido*: Modelo primario + secundario, strategy-independent sizing, sizing from probabilities, averaging active bets, discretization.

24. **Control de overfitting en conjunto de prueba**
    - *Justificación*: Múltiples backtests inflan errores Tipo I; requiere control estadístico riguroso. Específico de finanzas por iteración de estrategias.
    - *Contenido*: Deflated Sharpe Ratio, FWER, corrección de Šidàk, False Strategy Theorem, CPCV.

25. **Modelos generativos: determinismo y temperatura**
    - *Justificación*: Desmitifica la "creatividad" de modelos como ChatGPT. Requiere entender modelos autorregresivos y el rol de hiperparámetros.
    - *Contenido*: Temperatura, determinismo, modelos autorregresivos, tokens, limitaciones fundamentales.

26. **ChatGPT: arquitectura, entrenamiento y limitaciones**
    - *Justificación*: Caso práctico completo que integra tokens, temperatura, coste computacional, RLHF. Ilustra la brecha entre marketing y realidad técnica.
    - *Contenido*: Arquitectura transformer, RLHF, prompt del sistema, coste de infraestructura (€300,000), limitaciones de contexto.

27. **Datos sintéticos: potencial y limitaciones**
    - *Justificación*: Aplicación práctica de modelos generativos. El experimento (mejora con 2,000, empeora con 20,000) es lección crítica sobre calidad vs cantidad.
    - *Contenido*: Generación con LLMs, limitaciones fundamentales, experimento de degradación, nunca igualan originales.

28. **RAG (Retrieval Augmented Generation)**
    - *Justificación*: Técnica para superar limitaciones de contexto en LLMs. Requiere entender embeddings y modelos generativos.
    - *Contenido*: Fragmentación de documentos, embeddings, recuperación, limitaciones por pérdida de información.

29. **Agentes de IA y MCP**
    - *Justificación*: Nivel de abstracción superior que combina múltiples modelos. Representa la frontera actual de sistemas complejos.
    - *Contenido*: Orquestación de modelos, MCP (protocolo de comunicación), N8N, OpenClaude.

30. **Aprendizaje por refuerzo: teoría y advertencias prácticas**
    - *Justificación*: Paradigma más complejo y problemático. Se deja para el final porque requiere entender todos los conceptos previos para apreciar sus desafíos (catastrophic forgetting, dependencia de simuladores, aleatoriedad). Las advertencias sobre dificultad en producción son críticas.
    - *Contenido*: Función de recompensa, simuladores, RLHF, FinRL, catastrophic forgetting, limitaciones en producción.

31. **Meta-Estrategia y organización industrial**
    - *Justificación*: Integra todo en un paradigma organizacional que evita el "paradigma de Sísifo". Muestra cómo estructurar proyectos reales de ML financiero.
    - *Contenido*: Cadena de producción (Data Curators → Feature Analysts → Strategists → Backtesters → Deployment → Portfolio Oversight), Cursus Honorum.

32. **Baseline obligatorio y realismo en finanzas**
    - *Justificación*: Lección metodológica transversal. El caso de "buy and hold" ganando al modelo de refuerzo ilustra la importancia de comparaciones honestas. Cierre conceptual del bloque.
    - *Contenido*: Importancia de baselines simples, ejemplo TFM de Pablo, Renaissance Technologies y "datos limpios desde el origen".

## Patrones y observaciones

### 1. Tensión fundamental: Teoría vs Predicción vs Realidad Práctica
Los documentos convergen en tres niveles de tensión:

- **Teoría (Riesgo de Bayes) vs Predicción (Riesgo Empírico)**: López de Prado (2020) enfatiza: "Backtesting is not a research tool. Feature importance is." El ML debe usarse para **descubrir teorías** (identificar variables ocultas, relaciones complejas), no como oráculo predictivo. Esta filosofía permea todo el bloque: las técnicas más sofisticadas (denoising, CFI, NCO) buscan revelar estructura subyacente, no maximizar métricas de backtest.

- **Predicción vs Realidad Práctica**: El bloque enfatiza repetidamente la brecha entre capacidades teóricas y realidad de implementación. Ejemplos: aprendizaje por refuerzo (prometedor pero "extremadamente difícil en producción"), datos sintéticos (mejoran hasta ~2,000, luego degradan con 20,000), computación cuántica (algoritmos eficientes pero hardware limitado), modelos generativos (deterministas, no creativos). Esta tensión prepara expectativas realistas para el TFM.

- **Implicación práctica**: El flujo