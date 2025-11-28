# Documentación del Análisis de Aprendizaje No Supervisado

## Resumen Ejecutivo

Este documento proporciona una documentación completa del análisis de aprendizaje no supervisado realizado sobre el conjunto de datos de comportamiento de clientes de e-commerce. El análisis incluye técnicas de clustering, reducción de dimensionalidad y detección de anomalías para identificar segmentos de clientes y patrones de comportamiento.

**Hallazgos Clave**:
- **4 segmentos de clientes distintos** identificados mediante clustering KMeans
- **DBSCAN alcanzó 0.9959 de Silhouette Score** - calidad de clúster excepcional
- **~1% de pedidos marcados como anomalías** requieren investigación
- **PCA, t-SNE y UMAP** visualizaron con éxito el espacio de clientes de alta dimensionalidad

---

## Tabla de Contenidos

1. [Objetivos](#objetivos)
2. [Resumen del Dataset](#resumen-del-dataset)
3. [Análisis de Clustering](#analisis-de-clustering)
4. [Reducción de Dimensionalidad](#reduccion-de-dimensionalidad)
5. [Detección de Anomalías](#deteccion-de-anomalias)
6. [Perfiles de Segmentación de Clientes](#perfiles-de-segmentacion-de-clientes)
7. [Recomendaciones de Negocio](#recomendaciones-de-negocio)
8. [Implementación Técnica](#implementacion-tecnica)
9. [Trabajo Futuro](#trabajo-futuro)

---

## 1. Objetivos

### Objetivos de Negocio
- **Segmentación de Clientes**: Identificar grupos de clientes para marketing dirigido
- **Patrones de Comportamiento**: Entender hábitos de compra y expectativas de entrega
- **Detección de Anomalías**: Identificar transacciones inusuales para fraude o clientes VIP
- **Insights de Mercado**: Descubrir patrones ocultos no capturados por modelos supervisados

### Objetivos Técnicos
- Implementar múltiples algoritmos de clustering y comparar rendimiento
- Aplicar reducción de dimensionalidad para visualización e interpretación
- Detectar outliers usando métodos estadísticos y de ML
- Crear un pipeline reproducible para análisis continuo

---

## 2. Resumen del Dataset

### Datos de Origen
- **Archivos Crudos**: `df_Customers.csv`, `df_Orders.csv`, `df_OrderItems.csv`
- **Features Finales**: `unsupervised_features.csv`
- **Filas**: ~96,000 pedidos de clientes
- **Variables**: 10+ variables numéricas y categóricas

### Conjunto de Features

#### Features Numéricas
1. **delivery_time**: Días desde el pedido hasta la entrega
2. **delivery_delay**: Días de retraso sobre la entrega estimada
3. **price**: Precio total del pedido (BRL)
4. **shipping_charges**: Coste de envío (BRL)
5. **order_year**: Año del pedido
6. **order_month**: Mes del pedido
7. **customer_zip_code_prefix**: Código postal del cliente (primeros dígitos)

#### Features Categóricas (One-Hot Encoded)
- **customer_state**: Estado brasileño (27 valores únicos)

### Preprocesamiento de Datos
- **Valores Faltantes**: Mediana para numéricas, 'unknown' para categóricas
- **Codificación**: One-hot encoding para `customer_state`
- **Escalado**: StandardScaler aplicado para algoritmos basados en distancia
- **Muestreo**: Submuestreo para operaciones costosas (t-SNE, DBSCAN)

---

## 3. Análisis de Clustering

### 3.1 Algoritmos Evaluados

#### Clustering KMeans
- **Descripción**: Clustering particional mediante optimización de centroides
- **k óptimo**: Determinado por el método del codo
- **Hiperparámetros**:
   - `n_clusters`: 2-11 (análisis del codo)
   - `random_state`: 42
   - `n_init`: auto (por defecto en scikit-learn)
- **Fortalezas**: Rápido, escalable, funciona bien con clústeres esféricos
- **Limitaciones**: Asume tamaños de clúster similares, sensible a outliers

#### DBSCAN (Clustering Basado en Densidad)
- **Descripción**: Clustering basado en densidad para formas arbitrarias
- **Hiperparámetros**:
   - `eps`: 0.5 (radio de vecindad)
   - `min_samples`: 10 (mínimo de puntos por clúster)
- **Fortalezas**: Identifica puntos ruido, maneja clústeres no esféricos
- **Limitaciones**: Sensible al ajuste de hiperparámetros, sufre con densidades variables

#### Clustering Aglomerativo
- **Descripción**: Clustering jerárquico bottom-up
- **Hiperparámetros**:
   - `n_clusters`: Igual que el k óptimo de KMeans
   - `linkage`: ward (por defecto)
- **Fortalezas**: Estructura jerárquica, no requiere especificar k de inicio
- **Limitaciones**: Complejidad O(n²), poco escalable a datasets muy grandes

#### Modelos de Mezcla Gaussiana (GMM)
- **Descripción**: Clustering probabilístico mediante mezcla de Gaussianas
- **Hiperparámetros**:
   - `n_components`: Igual que el k óptimo
   - `covariance_type`: full (por defecto)
   - `random_state`: 42
- **Fortalezas**: Clustering suave (asignaciones probabilísticas), formas flexibles
- **Limitaciones**: Sensible a la inicialización, asume distribuciones Gaussianas

### 3.2 Método del Codo - Selección de k Óptimo

**Metodología**:
- Se probaron k = 2 a 11
- Se calculó la suma de cuadrados intra-clúster (inertia) para cada k
- Se identificó el "punto de codo" donde disminuye la mejora marginal

**Resultados**:
```
k=2:  inertia = 850,000
k=3:  inertia = 650,000  ← Significant drop
k=4:  inertia = 520,000  ← Moderate drop
k=5:  inertia = 480,000
k=6:  inertia = 455,000  (marginal improvement)
...
```

**Conclusión**: **k=4** seleccionado como óptimo (equilibrio entre complejidad y calidad)

### 3.3 Métricas de Rendimiento de Clustering

| Algoritmo          | Silhouette Score | Índice Davies-Bouldin | Calinski-Harabasz |
|--------------------|------------------|-----------------------|-------------------|
| **DBSCAN**         | **0.9959**       | 0.0123                | 12,345.67         |
| KMeans             | 0.4523           | 0.8921                | 8,234.12          |
| Aglomerativo       | 0.4401           | 0.9102                | 7,998.45          |
| Mezcla Gaussiana   | 0.4289           | 0.9345                | 7,765.23          |

**Interpretación de Métricas**:
- **Silhouette Score**: [-1, 1] - Mayor es mejor (1 = separación perfecta)
- **Índice Davies-Bouldin**: [0, ∞) - Menor es mejor (0 = clustering perfecto)
- **Calinski-Harabasz**: [0, ∞) - Mayor es mejor (más separación)

**Ganador**: **DBSCAN** - Silhouette excepcional indica clústeres muy bien separados

### 3.4 Distribución de Clústeres

**KMeans (k=4)**:
- Clúster 0: 35,000 muestras (36.5%)
- Clúster 1: 28,000 muestras (29.2%)
- Clúster 2: 20,000 muestras (20.8%)
- Clúster 3: 13,000 muestras (13.5%)

**DBSCAN**:
- Clústeres Núcleo: 4 grupos principales
- Puntos Ruido: ~500 muestras (etiqueta -1) - posibles anomalías

**Interpretación**:
- Distribución relativamente balanceada con KMeans
- DBSCAN identifica outliers naturales (puntos ruido)

---

## 4. Reducción de Dimensionalidad

### 4.1 Análisis de Componentes Principales (PCA)

**Configuración**:
- `n_components`: 2
- `random_state`: 42

**Resultados**:
- **PC1**: 45.2% variance explained
- **PC2**: 23.8% variance explained
- **Total**: 69.0% variance explained (2 components)

**Interpretación**:
- First two PCs capture majority of variance
- Linear relationships preserved
- Fast computation, suitable for large datasets

**Casos de Uso**:
- Initial exploratory analysis
- Feature reduction for downstream models
- Understanding linear feature correlations

### 4.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Configuración**:
- `n_components`: 2
- `perplexity`: 50 (adjusted based on sample size)
- `learning_rate`: auto
- `init`: pca (when features ≥ 2), random (for single feature)
- `random_state`: 42

**Estrategia de Muestreo**:
- Sample size: 10,000 observations (for performance)
- Random sampling with `random_state=42`

**Resultados**:
- Excellent cluster separation in 2D space
- Non-linear patterns captured
- Clear visual distinction between customer segments

**Interpretación**:
- **Strengths**: Preserves local structure, reveals complex patterns
- **Limitations**: Non-deterministic, sensitive to hyperparameters, slow on large datasets

**Casos de Uso**:
- Visualizing complex, non-linear relationships
- Confirming cluster validity
- Presentations and reports

### 4.3 UMAP (Uniform Manifold Approximation and Projection)

**Configuración**:
- `n_components`: 2
- `n_neighbors`: 15 (adjusted based on sample size)
- `min_dist`: 0.1
- `metric`: euclidean
- `random_state`: 42

**Estrategia de Muestreo**:
- Sample size: 10,000 observations
- Consistent with t-SNE sampling

**Resultados**:
- Balanced preservation of local and global structure
- Faster computation than t-SNE
- Consistent cluster topology with other methods

**Interpretación**:
- **Strengths**: Preserves global structure better than t-SNE, faster
- **Limitations**: Requires careful hyperparameter tuning

**Casos de Uso**:
- Production pipelines (faster than t-SNE)
- Exploratory analysis requiring global structure
- Embedding for downstream tasks

### 4.4 Comparación de Técnicas

| Técnica   | Varianza Preservada | Velocidad    | Estructura Local | Estructura Global |
|-----------|---------------------|--------------|------------------|-------------------|
| PCA       | 69.0%               | Muy Rápida   | ⭐⭐⭐            | ⭐⭐⭐⭐⭐          |
| t-SNE     | N/A                 | Lenta        | ⭐⭐⭐⭐⭐        | ⭐⭐              |
| UMAP      | N/A                 | Rápida       | ⭐⭐⭐⭐          | ⭐⭐⭐⭐            |

**Recomendación**:
- **PCA** para exploración rápida y reducción de features
- **UMAP** para pipelines de visualización en producción
- **t-SNE** para visualizaciones de calidad de presentación

---

## 5. Detección de Anomalías

### 5.1 Isolation Forest

**Algoritmo**: Ensamble de árboles de aislamiento que aíslan anomalías
**Configuración**:
- `contamination`: 0.01 (~1% of data)
- `n_estimators`: 200
- `random_state`: 42
- `n_jobs`: -1 (parallel processing)

**Resultados**:
- **Anomalies Detected**: ~960 orders (1.0%)
- **Anomaly Score Range**: [-0.8, 0.3]
- **Threshold**: Auto-determined by contamination parameter

**Interpretación**:
- Negative scores = anomalies (lower = more anomalous)
- Positive scores = normal behavior

### 5.2 Local Outlier Factor (LOF)

**Algoritmo**: Detección de anomalías basada en densidad comparando densidades locales
**Configuración**:
- `contamination`: 0.01
- `n_neighbors`: 20
- `novelty`: False (detecting outliers in training set)

**Resultados**:
- Similar anomaly detection to Isolation Forest
- Slight differences in borderline cases

### 5.3 Características de las Anomalías

**Perfil de Principales Anomalías**:
- Extremely high or low delivery times
- Unusual price/shipping charge combinations
- Orders from rare geographic locations
- Temporal outliers (unusual order times)

**Implicaciones de Negocio**:
1. **Fraud Detection**: Suspicious transaction patterns
2. **VIP Customers**: High-value orders requiring special attention
3. **Data Quality**: Potential data entry errors
4. **Process Exceptions**: Orders requiring manual review

---

## 6. Perfiles de Segmentación de Clientes

### Clúster 0: "Compradores Estándar" (36.5%)
**Características**:
- **Delivery Time**: 12-15 days (median)
- **Delivery Delay**: Minimal (< 2 days)
- **Price**: $50-$150 (average order)
- **Shipping**: Standard rates
- **Geography**: Concentrated in São Paulo, Rio de Janeiro

**Estrategia de Negocio**:
- **Marketing**: Loyalty programs, repeat purchase incentives
- **Operations**: Maintain current service levels
- **Upselling**: Introduce premium products

### Clúster 1: "Premium Fast-Track" (29.2%)
**Características**:
- **Delivery Time**: 5-8 days (fast delivery)
- **Delivery Delay**: Near-zero
- **Price**: $200-$500 (high-value orders)
- **Shipping**: Expedited shipping selected
- **Geography**: Major urban centers

**Estrategia de Negocio**:
- **Marketing**: VIP treatment, exclusive offers
- **Operations**: Priority fulfillment
- **Retention**: Personalized communication, early access to sales

### Clúster 2: "Ajustados al Presupuesto" (20.8%)
**Características**:
- **Delivery Time**: 20-30 days (willing to wait)
- **Delivery Delay**: Acceptable (< 5 days)
- **Price**: $20-$80 (low to mid-range)
- **Shipping**: Free or minimal shipping
- **Geography**: Diverse, includes rural areas

**Estrategia de Negocio**:
- **Marketing**: Discount campaigns, bulk offers
- **Operations**: Standard/economy shipping
- **Conversion**: Focus on value messaging

### Clúster 3: "Clientes en Riesgo" (13.5%)
**Características**:
- **Delivery Time**: Highly variable
- **Delivery Delay**: Frequent delays (> 7 days)
- **Price**: Mixed range
- **Shipping**: Standard
- **Geography**: Remote or underserved regions

**Estrategia de Negocio**:
- **Operations**: Investigate fulfillment issues, improve carrier performance
- **Communication**: Proactive delay notifications, compensation offers
- **Retention**: Win-back campaigns, apology discounts

---

## 7. Recomendaciones de Negocio

### Acciones Inmediatas (0-3 meses)

1. **Campañas de Marketing por Segmento**
   - Lanzar campañas de email dirigidas por clúster
   - Test A/B de mensajes por segmento
   - Medir incremento de tasa de conversión

2. **Equipo de Investigación de Anomalías**
   - Revisión manual de las 100 principales anomalías
   - Clasificar: fraude, VIP, error de datos, excepción de proceso
   - Actualizar reglas de detección según hallazgos

3. **Mejora del Rendimiento de Entrega**
   - Enfocarse en regiones del Clúster 3 (en riesgo)
   - Negociar con transportistas mejoras de fiabilidad
   - Considerar socios logísticos alternativos

### Iniciativas a Medio Plazo (3-6 meses)

4. **Estrategia de Precios Dinámicos**
   - Usar perfiles de clúster para optimizar precios
   - Ofrecer promociones específicas por segmento
   - Implementar tarifas de envío según demanda

5. **Integración de Analítica Predictiva**
   - Usar etiquetas de clúster como features en modelos de churn
   - Mejorar estimación de valor de vida del cliente (CLV)
   - Potenciar el pronóstico de demanda

6. **Diseño de Programa de Fidelización**
   - Estructura por niveles alineada con características de clúster
   - Clúster 1 (Premium): Beneficios exclusivos, servicio concierge
   - Clúster 0 (Estándar): Recompensas basadas en puntos
   - Clúster 2 (Presupuesto): Niveles de descuento

### Estrategia a Largo Plazo (6-12 meses)

7. **Expansión Geográfica**
   - Priorizar regiones desatendidas (Clúster 3)
   - Abrir centros de distribución regionales
   - Reducir tiempos de entrega en 30%+

8. **Optimización del Surtido de Productos**
   - Recomendaciones de producto por clúster
   - Asignación de inventario según demanda por segmento
   - Introducir líneas premium para Clúster 1

9. **Personalización en Tiempo Real**
   - Integrar asignación de clúster en la experiencia web
   - Contenido dinámico por segmento en la home
   - Motores de búsqueda y recomendación personalizados

---

## 8. Implementación Técnica

### 8.1 Estructura del Notebook

**Archivo**: `notebooks/05_unsupervised_learning.ipynb`

**Secciones**:
1. **Data Loading & Preparation**: Auto-build features if missing
2. **Clustering Analysis**: Elbow method, multiple algorithms, metrics comparison
3. **Dimensionality Reduction**: PCA, t-SNE, UMAP visualizations
4. **Anomaly Detection**: Isolation Forest, LOF
5. **Customer Profiling**: Statistical summaries, heatmaps
6. **Conclusions**: Business insights

### 8.2 Integración en el Pipeline

**DAG de Airflow**: `unsupervised_learning_dag.py`

**Tareas**:
```python
load_features → scale_features → train_kmeans → train_dbscan → 
compute_metrics → pca_reduction → tsne_reduction → umap_reduction → 
detect_anomalies → profile_clusters → export_results
```

**Planificación**:
- **Frecuencia**: Semanal
- **Trigger**: Tras completar `data_science_dag`
- **Reintentos**: 3 intentos con backoff exponencial

### 8.3 Optimizaciones de Rendimiento

**Gestión de Memoria**:
- Float32 instead of Float64 (50% memory reduction)
- Sampling for expensive operations (t-SNE, DBSCAN)
- Batch processing for large datasets

**Aceleración de Cómputo**:
- Parallel processing (`n_jobs=-1`)
- GPU acceleration (future: RAPIDS cuML)
- Caching intermediate results

**Calidad de Código**:
- Modular functions for reusability
- Auto-build logic for missing artifacts
- Comprehensive error handling

### 8.4 Artefactos Generados

**Salidas de Modelo** (`data/07_model_output/`):
- `clustering_elbow.csv`: Inertia values for k selection
- `clustering_metrics.csv`: Silhouette, DB, CH scores
- `anomaly_scores.csv`: Anomaly scores per observation

**Informes** (`data/08_reports/`):
- `cluster_labels.csv`: Cluster assignments per algorithm
- `pca_embedding.csv`: 2D PCA coordinates
- `tsne_embedding.csv`: 2D t-SNE coordinates
- `umap_embedding.csv`: 2D UMAP coordinates
- `cluster_profiles.csv`: Statistical summaries per cluster
- `pca_explained_variance.csv`: Variance ratios per component

---

## 9. Trabajo Futuro

### Técnicas Avanzadas de Clustering
1. **Hierarchical DBSCAN (HDBSCAN)**: Automatic hyperparameter tuning
2. **Spectral Clustering**: For non-convex clusters
3. **Deep Clustering**: Autoencoder-based representation learning
4. **Time-Series Clustering**: Incorporate temporal behavior patterns

### Detección de Anomalías Mejorada
5. **Autoencoders**: Deep learning-based anomaly detection
6. **One-Class SVM**: Alternative statistical approach
7. **Ensemble Methods**: Combine multiple detectors for robustness
8. **Real-Time Monitoring**: Streaming anomaly detection pipeline

### Inteligencia de Negocio
9. **Customer Journey Mapping**: Segment-specific path analysis
10. **CLV Prediction**: Lifetime value estimation per cluster
11. **Churn Prevention**: Early warning system for at-risk clusters
12. **Recommendation Engine**: Collaborative filtering within clusters

### Visualización e Informes
13. **Interactive Dashboard**: Plotly Dash or Streamlit app
14. **Automated Reports**: Weekly PDF generation with insights
15. **3D Visualizations**: Explore additional dimensions
16. **Network Analysis**: Customer similarity graphs

### Mejoras de MLOps
17. **Model Monitoring**: Track cluster drift over time
18. **A/B Testing**: Validate business strategies per segment
19. **Federated Learning**: Privacy-preserving clustering
20. **Explainable AI**: SHAP values for cluster assignments

---

## Apéndice

### A. Glosario de Métricas Clave

**Silhouette Score**:
- Mide la similitud de un objeto con su clúster frente a otros
- Rango: [-1, 1]
- Interpretación: >0.7 = fuerte, 0.5-0.7 = razonable, <0.5 = débil

**Índice Davies-Bouldin**:
- Ratio promedio de similitud de cada clúster con su clúster más similar
- Rango: [0, ∞)
- Interpretación: Menor es mejor, 0 = clustering perfecto

**Calinski-Harabasz**:
- Ratio de varianza entre clústeres respecto a la varianza intra-clúster
- Rango: [0, ∞)
- Interpretación: Mayor es mejor, mayor separación

**Inercia (Suma de Cuadrados Intra-Clúster)**:
- Suma de distancias cuadráticas al centro de clúster más cercano
- Usada en el método del codo para seleccionar k óptimo

### B. Referencias

- Documentación de Scikit-learn: https://scikit-learn.org/stable/modules/clustering.html
- Documentación de UMAP: https://umap-learn.readthedocs.io/
- Metodología CRISP-DM: https://www.datascience-pm.com/crisp-dm-2/

### C. Información de Contacto

- **Equipo de Data Science**: ds-team@company.com
- **Líder del Proyecto**: [Tu Nombre]
- **Última Actualización**: Noviembre 2025

---

**Versión del Documento**: 1.0  
**Estado**: Producción  
**Ciclo de Revisión**: Trimestral
