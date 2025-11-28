# Arquitectura del Proyecto - Pipeline de ML

## Visión General

Este documento describe la arquitectura del pipeline de Machine Learning para el proyecto de análisis del comportamiento de clientes de e-commerce. El proyecto sigue una estructura modular y lista para producción, utilizando buenas prácticas de la industria.

## Stack Tecnológico

### Tecnologías Principales
- **Python 3.11+**: Lenguaje de programación principal
- **Kedro**: Orquestación de pipelines y framework de ingeniería de datos
- **Apache Airflow**: Planificación y monitorización de workflows
- **Docker & Docker Compose**: Contenerización y despliegue
- **DVC (Data Version Control)**: Versionado de datos y modelos

### Bibliotecas de ML y Ciencia de Datos
- **scikit-learn**: Algoritmos de aprendizaje (supervisado y no supervisado)
- **pandas**: Manipulación y análisis de datos
- **numpy**: Cálculo numérico
- **matplotlib, seaborn, plotly**: Visualización de datos
- **UMAP, t-SNE**: Técnicas de reducción de dimensionalidad

### Testing y Calidad
- **pytest**: Pruebas unitarias e integración
- **pytest-cov**: Informes de cobertura de código

## Estructura del Proyecto

```
proyecto-ml/
├── airflow/                      # Configuración de Apache Airflow
│   ├── dags/                     # Definiciones de DAG
│   │   ├── data_engineering_dag.py
│   │   ├── data_science_dag.py
│   │   ├── unsupervised_learning_dag.py
│   │   ├── reporting_dag.py
│   │   └── master_pipeline_dag.py
│   ├── logs/                     # Logs de ejecución de Airflow
│   └── plugins/                  # Plugins personalizados de Airflow
│
├── conf/                         # Archivos de configuración (Kedro)
│   ├── base/                     # Configuración base
│   │   ├── catalog.yml           # Definiciones del catálogo de datos
│   │   ├── parameters.yml        # Parámetros del pipeline
│   │   └── logging.yml           # Configuración de logging
│   └── local/                    # Overrides locales
│
├── data/                         # Capas de datos (convención Kedro)
│   ├── 01_raw/                   # Datos crudos (inmutables)
│   ├── 02_intermediate/          # Datos limpiados
│   ├── 03_primary/               # Conjuntos de datos primarios
│   ├── 05_model_input/           # Features listas para ML
│   ├── 06_models/                # Artefactos de modelos entrenados
│   ├── 07_model_output/          # Predicciones y métricas
│   └── 08_reports/               # Informes y visualizaciones
│
├── notebooks/                    # Jupyter notebooks (análisis y exploración)
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_modeling_regression.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_final_analysis.ipynb
│
├── src/                          # Código fuente (Kedro pipelines y nodos)
│   └── proyecto_ml/
│       ├── pipelines/            # Definiciones modulares de pipelines
│       │   ├── data_engineering/
│       │   ├── data_science/
│       │   ├── unsupervised_learning/
│       │   └── reporting/
│       └── nodes/                # Nodos de procesamiento individuales
│
├── tests/                        # Suite de tests
│   ├── unit/
│   └── integration/
│
├── docs/                         # Documentación
│   ├── architecture.md           # This file
│   └── unsupervised_analysis.md  # Documentación de aprendizaje no supervisado
│
├── docker-compose.yml            # Orquestación multi-contenedor
├── Dockerfile                    # Definición de contenedor
├── dvc.yaml                      # Definición de pipeline DVC
├── pyproject.toml                # Metadatos y dependencias del proyecto
├── requirements.txt              # Dependencias de Python
└── README.md                     # Resumen del proyecto
```

## Arquitectura de Flujo de Datos

### Procesamiento por Capas

```
01_raw → 02_intermediate → 03_primary → 05_model_input → 06_models → 07_model_output → 08_reports
```

#### 1. Capa de Datos Crudos (`01_raw/`)
- **Origen**: Archivos CSV originales de la plataforma de e-commerce
- **Archivos**: `df_Customers.csv`, `df_Orders.csv`, `df_OrderItems.csv`
- **Características**: Inmutables, nunca se modifican
- **Versionado**: Seguimiento con DVC

#### 2. Capa Intermedia (`02_intermediate/`)
- **Propósito**: Limpieza de datos y controles de calidad
- **Transformaciones**:
  - Manejo de valores faltantes
  - Eliminación de duplicados
  - Corrección de tipos de datos
  - Validación básica
- **Archivos**: `customers_clean.csv`, `orders_clean.csv`, `order_items_clean.csv`

#### 3. Capa Primaria (`03_primary/`)
- **Propósito**: Integración de datos y lógica de negocio
- **Transformaciones**:
  - Unión de múltiples tablas
  - Creación de variables derivadas (p. ej., delivery_time, delivery_delay)
  - Aplicación de reglas de negocio
- **Archivos**: `full_dataset.csv`, `full_orders.csv`

#### 4. Capa de Entrada al Modelo (`05_model_input/`)
- **Propósito**: Ingeniería de características lista para ML
- **Transformaciones**:
  - Codificación de features (one-hot, label encoding)
  - División train/test
  - Escalado/normalización de características
  - Integración de etiquetas de clúster
- **Archivos**:
  - `model_input.csv`, `model_input_encoded.csv`
  - `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
  - `unsupervised_features.csv`, `cluster_labels.csv`

#### 5. Capa de Modelos (`06_models/`)
- **Propósito**: Modelos entrenados serializados
- **Formato**: Archivos Pickle (`.pkl`)
- **Versionado**: Seguimiento con DVC
- **Modelos**: Regresión, Clasificación, Clustering, Detección de Anomalías

#### 6. Capa de Salida del Modelo (`07_model_output/`)
- **Propósito**: Predicciones y métricas de evaluación
- **Archivos**:
  - `model_metrics.csv` (métricas de regresión)
  - `classification_metrics.csv`
  - `clustering_metrics.csv`
  - `anomaly_scores.csv`
  - `feature_importances.csv`

#### 7. Capa de Informes (`08_reports/`)
- **Propósito**: Insights de negocio y visualizaciones
- **Archivos**:
  - `cluster_profiles.csv`
  - `pca_embedding.csv`, `tsne_embedding.csv`, `umap_embedding.csv`
  - `correlation_spearman.csv`
  - `model_performance_summary.txt`

## Arquitectura de Pipelines

### Módulos de Pipeline

#### 1. Pipeline de Ingeniería de Datos
- **DAG**: `data_engineering_dag.py`
- **Pasos**:
  1. Cargar datos crudos
  2. Limpiar y validar
  3. Unir datasets
  4. Ingeniería de características
  5. Exportar a la capa primaria

#### 2. Pipeline de Ciencia de Datos (Aprendizaje Supervisado)
- **DAG**: `data_science_dag.py`
- **Pasos**:
  1. Cargar entrada del modelo
  2. División train/test
  3. Entrenar modelos de regresión
  4. Entrenar modelos de clasificación
  5. Evaluar y guardar métricas

#### 3. Pipeline de Aprendizaje No Supervisado
- **DAG**: `unsupervised_learning_dag.py`
- **Pasos**:
  1. Preparación y escalado de características
  2. Clustering (KMeans, DBSCAN, Aglomerativo, GMM)
  3. Reducción de dimensionalidad (PCA, t-SNE, UMAP)
  4. Detección de anomalías (Isolation Forest, LOF)
  5. Perfilado de clústeres

#### 4. Pipeline de Reporting
- **DAG**: `reporting_dag.py`
- **Pasos**:
  1. Agregar métricas
  2. Generar visualizaciones
  3. Crear resúmenes de negocio
  4. Exportar informes

#### 5. Pipeline Maestro
- **DAG**: `master_pipeline_dag.py`
- **Propósito**: Orquesta todos los sub-pipelines en secuencia
- **Schedule**: Diario/semanal (configurable)

## Orquestación con Apache Airflow

### Configuración de DAGs
- **Ejecución**: Secuencial con gestión de dependencias
- **Monitorización**: Web UI en `http://localhost:8080`
- **Alertas**: Notificaciones por email en caso de fallo
- **Reintentos**: Reintentos automáticos con backoff exponencial

### Dependencias de Tareas

```
data_engineering_dag → data_science_dag → unsupervised_learning_dag → reporting_dag
```

## Arquitectura de Despliegue

### Contenedores Docker
- **app**: Aplicación principal de ML
- **airflow-webserver**: UI de Airflow
- **airflow-scheduler**: Planificador de DAGs
- **postgres**: Base de datos de metadatos (opcional)

### Variables de Entorno
- `AIRFLOW_HOME`: Directorio de configuración de Airflow
- `PYTHONPATH`: Ruta de búsqueda de módulos Python
- `DATA_DIR`: Ubicación de almacenamiento de datos

### Volúmenes (Docker Compose)
- `./data:/app/data`: Almacenamiento persistente de datos
- `./airflow/dags:/opt/airflow/dags`: Definiciones de DAGs
- `./airflow/logs:/opt/airflow/logs`: Logs de ejecución

## Escalabilidad y Buenas Prácticas

### Optimización de Rendimiento
1. **Muestreo**: Datasets grandes se muestrean para operaciones costosas (t-SNE, DBSCAN)
2. **Procesamiento en Paralelo**: Tareas independientes corren en paralelo cuando es posible
3. **Caché**: Resultados intermedios en caché para evitar recomputación
4. **Gestión de Memoria**: Se utiliza Float32 en lugar de Float64 para arrays grandes

### Calidad de Código
- **Diseño Modular**: Nodos y pipelines reutilizables
- **Type Hints**: Tipado estático para claridad
- **Documentación**: Docstrings en todas las funciones
- **Testing**: Pruebas unitarias con pytest (objetivo: >80% cobertura)

### Control de Versiones
- **Git**: Versionado del código fuente
- **DVC**: Versionado de datos y modelos
- **Git LFS**: Almacenamiento de archivos grandes (opcional)

## Metodología de Machine Learning

### Fases CRISP-DM (Reflejadas en los Notebooks)
1. **Comprensión del Negocio** (`01_business_understanding.ipynb`)
2. **Comprensión de los Datos** (`02_data_understanding.ipynb`)
3. **Preparación de los Datos** (`03_data_preparation.ipynb`)
4. **Modelado** (`04_modeling_regression.ipynb`, `05_unsupervised_learning.ipynb`)
5. **Evaluación** (`06_final_analysis.ipynb`)
6. **Despliegue** (DAGs de Airflow + Docker)

### Tipos de Modelos Implementados

#### Aprendizaje Supervisado
- **Regresión**: Predecir tiempo de entrega, valor de pedido
  - Regresión Lineal
  - Random Forest Regressor
  - Gradient Boosting Regressor
- **Clasificación**: Predecir churn de clientes, estado de pedido
  - Regresión Logística
  - Random Forest Classifier
  - XGBoost

#### Aprendizaje No Supervisado
- **Clustering**: Segmentación de clientes
  - KMeans (k óptimo vía método del codo)
  - DBSCAN (basado en densidad)
  - Clustering Aglomerativo (jerárquico)
  - Gaussian Mixture Models
- **Reducción de Dimensionalidad**: Visualización
  - PCA (lineal)
  - t-SNE (no lineal, estructura local)
  - UMAP (no lineal, estructura global + local)
- **Detección de Anomalías**: Identificación de outliers
  - Isolation Forest
  - Local Outlier Factor (LOF)

## Monitorización y Mantenimiento

### Seguimiento de Métricas
- **Rendimiento de Modelos**: Registrado en `07_model_output/`
- **Salud del Pipeline**: Logs de Airflow en `airflow/logs/`
- **Calidad de Datos**: Chequeos de validación en el pipeline de ingeniería de datos

### Frecuencia de Actualización
- **Diaria**: Actualizaciones incrementales de datos
- **Semanal**: Reentrenamiento de modelos
- **Mensual**: Validación completa del pipeline
- **Trimestral**: Revisión de arquitectura

## Mejoras Futuras

1. **Integración MLOps**:
  - MLflow para seguimiento de experimentos
  - Registro de modelos para despliegue en producción
  - Framework de A/B testing

2. **Funcionalidades Avanzadas**:
  - Modelos de deep learning (redes neuronales)
  - API de predicción en tiempo real (FastAPI)
  - Tuning automático de hiperparámetros (Optuna)

3. **Infraestructura**:
  - Despliegue en Kubernetes
  - Almacenamiento en la nube (S3, GCS)
  - Pipeline de CI/CD (GitHub Actions)

## Contacto y Recursos

- **Repositorio**: [Enlace a GitHub]
- **Documentación**: Carpeta `docs/`
- **Issues**: GitHub Issues
- **Contribuidores**: Ver `README.md`

---

**Última Actualización**: Noviembre 2025  
**Versión**: 1.0  
**Mantenedor**: Equipo de ML
