# proyecto_ml

(admin - admin) <-- credenciales airflow ui

https://drive.google.com/file/d/10cF-siIP8vxZaxebEAVBVGos25U1H-7E/view?usp=sharing --> video
                                                                                   --> video 2

Proyecto Kedro para procesamiento y modelado de datos del dominio de pedidos/clientes, implementando etapas de CRISP-DM (comprensión de datos, preparación, modelado) y reportes reproducibles. Este README reúne los pasos y comandos clave para ponerlo en marcha rápidamente en Windows (PowerShell) y fusiona la guía de ejecución, DVC y Airflow.

## Índice
- Requisitos e Instalación
- Uso rápido (Kedro / Docker+Airflow / DVC)
- Estructura de datos
- Ejecución de pipelines y nodos
- Pipelines del proyecto (resumen)
- Verificación de resultados esperados
- DVC: reproducibilidad y métricas
- Airflow: orquestación con Docker
- Pruebas
- Troubleshooting
- Comandos útiles y Checklist de defensa
- Tiempos de ejecución
- Notebooks (Jupyter) y buenas prácticas
- Extender el proyecto

---

## Requisitos e Instalación

1) Clona el repositorio:
```
git clone https://github.com/benmancilla/proyecto-ml.git
cd /proyecto-ml
```

2) Instala dependencias:
```
pip install -r requirements.txt
```

3) Descarga datasets y colócalos en data/ (ver “Estructura de datos”).

> Nota: El proyecto está configurado para Python 3.10/3.11 y Kedro 0.19.11.

---

## Uso rápido (3 opciones)

Opción 1: Kedro local
```
kedro run
kedro viz
```

Opción 2: Docker + Airflow (recomendado en Windows)
```
docker compose up -d
start http://localhost:8080  # usuario: admin / pass: admin
```

Opción 3: DVC (reproducibilidad completa)
```
dvc init
dvc repro
dvc metrics show
dvc dag
```

Tiempo estimado end-to-end: 30–60 min (ver sección “Tiempos”).

---

## Estructura de datos

La carpeta data no está incluida en este repositorio.  
Descarga los datasets desde Kaggle (train folder):
https://www.kaggle.com/datasets/bytadit/ecommerce-order-dataset?resource=download

Estructura esperada:
```
ev_machine/
└── proyecto-ml/
    └── data/
        ├── 01_raw/
        │   ├── df_Customers.csv
        │   ├── df_OrderItems.csv
        │   └── df_Orders.csv
        ├── 02_intermediate/
        ├── 03_primary/
        ├── 04_feature/
        ├── 05_model_input/
        ├── 06_models/
        ├── 07_model_output/
        └── 08_reports/
```

> Si necesitas ayuda, contacta a [ben.mancilla@duocuc.cl].

---

## Ejecución de pipelines y nodos

Pipelines específicos:
```
kedro run --pipeline=data_engineering
kedro run --pipeline=data_science
kedro run --pipeline=reporting
```

Nodo específico (con dependencias):
```
kedro run --node=encode_features_node
```

Listar datasets del catálogo:
```
kedro catalog list
```

---

## Pipelines del proyecto (resumen)

- Data Engineering (`src/proyecto_ml/pipelines/data_engineering/`)
  - clean_orders(orders) -> orders_clean: convierte timestamps y deriva delivery_delay, delivery_time, order_year_month.
  - clean_customers(customers) -> customers_clean: deduplicación por customer_id.
  - merge_orders_customers(orders_clean, customers_clean) -> full_orders: integración de fuentes.
  - build_model_input(full_orders) -> model_input: selección/formateo de features iniciales.

- Data Science (`src/proyecto_ml/pipelines/data_science/`)
  - add_loyalty_target(full_orders) -> model_input_with_target: crea target binario loyal_customer (>1 pedidos).
  - encode_features(model_input_with_target) -> model_input_encoded: one-hot encoding de customer_state (y manejo de compatibilidad).
  - split_data(model_input_encoded, params:data_science) -> X_train, X_test, y_train, y_test.
  - train_model(X_train, y_train) -> rf_model: RandomForestClassifier (pipeline legado).
  - evaluate_model(rf_model, X_test, y_test) -> model_metrics.

- Classification / Regression (pipelines dedicados)
  - Entrenan 7 modelos por tarea con GridSearchCV + CV y guardan métricas extendidas.

- Reporting (`src/proyecto_ml/pipelines/reporting/`)
  - describe_dataset(model_input) -> model_input_stats
  - count_missing_values(model_input) -> model_input_missing
  - distribution_by_state(customers_clean) -> customers_by_state
  - compute_spearman_correlation(model_input_with_target) -> correlation_spearman
  - compute_vif(model_input_with_target) -> vif_analysis

Salidas típicas:  
- data/05_model_input/*  
- data/06_models/* (p. ej. rf_model.pkl, classification_trained_models.pkl, regression_trained_models.pkl)  
- data/07_model_output/* (métricas y feature importance)  
- data/08_reports/* (estadísticos, VIF, correlaciones, reportes)

Parámetros claves: conf/base/parameters.yml

---

## Verificación de resultados esperados

1) Métricas de Clasificación
```
type data\07_model_output\classification_metrics_extended.csv
```
Salida esperada (ejemplo):
```
model,accuracy,precision,recall,f1_score,cv_mean,cv_std
RandomForest,0.85+,0.84+,0.86+,0.85+,0.84,0.02
...
```

2) Métricas de Regresión
```
type data\07_model_output\regression_metrics_extended.csv
```
Salida esperada (ejemplo):
```
model,r2,mae,rmse,cv_r2_mean,cv_r2_std
GradientBoosting,0.89+,2.1+,2.7+,0.88,0.02
...
```

3) Modelos entrenados
```
ls data\06_models\
# Debe incluir:
# - classification_trained_models.pkl
# - regression_trained_models.pkl
# - rf_model.pkl (pipeline legacy data_science)
```

---

## DVC: reproducibilidad y métricas

Inicialización y pipeline:
```
dvc init
dvc add data/01_raw
git add data/01_raw.dvc data/.gitignore
git commit -m "Track raw data with DVC"

dvc repro            # ejecuta stages (classification, regression, data_science, reporting)
dvc dag              # DAG del pipeline
dvc metrics show     # ver métricas
dvc metrics diff     # comparar contra commit previo
```

Flujos típicos:
- Nuevo experimento: editar conf/base/parameters.yml (cv_folds, max_rows), correr dvc repro, revisar dvc metrics diff, commitear y dvc push.
- Volver a experimento previo: git checkout <commit>, dvc checkout, dvc pull, dvc metrics show.

Buenas prácticas:
- Git para código / DVC para datos-modelos
- Versionar parámetros en dvc.yaml
- No commitear data/ a Git (solo .dvc)

---

## Airflow: orquestación con Docker

Arranque con Docker Compose:
```
docker compose up -d
start http://localhost:8080  # admin/admin
```

Volúmenes mapeados (ver docker-compose.yml):
- ./data -> /opt/airflow/data
- ./src -> /opt/airflow/src
- ./airflow/dags -> /opt/airflow/dags

DAGs incluidos:
- data_engineering_pipeline
- data_science_pipeline
- reporting_pipeline

Logs:
```
docker compose logs -f
```

Probar DAGs:
```
airflow dags test data_engineering_pipeline 2024-01-01
airflow tasks test data_engineering_pipeline clean_orders 2024-01-01
```

Notas Windows:
- Usar Docker (Airflow no corre nativo en Windows). Alternativa: WSL2.

---

## Pruebas

Ejecutar:
```
pytest
```
Config: pytest.ini (testpaths) y pyproject.toml (coverage).

---

## Troubleshooting

- Import Error en Kedro
  ```
  pip install -e .
  ```
- Airflow no inicia
  ```
  docker-compose down -v
  docker-compose up --build
  ```
- DVC no encuentra archivos
  ```
  ls data\01_raw\
  # Si faltan, descargar y ubicar en data/01_raw/
  ```
- GridSearchCV muy lento
  - Reducir espacio de búsqueda o cv_folds en conf/base/parameters.yml (ej: 3).
- Memoria insuficiente
  - Reducir data_science.max_rows (ej: 5000).
- Permisos Docker (Windows)
  ```
  $env:AIRFLOW_UID = 50000
  docker compose down -v
  docker compose up -d
  ```

---

## Comandos útiles y Checklist de defensa

Arquitectura y DAG:
```
kedro registry list
kedro viz
dvc dag
```

Reproducibilidad:
```
dvc repro
dvc metrics show
dvc metrics diff HEAD~1
```

Checklist:
- Pipelines sin errores:
  ```
  kedro run --pipeline=classification
  kedro run --pipeline=regression
  ```
- Métricas generadas:
  ```
  Test-Path data\07_model_output\classification_metrics_extended.csv
  Test-Path data\07_model_output\regression_metrics_extended.csv
  ```
- ≥ 5 modelos por pipeline:
  ```
  (Import-Csv data\07_model_output\classification_metrics_extended.csv).Count
  ```
- GridSearchCV + CV aplicados (cv_mean/std presentes)
- Airflow DAG visible:
  ```
  start http://localhost:8080
  ```
- DVC configurado:
  ```
  Test-Path dvc.yaml
  dvc dag
  ```
- Docker funcionando:
  ```
  docker-compose ps
  ```

---

## Tiempos de ejecución estimados

- Data Engineering: 2–5 min
- Classification (7 modelos): 10–30 min
- Regression (7 modelos): 10–30 min
- Data Science (legacy): 5–15 min
- Reporting: 1–3 min
- Total: 30–60 min (8GB RAM, 4+ cores)

Acelerar para testing (conf/base/parameters.yml):
```
classification:
  cv_folds: 3
regression:
  cv_folds: 3
data_science:
  max_rows: 5000
```

---

## Notebooks (Jupyter) y buenas prácticas

Instalar e iniciar:
```
pip install jupyter jupyterlab
kedro jupyter notebook
kedro jupyter lab
kedro ipython
```

Ignorar salidas en git:
```
nbstripout --install
```

---

## Parámetros y credenciales

- conf/base/parameters.yml: parámetros globales (ej. data_science: { test_size: 0.2, random_state: 42 }).
- conf/local/credentials.yml: credenciales locales (no versionar).

---

## Cómo extender el proyecto

- Crear pipeline: `kedro pipeline create <nombre_pipeline>`
- Añadir nodos: funciones puras en nodes.py y agrégalas en pipeline.py con kedro.pipeline.node
- Registrar en pipeline_registry.py y (opcional) agregar al __default__
- Configurar datasets (CSV/Parquet/Excel) en conf/base/catalog.yml

[Más info sobre empaquetado](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)

## Airflow (detalles y mejores prácticas)

- No usar SequentialExecutor en producción
  - Recomendado: LocalExecutor + Postgres.
  - Ventaja: paralelismo real (múltiples tareas en paralelo) y scheduler más estable.

- Ajustes mínimos en docker-compose (variables de entorno)
  ```yaml
  services:
    airflow-webserver:
      environment:
        AIRFLOW__CORE__EXECUTOR: LocalExecutor
        AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres:5432/airflow
        AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'false'   # opcional, DAGs activos al crear
    airflow-scheduler:
      environment:
        AIRFLOW__CORE__EXECUTOR: LocalExecutor
        AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres:5432/airflow
    postgres:
      image: postgres:15
      environment:
        POSTGRES_USER: airflow
        POSTGRES_PASSWORD: airflow
        POSTGRES_DB: airflow
      ports:
        - "5432:5432"
  ```

- Montajes de volúmenes (confirmación)
  - ./airflow/dags -> /opt/airflow/dags
  - ./data -> /opt/airflow/data
  - ./src -> /opt/airflow/src

- DAGs vacíos o en pausa
  - Verificar parse: docker compose logs --tail=200 airflow-scheduler
  - Activar DAGs: en UI (toggle) o CLI:
    ```
    docker compose exec airflow-webserver airflow dags list
    docker compose exec airflow-webserver airflow dags unpause <DAG_ID>
    ```
  - Filtro UI: “All” para ver DAGs pausados.

- Comandos útiles
  ```
  docker compose ps
  docker compose logs --tail=200 airflow-webserver
  docker compose logs --tail=200 airflow-scheduler
  docker compose exec airflow-webserver airflow dags test data_engineering_pipeline 2025-01-01
  docker compose exec airflow-webserver airflow tasks test data_engineering_pipeline clean_orders 2025-01-01
  ```

- Rendimiento en Windows (Docker Desktop)
  - Preferir WSL2/VirtioFS o mover el proyecto a \\wsl$… para acelerar I/O.
  - Usar volumes named para logs/outputs pesados.
  - Aumentar parallelism/dag_concurrency al usar LocalExecutor.

- Buenas prácticas
  - Variables y conexiones: Admin > Connections / Variables (credenciales fuera del repo).
  - Versionar DAGs y requirements en Git; no versionar logs del scheduler.
  - Health-check: revisar “Browse > DAG Runs” y vistas Graph/Gantt para ruta crítica y duración.