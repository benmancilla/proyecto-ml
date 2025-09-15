# proyecto_ml

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

https://drive.google.com/file/d/10cF-siIP8vxZaxebEAVBVGos25U1H-7E/view?usp=sharing --> video

Proyecto Kedro para procesamiento y modelado de datos del dominio de pedidos/clientes, implementando etapas de CRISP-DM (comprensión de datos, preparación, modelado) y reportes reproducibles. Este README reúne los pasos y comandos clave para ponerlo en marcha rápidamente en Windows (PowerShell).

## Instalación

1. Clona el repositorio:
    ```
    git clone https://github.com/benmancilla/proyecto-ml.git
    cd /proyecto-ml
    ```

2. Instala las dependencias:
    ```
    pip install -r requirements.txt
    ```

3. Descarga los datasets requeridos y colócalos en la carpeta `data/` como se indica abajo.

## Uso

Para ejecutar el pipeline principal:
```
kedro run
```

Para visualizar el pipeline:
```
kedro viz
```

Para ejecutar los tests:
```
pytest
```

## Ejecutar pipelines específicos y nodos
- Ejecutar un pipeline por nombre (según `src/proyecto_ml/pipeline_registry.py`):
```
kedro run --pipeline=data_engineering
kedro run --pipeline=data_science
kedro run --pipeline=reporting
```

- Ejecutar un nodo concreto y sus dependencias (ejemplo):
```
kedro run --node=encode_features_node
```

## Estructura de datos

La carpeta `data` no está incluida en este repositorio.  
Para ejecutar el proyecto, descarga los datasets requeridos desde la fuente correspondiente (https://www.kaggle.com/datasets/bytadit/ecommerce-order-dataset?resource=download, train folder) y colócalos en la carpeta `data` siguiendo la estructura:

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

> **Nota:** Si necesitas ayuda, contacta a [ben.mancilla@duocuc.cl].

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Parámetros y credenciales
- `conf/base/parameters.yml`: agrega aquí parámetros globales del proyecto (por ejemplo `data_science: { test_size: 0.2, random_state: 42 }`).
- `conf/local/credentials.yml`: credenciales locales para fuentes protegidas (no versionar). Mantén `conf/local/` fuera del repo.

Listar datasets configurados en el catálogo:
```
kedro catalog list
```

## Descripción general

Este es tu nuevo proyecto Kedro, generado con `kedro 0.19.11`.

Consulta la [documentación de Kedro](https://docs.kedro.org) para comenzar.

## Reglas y buenas prácticas

- No elimines líneas del archivo `.gitignore` proporcionado.
- Asegúrate de que tus resultados sean reproducibles siguiendo convenciones de ingeniería de datos.
- No subas datos al repositorio.
- No subas credenciales ni configuraciones locales. Mantén todo en `conf/local/`.

## Instalación de dependencias

Declara las dependencias en `requirements.txt` para instalación con `pip`.

Para instalarlas, ejecuta:

```
pip install -r requirements.txt
```

## Ejecución del pipeline Kedro

Puedes ejecutar el proyecto con:

```
kedro run
```

Para visualizar el DAG de extremo a extremo e inspeccionar entradas/salidas:
```
kedro viz
```

## Pipelines del proyecto (resumen)
El registro en `src/proyecto_ml/pipeline_registry.py` combina por defecto:
- `data_engineering.create_pipeline()`
- `data_science.create_pipeline()`
- `reporting.create_pipeline()`

Principales nodos y salidas:

1) Data Engineering (`src/proyecto_ml/pipelines/data_engineering/`)
- `clean_orders(orders) -> orders_clean`: convierte timestamps y deriva `delivery_delay`, `delivery_time`, `order_year_month`.
- `clean_customers(customers) -> customers_clean`: deduplicación por `customer_id`.
- `merge_orders_customers(orders_clean, customers_clean) -> full_orders`: integración de fuentes.
- `build_model_input(full_orders) -> model_input`: selección/formateo de features iniciales.

2) Data Science (`src/proyecto_ml/pipelines/data_science/`)
- `add_loyalty_target(full_orders) -> model_input_with_target`: crea target binario `loyal_customer` (>1 pedidos).
- `encode_features(model_input_with_target) -> model_input_encoded`: one-hot encoding de `customer_state` y `customer_city`.
- `split_data(model_input_encoded, params:data_science) -> X_train, X_test, y_train, y_test`.
- `train_model(X_train, y_train) -> rf_model`: `RandomForestClassifier`.
- `evaluate_model(rf_model, X_test, y_test) -> model_metrics`: accuracy y, si aplica, importancias de features.

3) Reporting (`src/proyecto_ml/pipelines/reporting/`)
- `describe_dataset(model_input) -> model_input_stats`
- `count_missing_values(model_input) -> model_input_missing`
- `distribution_by_state(customers_clean) -> customers_by_state`

Salidas típicas: `data/05_model_input/*`, `data/06_models/rf_model.pkl`, `data/07_model_output/model_metrics.csv`, `data/08_reports/*`.

## Pruebas

Consulta el archivo `src/tests/test_run.py` para instrucciones sobre cómo escribir tus tests. Ejecuta las pruebas con:

```
pytest
```

Puedes configurar el umbral de cobertura en el archivo `pyproject.toml` bajo la sección `[tool.coverage.report]`.

## Dependencias del proyecto

Para ver y actualizar los requisitos de dependencias, usa `requirements.txt`. Instala los requisitos con:

```
pip install -r requirements.txt
```

[Más información sobre dependencias](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## Solución de problemas frecuentes
- Archivo/dataset no encontrado (MissingDataSet): verifica que los CSV estén en `data/01_raw/` y que las rutas del `catalog.yml` coincidan.
- Conflictos de versiones (por ejemplo `OneHotEncoder.sparse_output`): usa las versiones de `requirements.txt` incluidas; el código maneja compatibilidad en los nodos.
- El kernel de Jupyter no aparece: tras activar el venv, ejecuta `python -m ipykernel install --user --name proyecto-ml` y selecciona ese kernel en Jupyter.
- Kedro Viz no abre automáticamente: copia la URL/puerto que muestra la consola en tu navegador.

## Cómo extender el proyecto
- Crear nuevo pipeline: `kedro pipeline create <nombre_pipeline>`.
- Añadir nodos: define funciones puras en `nodes.py` y añádelas a `pipeline.py` con `kedro.pipeline.node`.
- Registrar en `pipeline_registry.py` y (opcional) agregar al `__default__`.
- Configurar nuevos datasets (CSV/Parquet/Excel) en `conf/base/catalog.yml`.

## Uso de Kedro con notebooks

> Usando `kedro jupyter` o `kedro ipython` tendrás disponibles las variables: `context`, `session`, `catalog` y `pipelines`.
>
> Jupyter, JupyterLab e IPython están incluidos en los requisitos del proyecto. Tras instalar dependencias, puedes usarlos directamente.

### Jupyter
Instala Jupyter si no lo tienes:

```
pip install jupyter
```

Inicia el servidor de notebooks:

```
kedro jupyter notebook
```

### JupyterLab
Instala JupyterLab:

```
pip install jupyterlab
```

Inicia JupyterLab:

```
kedro jupyter lab
```

### IPython
Para iniciar una sesión IPython:

```
kedro ipython
```

### Ignorar celdas de salida de notebooks en `git`
Para eliminar automáticamente las salidas de las celdas antes de hacer commit, usa [`nbstripout`](https://github.com/kynan/nbstripout):

```
nbstripout --install
```

> *Nota:* Las salidas se mantienen localmente.

## Empaquetar tu proyecto Kedro

[Más información sobre documentación y empaquetado](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)