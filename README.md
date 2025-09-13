# proyecto_ml

## Instalación

1. Clona el repositorio:
    ```
    git clone https://github.com/tu_usuario/ev_machine.git
    cd ev_machine/proyecto-ml
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

## Estructura de datos

La carpeta `data` no está incluida en este repositorio.  
Para ejecutar el proyecto, descarga los datasets requeridos desde la fuente correspondiente y colócalos en la carpeta `data` siguiendo la estructura:

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

> **Nota:** Si necesitas ayuda, contacta a [tu_email@dominio.com].

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

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