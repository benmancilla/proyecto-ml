"""
DAG de Airflow para el pipeline de Ingeniería de Datos
Ejecuta la limpieza y preparación de datos raw hacia datasets procesados.
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

# Agregar el directorio src al path para importar los pipelines
project_root = Path('/opt/airflow')
sys.path.insert(0, str(project_root / 'src'))

from proyecto_ml.pipelines.data_engineering import nodes as de_nodes
import pandas as pd


# Configuración de paths
DATA_ROOT = project_root / 'data'
RAW_PATH = DATA_ROOT / '01_raw'
INTERMEDIATE_PATH = DATA_ROOT / '02_intermediate'
PRIMARY_PATH = DATA_ROOT / '03_primary'
MODEL_INPUT_PATH = DATA_ROOT / '05_model_input'

# Asegurar que los directorios existen
INTERMEDIATE_PATH.mkdir(parents=True, exist_ok=True)
PRIMARY_PATH.mkdir(parents=True, exist_ok=True)
MODEL_INPUT_PATH.mkdir(parents=True, exist_ok=True)


def load_and_clean_orders():
    """Carga y limpia datos de órdenes"""
    print("Cargando datos de órdenes...")
    orders = pd.read_csv(RAW_PATH / 'df_Orders.csv')
    print(f"Órdenes cargadas: {len(orders)} registros")
    
    orders_clean = de_nodes.clean_orders(orders)
    orders_clean.to_csv(INTERMEDIATE_PATH / 'orders_clean.csv', index=False)
    print(f"Órdenes limpiadas guardadas: {len(orders_clean)} registros")


def load_and_clean_customers():
    """Carga y limpia datos de clientes"""
    print("Cargando datos de clientes...")
    customers = pd.read_csv(RAW_PATH / 'df_Customers.csv')
    print(f"Clientes cargados: {len(customers)} registros")
    
    customers_clean = de_nodes.clean_customers(customers)
    customers_clean.to_csv(INTERMEDIATE_PATH / 'customers_clean.csv', index=False)
    print(f"Clientes limpios guardados: {len(customers_clean)} registros")


def load_and_clean_order_items():
    """Carga y limpia datos de items de órdenes"""
    print("Cargando datos de items de órdenes...")
    order_items = pd.read_csv(RAW_PATH / 'df_OrderItems.csv')
    print(f"Items cargados: {len(order_items)} registros")
    
    # Limpieza básica (puedes extender según necesites)
    order_items_clean = order_items.copy()
    order_items_clean.to_csv(INTERMEDIATE_PATH / 'order_items_clean.csv', index=False)
    print(f"Items limpios guardados: {len(order_items_clean)} registros")


def merge_datasets():
    """Une órdenes con clientes para crear dataset completo"""
    print("Uniendo datasets...")
    orders_clean = pd.read_csv(INTERMEDIATE_PATH / 'orders_clean.csv')
    customers_clean = pd.read_csv(INTERMEDIATE_PATH / 'customers_clean.csv')
    
    full_orders = de_nodes.merge_orders_customers(orders_clean, customers_clean)
    full_orders.to_csv(PRIMARY_PATH / 'full_orders.csv', index=False)
    print(f"Dataset completo guardado: {len(full_orders)} registros")


def create_model_input():
    """Genera el dataset de entrada para modelos ML"""
    print("Creando input para modelos...")
    full_orders = pd.read_csv(PRIMARY_PATH / 'full_orders.csv')
    
    model_input = de_nodes.build_model_input(full_orders)
    model_input.to_csv(MODEL_INPUT_PATH / 'model_input.csv', index=False)
    
    # Crear versión con target para entrenamiento
    model_input_with_target = model_input.copy()
    if 'delivery_delay' in model_input_with_target.columns:
        model_input_with_target['target'] = (
            model_input_with_target['delivery_delay'] > 0
        ).astype(int)
    model_input_with_target.to_csv(
        MODEL_INPUT_PATH / 'model_input_with_target.csv', 
        index=False
    )
    print(f"Model input guardado: {len(model_input)} registros")


# Argumentos por defecto del DAG
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Definición del DAG
dag = DAG(
    'data_engineering_pipeline',
    default_args=default_args,
    description='Pipeline de ingeniería de datos: limpieza y preparación',
    schedule_interval='@daily',  # Ajusta según necesidad (ej: @weekly, None para manual)
    catchup=False,
    tags=['data-engineering', 'etl', 'preprocessing'],
)

# Definición de tareas
task_clean_orders = PythonOperator(
    task_id='clean_orders',
    python_callable=load_and_clean_orders,
    dag=dag,
)

task_clean_customers = PythonOperator(
    task_id='clean_customers',
    python_callable=load_and_clean_customers,
    dag=dag,
)

task_clean_order_items = PythonOperator(
    task_id='clean_order_items',
    python_callable=load_and_clean_order_items,
    dag=dag,
)

task_merge = PythonOperator(
    task_id='merge_datasets',
    python_callable=merge_datasets,
    dag=dag,
)

task_model_input = PythonOperator(
    task_id='create_model_input',
    python_callable=create_model_input,
    dag=dag,
)

# Definir dependencias (orden de ejecución)
[task_clean_orders, task_clean_customers, task_clean_order_items] >> task_merge >> task_model_input
