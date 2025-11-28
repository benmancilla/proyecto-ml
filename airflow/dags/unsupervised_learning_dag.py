"""
DAG para ejecutar el pipeline de aprendizaje no supervisado (clustering + reducción de dimensionalidad + anomalías).
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "ml-ops-team",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="unsupervised_learning_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["unsupervised", "mlops", "clustering"],
) as dag:
    run_unsupervised = BashOperator(
        task_id="run_unsupervised_pipeline",
        bash_command="cd /opt/airflow && kedro run --pipeline=unsupervised_learning",
    )

    run_unsupervised
