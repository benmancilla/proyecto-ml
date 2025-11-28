"""
Master DAG that orchestrates the full ML pipeline end-to-end:
data engineering -> unsupervised -> classification/regression -> data science -> reporting -> DVC.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
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

dag = DAG(
    "master_ml_pipeline",
    default_args=default_args,
    description="Pipeline maestro de ML: Data Engineering + Unsupervised + Classification + Regression + Reporting",
    schedule_interval="@weekly",
    catchup=False,
    tags=["master", "ml-ops", "end-to-end", "evaluacion-final"],
)

trigger_data_engineering = TriggerDagRunOperator(
    task_id="trigger_data_engineering",
    trigger_dag_id="data_engineering_pipeline",
    wait_for_completion=True,
    poke_interval=30,
    dag=dag,
)

trigger_unsupervised = BashOperator(
    task_id="trigger_unsupervised",
    bash_command="cd /opt/airflow && kedro run --pipeline=unsupervised_learning",
    dag=dag,
)

trigger_classification = BashOperator(
    task_id="trigger_classification",
    bash_command="cd /opt/airflow && kedro run --pipeline=classification",
    dag=dag,
)

trigger_regression = BashOperator(
    task_id="trigger_regression",
    bash_command="cd /opt/airflow && kedro run --pipeline=regression",
    dag=dag,
)

trigger_data_science = TriggerDagRunOperator(
    task_id="trigger_data_science",
    trigger_dag_id="data_science_pipeline",
    wait_for_completion=True,
    poke_interval=30,
    dag=dag,
)

trigger_reporting = TriggerDagRunOperator(
    task_id="trigger_reporting",
    trigger_dag_id="reporting_pipeline",
    wait_for_completion=True,
    poke_interval=30,
    dag=dag,
)

dvc_push = BashOperator(
    task_id="dvc_push",
    bash_command='cd /opt/airflow && dvc push || echo "DVC push skipped (remote not configured)"',
    dag=dag,
)

trigger_data_engineering >> trigger_unsupervised >> [trigger_classification, trigger_regression]
[trigger_classification, trigger_regression] >> trigger_data_science >> trigger_reporting >> dvc_push
