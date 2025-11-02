"""
DAG de Airflow para el pipeline de Reporting
Genera reportes y análisis estadísticos sobre los datos procesados.
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

# Agregar el directorio src al path
project_root = Path('/opt/airflow')
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np
from scipy import stats


# Configuración de paths
DATA_ROOT = project_root / 'data'
MODEL_INPUT_PATH = DATA_ROOT / '05_model_input'
REPORTS_PATH = DATA_ROOT / '08_reports'

# Asegurar que el directorio existe
REPORTS_PATH.mkdir(parents=True, exist_ok=True)


def generate_data_quality_report():
    """Genera reporte de calidad de datos"""
    print("Generando reporte de calidad de datos...")
    
    df = pd.read_csv(MODEL_INPUT_PATH / 'model_input_with_target.csv')
    
    # Estadísticas descriptivas
    stats_df = df.describe().T
    stats_df.to_csv(REPORTS_PATH / 'model_input_stats.csv')
    
    # Valores faltantes
    missing_df = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2)
    }).sort_values('missing_count', ascending=False)
    missing_df.to_csv(REPORTS_PATH / 'model_input_missing.csv', index=False)
    
    print(f"Reporte de calidad guardado en {REPORTS_PATH}")


def generate_correlation_report():
    """Genera reporte de correlaciones"""
    print("Generando reporte de correlaciones...")
    
    df = pd.read_csv(MODEL_INPUT_PATH / 'model_input_with_target.csv')
    
    # Seleccionar columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1:
        # Correlación de Spearman (robusta a outliers)
        corr_spearman = df[numeric_cols].corr(method='spearman')
        corr_spearman.to_csv(REPORTS_PATH / 'correlation_spearman.csv')
        print(f"Correlación Spearman guardada: {len(numeric_cols)} variables")
    else:
        print("Insuficientes variables numéricas para correlación")


def generate_geographic_report():
    """Genera reporte geográfico de clientes"""
    print("Generando reporte geográfico...")
    
    df = pd.read_csv(MODEL_INPUT_PATH / 'model_input_with_target.csv')
    
    if 'customer_state' in df.columns:
        state_summary = df.groupby('customer_state').agg({
            'customer_id': 'count',
            'delivery_delay': ['mean', 'median', 'std']
        }).round(2)
        state_summary.columns = ['_'.join(col).strip() for col in state_summary.columns.values]
        state_summary = state_summary.sort_values('customer_id_count', ascending=False)
        state_summary.to_csv(REPORTS_PATH / 'customers_by_state.csv')
        print(f"Reporte geográfico guardado: {len(state_summary)} estados")
    else:
        print("Columna customer_state no disponible")


def generate_vif_report():
    """Genera reporte de VIF (Variance Inflation Factor) para detectar multicolinealidad"""
    print("Generando reporte VIF...")
    
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        df = pd.read_csv(MODEL_INPUT_PATH / 'model_input_with_target.csv')
        
        # Seleccionar variables numéricas (excluyendo target)
        numeric_cols = ['delivery_time', 'customer_zip_code_prefix', 'order_month', 'order_year']
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(numeric_cols) > 1:
            X = df[numeric_cols].dropna()
            
            vif_data = pd.DataFrame()
            vif_data["feature"] = numeric_cols
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(numeric_cols))]
            vif_data = vif_data.sort_values('VIF', ascending=False)
            vif_data.to_csv(REPORTS_PATH / 'vif_analysis.csv', index=False)
            
            print(f"VIF analysis guardado: {len(numeric_cols)} variables")
        else:
            print("Insuficientes variables numéricas para VIF")
    except ImportError:
        print("statsmodels no disponible, saltando VIF analysis")
    except Exception as e:
        print(f"Error en VIF analysis: {e}")


def generate_model_performance_summary():
    """Genera resumen de desempeño de modelos"""
    print("Generando resumen de desempeño de modelos...")
    
    output_path = DATA_ROOT / '07_model_output'
    
    summary_lines = []
    
    # Leer métricas de regresión
    try:
        reg_metrics = pd.read_csv(output_path / 'regression_metrics.csv')
        best_reg = reg_metrics.loc[reg_metrics['rmse'].idxmin()]
        summary_lines.append(f"Mejor modelo regresión: {best_reg['model']} (RMSE: {best_reg['rmse']:.2f})")
    except Exception as e:
        summary_lines.append(f"No se pudieron leer métricas de regresión: {e}")
    
    # Leer métricas de clasificación
    try:
        clf_metrics = pd.read_csv(output_path / 'classification_metrics.csv')
        best_clf = clf_metrics.loc[clf_metrics['f1'].idxmax()]
        summary_lines.append(f"Mejor modelo clasificación: {best_clf['model']} (F1: {best_clf['f1']:.3f})")
    except Exception as e:
        summary_lines.append(f"No se pudieron leer métricas de clasificación: {e}")
    
    # Guardar resumen
    with open(REPORTS_PATH / 'model_performance_summary.txt', 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print("Resumen de desempeño guardado")


# Argumentos por defecto del DAG
default_args = {
    'owner': 'analytics-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Definición del DAG
dag = DAG(
    'reporting_pipeline',
    default_args=default_args,
    description='Pipeline de reporting: generación de reportes y análisis',
    schedule_interval='@weekly',  # Generar reportes semanalmente
    catchup=False,
    tags=['reporting', 'analytics', 'eda'],
)

# Definición de tareas
task_data_quality = PythonOperator(
    task_id='generate_data_quality_report',
    python_callable=generate_data_quality_report,
    dag=dag,
)

task_correlation = PythonOperator(
    task_id='generate_correlation_report',
    python_callable=generate_correlation_report,
    dag=dag,
)

task_geographic = PythonOperator(
    task_id='generate_geographic_report',
    python_callable=generate_geographic_report,
    dag=dag,
)

task_vif = PythonOperator(
    task_id='generate_vif_report',
    python_callable=generate_vif_report,
    dag=dag,
)

task_performance = PythonOperator(
    task_id='generate_model_performance_summary',
    python_callable=generate_model_performance_summary,
    dag=dag,
)

# Todas las tareas pueden ejecutarse en paralelo
[task_data_quality, task_correlation, task_geographic, task_vif, task_performance]
