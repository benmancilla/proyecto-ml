"""
DAG de Airflow para el pipeline de Ciencia de Datos
Ejecuta el entrenamiento de modelos de regresión y clasificación.
"""
from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable

# Agregar el directorio src al path
project_root = Path('/opt/airflow')
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import joblib


# Configuración de paths
DATA_ROOT = project_root / 'data'
MODEL_INPUT_PATH = DATA_ROOT / '05_model_input'
MODELS_PATH = DATA_ROOT / '06_models'
OUTPUT_PATH = DATA_ROOT / '07_model_output'

# Asegurar que los directorios existen
MODELS_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Variables globales para compartir entre tareas
FEATURE_COLUMNS = ['delivery_time', 'customer_zip_code_prefix', 'order_month', 'order_year', 'customer_state']
NUMERIC_FEATURES = ['delivery_time', 'customer_zip_code_prefix', 'order_month', 'order_year']
CATEGORICAL_FEATURES = ['customer_state']


def prepare_data():
    """Prepara los datos para modelado"""
    print("Preparando datos para modelado...")
    
    df = pd.read_csv(MODEL_INPUT_PATH / 'model_input_with_target.csv')
    df = df[df['order_status'] == 'delivered'].copy()
    df = df.dropna(subset=['delivery_delay', 'delivery_time', 'customer_state', 'order_purchase_timestamp'])
    
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_year'] = df['order_purchase_timestamp'].dt.year
    df['order_month'] = df['order_purchase_timestamp'].dt.month
    df['customer_zip_code_prefix'] = pd.to_numeric(df['customer_zip_code_prefix'], errors='coerce')
    df = df.dropna(subset=['customer_zip_code_prefix'])
    
    # Target de clasificación
    if 'target' not in df.columns:
        df['target'] = (df['delivery_delay'] > 0).astype(int)
    else:
        df['target'] = pd.to_numeric(df['target'], errors='coerce').fillna(0).astype(int)
    
    # Imputación
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(df[NUMERIC_FEATURES].median())
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna('unknown')
    
    # Muestreo opcional controlado por Variable de Airflow (DS_SAMPLE_FRAC)
    try:
        sample_frac = float(Variable.get("DS_SAMPLE_FRAC", default_var="1.0"))
    except Exception:
        sample_frac = 1.0
    if 0 < sample_frac < 1.0:
        before = len(df)
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        print(f"Sampling enabled: using {len(df)} rows ({sample_frac:.0%}) out of {before}.")
    else:
        print(f"Sampling disabled: using full dataset with {len(df)} rows.")

    X = df[FEATURE_COLUMNS]
    y_regression = df['delivery_delay']
    y_classification = df['target']
    
    # Splits para regresión
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )
    
    # Splits para clasificación (estratificado)
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X, y_classification, test_size=0.2, stratify=y_classification, random_state=42
    )
    
    # Guardar splits
    X_train.to_csv(MODEL_INPUT_PATH / 'X_train.csv', index=False)
    X_test.to_csv(MODEL_INPUT_PATH / 'X_test.csv', index=False)
    y_train.to_csv(MODEL_INPUT_PATH / 'y_train.csv', index=False)
    y_test.to_csv(MODEL_INPUT_PATH / 'y_test.csv', index=False)
    
    X_clf_train.to_csv(MODEL_INPUT_PATH / 'X_clf_train.csv', index=False)
    X_clf_test.to_csv(MODEL_INPUT_PATH / 'X_clf_test.csv', index=False)
    y_clf_train.to_csv(MODEL_INPUT_PATH / 'y_clf_train.csv', index=False)
    y_clf_test.to_csv(MODEL_INPUT_PATH / 'y_clf_test.csv', index=False)
    
    print(f"Datos preparados: {len(X_train)} train, {len(X_test)} test")


def train_regression_models():
    """Entrena modelos de regresión"""
    print("Entrenando modelos de regresión...")
    
    # Cargar datos
    X_train = pd.read_csv(MODEL_INPUT_PATH / 'X_train.csv')
    X_test = pd.read_csv(MODEL_INPUT_PATH / 'X_test.csv')
    y_train = pd.read_csv(MODEL_INPUT_PATH / 'y_train.csv').squeeze()
    y_test = pd.read_csv(MODEL_INPUT_PATH / 'y_test.csv').squeeze()
    
    # Preprocesador
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ])
    
    # Modelos
    # Permitir configurar número de árboles/estimadores vía Variable (DS_N_ESTIMATORS)
    try:
        n_estimators_cfg = int(float(Variable.get("DS_N_ESTIMATORS", default_var="200")))
    except Exception:
        n_estimators_cfg = 200

    models = {
        'LinearRegression': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ]),
        'RandomForest': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42, n_estimators=n_estimators_cfg))
        ]),
        'GradientBoosting': Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(random_state=42, n_estimators=n_estimators_cfg))
        ])
    }
    
    metrics = []
    
    for name, model in models.items():
        print(f"Entrenando {name}...")
        model.fit(X_train, y_train)
        
        # Guardar modelo
        joblib.dump(model, MODELS_PATH / f'regression_{name}.pkl')
        
        # Evaluar
        y_pred = model.predict(X_test)
        metrics.append({
            'model': name,
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        })
        print(f"{name} - R²: {metrics[-1]['r2']:.3f}, RMSE: {metrics[-1]['rmse']:.2f}")
    
    # Guardar métricas
    pd.DataFrame(metrics).to_csv(OUTPUT_PATH / 'regression_metrics.csv', index=False)
    print("Modelos de regresión guardados")


def train_classification_models():
    """Entrena modelos de clasificación"""
    print("Entrenando modelos de clasificación...")
    
    # Cargar datos
    X_train = pd.read_csv(MODEL_INPUT_PATH / 'X_clf_train.csv')
    X_test = pd.read_csv(MODEL_INPUT_PATH / 'X_clf_test.csv')
    y_train = pd.read_csv(MODEL_INPUT_PATH / 'y_clf_train.csv').squeeze()
    y_test = pd.read_csv(MODEL_INPUT_PATH / 'y_clf_test.csv').squeeze()
    
    # Preprocesador
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ])
    
    # Modelos
    # Permitir configurar número de árboles vía Variable (DS_N_ESTIMATORS)
    try:
        n_estimators_cfg = int(float(Variable.get("DS_N_ESTIMATORS", default_var="200")))
    except Exception:
        n_estimators_cfg = 200

    models = {
        'LogisticRegression': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ]),
        'RandomForestClassifier': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=n_estimators_cfg))
        ])
    }
    
    metrics = []
    confusion_matrices = []
    
    for name, model in models.items():
        print(f"Entrenando {name}...")
        model.fit(X_train, y_train)
        
        # Guardar modelo
        joblib.dump(model, MODELS_PATH / f'classification_{name}.pkl')
        
        # Evaluar
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        metrics.append({
            'model': name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        })
        
        confusion_matrices.append({
            'model': name,
            'tn': cm[0, 0],
            'fp': cm[0, 1],
            'fn': cm[1, 0],
            'tp': cm[1, 1]
        })
        
        print(f"{name} - F1: {metrics[-1]['f1']:.3f}, Accuracy: {metrics[-1]['accuracy']:.3f}")
    
    # Guardar métricas
    pd.DataFrame(metrics).to_csv(OUTPUT_PATH / 'classification_metrics.csv', index=False)
    pd.DataFrame(confusion_matrices).to_csv(OUTPUT_PATH / 'classification_confusion_matrices.csv', index=False)
    print("Modelos de clasificación guardados")


# Argumentos por defecto del DAG
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Definición del DAG
dag = DAG(
    'data_science_pipeline',
    default_args=default_args,
    description='Pipeline de ciencia de datos: entrenamiento de modelos ML',
    schedule_interval='@weekly',  # Entrenar modelos semanalmente
    catchup=False,
    tags=['machine-learning', 'training', 'regression', 'classification'],
)

# Definición de tareas
task_prepare = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_data,
    dag=dag,
)

task_regression = PythonOperator(
    task_id='train_regression_models',
    python_callable=train_regression_models,
    dag=dag,
)

task_classification = PythonOperator(
    task_id='train_classification_models',
    python_callable=train_classification_models,
    dag=dag,
)

# Definir dependencias
task_prepare >> [task_regression, task_classification]
