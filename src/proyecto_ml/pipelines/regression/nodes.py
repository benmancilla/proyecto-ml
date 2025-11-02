"""
Nodes for regression pipeline with multiple models and GridSearchCV.
Implements 5+ regression models with hyperparameter tuning and cross-validation.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def prepare_regression_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare and scale data for regression models.
    Handles categorical variables by encoding them before scaling.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training targets
        y_test: Testing targets
        
    Returns:
        Scaled X_train, X_test, y_train, y_test
    """
    logger.info(f"Preparing regression data: Train shape={X_train.shape}, Test shape={X_test.shape}")
    
    # Make copies to avoid modifying original data
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    
    logger.info(f"Found {len(categorical_cols)} categorical columns and {len(numerical_cols)} numerical columns")
    
    # Encode categorical columns using one-hot encoding
    if categorical_cols:
        logger.info(f"Encoding categorical columns: {categorical_cols}")
        X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
        
        # Ensure test set has same columns as train set
        missing_cols = set(X_train.columns) - set(X_test.columns)
        for col in missing_cols:
            X_test[col] = 0
        
        # Remove extra columns in test set
        extra_cols = set(X_test.columns) - set(X_train.columns)
        X_test = X_test.drop(columns=extra_cols)
        
        # Reorder columns to match train set
        X_test = X_test[X_train.columns]
    
    # Convert y to 1D arrays if they are DataFrames
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]
    
    # Ensure y is numpy array (ravel to 1D)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    
    # Scale all features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    logger.info(f"Final scaled shape: Train={X_train_scaled.shape}, Test={X_test_scaled.shape}")
    
    # Calculate target statistics safely
    target_mean = np.mean(y_train)
    target_std = np.std(y_train)
    logger.info(f"Target statistics - Mean: {target_mean:.2f}, Std: {target_std:.2f}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def train_regression_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Train multiple regression models with GridSearchCV and k-fold CV.
    
    Implements at least 5 regression models:
    1. Random Forest Regressor
    2. Gradient Boosting Regressor
    3. Linear Regression
    4. Ridge Regression
    5. Lasso Regression
    6. ElasticNet (bonus)
    7. Extra Trees (bonus)
    
    Args:
        X_train: Scaled training features
        X_test: Scaled testing features
        y_train: Training targets
        y_test: Testing targets
        params: Configuration parameters including cv_folds, random_state
        
    Returns:
        metrics_df: DataFrame with model performance metrics (mean ± std)
        feature_importances_df: DataFrame with feature importances for applicable models
        trained_models: Dictionary of trained model objects
    """
    cv_folds = params.get("cv_folds", 5)
    random_state = params.get("random_state", 42)
    n_jobs = params.get("n_jobs", -1)
    
    logger.info(f"Training regression models with {cv_folds}-fold cross-validation")
    
    # Sample data if too large
    sample_size = params.get("sample_size", None)
    if sample_size and len(X_train) > sample_size:
        logger.info(f"Sampling {sample_size} rows from {len(X_train)} for faster training")
        # Create random sample indices
        np.random.seed(random_state)
        sample_indices = np.random.choice(len(X_train), size=sample_size, replace=False)
        X_train = X_train.iloc[sample_indices]
        
        # Handle y_train whether it's a Series or ndarray
        if isinstance(y_train, pd.Series):
            y_train = y_train.iloc[sample_indices]
        else:
            y_train = y_train[sample_indices]
    
    # Define models with REDUCED hyperparameter grids for faster training
    models_config = {
        "RandomForest": {
            "model": RandomForestRegressor(random_state=random_state, n_jobs=1),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [10, 20],
                "min_samples_split": [2, 5]
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor(random_state=random_state),
            "params": {
                "n_estimators": [50, 100],
                "learning_rate": [0.1, 0.2],
                "max_depth": [3, 5]
            }
        },
        "LinearRegression": {
            "model": LinearRegression(),
            "params": {
                "fit_intercept": [True, False]
            }
        },
        "Ridge": {
            "model": Ridge(random_state=random_state),
            "params": {
                "alpha": [0.1, 1.0, 10.0],
                "solver": ["auto", "svd"]
            }
        },
        "Lasso": {
            "model": Lasso(random_state=random_state, max_iter=2000),
            "params": {
                "alpha": [0.1, 1.0, 10.0]
            }
        },
        "ElasticNet": {
            "model": ElasticNet(random_state=random_state, max_iter=2000),
            "params": {
                "alpha": [0.1, 1.0],
                "l1_ratio": [0.3, 0.5, 0.7]
            }
        },
        "ExtraTrees": {
            "model": ExtraTreesRegressor(random_state=random_state, n_jobs=1),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [10, 20]
            }
        }
    }
    
    metrics_list = []
    trained_models = {}
    
    for model_name, config in models_config.items():
        logger.info(f"Training {model_name}...")
        
        try:
            # GridSearchCV with k-fold cross-validation
            grid_search = GridSearchCV(
                estimator=config["model"],
                param_grid=config["params"],
                cv=cv_folds,
                scoring="neg_mean_squared_error",
                n_jobs=n_jobs,
                verbose=0
            )
            
            # Fit the model
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Cross-validation scores on training data
            cv_scores_r2 = cross_val_score(
                best_model, X_train, y_train,
                cv=cv_folds,
                scoring="r2",
                n_jobs=n_jobs
            )
            
            cv_scores_neg_mse = cross_val_score(
                best_model, X_train, y_train,
                cv=cv_folds,
                scoring="neg_mean_squared_error",
                n_jobs=n_jobs
            )
            
            cv_scores_neg_mae = cross_val_score(
                best_model, X_train, y_train,
                cv=cv_folds,
                scoring="neg_mean_absolute_error",
                n_jobs=n_jobs
            )
            
            # Predictions
            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)
            
            # Calculate metrics for test set
            mae = mean_absolute_error(y_test, y_pred_test)
            mse = mean_squared_error(y_test, y_pred_test)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred_test)
            
            # Calculate MAPE safely
            try:
                mape = mean_absolute_percentage_error(y_test, y_pred_test)
            except Exception:
                mape = np.nan
            
            # Train metrics
            train_r2 = r2_score(y_train, y_pred_train)
            train_mae = mean_absolute_error(y_train, y_pred_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            
            # Store metrics
            metrics_list.append({
                "model": model_name,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "mape": mape,
                "cv_r2_mean": cv_scores_r2.mean(),
                "cv_r2_std": cv_scores_r2.std(),
                "cv_rmse_mean": np.sqrt(-cv_scores_neg_mse.mean()),
                "cv_rmse_std": np.sqrt(cv_scores_neg_mse.std()),
                "cv_mae_mean": -cv_scores_neg_mae.mean(),
                "cv_mae_std": cv_scores_neg_mae.std(),
                "cv_scores": f"{cv_scores_r2.mean():.4f} ± {cv_scores_r2.std():.4f}",
                "best_params": str(grid_search.best_params_),
                "train_r2": train_r2,
                "train_mae": train_mae,
                "train_rmse": train_rmse
            })
            
            # Store trained model
            trained_models[model_name] = best_model
            
            logger.info(f"{model_name} - Test R²: {r2:.4f}, RMSE: {rmse:.4f}, CV R²: {cv_scores_r2.mean():.4f} ± {cv_scores_r2.std():.4f}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            continue
    
    # Create DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    
    # Sort by R² score (descending)
    metrics_df = metrics_df.sort_values("r2", ascending=False)
    
    logger.info(f"Successfully trained {len(trained_models)} regression models")
    logger.info(f"Best model by R²: {metrics_df.iloc[0]['model']} with R²={metrics_df.iloc[0]['r2']:.4f}")
    
    # Generate feature importances
    feature_importances_df = generate_regression_feature_importances(trained_models, X_train)
    
    return metrics_df, feature_importances_df, trained_models


def generate_regression_feature_importances(
    trained_models: Dict,
    X_train: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract feature importances from tree-based regression models.
    
    Args:
        trained_models: Dictionary of trained model objects
        X_train: Training features for reference
        
    Returns:
        DataFrame with feature importances for applicable models
    """
    importance_list = []
    
    for model_name, model in trained_models.items():
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for feature, importance in zip(X_train.columns, importances):
                importance_list.append({
                    "model": model_name,
                    "feature": feature,
                    "importance": importance
                })
            logger.info(f"Extracted feature importances for {model_name}")
        elif hasattr(model, "coef_"):
            # For linear models, use absolute coefficient values as importance
            coefs = np.abs(model.coef_)
            for feature, coef in zip(X_train.columns, coefs):
                importance_list.append({
                    "model": model_name,
                    "feature": feature,
                    "importance": coef
                })
            logger.info(f"Extracted coefficients for {model_name}")
    
    if importance_list:
        importance_df = pd.DataFrame(importance_list)
        importance_df = importance_df.sort_values(["model", "importance"], ascending=[True, False])
        return importance_df
    else:
        logger.warning("No feature importances extracted from models")
        return pd.DataFrame(columns=["model", "feature", "importance"])
