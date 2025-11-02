"""
Nodes for classification pipeline with multiple models and GridSearchCV.
Implements 5+ classification models with hyperparameter tuning and cross-validation.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def prepare_classification_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare and scale data for classification models.
    Handles categorical variables by encoding them before scaling.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
        
    Returns:
        Scaled X_train, X_test, y_train, y_test
    """
    logger.info(f"Preparing classification data: Train shape={X_train.shape}, Test shape={X_test.shape}")
    
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
    
    # Get class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique, counts))
    logger.info(f"Class distribution in training: {class_dist}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def train_classification_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    params: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Train multiple classification models with GridSearchCV and k-fold CV.
    
    Implements at least 5 classification models:
    1. Random Forest
    2. Gradient Boosting
    3. Logistic Regression
    4. Support Vector Classifier
    5. K-Nearest Neighbors
    6. AdaBoost (bonus)
    7. Extra Trees (bonus)
    
    Args:
        X_train: Scaled training features
        X_test: Scaled testing features
        y_train: Training labels
        y_test: Testing labels
        params: Configuration parameters including cv_folds, random_state
        
    Returns:
        metrics_df: DataFrame with model performance metrics (mean ± std)
        confusion_matrices_df: DataFrame with confusion matrices for all models
        trained_models: Dictionary of trained model objects
    """
    cv_folds = params.get("cv_folds", 5)
    random_state = params.get("random_state", 42)
    n_jobs = params.get("n_jobs", -1)
    
    logger.info(f"Training classification models with {cv_folds}-fold cross-validation")
    
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
            "model": RandomForestClassifier(random_state=random_state, class_weight="balanced", n_jobs=1),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [10, 20],
                "min_samples_split": [2, 5]
            }
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=random_state),
            "params": {
                "n_estimators": [50, 100],
                "learning_rate": [0.1, 0.2],
                "max_depth": [3, 5]
            }
        },
        "LogisticRegression": {
            "model": LogisticRegression(random_state=random_state, max_iter=500, class_weight="balanced"),
            "params": {
                "C": [0.1, 1, 10],
                "solver": ["lbfgs", "liblinear"]
            }
        },
        "SVC": {
            "model": SVC(random_state=random_state, probability=True, class_weight="balanced", max_iter=1000),
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale"]
            }
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"]
            }
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(random_state=random_state),
            "params": {
                "n_estimators": [50, 100],
                "learning_rate": [0.1, 0.5, 1.0]
            }
        },
        "ExtraTrees": {
            "model": ExtraTreesClassifier(random_state=random_state, class_weight="balanced", n_jobs=1),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [10, 20]
            }
        }
    }
    
    metrics_list = []
    confusion_list = []
    trained_models = {}
    
    for model_name, config in models_config.items():
        logger.info(f"Training {model_name}...")
        
        try:
            # GridSearchCV with k-fold cross-validation
            grid_search = GridSearchCV(
                estimator=config["model"],
                param_grid=config["params"],
                cv=cv_folds,
                scoring="f1_weighted",
                n_jobs=n_jobs,
                verbose=0
            )
            
            # Fit the model
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Cross-validation scores on training data
            cv_scores = cross_val_score(
                best_model, X_train, y_train,
                cv=cv_folds,
                scoring="f1_weighted",
                n_jobs=n_jobs
            )
            
            # Predictions
            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)
            
            # Calculate metrics for test set
            accuracy = accuracy_score(y_test, y_pred_test)
            precision = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred_test, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
            
            # Try to calculate ROC AUC if possible
            try:
                if hasattr(best_model, "predict_proba"):
                    y_pred_proba = best_model.predict_proba(X_test)
                    if len(np.unique(y_test)) == 2:
                        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                    else:
                        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="weighted")
                else:
                    roc_auc = None
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC for {model_name}: {e}")
                roc_auc = None
            
            # Store metrics
            metrics_list.append({
                "model": model_name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc if roc_auc is not None else np.nan,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "cv_scores": f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
                "best_params": str(grid_search.best_params_),
                "train_accuracy": accuracy_score(y_train, y_pred_train)
            })
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred_test)
            cm_dict = {
                "model": model_name,
                "confusion_matrix": str(cm.tolist())
            }
            
            # Add individual confusion matrix values
            if cm.shape == (2, 2):
                cm_dict.update({
                    "tn": int(cm[0, 0]),
                    "fp": int(cm[0, 1]),
                    "fn": int(cm[1, 0]),
                    "tp": int(cm[1, 1])
                })
            
            confusion_list.append(cm_dict)
            
            # Store trained model
            trained_models[model_name] = best_model
            
            logger.info(f"{model_name} - Test F1: {f1:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            continue
    
    # Create DataFrames
    metrics_df = pd.DataFrame(metrics_list)
    confusion_df = pd.DataFrame(confusion_list)
    
    # Sort by F1 score
    metrics_df = metrics_df.sort_values("f1_score", ascending=False)
    
    logger.info(f"Successfully trained {len(trained_models)} classification models")
    logger.info(f"Best model by F1 score: {metrics_df.iloc[0]['model']} with F1={metrics_df.iloc[0]['f1_score']:.4f}")
    
    return metrics_df, confusion_df, trained_models


def generate_classification_feature_importances(
    trained_models: Dict,
    X_train: pd.DataFrame
) -> pd.DataFrame:
    """
    Extract feature importances from tree-based models.
    
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
    
    if importance_list:
        importance_df = pd.DataFrame(importance_list)
        importance_df = importance_df.sort_values(["model", "importance"], ascending=[True, False])
        return importance_df
    else:
        logger.warning("No feature importances extracted from models")
        return pd.DataFrame(columns=["model", "feature", "importance"])
