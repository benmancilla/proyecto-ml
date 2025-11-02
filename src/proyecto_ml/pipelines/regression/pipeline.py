"""
Pipeline definition for regression models.
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    prepare_regression_data,
    train_regression_models
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the regression pipeline with multiple models and GridSearchCV.
    
    Returns:
        Pipeline object with regression nodes
    """
    return pipeline([
        node(
            func=prepare_regression_data,
            inputs=["X_train", "X_test", "y_train", "y_test"],
            outputs=["X_train_scaled", "X_test_scaled", "y_train_prepared", "y_test_prepared"],
            name="prepare_regression_data_node",
        ),
        node(
            func=train_regression_models,
            inputs=[
                "X_train_scaled",
                "X_test_scaled",
                "y_train_prepared",
                "y_test_prepared",
                "params:regression"
            ],
            outputs=[
                "regression_metrics_extended",
                "regression_feature_importances_extended",
                "regression_trained_models"
            ],
            name="train_regression_models_node",
        ),
    ])
