"""
Pipeline definition for classification models.
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    prepare_classification_data,
    train_classification_models,
    generate_classification_feature_importances
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the classification pipeline with multiple models and GridSearchCV.
    
    Returns:
        Pipeline object with classification nodes
    """
    return pipeline([
        node(
            func=prepare_classification_data,
            inputs=["X_clf_train", "X_clf_test", "y_clf_train", "y_clf_test"],
            outputs=["X_clf_train_scaled", "X_clf_test_scaled", "y_clf_train_prepared", "y_clf_test_prepared"],
            name="prepare_classification_data_node",
        ),
        node(
            func=train_classification_models,
            inputs=[
                "X_clf_train_scaled",
                "X_clf_test_scaled",
                "y_clf_train_prepared",
                "y_clf_test_prepared",
                "params:classification"
            ],
            outputs=[
                "classification_metrics_extended",
                "classification_confusion_matrices_extended",
                "classification_trained_models"
            ],
            name="train_classification_models_node",
        ),
        node(
            func=generate_classification_feature_importances,
            inputs=["classification_trained_models", "X_clf_train_scaled"],
            outputs="classification_feature_importances_extended",
            name="generate_classification_feature_importances_node",
        ),
    ])
