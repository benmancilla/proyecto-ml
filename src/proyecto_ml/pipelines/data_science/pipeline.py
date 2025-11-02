from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    add_loyalty_target,
    encode_features,
    split_data,
    train_model,
    evaluate_model,
    model_delivery_tasks,
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=add_loyalty_target,
            inputs="full_orders",
            outputs="model_input_with_target",
            name="add_loyalty_target_node",
        ),
        node(
            func=encode_features,
            inputs="model_input_with_target",
            outputs="model_input_encoded",
            name="encode_features_node",
        ),
        node(
            func=split_data,
            inputs=["model_input_encoded", "params:data_science"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data_node",
        ),
        node(
            func=train_model,
            inputs=["X_train", "y_train"],
            outputs="rf_model",
            name="train_model_node",
        ),
        node(
            func=evaluate_model,
            inputs=["rf_model", "X_test", "y_test"],
            outputs="model_metrics",
            name="evaluate_model_node",
        ),
        # ---------
        node(
            func=model_delivery_tasks,
            inputs=["model_input_with_target", "params:data_science"],
            outputs=[
                "regression_metrics",
                "classification_metrics",
                "classification_confusion_matrices",
                "feature_importances",
                "classification_feature_importances",
            ],
            name="model_delivery_tasks_node",
        ),
    ])
