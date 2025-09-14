from kedro.pipeline import Pipeline, node, pipeline
from .nodes import describe_dataset, count_missing_values, distribution_by_state


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=describe_dataset,
            inputs="model_input",
            outputs="model_input_stats",
            name="describe_dataset_node",
        ),
        node(
            func=count_missing_values,
            inputs="model_input",
            outputs="model_input_missing",
            name="count_missing_values_node",
        ),
        node(
            func=distribution_by_state,
            inputs="customers_clean",
            outputs="customers_by_state",
            name="distribution_by_state_node",
        ),
    ])
