from kedro.pipeline import Pipeline, node, pipeline
from .nodes import clean_orders, clean_customers, merge_orders_customers, build_model_input


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_orders,
            inputs="orders",
            outputs="orders_clean",
            name="clean_orders_node",
        ),
        node(
            func=clean_customers,
            inputs="customers",
            outputs="customers_clean",
            name="clean_customers_node",
        ),
        node(
            func=merge_orders_customers,
            inputs=["orders_clean", "customers_clean"],
            outputs="full_orders",
            name="merge_orders_customers_node",
        ),
        node(
            func=build_model_input,
            inputs="full_orders",
            outputs="model_input",
            name="build_model_input_node",
        ),
    ])
