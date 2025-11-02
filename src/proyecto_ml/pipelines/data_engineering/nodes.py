import pandas as pd


def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    orders = orders.copy()
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    orders["order_delivered_timestamp"] = pd.to_datetime(orders["order_delivered_timestamp"])
    orders["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"])

    # Derived variables
    orders["delivery_delay"] = (
        (orders["order_delivered_timestamp"] - orders["order_estimated_delivery_date"]).dt.days
    )
    orders["delivery_time"] = (
        (orders["order_delivered_timestamp"] - orders["order_purchase_timestamp"]).dt.days
    )
    orders["order_year_month"] = orders["order_purchase_timestamp"].dt.to_period("M")
    # Additional numeric splits for modeling
    orders["order_year"] = orders["order_purchase_timestamp"].dt.year
    orders["order_month"] = orders["order_purchase_timestamp"].dt.month

    return orders


def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    customers = customers.copy()
    customers = customers.drop_duplicates(subset=["customer_id"])
    return customers


def merge_orders_customers(
    orders_clean: pd.DataFrame, customers_clean: pd.DataFrame
) -> pd.DataFrame:
    return orders_clean.merge(customers_clean, on="customer_id", how="inner")


def build_model_input(full_orders: pd.DataFrame) -> pd.DataFrame:
    """Select and format final features for ML model input."""
    model_input = full_orders.copy()

    # Example feature selection (can be extended later)
    keep_cols = [
        "customer_id",
        "customer_city",
        "customer_state",
        "customer_zip_code_prefix",
        "delivery_delay",
        "delivery_time",
        "order_year_month",
        "order_year",
        "order_month",
        "order_status",
        "order_purchase_timestamp",
    ]
    model_input = model_input[keep_cols]

    return model_input
