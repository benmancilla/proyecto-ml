import pandas as pd
from proyecto_ml.pipelines.data_science import nodes


def test_add_loyalty_target_creates_flag():
    df = pd.DataFrame({
        "customer_id": [1, 1, 2],
        "order_id": [10, 11, 12]
    })
    out = nodes.add_loyalty_target(df)
    assert "loyal_customer" in out.columns
    ratio = out.groupby("customer_id")["loyal_customer"].first().tolist()
    assert ratio == [1, 0]


essential_cols = ["order_id", "customer_id", "loyal_customer", "num_feat"]

def test_split_data_returns_expected_shapes():
    df = pd.DataFrame({
        "order_id": [1, 2, 3, 4],
        "customer_id": [1, 1, 2, 3],
        "loyal_customer": [0, 0, 1, 0],
        "num_feat": [0.1, 0.2, 0.3, 0.4],
    })

    X_train, X_test, y_train, y_test = nodes.split_data(df, params={"max_rows": 4, "test_size": 0.5, "random_state": 42})
    assert X_train.shape[0] + X_test.shape[0] == 4
    assert list(X_train.columns) == ["num_feat"]


def test_encode_features_raises_on_missing_cols():
    df = pd.DataFrame({"customer_state": ["S"], "other": [1]})
    try:
        nodes.encode_features(df)
        assert False, "Expected ValueError for missing 'customer_city'"
    except ValueError:
        pass
