import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


# 1. Add target for loyalty (repeat purchase)
def add_loyalty_target(full_orders: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary target 'loyal_customer' = 1 if a customer has > 1 orders.

    Args:
        full_orders: DataFrame with at least 'customer_id' and 'order_id'.

    Returns:
        DataFrame including the new 'loyal_customer' column.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {"customer_id", "order_id"}
    missing = required - set(full_orders.columns)
    if missing:
        raise ValueError(f"Missing required columns in full_orders: {missing}")

    logger.info("Building loyalty target from %d rows", len(full_orders))
    customer_order_counts = (
        full_orders.groupby("customer_id")["order_id"].count().reset_index()
    )
    customer_order_counts["loyal_customer"] = (
        customer_order_counts["order_id"] > 1
    ).astype(int)
    df = full_orders.merge(
        customer_order_counts[["customer_id", "loyal_customer"]],
        on="customer_id"
    )
    logger.info(
        "Loyal customers ratio: %.3f",
        df["loyal_customer"].mean() if "loyal_customer" in df.columns else float("nan"),
    )
    return df


# 2. Encode categorical features
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode selected categorical columns and drop the originals.

    Currently encodes: 'customer_state', 'customer_city'.

    Args:
        df: Input DataFrame containing the categorical columns.

    Returns:
        Transformed DataFrame with encoded columns.

    Raises:
        ValueError: If the expected categorical columns are missing.
    """
    df_encoded = df.copy()
    cat_cols = ["customer_state", "customer_city"]
    missing = [c for c in cat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing categorical columns for encoding: {missing}")

    # Compatibility with scikit-learn >=1.2 (sparse_output) and older versions (sparse)
    try:
        encoder = OneHotEncoder(sparse_output=False, drop="first")
    except TypeError:
        encoder = OneHotEncoder(sparse=False, drop="first")
    logger.info("Encoding categorical columns: %s", cat_cols)
    encoded = pd.DataFrame(
        encoder.fit_transform(df_encoded[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols),
        index=df_encoded.index,
    )
    df_encoded = pd.concat([df_encoded.drop(columns=cat_cols), encoded], axis=1)
    logger.info("Encoded shape: %s", df_encoded.shape)
    return df_encoded


# 3. Split into train/test
def split_data(df: pd.DataFrame, params: dict):
    """
    Prepare feature matrix and split into train/test with optional downsampling.

    Args:
        df: DataFrame including 'loyal_customer' target.
        params: Dictionary that may include 'max_rows', 'test_size', 'random_state'.

    Returns:
        X_train, X_test, y_train, y_test

    Raises:
        ValueError: If 'loyal_customer' is missing.
    """
    if "loyal_customer" not in df.columns:
        raise ValueError("Target column 'loyal_customer' not found.")

    # Extract parameters with safe defaults
    max_rows = int(params.get("max_rows", 10000)) if isinstance(params, dict) else 10000
    test_size = float(params.get("test_size", 0.2)) if isinstance(params, dict) else 0.2
    random_state = int(params.get("random_state", 42)) if isinstance(params, dict) else 42

    logger.info(
        "Splitting data: rows=%d, max_rows=%s, test_size=%.2f, random_state=%d",
        len(df), str(max_rows), test_size, random_state
    )

    # Optional downsample (preserve class balance when possible)
    if max_rows and max_rows > 0 and len(df) > max_rows:
        # Use stratify only if there are at least 2 samples per class
        y_down = df["loyal_customer"] if "loyal_customer" in df.columns else None
        if y_down is not None and y_down.nunique() > 1 and y_down.value_counts().min() >= 2:
            stratify_down = y_down
        else:
            stratify_down = None
        df, _ = train_test_split(
            df,
            train_size=max_rows,
            random_state=random_state,
            stratify=stratify_down,
        )
        logger.info("Downsampled to %d rows", len(df))

    # Drop non-feature columns if present
    drop_cols = [c for c in ["order_id", "customer_id", "loyal_customer", "order_status", "order_year_month"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["loyal_customer"]

    # Keep only numeric/bool features (avoid string/datetime conversion errors)
    X = X.select_dtypes(include=["number", "bool"]).copy()
    # Convert bool to int and handle missing values
    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(int)
    X = X.fillna(0)

    # Use stratify only if there are at least 2 samples per class
    stratify_y = y if (y.nunique() > 1 and y.value_counts().min() >= 2) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
    )
    logger.info(
        "Train/Test shapes: X_train=%s, X_test=%s",
        X_train.shape, X_test.shape
    )
    return X_train, X_test, y_train, y_test


# 4. Train Random Forest
def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier with fixed hyperparameters.

    Args:
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Fitted RandomForestClassifier instance.
    """
    logger.info("Training RandomForestClassifier on %d samples, %d features", *X_train.shape)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# 5. Evaluate model
def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Evaluate model accuracy and return a long-form metrics DataFrame.

    If the model exposes feature_importances_, include them as 'fi__<feature>'.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info("Accuracy: %.4f", accuracy)

    # Prepare a long-form DataFrame suitable for CSV saving
    metrics = [{"metric": "accuracy", "value": float(accuracy)}]
    if hasattr(model, "feature_importances_"):
        for feat, imp in zip(X_test.columns, model.feature_importances_):
            metrics.append({"metric": f"fi__{feat}", "value": float(imp)})

    return pd.DataFrame(metrics)
