import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
)

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

    Expects both 'customer_state' and 'customer_city' to be present for schema
    validation; encodes only 'customer_state' and drops 'customer_city' to avoid
    high cardinality.

    Args:
        df: Input DataFrame containing the categorical columns.

    Returns:
        Transformed DataFrame with encoded columns.

    Raises:
        ValueError: If the expected categorical columns are missing.
    """
    df_encoded = df.copy()
    
    # Sample data to reduce processing time (use 10,000 rows max)
    max_sample = 10000
    if len(df_encoded) > max_sample:
        df_encoded = df_encoded.sample(n=max_sample, random_state=42)
        logger.info(f"Sampled {max_sample} rows from {len(df)} for faster processing")
    
    # Require both columns for validation per tests, but only encode state
    required_cols = ["customer_state", "customer_city"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing categorical columns for encoding: {missing}")

    # Only encode customer_state, drop customer_city (too many unique values)
    cat_cols = ["customer_state"]

    # Drop customer_city to avoid high cardinality issues
    if "customer_city" in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=["customer_city"])
        logger.info("Dropped 'customer_city' column to avoid high cardinality")

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


# 6. Unified modeling for delivery_delay (regression) and delay_flag (classification)
def model_delivery_tasks(df: pd.DataFrame, params: dict) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Train and evaluate regression (delivery_delay) and classification (delay_flag)
    models as implemented in the notebooks, returning metrics and importances.

    Args:
        df: Full orders merged dataset with delivery fields and customer info.
        params: Dict with optional keys: 'sample_frac', 'cv_folds', 'random_state'

    Returns:
        (regression_metrics, classification_metrics, classification_confusions,
         regression_feature_importances, classification_feature_importances)
    """
    rng = int(params.get("random_state", 42)) if isinstance(params, dict) else 42
    sample_frac = float(params.get("sample_frac", 0.4)) if isinstance(params, dict) else 0.4
    cv_folds = int(params.get("cv_folds", 5)) if isinstance(params, dict) else 5

    data = df.copy()
    # Keep delivered and required columns; derive year/month
    data = data[data.get("order_status").eq("delivered")] if "order_status" in data.columns else data
    if "order_purchase_timestamp" in data.columns:
        data["order_purchase_timestamp"] = pd.to_datetime(data["order_purchase_timestamp"])  # type: ignore
        data["order_year"] = data["order_purchase_timestamp"].dt.year
        data["order_month"] = data["order_purchase_timestamp"].dt.month

    # Targets
    required_cols = [
        "delivery_delay", "delivery_time", "customer_state", "customer_zip_code_prefix",
        "order_year", "order_month"
    ]
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns for modeling: {missing}")

    data = data.dropna(subset=["delivery_delay", "delivery_time", "customer_state", "customer_zip_code_prefix"])  # type: ignore
    data["customer_zip_code_prefix"] = pd.to_numeric(data["customer_zip_code_prefix"], errors="coerce")
    data = data.dropna(subset=["customer_zip_code_prefix"])  # type: ignore
    data["delay_flag"] = (data["delivery_delay"] > 0).astype(int)

    if 0 < sample_frac < 1.0:
        data = data.sample(frac=sample_frac, random_state=rng)

    feature_columns = [
        "delivery_time", "customer_zip_code_prefix", "order_month", "order_year", "customer_state"
    ]
    numeric_features = ["delivery_time", "customer_zip_code_prefix", "order_month", "order_year"]
    categorical_features = ["customer_state"]

    # Preprocessor
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Prepare splits
    X = data[feature_columns]
    y_reg = data["delivery_delay"]
    y_clf = data["delay_flag"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=rng
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=rng
    )

    X_clf_train_full, X_clf_test, y_clf_train_full, y_clf_test = train_test_split(
        X, y_clf, test_size=0.2, stratify=y_clf, random_state=rng
    )
    X_clf_train, X_clf_val, y_clf_train, y_clf_val = train_test_split(
        X_clf_train_full, y_clf_train_full, test_size=0.2, stratify=y_clf_train_full, random_state=rng
    )

    # Regression models
    reg_models = {
        "LinearRegression": Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())]),
        "RandomForest": Pipeline([("preprocessor", preprocessor), ("regressor", RandomForestRegressor(random_state=rng, n_estimators=300))]),
        "GradientBoosting": Pipeline([("preprocessor", preprocessor), ("regressor", GradientBoostingRegressor(random_state=rng))]),
    }
    reg_params = {
        "RandomForest": {
            "regressor__n_estimators": [200, 300],
            "regressor__max_depth": [None, 12],
        },
        "GradientBoosting": {
            "regressor__n_estimators": [200, 300],
            "regressor__learning_rate": [0.05, 0.1],
        },
        "LinearRegression": {},
    }

    regression_metrics = []
    feature_importances_rows = []

    for name, pipe in reg_models.items():
        if reg_params[name]:
            search = RandomizedSearchCV(pipe, reg_params[name], n_iter=4, cv=cv_folds, scoring="neg_root_mean_squared_error", n_jobs=-1, random_state=rng)
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
        else:
            best_model = pipe.fit(X_train, y_train)

        # Train on full train
        best_model.fit(X_train, y_train)
        # CV on training only (R2)
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring="r2", n_jobs=-1)
        # Metrics
        for split_name, X_split, y_split in [("validation", X_val, y_val), ("test", X_test, y_test)]:
            y_pred = best_model.predict(X_split)
            regression_metrics.append({
                "model": name,
                "split": split_name,
                "r2": r2_score(y_split, y_pred),
                "mae": mean_absolute_error(y_split, y_pred),
                "rmse": float(np.sqrt(mean_squared_error(y_split, y_pred))),
                "cv_r2_mean": float(cv_scores.mean()),
                "cv_r2_std": float(cv_scores.std()),
            })

        # Store permutation-like approximate importances via RF feature_importances_ when available
        if name == "RandomForest":
            # Get feature names after preprocessor
            pre = best_model.named_steps["preprocessor"]
            try:
                feat_names = pre.get_feature_names_out(feature_columns)
            except Exception:
                feat_names = pre.get_feature_names_out()
            importances = best_model.named_steps["regressor"].feature_importances_
            for f, imp in zip(feat_names, importances):
                feature_importances_rows.append({"feature": f, "importance_mean": float(imp)})

    regression_metrics_df = pd.DataFrame(regression_metrics)
    feature_importances_df = pd.DataFrame(feature_importances_rows).sort_values("importance_mean", ascending=False)

    # Classification models (exclude KNN to avoid memory issues with large sparse matrices)
    clf_models = {
        "LogisticRegression": Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=rng))]),
        "RandomForestClassifier": Pipeline([("preprocessor", preprocessor), ("classifier", RandomForestClassifier(random_state=rng, class_weight="balanced"))]),
        "SVC": Pipeline([("preprocessor", preprocessor), ("classifier", SVC(probability=True, class_weight="balanced", random_state=rng))]),
    }
    clf_params = {
        "LogisticRegression": {"classifier__C": [0.1, 1.0, 10.0]},
        "RandomForestClassifier": {"classifier__n_estimators": [200, 300], "classifier__max_depth": [None, 12]},
        "SVC": {"classifier__C": [0.5, 1.0], "classifier__gamma": ["scale", "auto"]},
    }

    # Optional simple oversampling if highly imbalanced
    cls_counts = y_clf_train.value_counts()
    if cls_counts.min() / cls_counts.max() < 0.8:
        # naive oversampling on training features only (no leak):
        train_clf_df = X_clf_train.copy()
        train_clf_df["delay_flag"] = y_clf_train.values
        maj = cls_counts.idxmax()
        frames = [train_clf_df[train_clf_df["delay_flag"] == maj]]
        for label, subset in train_clf_df.groupby("delay_flag"):
            if label == maj:
                continue
            frames.append(subset.sample(n=int(cls_counts.max()), replace=True, random_state=rng))
        train_balanced_df = pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=rng)
        X_clf_train_use = train_balanced_df.drop(columns="delay_flag")
        y_clf_train_use = train_balanced_df["delay_flag"]
    else:
        X_clf_train_use = X_clf_train
        y_clf_train_use = y_clf_train

    classification_metrics = []
    confusion_rows = []
    clf_importances_rows = []

    for name, pipe in clf_models.items():
        search = RandomizedSearchCV(pipe, clf_params[name], n_iter=4, cv=cv_folds, scoring="f1", n_jobs=-1, random_state=rng)
        search.fit(X_clf_train_use, y_clf_train_use)
        best_model = search.best_estimator_

        cv_scores = cross_val_score(best_model, X_clf_train_use, y_clf_train_use, cv=cv_folds, scoring="f1", n_jobs=-1)

        y_val_pred = best_model.predict(X_clf_val)
        y_test_pred = best_model.predict(X_clf_test)
        for split_name, y_true, y_pred in [("validation", y_clf_val, y_val_pred), ("test", y_clf_test, y_test_pred)]:
            classification_metrics.append({
                "model": name,
                "split": split_name,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "cv_f1_mean": float(cv_scores.mean()),
                "cv_f1_std": float(cv_scores.std()),
            })

        cm = confusion_matrix(y_clf_test, y_test_pred)
        confusion_rows.append({
            "model": name,
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        })

        if name == "RandomForestClassifier":
            pre = best_model.named_steps["preprocessor"]
            try:
                feat_names = pre.get_feature_names_out(feature_columns)
            except Exception:
                feat_names = pre.get_feature_names_out()
            importances = best_model.named_steps["classifier"].feature_importances_
            for f, imp in zip(feat_names, importances):
                clf_importances_rows.append({"feature": f, "importance": float(imp)})

    classification_metrics_df = pd.DataFrame(classification_metrics)
    confusion_df = pd.DataFrame(confusion_rows)
    clf_importances_df = pd.DataFrame(clf_importances_rows).sort_values("importance", ascending=False)

    return (
        regression_metrics_df,
        classification_metrics_df,
        confusion_df,
        feature_importances_df,
        clf_importances_df,
    )
