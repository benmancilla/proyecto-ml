import pandas as pd
import numpy as np


def compute_spearman_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Spearman correlation matrix for numeric columns.

    Args:
        df: Input DataFrame with mixed dtypes.

    Returns:
        A DataFrame (square matrix) with Spearman correlation between numeric columns.
    """
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        # Return empty DataFrame to avoid breaking the pipeline
        return pd.DataFrame()
    corr = df[num_cols].corr(method="spearman")
    return corr


def describe_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Generate basic descriptive statistics (numerical summary)."""
    return df.describe(include="all").transpose()


def count_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Count missing values per column."""
    missing = df.isnull().sum().reset_index()
    missing.columns = ["column", "missing_values"]
    return missing


def distribution_by_state(customers_clean: pd.DataFrame) -> pd.DataFrame:
    """Example: count customers per state."""
    return customers_clean.groupby("customer_state").size().reset_index(name="count")


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor (VIF) for numeric columns.

    Note: requires statsmodels to be installed.

    Args:
        df: Input DataFrame with numeric columns.

    Returns:
        DataFrame with columns [feature, VIF], sorted descending by VIF.
    """
    try:
        import statsmodels.api as sm  # type: ignore
        from statsmodels.stats.outliers_influence import variance_inflation_factor  # type: ignore
    except Exception:
        # Return empty DataFrame with expected columns if statsmodels is unavailable
        return pd.DataFrame({"feature": [], "VIF": []})

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        return pd.DataFrame({"feature": [], "VIF": []})

    X = df[num_cols].copy()
    # Fill missing with median to allow VIF computation
    X = X.fillna(X.median(numeric_only=True))
    X_const = sm.add_constant(X, has_constant="add")
    vif_values = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
    vif_df = pd.DataFrame({"feature": X_const.columns, "VIF": vif_values})
    vif_df = vif_df[vif_df["feature"] != "const"].sort_values("VIF", ascending=False).reset_index(drop=True)
    return vif_df
