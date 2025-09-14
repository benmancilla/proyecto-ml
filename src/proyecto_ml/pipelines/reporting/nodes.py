import pandas as pd


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
