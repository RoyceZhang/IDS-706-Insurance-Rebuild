"""Data validation, descriptive analysis, and modeling functions."""

from math import isfinite, sqrt
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region",
    "charges",
]
NUMERIC_COLUMNS = ["age", "bmi", "children", "charges"]
CATEGORICAL_COLUMNS = ["sex", "smoker", "region"]
SMOKER_MAPPING = {"no": 0, "yes": 1}
MINIMUM_MODEL_ROWS = 10


# Data loading and validation


def _check_required_columns(data):
    """Raise a clear error when one or more required columns are absent."""
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in data.columns
    ]
    if missing_columns:
        names = ", ".join(missing_columns)
        raise ValueError(f"Missing required column(s): {names}")


def load_and_validate_data(csv_path):
    """Load a CSV file, remove exact duplicates, and validate its values."""
    path = Path(csv_path)

    try:
        data = pd.read_csv(path)
    except FileNotFoundError as error:
        raise ValueError(f"Data file does not exist: {path}") from error
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Data file is empty: {path}") from error
    except (OSError, pd.errors.ParserError) as error:
        raise ValueError(f"Could not read data file {path}: {error}") from error

    if data.empty:
        raise ValueError(f"Data file contains no data rows: {path}")

    _check_required_columns(data)
    rows_before = len(data)
    cleaned = data.drop_duplicates().copy()
    duplicates_removed = rows_before - len(cleaned)

    cleaned = validate_data(cleaned)
    cleaning_info = {
        "rows_before_cleaning": rows_before,
        "rows_after_cleaning": len(cleaned),
        "duplicates_removed": duplicates_removed,
    }
    return cleaned, cleaning_info


def validate_data(data):
    """Return a validated copy of an insurance DataFrame."""
    _check_required_columns(data)

    validated = data.copy()

    for column in CATEGORICAL_COLUMNS:
        validated[column] = validated[column].astype("string").str.strip().str.lower()

    missing_value_columns = []
    for column in REQUIRED_COLUMNS:
        values = validated[column]
        has_missing = values.isna().any()
        if column in CATEGORICAL_COLUMNS:
            has_missing = has_missing or values.eq("").any()
        if has_missing:
            missing_value_columns.append(column)

    if missing_value_columns:
        names = ", ".join(missing_value_columns)
        raise ValueError(f"Missing required value(s) in column(s): {names}")

    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(validated[column], errors="coerce")
        if converted.isna().any() or not converted.map(isfinite).all():
            raise ValueError(f"Column '{column}' must contain finite numeric values")
        validated[column] = converted

    if (validated["age"] < 0).any():
        raise ValueError("Column 'age' must contain nonnegative values")
    if (validated["bmi"] <= 0).any():
        raise ValueError("Column 'bmi' must contain positive values")
    if (validated["children"] < 0).any():
        raise ValueError("Column 'children' must contain nonnegative values")
    if not validated["children"].mod(1).eq(0).all():
        raise ValueError("Column 'children' must contain integer values")
    if (validated["charges"] < 0).any():
        raise ValueError("Column 'charges' must contain nonnegative values")

    if not validated["smoker"].isin(SMOKER_MAPPING).all():
        raise ValueError("Column 'smoker' must contain only 'yes' or 'no'")

    return validated


# Descriptive analysis


def _json_number(value):
    """Return a regular float, or None for a value JSON cannot represent."""
    if pd.isna(value) or not isfinite(value):
        return None
    return float(value)


def _numeric_summary(series):
    """Calculate the selected statistics used in the JSON summary."""
    return {
        "count": int(series.count()),
        "mean": _json_number(series.mean()),
        "median": _json_number(series.median()),
        "standard_deviation": _json_number(series.std()),
        "minimum": _json_number(series.min()),
        "maximum": _json_number(series.max()),
    }


def _charge_groups(data, column):
    """Calculate counts and mean charges for one categorical column."""
    results = {}
    for name, group in data.groupby(column):
        results[str(name)] = {
            "count": len(group),
            "mean_charges": float(group["charges"].mean()),
        }
    return results


def summarize_data(data, cleaning_info=None):
    """Build a small descriptive summary of validated insurance data."""
    if data.empty:
        raise ValueError("Cannot summarize an empty dataset")

    if cleaning_info is None:
        cleaning_info = {
            "rows_before_cleaning": len(data),
            "rows_after_cleaning": len(data),
            "duplicates_removed": 0,
        }

    if data["bmi"].nunique() < 2 or data["charges"].nunique() < 2:
        correlation = None
    else:
        correlation = data["bmi"].corr(data["charges"])
    correlation = _json_number(correlation)

    numeric_summary = {}
    for column in ["age", "bmi", "charges"]:
        numeric_summary[column] = _numeric_summary(data[column])

    return {
        "cleaning": cleaning_info,
        "numeric_summary": numeric_summary,
        "charges_by_smoker": _charge_groups(data, "smoker"),
        "charges_by_region": _charge_groups(data, "region"),
        "bmi_charges_correlation": correlation,
    }


# Feature preparation and modeling


def prepare_features(data):
    """Prepare the four model features from validated data."""
    features = data[["age", "bmi", "children"]].copy()
    features["smoker_encoded"] = data["smoker"].map(SMOKER_MAPPING)
    target = data["charges"].copy()
    return features, target


def train_model(data):
    """Train and evaluate deterministic linear regression on validated data."""
    if len(data) < MINIMUM_MODEL_ROWS:
        raise ValueError(
            f"At least {MINIMUM_MODEL_ROWS} cleaned rows are required for modeling; "
            f"received {len(data)}"
        )

    features, target = prepare_features(data)
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(train_features, train_target)
    predicted = model.predict(test_features)

    mse = mean_squared_error(test_target, predicted)
    metrics = {
        "train_rows": len(train_features),
        "test_rows": len(test_features),
        "mean_absolute_error": float(mean_absolute_error(test_target, predicted)),
        "mean_squared_error": float(mse),
        "root_mean_squared_error": float(sqrt(mse)),
        "r_squared": float(r2_score(test_target, predicted)),
    }

    predictions = pd.DataFrame(
        {
            "original_index": test_target.index,
            "actual_charges": test_target.to_numpy(),
            "predicted_charges": predicted,
        }
    )
    predictions = predictions.sort_values("original_index")
    predictions = predictions.reset_index(drop=True)

    return metrics, predictions
