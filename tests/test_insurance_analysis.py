"""Tests for insurance data validation, analysis, and modeling."""

import json
from math import isfinite

import pandas as pd
import pytest

from src.insurance_analysis import (
    load_and_validate_data,
    prepare_features,
    summarize_data,
    train_model,
    validate_data,
)
from src.main import run_workflow


def sample_data(row_count=12):
    """Create valid deterministic insurance records for tests."""
    rows = []
    regions = ["northeast", "northwest", "southeast", "southwest"]
    for index in range(row_count):
        age = 20 + index
        bmi = 22.0 + index * 0.7
        children = index % 4
        smoker = "yes" if index % 3 == 0 else "no"
        smoker_value = 1 if smoker == "yes" else 0
        charges = 1000 + 120 * age + 80 * bmi + 300 * children + 9000 * smoker_value
        rows.append(
            {
                "age": age,
                "sex": "female" if index % 2 == 0 else "male",
                "bmi": bmi,
                "children": children,
                "smoker": smoker,
                "region": regions[index % len(regions)],
                "charges": charges,
            }
        )
    return pd.DataFrame(rows)


def test_valid_csv_loads_and_normalizes_categories(tmp_path):
    data = sample_data(10)
    data.loc[0, "smoker"] = " YES "
    csv_path = tmp_path / "insurance.csv"
    data.to_csv(csv_path, index=False)

    loaded, info = load_and_validate_data(csv_path)

    assert list(loaded.columns) == list(data.columns)
    assert len(loaded) == 10
    assert loaded.loc[0, "smoker"] == "yes"
    assert info == {
        "rows_before_cleaning": 10,
        "rows_after_cleaning": 10,
        "duplicates_removed": 0,
    }


def test_exact_duplicate_rows_are_removed(tmp_path):
    data = sample_data(10)
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    csv_path = tmp_path / "insurance.csv"
    data.to_csv(csv_path, index=False)

    loaded, info = load_and_validate_data(csv_path)

    assert len(loaded) == 10
    assert info["duplicates_removed"] == 1


def test_missing_required_column_names_the_column(tmp_path):
    csv_path = tmp_path / "insurance.csv"
    sample_data().drop(columns="region").to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required column.*region"):
        load_and_validate_data(csv_path)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("bmi", float("nan"), "Missing required value.*bmi"),
        ("smoker", "sometimes", "only 'yes' or 'no'"),
        ("age", -1, "age.*nonnegative"),
        ("bmi", 0, "bmi.*positive"),
        ("children", 1.5, "children.*integer"),
        ("charges", -1, "charges.*nonnegative"),
        ("region", " ", "Missing required value.*region"),
    ],
)
def test_invalid_required_values_are_rejected(column, value, message):
    data = sample_data()
    if column == "children":
        data[column] = data[column].astype(float)
    data.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        validate_data(data)


def test_smoker_region_summaries_and_correlation():
    data = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "sex": ["female", "male", "female", "male"],
            "bmi": [20.0, 25.0, 30.0, 35.0],
            "children": [0, 1, 0, 1],
            "smoker": ["no", "no", "yes", "yes"],
            "region": ["east", "west", "east", "west"],
            "charges": [100.0, 200.0, 300.0, 400.0],
        }
    )

    summary = summarize_data(data)

    assert summary["charges_by_smoker"]["no"] == {
        "count": 2,
        "mean_charges": 150.0,
    }
    assert summary["charges_by_region"]["east"]["mean_charges"] == 200.0
    assert summary["bmi_charges_correlation"] == pytest.approx(1.0)


def test_no_smokers_returns_only_observed_group():
    data = sample_data()
    data["smoker"] = "no"

    groups = summarize_data(data)["charges_by_smoker"]

    assert list(groups) == ["no"]
    assert groups["no"]["count"] == len(data)


def test_constant_bmi_correlation_is_json_compatible_null():
    data = sample_data()
    data["bmi"] = 25.0

    summary = summarize_data(data)

    assert summary["bmi_charges_correlation"] is None
    assert "null" in json.dumps(summary, allow_nan=False)


def test_feature_preparation_has_exact_order_and_encoding():
    data = sample_data()

    features, target = prepare_features(data)

    assert list(features.columns) == ["age", "bmi", "children", "smoker_encoded"]
    assert features.loc[0, "smoker_encoded"] == 1
    assert features.loc[1, "smoker_encoded"] == 0
    assert target.equals(data["charges"])


def test_model_workflow_is_finite_repeatable_and_nearly_perfect():
    data = sample_data(30)

    first_metrics, first_predictions = train_model(data)
    second_metrics, second_predictions = train_model(data)

    assert first_metrics["train_rows"] == 24
    assert first_metrics["test_rows"] == 6
    assert len(first_predictions) == first_metrics["test_rows"]
    assert all(isfinite(value) for value in first_metrics.values())
    assert first_metrics["mean_squared_error"] == pytest.approx(0.0, abs=1e-15)
    assert first_metrics["r_squared"] == pytest.approx(1.0)
    pd.testing.assert_frame_equal(first_predictions, second_predictions)
    assert first_metrics == second_metrics


def test_too_few_rows_are_rejected():
    with pytest.raises(ValueError, match="At least 10 cleaned rows"):
        train_model(sample_data(9))


def test_complete_workflow_writes_parseable_outputs(tmp_path):
    csv_path = tmp_path / "insurance.csv"
    output_path = tmp_path / "results"
    sample_data(20).to_csv(csv_path, index=False)

    run_workflow(csv_path, output_path)

    with (output_path / "analysis_summary.json").open(encoding="utf-8") as file:
        analysis = json.load(file)
    with (output_path / "model_metrics.json").open(encoding="utf-8") as file:
        metrics = json.load(file)
    predictions = pd.read_csv(output_path / "test_predictions.csv")

    assert analysis["cleaning"]["rows_after_cleaning"] == 20
    assert metrics["test_rows"] == 4
    assert list(predictions.columns) == [
        "original_index",
        "actual_charges",
        "predicted_charges",
    ]
    assert predictions["original_index"].is_monotonic_increasing
