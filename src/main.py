"""Command-line entry point for the insurance analysis workflow."""

import argparse
import json
import sys
from pathlib import Path

from src.insurance_analysis import (
    load_and_validate_data,
    summarize_data,
    train_model,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze medical insurance data and train a linear model."
    )
    parser.add_argument(
        "--data",
        default="data/insurance.csv",
        help="Path to the input CSV (default: data/insurance.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for generated results (default: outputs)",
    )
    return parser.parse_args()


def run_workflow(data_path, output_dir):
    """Run analysis and modeling, then write the three output artifacts."""
    data, cleaning_info = load_and_validate_data(data_path)
    analysis_summary = summarize_data(data, cleaning_info)
    model_metrics, predictions = train_model(data)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with (output_path / "analysis_summary.json").open("w", encoding="utf-8") as file:
        json.dump(analysis_summary, file, indent=2, allow_nan=False)

    with (output_path / "model_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(model_metrics, file, indent=2, allow_nan=False)

    predictions.to_csv(output_path / "test_predictions.csv", index=False)
    return analysis_summary, model_metrics


def main():
    """Run the command-line workflow and return its exit code."""
    args = parse_args()
    try:
        summary, metrics = run_workflow(args.data, args.output_dir)
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    cleaning = summary["cleaning"]
    correlation = summary["bmi_charges_correlation"]
    correlation_text = "undefined" if correlation is None else f"{correlation:.4f}"
    print("Insurance analysis complete")
    print(
        "Rows: "
        f"{cleaning['rows_after_cleaning']} cleaned "
        f"({cleaning['duplicates_removed']} duplicate(s) removed)"
    )
    print(f"BMI/charges correlation: {correlation_text}")
    print(f"Test MSE: {metrics['mean_squared_error']:.2f}")
    print(f"Test R-squared: {metrics['r_squared']:.4f}")
    print(f"Results written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
