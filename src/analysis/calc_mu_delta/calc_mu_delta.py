from pathlib import Path
import argparse

import pandas as pd


def calculate_mu_delta(
    baseline_path: str | Path = "outputs/baseline_summary.csv",
    closest_path: str | Path = "outputs/closest_example_summary.csv",
) -> pd.DataFrame:
    baseline_df = pd.read_csv(baseline_path)
    closest_df = pd.read_csv(closest_path)

    required_baseline = {"model_name", "eco_mean", "soc_mean"}
    required_closest = {"model_name", "eco_ideology", "soc_ideology", "example_num", "eco_mean", "soc_mean"}
    if not required_baseline.issubset(baseline_df.columns):
        missing = sorted(required_baseline - set(baseline_df.columns))
        raise ValueError(f"Missing required columns in baseline CSV: {missing}")
    if not required_closest.issubset(closest_df.columns):
        missing = sorted(required_closest - set(closest_df.columns))
        raise ValueError(f"Missing required columns in closest CSV: {missing}")

    baseline_means = baseline_df.loc[:, ["model_name", "eco_mean", "soc_mean"]].rename(
        columns={
            "eco_mean": "baseline_eco_mean",
            "soc_mean": "baseline_soc_mean",
        }
    )
    merged = closest_df.merge(baseline_means, on="model_name", how="left", validate="m:1")

    merged["has_baseline"] = ~(
        merged["baseline_eco_mean"].isna() | merged["baseline_soc_mean"].isna()
    )
    filtered = merged.loc[merged["has_baseline"]].copy()
    filtered["delta_eco_mean"] = filtered["eco_mean"] - filtered["baseline_eco_mean"]
    filtered["delta_soc_mean"] = filtered["soc_mean"] - filtered["baseline_soc_mean"]
    return filtered.sort_values(
        ["model_name", "eco_ideology", "soc_ideology", "example_num"]
    )


def export_mu_delta_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Calculate baseline deltas from baseline and closest summary CSV files."
    )
    parser.add_argument(
        "--baseline",
        default="outputs/baseline_summary.csv",
        help="Path to baseline summary CSV.",
    )
    parser.add_argument(
        "--closest",
        default="outputs/closest_example_summary.csv",
        help="Path to closest example summary CSV.",
    )
    parser.add_argument(
        "--output",
        default="outputs/mu_delta_summary.csv",
        help="Path to output CSV.",
    )
    args = parser.parse_args()

    delta_df = calculate_mu_delta(
        baseline_path=args.baseline,
        closest_path=args.closest,
    )
    output_path = export_mu_delta_csv(delta_df, args.output)
    print(f"Saved CSV: {output_path}")


if __name__ == "__main__":
    main()
