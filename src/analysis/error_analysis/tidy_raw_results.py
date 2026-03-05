# python -m src.error_analysis.tidy_raw_results

from __future__ import annotations

from pathlib import Path

import pandas as pd


def tidy_raw_results(
    raw_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/tidied",
) -> None:
    base_dir = Path(__file__).resolve().parents[2]
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    if not raw_dir.is_absolute():
        raw_dir = base_dir / raw_dir
    if not output_dir.is_absolute():
        output_dir = base_dir / output_dir

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    for model_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        target_dir = output_dir / model_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        for csv_path in sorted(model_dir.glob("*.csv")):
            if not csv_path.name.endswith(".csv"):
                continue

            df = pd.read_csv(csv_path)
            if "eco_score" not in df.columns:
                raise ValueError(f"eco_score column missing in {csv_path}")

            cleaned = df[df["eco_score"] != "Failed"].copy()
            output_path = target_dir / csv_path.name
            cleaned.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    tidy_raw_results()
    print("Tidied CSVs written to data/tidied/{model_name}.")


if __name__ == "__main__":
    main()
