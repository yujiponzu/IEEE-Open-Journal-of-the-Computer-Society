# python -m src.error_analysis.count_rows

import argparse
import csv
import sys
from pathlib import Path

from ..result_plot.closest_example_stats import pick_nearest_examples


def _count_csv_rows(csv_path: Path) -> int:
    # Increase field size limit for large CSV cells.
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            break
        except OverflowError:
            max_size = max_size // 10

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return 0
        return sum(1 for _ in reader)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count rows for each picked example pattern "
            "(model_name, eco_ideology, soc_ideology, example_num)."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="../data/analysis",
        help="Base directory that contains per-model CSV folders.",
    )
    parser.add_argument(
        "--output",
        default="outputs/error_analysis/row_counts_by_pattern.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[2]
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir not found: {data_dir}")

    picked = pick_nearest_examples(data_dir=data_dir)
    if not picked:
        print("No picked examples found.")
        return

    rows: list[tuple[str, str, str, int, int]] = []
    for item in picked:
        count = _count_csv_rows(item.path)
        rows.append(
            (
                item.model_name,
                item.eco_ideology,
                item.soc_ideology,
                item.example_num,
                count,
            )
        )
        print(
            f"{item.model_name},{item.eco_ideology},{item.soc_ideology},{item.example_num}: {count}"
        )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model_name", "eco_ideology", "soc_ideology", "example_num", "row_count"]
        )
        writer.writerows(rows)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
