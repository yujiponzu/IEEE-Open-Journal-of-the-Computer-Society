# python -m src.error_analysis.axis_match_rate

from pathlib import Path

import numpy as np
import pandas as pd

from ..result_plot.closest_example_stats import pick_nearest_examples

def _load_axis_stance_map(csv_path: Path) -> dict[int, tuple[str, str, str]]:
    df = pd.read_csv(csv_path)
    df["question"] = pd.to_numeric(df["question"], errors="coerce")
    df = df.dropna(subset=["question", "axis", "stance", "statement"])
    mapping: dict[int, tuple[str, str, str]] = {}
    for _, row in df.iterrows():
        mapping[int(row["question"])] = (
            str(row["axis"]),
            str(row["stance"]),
            str(row["statement"]),
        )
    if len(mapping) != 62:
        raise ValueError("Expected 62 questions in PCT_axis_stance_when_agree.csv.")
    return mapping


def _extract_question_columns(df: pd.DataFrame) -> list[str]:
    # Use explicit result_1..result_62 columns to avoid schema differences.
    result_cols = [c for c in df.columns if c.startswith("result_")]
    if not result_cols:
        raise ValueError("No result_* columns found in result CSV.")

    def _result_key(name: str) -> int:
        try:
            return int(name.split("_", 1)[1])
        except (IndexError, ValueError):
            return 0

    result_cols = sorted(result_cols, key=_result_key)
    if len(result_cols) != 62:
        raise ValueError(
            f"Expected 62 result_* columns, found {len(result_cols)}."
        )
    return result_cols


def _count_matches(
    csv_paths: list[Path],
    axis_stance_map: dict[int, tuple[str, str, str]],
    eco_ideology: str,
    soc_ideology: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = np.zeros(62, dtype=int)
    match = np.zeros(62, dtype=int)
    agree = np.zeros(62, dtype=int)
    disagree = np.zeros(62, dtype=int)
    valid_values = {1, 2, 3, 4}

    for path in csv_paths:
        df = pd.read_csv(path)
        q_cols = _extract_question_columns(df)
        for idx, col in enumerate(q_cols, start=1):
            question = idx
            axis, stance, _statement = axis_stance_map[question]
            desired_stance = eco_ideology if axis == "eco" else soc_ideology
            agree_with_statement = stance == desired_stance

            answers = pd.to_numeric(df[col], errors="coerce")
            valid = answers.isin(valid_values)
            total[question - 1] += int(valid.sum())

            agree_count = int(answers.isin({3, 4}).sum())
            disagree_count = int(answers.isin({1, 2}).sum())
            agree[question - 1] += agree_count
            disagree[question - 1] += disagree_count

            if agree_with_statement:
                match[question - 1] += agree_count
            else:
                match[question - 1] += disagree_count

    return total, match, agree, disagree


def compute_match_rates(
    data_dir: str | Path = "../data/analysis",
    pct_path: str | Path = "src/error_analysis/data/PCT_axis_stance_when_agree.csv",
) -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = Path(data_dir)
    pct_path = Path(pct_path)
    if not data_dir.is_absolute():
        data_dir = base_dir / data_dir
    if not pct_path.is_absolute():
        pct_path = base_dir / pct_path
    axis_stance_map = _load_axis_stance_map(pct_path)

    picked = pick_nearest_examples(data_dir=data_dir)

    rows: list[dict[str, int | float | str]] = []
    for item in picked:
        total, match, agree, disagree = _count_matches(
            [item.path], axis_stance_map, item.eco_ideology, item.soc_ideology
        )

        for question in range(1, 63):
            axis, stance, statement = axis_stance_map[question]
            total_count = int(total[question - 1])
            match_count = int(match[question - 1])
            agree_count = int(agree[question - 1])
            disagree_count = int(disagree[question - 1])
            match_rate = match_count / total_count if total_count else float("nan")
            agree_rate = agree_count / total_count if total_count else float("nan")
            disagree_rate = disagree_count / total_count if total_count else float("nan")
            desired_stance = item.eco_ideology if axis == "eco" else item.soc_ideology
            expected_agree = stance == desired_stance
            rows.append(
                {
                    "model_name": item.model_name,
                    "example_num": item.example_num,
                    "eco_ideology": item.eco_ideology,
                    "soc_ideology": item.soc_ideology,
                    "question": question,
                    "statement": statement,
                    "axis": axis,
                    "stance": stance,
                    "expected_agree": expected_agree,
                    "match_count": match_count,
                    "agree_count": agree_count,
                    "disagree_count": disagree_count,
                    "match_rate": match_rate,
                    "agree_rate": agree_rate,
                    "disagree_rate": disagree_rate,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    df = compute_match_rates()
    output_dir = Path("outputs") / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "axis_match_rates.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
