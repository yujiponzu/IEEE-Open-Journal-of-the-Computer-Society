# python -m src.error_analysis.statement_match_rate_cross_ideology

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

from .axis_match_rate import _count_matches, _load_axis_stance_map
from ..result_plot.closest_example_stats import pick_nearest_examples

NUM_QUESTIONS = 62
RIGHT = "right-wing"
LEFT = "left-wing"
AXES = ("eco", "soc")
SIDES = (RIGHT, LEFT)


def _to_absolute_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    base_dir = Path(__file__).resolve().parents[2]
    return base_dir / path


def _safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def _format_percent(value: float) -> str:
    if pd.isna(value):
        return ""
    rounded = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{rounded:.2f}%"


def _init_axis_counts() -> dict[str, dict[str, dict[str, list[int]]]]:
    return {
        axis: {
            side: {
                "total": [0] * NUM_QUESTIONS,
                "agree": [0] * NUM_QUESTIONS,
                "disagree": [0] * NUM_QUESTIONS,
            }
            for side in SIDES
        }
        for axis in AXES
    }


def _filter_picked(
    picked,
    eco_ideology: str,
    soc_ideology: str,
):
    return [
        item
        for item in picked
        if item.eco_ideology == eco_ideology and item.soc_ideology == soc_ideology
    ]


def compute_statement_match_rates_with_picked(
    picked,
    pct_path: str | Path,
) -> pd.DataFrame:
    pct_path = _to_absolute_path(pct_path)
    axis_stance_map = _load_axis_stance_map(pct_path)

    total_sum = [0] * NUM_QUESTIONS
    axis_counts = _init_axis_counts()

    for item in picked:
        total, _match, agree, disagree = _count_matches(
            [item.path], axis_stance_map, item.eco_ideology, item.soc_ideology
        )
        ideology_by_axis = {
            "eco": item.eco_ideology,
            "soc": item.soc_ideology,
        }
        for idx in range(NUM_QUESTIONS):
            total_i = int(total[idx])
            agree_i = int(agree[idx])
            disagree_i = int(disagree[idx])
            total_sum[idx] += total_i
            for axis in AXES:
                ideology = ideology_by_axis[axis]
                if ideology in SIDES:
                    axis_counts[axis][ideology]["total"][idx] += total_i
                    axis_counts[axis][ideology]["agree"][idx] += agree_i
                    axis_counts[axis][ideology]["disagree"][idx] += disagree_i

    rows: list[dict[str, int | float | str]] = []
    for question in range(1, NUM_QUESTIONS + 1):
        axis, stance, statement = axis_stance_map[question]
        total_count = int(total_sum[question - 1])
        idx = question - 1

        soc_right_total_count = int(axis_counts["soc"][RIGHT]["total"][idx])
        soc_right_agree_count = int(axis_counts["soc"][RIGHT]["agree"][idx])
        soc_right_disagree_count = int(axis_counts["soc"][RIGHT]["disagree"][idx])
        soc_left_total_count = int(axis_counts["soc"][LEFT]["total"][idx])
        soc_left_agree_count = int(axis_counts["soc"][LEFT]["agree"][idx])
        soc_left_disagree_count = int(axis_counts["soc"][LEFT]["disagree"][idx])
        eco_right_total_count = int(axis_counts["eco"][RIGHT]["total"][idx])
        eco_right_agree_count = int(axis_counts["eco"][RIGHT]["agree"][idx])
        eco_right_disagree_count = int(axis_counts["eco"][RIGHT]["disagree"][idx])
        eco_left_total_count = int(axis_counts["eco"][LEFT]["total"][idx])
        eco_left_agree_count = int(axis_counts["eco"][LEFT]["agree"][idx])
        eco_left_disagree_count = int(axis_counts["eco"][LEFT]["disagree"][idx])

        if axis == "soc" and stance == "left":
            match_count = soc_left_agree_count + soc_right_disagree_count
        elif axis == "soc" and stance == "right":
            match_count = soc_right_agree_count + soc_left_disagree_count
        elif axis == "eco" and stance == "left":
            match_count = eco_left_agree_count + eco_right_disagree_count
        elif axis == "eco" and stance == "right":
            match_count = eco_right_agree_count + eco_left_disagree_count
        else:
            match_count = 0
        match_rate = _safe_rate(match_count, total_count)

        eco_only = axis == "eco"
        soc_only = axis == "soc"

        if eco_only:
            eco_right_agree_rate = _safe_rate(eco_right_agree_count, eco_right_total_count)
            eco_right_disagree_rate = _safe_rate(
                eco_right_disagree_count, eco_right_total_count
            )
            eco_left_agree_rate = _safe_rate(eco_left_agree_count, eco_left_total_count)
            eco_left_disagree_rate = _safe_rate(eco_left_disagree_count, eco_left_total_count)
        else:
            eco_right_agree_count = 0
            eco_right_disagree_count = 0
            eco_left_agree_count = 0
            eco_left_disagree_count = 0
            eco_right_agree_rate = float("nan")
            eco_right_disagree_rate = float("nan")
            eco_left_agree_rate = float("nan")
            eco_left_disagree_rate = float("nan")

        if soc_only:
            soc_right_agree_rate = _safe_rate(soc_right_agree_count, soc_right_total_count)
            soc_right_disagree_rate = _safe_rate(
                soc_right_disagree_count, soc_right_total_count
            )
            soc_left_agree_rate = _safe_rate(soc_left_agree_count, soc_left_total_count)
            soc_left_disagree_rate = _safe_rate(soc_left_disagree_count, soc_left_total_count)
        else:
            soc_right_agree_count = 0
            soc_right_disagree_count = 0
            soc_left_agree_count = 0
            soc_left_disagree_count = 0
            soc_right_agree_rate = float("nan")
            soc_right_disagree_rate = float("nan")
            soc_left_agree_rate = float("nan")
            soc_left_disagree_rate = float("nan")

        rows.append(
            {
                "question": question,
                "statement": statement,
                "axis": axis,
                "stance": stance,
                "total_count": total_count,
                "match_count": match_count,
                "match_rate": match_rate,
                "eco_right_agree_count": eco_right_agree_count,
                "eco_right_agree_rate": eco_right_agree_rate,
                "eco_right_disagree_count": eco_right_disagree_count,
                "eco_right_disagree_rate": eco_right_disagree_rate,
                "eco_left_agree_count": eco_left_agree_count,
                "eco_left_agree_rate": eco_left_agree_rate,
                "eco_left_disagree_count": eco_left_disagree_count,
                "eco_left_disagree_rate": eco_left_disagree_rate,
                "soc_right_agree_count": soc_right_agree_count,
                "soc_right_agree_rate": soc_right_agree_rate,
                "soc_right_disagree_count": soc_right_disagree_count,
                "soc_right_disagree_rate": soc_right_disagree_rate,
                "soc_left_agree_count": soc_left_agree_count,
                "soc_left_agree_rate": soc_left_agree_rate,
                "soc_left_disagree_count": soc_left_disagree_count,
                "soc_left_disagree_rate": soc_left_disagree_rate,
            }
        )

    df = pd.DataFrame(rows).sort_values("match_rate", ascending=True, na_position="last")
    return df


def compute_statement_match_rates(
    data_dir: str | Path = "../data/analysis",
    pct_path: str | Path = "src/error_analysis/data/PCT_axis_stance_when_agree.csv",
    eco_ideology: str = RIGHT,
    soc_ideology: str = LEFT,
) -> pd.DataFrame:
    data_dir = _to_absolute_path(data_dir)
    pct_path = _to_absolute_path(pct_path)

    picked = [
        item
        for item in pick_nearest_examples(data_dir=data_dir)
        if "baseline" not in item.path.name
        and item.eco_ideology == eco_ideology
        and item.soc_ideology == soc_ideology
    ]
    return compute_statement_match_rates_with_picked(
        picked=picked,
        pct_path=pct_path,
    )


def _write_diff_csv(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_suffix: str,
    right_suffix: str,
    output_path: Path,
) -> None:
    merged = left_df.merge(
        right_df,
        on=["question", "statement", "axis", "stance"],
        suffixes=(f"_{left_suffix}", f"_{right_suffix}"),
    )
    left_col = f"match_count_{left_suffix}"
    right_col = f"match_count_{right_suffix}"
    left_rate_col = f"match_rate_{left_suffix}"
    right_rate_col = f"match_rate_{right_suffix}"
    merged["match_rate_diff"] = (merged[left_rate_col] - merged[right_rate_col]) * 100
    merged = merged.sort_values("match_rate_diff", ascending=False, na_position="last")
    merged["match_rate_diff"] = merged["match_rate_diff"].map(_format_percent)
    merged = merged[
        [
            "question",
            "statement",
            "axis",
            "stance",
            left_col,
            right_col,
            left_rate_col,
            right_rate_col,
            "match_rate_diff",
        ]
    ]
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_path}")


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / ".." / "data" / "analysis"
    pct_path = (
        base_dir / "src" / "error_analysis" / "data" / "PCT_axis_stance_when_agree.csv"
    )

    all_picked = [
        item
        for item in pick_nearest_examples(data_dir=data_dir)
        if "baseline" not in item.path.name
    ]

    outputs = [
        (RIGHT, LEFT, "statement_match_rates_right-left.csv"),
        (LEFT, RIGHT, "statement_match_rates_left-right.csv"),
        (RIGHT, RIGHT, "statement_match_rates_right-right.csv"),
        (LEFT, LEFT, "statement_match_rates_left-left.csv"),
    ]

    output_dir = Path("outputs") / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    dfs_by_key: dict[tuple[str, str], pd.DataFrame] = {}
    for eco_ideology, soc_ideology, filename in outputs:
        picked = _filter_picked(all_picked, eco_ideology, soc_ideology)
        df = compute_statement_match_rates_with_picked(
            picked=picked,
            pct_path=pct_path,
        )
        dfs_by_key[(eco_ideology, soc_ideology)] = df
        output_path = output_dir / filename
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {output_path}")
        print("\nLowest 10 match rates:")
        print(df.head(10).to_string(index=False))
        print("\nHighest 10 match rates:")
        print(df.tail(10).sort_values("match_rate", ascending=False).to_string(index=False))

    right_right = dfs_by_key.get((RIGHT, RIGHT))
    right_left = dfs_by_key.get((RIGHT, LEFT))
    left_left = dfs_by_key.get((LEFT, LEFT))
    left_right = dfs_by_key.get((LEFT, RIGHT))

    if right_right is not None and right_left is not None:
        _write_diff_csv(
            left_df=right_right,
            right_df=right_left,
            left_suffix="right-right",
            right_suffix="right-left",
            output_path=output_dir
            / "statement_match_rates_diff_right-right_vs_right-left.csv",
        )

    if left_left is not None and left_right is not None:
        _write_diff_csv(
            left_df=left_left,
            right_df=left_right,
            left_suffix="left-left",
            right_suffix="left-right",
            output_path=output_dir
            / "statement_match_rates_diff_left-left_vs_left-right.csv",
        )

    if left_left is not None and right_left is not None:
        _write_diff_csv(
            left_df=left_left,
            right_df=right_left,
            left_suffix="left-left",
            right_suffix="right-left",
            output_path=output_dir
            / "statement_match_rates_diff_left-left_vs_right-left.csv",
        )

    if right_right is not None and left_right is not None:
        _write_diff_csv(
            left_df=right_right,
            right_df=left_right,
            left_suffix="right-right",
            right_suffix="left-right",
            output_path=output_dir
            / "statement_match_rates_diff_right-right_vs_left-right.csv",
        )


if __name__ == "__main__":
    main()
