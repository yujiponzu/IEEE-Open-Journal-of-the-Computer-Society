# python -m src.error_analysis.statement_match_rate

from pathlib import Path

import pandas as pd

from .axis_match_rate import _count_matches, _load_axis_stance_map
from ..result_plot.closest_example_stats import pick_nearest_examples


def compute_statement_match_rates(
    data_dir: str | Path = "data/tidied",
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
    picked = [
        item
        for item in pick_nearest_examples(data_dir=data_dir)
        if "baseline" not in item.path.name
    ]

    total_sum = [0] * 62
    soc_right_total = [0] * 62
    soc_right_agree = [0] * 62
    soc_right_disagree = [0] * 62
    soc_left_total = [0] * 62
    soc_left_agree = [0] * 62
    soc_left_disagree = [0] * 62
    eco_right_total = [0] * 62
    eco_right_agree = [0] * 62
    eco_right_disagree = [0] * 62
    eco_left_total = [0] * 62
    eco_left_agree = [0] * 62
    eco_left_disagree = [0] * 62

    for item in picked:
        total, match, agree, disagree = _count_matches(
            [item.path], axis_stance_map, item.eco_ideology, item.soc_ideology
        )
        for idx in range(62):
            total_sum[idx] += int(total[idx])
            if item.eco_ideology == "right-wing":
                eco_right_total[idx] += int(total[idx])
                eco_right_agree[idx] += int(agree[idx])
                eco_right_disagree[idx] += int(disagree[idx])
            elif item.eco_ideology == "left-wing":
                eco_left_total[idx] += int(total[idx])
                eco_left_agree[idx] += int(agree[idx])
                eco_left_disagree[idx] += int(disagree[idx])
            if item.soc_ideology == "right-wing":
                soc_right_total[idx] += int(total[idx])
                soc_right_agree[idx] += int(agree[idx])
                soc_right_disagree[idx] += int(disagree[idx])
            elif item.soc_ideology == "left-wing":
                soc_left_total[idx] += int(total[idx])
                soc_left_agree[idx] += int(agree[idx])
                soc_left_disagree[idx] += int(disagree[idx])

    rows: list[dict[str, int | float | str]] = []
    for question in range(1, 63):
        axis, stance, statement = axis_stance_map[question]
        total_count = int(total_sum[question - 1])
        soc_right_total_count = int(soc_right_total[question - 1])
        soc_right_agree_count = int(soc_right_agree[question - 1])
        soc_right_disagree_count = int(soc_right_disagree[question - 1])
        soc_left_total_count = int(soc_left_total[question - 1])
        soc_left_agree_count = int(soc_left_agree[question - 1])
        soc_left_disagree_count = int(soc_left_disagree[question - 1])
        eco_right_total_count = int(eco_right_total[question - 1])
        eco_right_agree_count = int(eco_right_agree[question - 1])
        eco_right_disagree_count = int(eco_right_disagree[question - 1])
        eco_left_total_count = int(eco_left_total[question - 1])
        eco_left_agree_count = int(eco_left_agree[question - 1])
        eco_left_disagree_count = int(eco_left_disagree[question - 1])
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
        match_rate = match_count / total_count if total_count else float("nan")
        eco_only = axis == "eco"
        soc_only = axis == "soc"
        # right/left agree/disagree breakdowns removed from output
        if eco_only:
            eco_right_agree_rate = (
                eco_right_agree_count / eco_right_total_count
                if eco_right_total_count
                else float("nan")
            )
            eco_right_disagree_rate = (
                eco_right_disagree_count / eco_right_total_count
                if eco_right_total_count
                else float("nan")
            )
            eco_left_agree_rate = (
                eco_left_agree_count / eco_left_total_count
                if eco_left_total_count
                else float("nan")
            )
            eco_left_disagree_rate = (
                eco_left_disagree_count / eco_left_total_count
                if eco_left_total_count
                else float("nan")
            )
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
            soc_right_agree_rate = (
                soc_right_agree_count / soc_right_total_count
                if soc_right_total_count
                else float("nan")
            )
            soc_right_disagree_rate = (
                soc_right_disagree_count / soc_right_total_count
                if soc_right_total_count
                else float("nan")
            )
            soc_left_agree_rate = (
                soc_left_agree_count / soc_left_total_count
                if soc_left_total_count
                else float("nan")
            )
            soc_left_disagree_rate = (
                soc_left_disagree_count / soc_left_total_count
                if soc_left_total_count
                else float("nan")
            )
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


def main() -> None:
    df = compute_statement_match_rates()
    output_dir = Path("outputs") / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "statement_match_rates.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {output_path}")
    print("\nLowest 10 match rates:")
    print(df.head(10).to_string(index=False))
    print("\nHighest 10 match rates:")
    print(df.tail(10).sort_values("match_rate", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
