# python -m src.result_plot.closest_example_stats

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.markers import MarkerStyle
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import Affine2D
from rich.console import Console
from rich.table import Table

from ..utils import _RESULT_PATTERN, get_model_color, load_all_results, t_confidence_interval, wrap_latex_table

@dataclass(frozen=True)
class PickedExample:
    model_name: str
    eco_ideology: str
    soc_ideology: str
    example_num: int
    path: Path


def _target_point(eco_ideology: str, soc_ideology: str) -> tuple[float, float]:
    if eco_ideology == "right-wing":
        eco_target = 10.0
    elif eco_ideology == "left-wing":
        eco_target = -10.0
    else:
        raise ValueError(f"Unexpected eco_ideology: {eco_ideology}")

    if soc_ideology == "right-wing":
        soc_target = 10.0
    elif soc_ideology == "left-wing":
        soc_target = -10.0
    else:
        raise ValueError(f"Unexpected soc_ideology: {soc_ideology}")

    return eco_target, soc_target


def _iter_result_files(data_dir: str | Path) -> list[tuple[str, str, str, int, Path]]:
    base_dir = Path(__file__).resolve().parents[2] / data_dir
    results: list[tuple[str, str, str, int, Path]] = []
    for model_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        for csv_path in sorted(model_dir.glob("*.csv")):
            name = csv_path.name
            if "debiased" in name or "baseline" in name:
                continue
            match = _RESULT_PATTERN.match(name)
            if match is None:
                continue
            results.append(
                (
                    model_dir.name,
                    match.group("eco"),
                    match.group("soc"),
                    int(match.group("example")),
                    csv_path,
                )
            )
    return results


def pick_nearest_examples(data_dir: str | Path = "../data/analysis") -> list[PickedExample]:
    best: dict[tuple[str, str, str], tuple[float, int, Path]] = {}
    for model_name, eco_ideology, soc_ideology, example_num, path in _iter_result_files(
        data_dir
    ):
        df = pd.read_csv(path)
        df["eco_score"] = pd.to_numeric(df["eco_score"], errors="coerce")
        df["soc_score"] = pd.to_numeric(df["soc_score"], errors="coerce")
        df = df.dropna(subset=["eco_score", "soc_score"])
        if df.empty:
            continue
        eco_mean = float(df["eco_score"].mean())
        soc_mean = float(df["soc_score"].mean())
        eco_target, soc_target = _target_point(eco_ideology, soc_ideology)
        distance = float(
            np.hypot(eco_mean - eco_target, soc_mean - soc_target)
        )

        key = (model_name, eco_ideology, soc_ideology)
        current = best.get(key)
        if current is None or distance < current[0] or (
            distance == current[0] and example_num < current[1]
        ):
            best[key] = (distance, example_num, path)

    picked: list[PickedExample] = []
    for (model_name, eco_ideology, soc_ideology), (_, example_num, path) in best.items():
        picked.append(
            PickedExample(
                model_name=model_name,
                eco_ideology=eco_ideology,
                soc_ideology=soc_ideology,
                example_num=example_num,
                path=path,
            )
        )
    return sorted(
        picked, key=lambda item: (item.model_name, item.eco_ideology, item.soc_ideology)
    )


def _permutation_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 10000,
    seed: int = 0,
) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    observed = abs(float(x.mean() - y.mean()))
    combined = np.concatenate([x, y])
    n_x = x.size
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        diff = abs(float(combined[:n_x].mean() - combined[n_x:].mean()))
        if diff >= observed:
            exceed += 1
    return (exceed + 1) / (n_perm + 1)


def _cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    s1 = float(x.var(ddof=1))
    s2 = float(y.var(ddof=1))
    n1 = x.size
    n2 = y.size
    pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    if pooled <= 0:
        return float("nan")
    return (float(x.mean()) - float(y.mean())) / float(np.sqrt(pooled))


def summarize_picked_examples(
    picked: list[PickedExample] | None = None,
    data_dir: str | Path = "../data/analysis",
    n_perm: int = 10000,
    seed: int = 0,
    confidence: float = 0.95,
) -> pd.DataFrame:
    if picked is None:
        picked = pick_nearest_examples(data_dir=data_dir)

    all_results = load_all_results(data_dir=data_dir)
    baseline = all_results[all_results["eco_ideology"] == "baseline"].copy()
    baseline["eco_score"] = pd.to_numeric(baseline["eco_score"], errors="coerce")
    baseline["soc_score"] = pd.to_numeric(baseline["soc_score"], errors="coerce")
    baseline = baseline.dropna(subset=["eco_score", "soc_score"])

    baseline_by_model: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_name, model_df in baseline.groupby("model_name"):
        baseline_by_model[model_name] = (
            model_df["eco_score"].to_numpy(),
            model_df["soc_score"].to_numpy(),
        )

    rows: list[dict[str, float | str | int]] = []
    for item in picked:
        df = pd.read_csv(item.path)
        df["eco_score"] = pd.to_numeric(df["eco_score"], errors="coerce")
        df["soc_score"] = pd.to_numeric(df["soc_score"], errors="coerce")
        df = df.dropna(subset=["eco_score", "soc_score"])

        eco_scores = df["eco_score"].to_numpy()
        soc_scores = df["soc_score"].to_numpy()
        eco_ci_low, eco_ci_high = t_confidence_interval(
            eco_scores, confidence=confidence
        )
        soc_ci_low, soc_ci_high = t_confidence_interval(
            soc_scores, confidence=confidence
        )

        row: dict[str, float | str | int] = {
            "model_name": item.model_name,
            "eco_ideology": item.eco_ideology,
            "soc_ideology": item.soc_ideology,
            "example_num": item.example_num,
            "N": int(eco_scores.size),
            "eco_mean": float(eco_scores.mean()) if eco_scores.size else float("nan"),
            "eco_std": float(eco_scores.std(ddof=0)) if eco_scores.size else float("nan"),
            "eco_var": float(eco_scores.var(ddof=0)) if eco_scores.size else float("nan"),
            "eco_ci_low": float(eco_ci_low),
            "eco_ci_high": float(eco_ci_high),
            "soc_mean": float(soc_scores.mean()) if soc_scores.size else float("nan"),
            "soc_std": float(soc_scores.std(ddof=0)) if soc_scores.size else float("nan"),
            "soc_var": float(soc_scores.var(ddof=0)) if soc_scores.size else float("nan"),
            "soc_ci_low": float(soc_ci_low),
            "soc_ci_high": float(soc_ci_high),
        }

        baseline_scores = baseline_by_model.get(item.model_name)
        if baseline_scores is not None:
            base_eco, base_soc = baseline_scores
            row["eco_p_value"] = _permutation_pvalue(
                eco_scores, base_eco, n_perm=n_perm, seed=seed
            )
            row["soc_p_value"] = _permutation_pvalue(
                soc_scores, base_soc, n_perm=n_perm, seed=seed
            )
            row["eco_cohen_d"] = _cohen_d(eco_scores, base_eco)
            row["soc_cohen_d"] = _cohen_d(soc_scores, base_soc)
        else:
            row["eco_p_value"] = float("nan")
            row["soc_p_value"] = float("nan")
            row["eco_cohen_d"] = float("nan")
            row["soc_cohen_d"] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["model_name", "eco_ideology", "soc_ideology", "example_num"]
    )


def plot_picked_means(
    summary_table: pd.DataFrame,
    output_path: str | Path | None = None,
    show: bool = True,
):
    def _add_marker_patch(
        ax: plt.Axes,
        x: float,
        y: float,
        marker: str,
        color: str,
        hatch: str | None,
        size: float = 0.65,
    ) -> PathPatch:
        style = MarkerStyle(marker)
        path = style.get_path().transformed(style.get_transform())
        transform = Affine2D().scale(size).translate(x, y) + ax.transData
        patch = PathPatch(
            path,
            transform=transform,
            facecolor="none",
            edgecolor=color,
            hatch=hatch,
            lw=1.0,
            alpha=0.8,
            zorder=3,
        )
        ax.add_patch(patch)
        return patch

    fig, ax = plt.subplots(figsize=(8.8, 7))
    fig.subplots_adjust(left=0.08, right=0.56, bottom=0.08, top=0.98)
    marker_map = {
        ("left-wing", "left-wing"): "^",
        ("left-wing", "right-wing"): "*",
        ("right-wing", "right-wing"): "o",
        ("right-wing", "left-wing"): "s",
    }
    hatch_map = {
        0: None,
        2: "||||||",
        4: "------",
        6: "//////",
    }

    for _, row in summary_table.iterrows():
        color = get_model_color(row["model_name"])
        eco_mean = float(row["eco_mean"])
        soc_mean = float(row["soc_mean"])
        marker = marker_map.get(
            (row["eco_ideology"], row["soc_ideology"]), "o"
        )
        example_num = int(row["example_num"])
        hatch = hatch_map.get(example_num)
        eco_std = float(row["eco_std"])
        soc_std = float(row["soc_std"])

        xerr = np.array([[eco_std], [eco_std]])
        yerr = np.array([[soc_std], [soc_std]])

        ax.errorbar(
            eco_mean,
            soc_mean,
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            alpha=0.8,
            capsize=3,
            color=color,
            ecolor=color,
        )
        label = f"{row['model_name']}:{row['eco_ideology']}/{row['soc_ideology']}"
        _add_marker_patch(ax, eco_mean, soc_mean, marker, color, hatch=hatch)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ideology_labels = [
        (r"$[L_{eco}, L_{soc}]$", "^"),
        (r"$[L_{eco}, R_{soc}]$", "*"),
        (r"$[R_{eco}, R_{soc}]$", "o"),
        (r"$[R_{eco}, L_{soc}]$", "s"),
    ]
    ideology_handles = [
        Line2D([0], [0], marker=marker, color="black", linestyle="None",
               markersize=8, label=label)
        for label, marker in ideology_labels
    ]
    example_handles = [
        Patch(facecolor="white", edgecolor="black", hatch="", label="0"),
        Patch(facecolor="white", edgecolor="black", hatch="||||||", label="2"),
        Patch(facecolor="white", edgecolor="black", hatch="------", label="4"),
        Patch(facecolor="white", edgecolor="black", hatch="//////", label="6"),
    ]
    legend_ideology = ax.legend(
        handles=ideology_handles,
        title="political stance",
        fontsize=7,
        title_fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.62),
        borderaxespad=0.0,
    )
    ax.add_artist(legend_ideology)
    ax.legend(
        handles=example_handles,
        title="#example",
        fontsize=7,
        title_fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.30),
        borderaxespad=0.0,
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.annotate(
        "",
        xy=(10, 0),
        xytext=(-10, 0),
        arrowprops={"arrowstyle": "-", "color": "black", "lw": 1.2},
    )
    ax.annotate(
        "",
        xy=(0, 10),
        xytext=(0, -10),
        arrowprops={"arrowstyle": "-", "color": "black", "lw": 1.2},
    )
    ax.text(
        8.7,
        0.45,
        "economy score",
        ha="right",
        va="bottom",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
    )
    ax.text(
        0.45,
        8.7,
        "society score",
        ha="left",
        va="top",
        rotation=90,
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1.5},
    )

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    if show:
        plt.show()
    return fig, ax


def export_summary_csv(
    summary_table: pd.DataFrame, output_path: str | Path
) -> pd.DataFrame:
    columns = [
        "model_name",
        "eco_ideology",
        "soc_ideology",
        "example_num",
        "eco_mean",
        "eco_std",
        "soc_mean",
        "soc_std",
    ]
    csv_table = summary_table.loc[:, columns].copy()

    def _round_2(value: float) -> float:
        if pd.isna(value):
            return value
        return float(
            Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )

    csv_table["eco_mean"] = csv_table["eco_mean"].apply(_round_2)
    csv_table["eco_std"] = csv_table["eco_std"].apply(_round_2)
    csv_table["soc_mean"] = csv_table["soc_mean"].apply(_round_2)
    csv_table["soc_std"] = csv_table["soc_std"].apply(_round_2)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_table.to_csv(output_path, index=False, float_format="%.2f")
    return csv_table


def export_full_summary_csv(
    summary_table: pd.DataFrame, output_path: str | Path
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(output_path, index=False)
    return output_path


def export_full_summary_latex(
    summary_table: pd.DataFrame,
    output_path: str | Path,
    caption: str = "Closest example summary",
    label: str = "tab:closest_example_summary",
    columns: list[str] | None = None,
) -> Path:
    table = summary_table.copy()

    def _short_ideology(value: str) -> str:
        if value == "left-wing":
            return "L"
        if value == "right-wing":
            return "R"
        return value

    table["stance"] = table.apply(
        lambda row: f"[{_short_ideology(row['eco_ideology'])}, "
        f"{_short_ideology(row['soc_ideology'])}]",
        axis=1,
    )

    def _mean_std_cell(mean: float, std: float) -> str:
        if pd.isna(mean) or pd.isna(std):
            return ""
        return f"{mean:.2f} (±{std:.2f})"

    table["eco_mean"] = table.apply(
        lambda row: _mean_std_cell(row["eco_mean"], row["eco_std"]), axis=1
    )
    table["soc_mean"] = table.apply(
        lambda row: _mean_std_cell(row["soc_mean"], row["soc_std"]), axis=1
    )

    default_columns = ["model_name", "stance", "eco_mean", "soc_mean", "example_num"]
    if columns is None:
        columns = default_columns
    table = table.loc[:, columns]
    table = table.rename(
        columns={
            "model_name": "model",
            "example_num": r"$i$",
            "eco_mean": r"$\mu_{eco}(\pm\sigma_{eco})$",
            "soc_mean": r"$\mu_{soc}(\pm\sigma_{soc})$",
        }
    )
    half = int(np.ceil(len(table) / 2))
    left = table.iloc[:half]
    right = table.iloc[half:]

    left_tabular = left.to_latex(index=False, escape=False, float_format="%.2f")
    right_tabular = right.to_latex(index=False, escape=False, float_format="%.2f")

    latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\begin{subtable}[t]{0.49\\linewidth}\n"
        "\\centering\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\resizebox{\\linewidth}{!}{%\n"
        f"{left_tabular}\n"
        "}\n"
        "\\end{subtable}\n"
        "\\hfill\n"
        "\\begin{subtable}[t]{0.49\\linewidth}\n"
        "\\centering\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        "\\resizebox{\\linewidth}{!}{%\n"
        f"{right_tabular}\n"
        "}\n"
        "\\end{subtable}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")
    return output_path


def print_rich_table(df: pd.DataFrame):
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")

    for col in df.columns:
        table.add_column(col)

    for _, row in df.iterrows():
        table.add_row(*[f"{v:.3f}" if isinstance(v, float) else str(v) for v in row])

    console.print(table)



if __name__ == "__main__":
    default_data_dir = "../data/analysis"
    output_dir = Path(__file__).resolve().parents[2] / "outputs"
    output_graph_dir = output_dir / "graphs"
    output_graph_dir.mkdir(parents=True, exist_ok=True)
    table = summarize_picked_examples(data_dir=default_data_dir)
    print_rich_table(table)
    export_full_summary_latex(
        table, output_dir / "closest_example_summary.tex"
    )
    export_full_summary_csv(
        table, output_dir / "closest_example_summary.csv"
    )
    export_summary_csv(
        table, output_dir / "result_csv" / "picked_example_summary.csv"
    )
    plot_picked_means(
        table,
        output_path=output_graph_dir / "closest_example_plot.png",
        show=False,
    )
