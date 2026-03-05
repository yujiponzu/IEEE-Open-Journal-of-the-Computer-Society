# uv run python -m src.analysis.baseline_plot.baseline_stats

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..utils import get_model_color, load_all_results, t_confidence_interval, wrap_latex_table


def get_baseline(data_dir: str | Path = "data") -> pd.DataFrame:
    all_results = load_all_results(data_dir=data_dir)
    baseline = all_results[all_results["eco_ideology"] == "baseline"].copy()
    baseline["eco_score"] = pd.to_numeric(baseline["eco_score"], errors="coerce")
    baseline["soc_score"] = pd.to_numeric(baseline["soc_score"], errors="coerce")
    baseline = baseline.dropna(subset=["eco_score", "soc_score"])
    return baseline

def summarize_baseline_by_model(
    confidence: float = 0.95, data_dir: str | Path = "data"
) -> pd.DataFrame:
    baseline = get_baseline(data_dir=data_dir)
    rows: list[dict[str, float | str]] = []

    for model_name, model_df in baseline.groupby("model_name"):
        eco_scores = model_df["eco_score"].to_numpy()
        soc_scores = model_df["soc_score"].to_numpy()
        eco_mean = float(eco_scores.mean())
        soc_mean = float(soc_scores.mean())
        eco_ci_low, eco_ci_high = t_confidence_interval(
            eco_scores, confidence=confidence
        )
        soc_ci_low, soc_ci_high = t_confidence_interval(
            soc_scores, confidence=confidence
        )

        rows.append(
            {
                "model_name": model_name,
                "eco_mean": float(eco_mean),
                "eco_std": float(eco_scores.std(ddof=0)),
                "eco_ci_low": float(eco_ci_low),
                "eco_ci_high": float(eco_ci_high),
                "soc_mean": float(soc_mean),
                "soc_std": float(soc_scores.std(ddof=0)),
                "soc_ci_low": float(soc_ci_low),
                "soc_ci_high": float(soc_ci_high),
            }
        )

    return pd.DataFrame(rows).sort_values("model_name")


def export_baseline_summary_latex(
    output_path: str | Path,
    confidence: float = 0.95,
    data_dir: str | Path = "data",
    caption: str = "Baseline summary by model",
    label: str = "tab:baseline_summary",
    columns: list[str] | None = None,
) -> Path:
    summary_table = summarize_baseline_by_model(
        confidence=confidence, data_dir=data_dir
    )
    if columns is not None:
        summary_table = summary_table.loc[:, columns]
    tabular = summary_table.to_latex(index=False, float_format="%.2f")
    latex = wrap_latex_table(tabular, caption=caption, label=label)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")
    return output_path


def export_baseline_summary_csv(
    output_path: str | Path,
    confidence: float = 0.95,
    data_dir: str | Path = "data",
) -> Path:
    summary_table = summarize_baseline_by_model(
        confidence=confidence, data_dir=data_dir
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(output_path, index=False)
    return output_path


def plot_baseline_scores_by_model(
    output_path: str | Path | None = None,
    show: bool = True,
    confidence: float = 0.95,
    data_dir: str | Path = "data",
):
    summary_table = summarize_baseline_by_model(
        confidence=confidence, data_dir=data_dir
    )
    fig, ax = plt.subplots(figsize=(7, 7))

    for _, row in summary_table.iterrows():
        eco_mean = float(row["eco_mean"])
        soc_mean = float(row["soc_mean"])
        color = get_model_color(row["model_name"])
        ax.scatter(eco_mean, soc_mean, color=color, label=row["model_name"])

    ax.set_xlim(-10, 2.5)
    ax.set_ylim(-10, 2.5)
    ax.set_xlabel("eco_score")
    ax.set_ylabel("soc_score")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(title="model name", fontsize=15, title_fontsize=15)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.annotate(
        "",
        xy=(2.4, 0),
        xytext=(-10, 0),
        arrowprops={"arrowstyle": "-", "color": "black", "lw": 1.2},
    )
    ax.annotate(
        "",
        xy=(0, 2.4),
        xytext=(0, -10),
        arrowprops={"arrowstyle": "-", "color": "black", "lw": 1.2},
    )
    ax.text(-7.5, -1.2, "economy left", ha="center", va="bottom", fontsize=20)
    ax.text(-3.8, -9.5, "society left", ha="left", va="center", fontsize=20)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


if __name__ == "__main__":
    default_data_dir = "../data/analysis"
    output_graph_dir = Path("outputs/graphs")
    output_graph_dir.mkdir(parents=True, exist_ok=True)
    default_plot_path = output_graph_dir / "baseline_scores_by_model.png"
    print(summarize_baseline_by_model(data_dir=default_data_dir).to_string(index=False))
    default_tex_path = output_graph_dir / "baseline_summary.tex"
    default_csv_path = output_graph_dir / "baseline_summary.csv"
    print(
        f"Saved LaTeX: {export_baseline_summary_latex(default_tex_path, data_dir=default_data_dir)}"
    )
    print(
        f"Saved CSV: {export_baseline_summary_csv(default_csv_path, data_dir=default_data_dir)}"
    )
    plot_baseline_scores_by_model(
        output_path=default_plot_path,
        show=False,
        data_dir=default_data_dir,
    )
    print(f"Saved plot: {default_plot_path}")
