import math
import re
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

_RESULT_PATTERN = re.compile(
    r"^result_(?P<eco>[^_]+)_(?P<soc>[^_]+)_(?P<example>\d+)_.*\.csv$"
)

MODEL_COLORS: dict[str, str] = {
    "GPT-5.2": "#FFBB78",
    "Gemini": "#2CA02C",
    "Claude": "#1F77B4",
    "Llama4": "#D62728",
    "DeepSeek": "#AEC7E8",
    "Qwen3": "#9467BD",
    "GLM-4.5": "#FF7F0E",
    "Kimi K2": "#98DF8A",
    "Marin": "#000000",
    "cogito": "#C49C94",
    "Refuel": "#C5B0D5",
    "Typhoon": "#8C564B",
    "Maestro": "#FF9896",
    "Gemini 2.5 Pro": "#2CA02C",
    "Claude Opus 4.1": "#1F77B4",
    "Llama 4 Maverick": "#D62728",
    "DeepSeek-V3-0324": "#AEC7E8",
    "Qwen3-235B-A22B-Instruct-2507 FP8": "#9467BD",
    "GLM-4.5-Air-FP8": "#FF7F0E",
    "Kimi K2 Instruct": "#98DF8A",
    "Marin-8B Instruct": "#000000",
    "cogito-v2-preview-llama-405B": "#C49C94",
    "Refuel LLM-2": "#C5B0D5",
    "Typhoon-v2.1-12B-Instruct": "#8C564B",
    "Arcee Maestro": "#FF9896",
}


def get_model_color(model_name: str) -> str | None:
    return MODEL_COLORS.get(model_name)


def wrap_latex_table(tabular: str, caption: str, label: str) -> str:
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )


# 役割: データディレクトリ配下の結果CSVを読み込み、解析に使いやすい形に整形して返す。
def load_all_results(data_dir: str | Path = "data") -> pd.DataFrame:
    data_dir = Path(__file__).resolve().parent.parent / data_dir
    frames: list[pd.DataFrame] = []

    for model_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for csv_path in sorted(model_dir.glob("*.csv")):
            if "debiased" in csv_path.name:
                continue
            match = _RESULT_PATTERN.match(csv_path.name)

            df = pd.read_csv(csv_path)
            df["model_name"] = model_dir.name
            if "baseline" in csv_path.name:
                df["eco_ideology"] = "baseline"
                df["soc_ideology"] = "baseline"
            else:
                df["eco_ideology"] = match.group("eco")
                df["soc_ideology"] = match.group("soc")
            df["example_num"] = int(match.group("example"))
            frames.append(df)

    all_results = pd.concat(frames, ignore_index=True)
    results = all_results[
        [
            "model_name",
            "eco_ideology",
            "soc_ideology",
            "example_num",
            "eco_score",
            "soc_score",
        ]
    ].copy()
    results["eco_score"] = pd.to_numeric(results["eco_score"], errors="coerce")
    results = results.dropna(subset=["eco_score"])

    return results


# 役割: 指定条件で結果を絞り込み、eco/socスコアの統計量をまとめて返す。
def get_score_stats(
    model_name: str,
    eco_ideology: str,
    soc_ideology: str,
    example_num: int,
    confidence: float = 0.95,
    data_dir: str | Path = "data",
) -> dict[str, dict[str, float | tuple[float, float]]]:
    all_results = load_all_results(data_dir)
    filtered = all_results[
        (all_results["model_name"] == model_name)
        & (all_results["eco_ideology"] == eco_ideology)
        & (all_results["soc_ideology"] == soc_ideology)
        & (all_results["example_num"] == example_num)
    ].copy()
    filtered["eco_score"] = pd.to_numeric(filtered["eco_score"], errors="coerce")
    filtered["soc_score"] = pd.to_numeric(filtered["soc_score"], errors="coerce")
    filtered = filtered.dropna(subset=["eco_score", "soc_score"])

    return {
        "eco_score": _score_summary(filtered["eco_score"].to_numpy(), confidence),
        "soc_score": _score_summary(filtered["soc_score"].to_numpy(), confidence),
    }


# 役割: t分布を仮定した平均の信頼区間を計算する。
def t_confidence_interval(
    values: np.ndarray, confidence: float = 0.95
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    count = values.size
    if not (0 < confidence < 1):
        raise ValueError("confidence must be between 0 and 1.")
    if count == 0:
        return float("nan"), float("nan")

    mean = float(values.mean())
    if count < 2:
        return mean, mean

    std = float(values.std(ddof=1))
    if std == 0.0:
        return mean, mean

    df = count - 1
    alpha = 1.0 - confidence
    t_crit = _t_critical_value(df, 1.0 - alpha / 2.0)
    margin = t_crit * std / math.sqrt(count)
    return mean - margin, mean + margin


def _t_critical_value(df: int, p: float) -> float:
    if df <= 0:
        raise ValueError("df must be positive.")
    if not (0 < p < 1):
        raise ValueError("p must be between 0 and 1.")

    z = NormalDist().inv_cdf(p)
    z2 = z * z
    z3 = z2 * z
    z5 = z3 * z2
    z7 = z5 * z2
    z9 = z7 * z2
    df1 = float(df)
    df2 = df1 * df1
    df3 = df2 * df1
    df4 = df3 * df1

    return (
        z
        + (z3 + z) / (4.0 * df1)
        + (5.0 * z5 + 16.0 * z3 + 3.0 * z) / (96.0 * df2)
        + (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z) / (384.0 * df3)
        + (79.0 * z9 + 776.0 * z7 + 1482.0 * z5 - 1920.0 * z3 - 945.0 * z)
        / (92160.0 * df4)
    )


# 役割: スコア配列の件数・平均・分散・信頼区間を計算して要約する。
def _score_summary(
    values: np.ndarray, confidence: float
) -> dict[str, float | tuple[float, float]]:
    count = values.size
    mean = float(values.mean())
    if count < 2:
        variance = 0.0
        ci = (mean, mean)
        return {"count": count, "mean": mean, "variance": variance, "ci": ci}

    variance = float(values.var(ddof=0))
    delta = 1 - confidence
    if not (0 < delta < 1):
        raise ValueError("confidence must be between 0 and 1.")

    ci_lower, ci_upper = _calc_ci(variance, mean, count, delta, lower=-10.0, upper=10.0)
    return {"count": count, "mean": mean, "variance": variance, "ci_lowwer": ci_lower, "ci_upper": ci_upper}


# 役割: エンピリカル・ベルンシュタイン不等式に基づく信頼区間を計算する。
def _calc_ci(
    variance: float, mean: float, count: int, delta: float, lower: float = -10, upper: float = 10
) -> tuple[float, float]:
    """
    Empirical Bernstein radius for bounded variables in [lower, upper].

    Assumptions:
    - variance is the empirical variance with ddof=0:
        variance = np.mean((x - x.mean())**2)
    - two-sided confidence interval with coverage 1 - delta
    """
    # 分散の計算は 必ず ddof=0
    range_width = upper - lower

    # two-sided CI: use delta/2 via union bound
    log_term = np.log(6.0 / delta)

    radius = np.sqrt(2.0 * variance * log_term / count) + 3.0 * range_width * log_term / count
    ci_lower = max(lower, mean - radius)
    ci_upper = min(upper, mean + radius)

    return ci_lower, ci_upper
