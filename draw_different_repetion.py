#!/usr/bin/env python3
"""Plot average query QPS for different query-batch repetition ratios."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import StrMethodFormatter


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "sift" / "streamseed_hybrid"
CSV_NAME = "ef-120_batch_query_qps.csv"
TRIALS = (1, 2, 3)


@dataclass(frozen=True)
class RepetitionConfig:
    repetition: int
    pool_size: int
    sample_size: int
    sample_mode: str

    def result_dir_name(self, trial: int) -> str:
        return (
            f"test{trial}_different_repetition_{self.repetition}_"
            f"{self.pool_size}_{self.sample_size}_{self.sample_mode}"
        )


CONFIGS = (
    RepetitionConfig(20, 30000, 6000, "per_batch"),
    RepetitionConfig(25, 24000, 6000, "per_batch"),
    RepetitionConfig(50, 12000, 6000, "per_batch"),
    RepetitionConfig(75, 8000, 6000, "per_batch"),
    RepetitionConfig(100, 30000, 6000, "fixed"),
)


def load_trial(csv_path: Path, trial: int) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing input CSV: {csv_path}")

    frame = pd.read_csv(csv_path)
    required_columns = {"batch_idx", "query_qps"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing_columns)}")
    if frame.empty:
        raise ValueError(f"input CSV is empty: {csv_path}")
    if frame["batch_idx"].duplicated().any():
        raise ValueError(f"duplicate batch_idx values in {csv_path}")
    if frame[["batch_idx", "query_qps"]].isna().any().any():
        raise ValueError(f"batch_idx or query_qps contains missing values in {csv_path}")

    result = frame[["batch_idx", "query_qps"]].copy()
    result["trial"] = trial
    return result


def aggregate_config(
    results_dir: Path, config: RepetitionConfig
) -> tuple[float, pd.DataFrame]:
    trial_frames = []
    expected_batch_indices = None

    for trial in TRIALS:
        csv_path = results_dir / config.result_dir_name(trial) / CSV_NAME
        frame = load_trial(csv_path, trial)
        batch_indices = set(frame["batch_idx"].tolist())
        if expected_batch_indices is None:
            expected_batch_indices = batch_indices
        elif batch_indices != expected_batch_indices:
            raise ValueError(
                f"batch_idx values differ across trials for {config.repetition}% repetition"
            )
        trial_frames.append(frame)

    all_trials = pd.concat(trial_frames, ignore_index=True)

    # First take the median of the three trials for each batch_idx.
    batch_medians = (
        all_trials.groupby("batch_idx", as_index=False)["query_qps"]
        .median()
        .rename(columns={"query_qps": "median_query_qps"})
        .sort_values("batch_idx")
    )

    # Then average the per-batch medians across all batch_idx values.
    overall_mean_qps = float(batch_medians["median_query_qps"].mean())
    return overall_mean_qps, batch_medians


def collect_summary(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    batch_rows = []

    for config in CONFIGS:
        mean_qps, batch_medians = aggregate_config(results_dir, config)
        summary_rows.append(
            {
                "repetition_percent": config.repetition,
                "query_pool_size": config.pool_size,
                "query_sample_size": config.sample_size,
                "query_sample_mode": config.sample_mode,
                "num_trials": len(TRIALS),
                "num_batch_indices": len(batch_medians),
                "mean_query_qps": mean_qps,
            }
        )

        detailed = batch_medians.copy()
        detailed.insert(0, "repetition_percent", config.repetition)
        batch_rows.append(detailed)

    return pd.DataFrame(summary_rows), pd.concat(batch_rows, ignore_index=True)


def draw(summary: pd.DataFrame, output_prefix: Path) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axis = plt.subplots(figsize=(5.4, 3.5))
    axis.plot(
        summary["repetition_percent"],
        summary["mean_query_qps"],
        color="#0072B2",
        linewidth=2.0,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.8,
    )

    axis.set_xlabel("Query-batch overlap (%)")
    axis.set_ylabel("Average QPS")
    axis.set_xticks(summary["repetition_percent"].tolist())
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.grid(axis="y", color="#D0D0D0", linewidth=0.8, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.margins(x=0.04, y=0.12)

    for row in summary.itertuples(index=False):
        axis.annotate(
            f"{row.mean_query_qps:,.0f}",
            (row.repetition_percent, row.mean_query_qps),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    figure.tight_layout()
    pdf_path = output_prefix.with_suffix(".pdf")
    png_path = output_prefix.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return pdf_path, png_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot two-stage average query QPS against query-batch overlap."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"directory containing repetition result folders (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "different_repetition_qps",
        help="output path without a suffix",
    )
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary, batch_medians = collect_summary(results_dir)
    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.csv")
    batch_medians_path = output_prefix.with_name(
        f"{output_prefix.name}_batch_medians.csv"
    )
    summary.to_csv(summary_path, index=False)
    batch_medians.to_csv(batch_medians_path, index=False)
    pdf_path, png_path = draw(summary, output_prefix)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved per-batch medians: {batch_medians_path}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved figure: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
