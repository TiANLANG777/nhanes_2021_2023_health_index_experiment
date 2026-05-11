"""Generate Chapter 4-ready figures from saved experiment artifacts.
基于已保存实验产物生成可用于第4章的图表。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_utils import DEFAULT_OUTPUT_DIR


MODEL_ORDER = [
    "Ridge",
    "ElasticNet",
    "GradientBoostingRegressor",
    "RandomForestRegressor",
    "XGBRegressor",
    "LGBMRegressor",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. / 解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Generate Chapter 4-ready figures from saved outputs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory containing experiment outputs.")
    return parser.parse_args()


def read_csv_required(path: Path) -> pd.DataFrame:
    """Read CSV and raise if missing. / 读取 CSV，缺失则报错。"""
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV file: {path}")
    return pd.read_csv(path)


def plot_model_comparison(metric: str, ylabel: str, title: str, output_path: Path, hv2_table: pd.DataFrame) -> None:
    """Plot grouped model comparison bars. / 绘制模型对比柱状图。"""
    success = hv2_table.loc[hv2_table["status"] == "success"].copy()
    if success.empty:
        raise ValueError("No successful Hv2 model rows are available for plotting.")
    success["model_plot"] = success["model_label"].fillna(success["model"])
    model_names = [name for name in MODEL_ORDER if name in success["model_plot"].tolist()]
    feature_sets = [item for item in ["full", "reduced"] if item in success["feature_set"].tolist()]

    x = np.arange(len(model_names))
    width = 0.35 if len(feature_sets) > 1 else 0.5
    plt.figure(figsize=(11, 6))
    for offset_index, feature_set in enumerate(feature_sets):
        subset = success.loc[success["feature_set"] == feature_set].set_index("model_plot")
        values = [subset.loc[name, metric] if name in subset.index else np.nan for name in model_names]
        offset = (offset_index - (len(feature_sets) - 1) / 2) * width
        plt.bar(x + offset, values, width=width, label=feature_set)
    plt.xticks(x, model_names, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    if len(feature_sets) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_predicted_vs_actual(prediction_frame: pd.DataFrame, title: str, output_path: Path) -> None:
    """Plot predicted vs actual scatter. / 绘制预测值与真实值散点图。"""
    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(prediction_frame["H_v2_true"], prediction_frame["H_v2_pred"], alpha=0.6)
    min_value = min(prediction_frame["H_v2_true"].min(), prediction_frame["H_v2_pred"].min())
    max_value = max(prediction_frame["H_v2_true"].max(), prediction_frame["H_v2_pred"].max())
    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    plt.xlabel("Actual H_v2")
    plt.ylabel("Predicted H_v2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_residual_distribution(prediction_frame: pd.DataFrame, title: str, output_path: Path) -> None:
    """Plot residual histogram. / 绘制残差直方图。"""
    plt.figure(figsize=(7, 5))
    plt.hist(prediction_frame["residual"], bins=30)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_residual_vs_predicted(prediction_frame: pd.DataFrame, title: str, output_path: Path) -> None:
    """Plot predicted value vs residual. / 绘制预测值与残差散点图。"""
    plt.figure(figsize=(7, 5))
    plt.scatter(prediction_frame["H_v2_pred"], prediction_frame["residual"], alpha=0.6)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Predicted H_v2")
    plt.ylabel("Residual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_age_group_mae(age_group_table: pd.DataFrame, output_path: Path) -> None:
    """Plot age-group MAE comparison. / 绘制年龄组 MAE 对比图。"""
    subset = age_group_table.loc[age_group_table["age_group"] != "overall"].copy()
    if subset.empty:
        raise ValueError("No non-overall age-group rows are available for plotting.")
    feature_sets = [item for item in ["full", "reduced"] if item in subset["feature_set"].tolist()]
    age_groups = list(dict.fromkeys(subset["age_group"].tolist()))
    x = np.arange(len(age_groups))
    width = 0.35 if len(feature_sets) > 1 else 0.5
    plt.figure(figsize=(10, 6))
    for offset_index, feature_set in enumerate(feature_sets):
        feature_subset = subset.loc[subset["feature_set"] == feature_set].set_index("age_group")
        values = [feature_subset.loc[group, "mae"] if group in feature_subset.index else np.nan for group in age_groups]
        offset = (offset_index - (len(feature_sets) - 1) / 2) * width
        plt.bar(x + offset, values, width=width, label=feature_set)
    plt.xticks(x, age_groups, rotation=20, ha="right")
    plt.ylabel("MAE")
    plt.title("Age-group stability (MAE)")
    if len(feature_sets) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_hv1_vs_hv2_rmse(hv2_table: pd.DataFrame, hv1_table: pd.DataFrame, output_path: Path) -> None:
    """Plot Hv1 vs Hv2 RMSE comparison. / 绘制 Hv1 与 Hv2 的 RMSE 对比图。"""
    hv2_success = hv2_table.loc[hv2_table["status"] == "success"].copy()
    hv1_success = hv1_table.loc[hv1_table["status"] == "success"].copy()
    if hv2_success.empty or hv1_success.empty:
        raise ValueError("Hv1 or Hv2 successful rows are missing for RMSE comparison.")

    hv2_success["model_plot"] = hv2_success["model_label"].fillna(hv2_success["model"])
    hv1_success["model_plot"] = hv1_success["model_label"].fillna(hv1_success["model"])
    hv2_success["label"] = hv2_success["feature_set"] + " | " + hv2_success["model_plot"]
    hv1_success["label"] = hv1_success["feature_set"] + " | " + hv1_success["model_plot"]

    merged = hv2_success[["label", "holdout_rmse"]].merge(
        hv1_success[["label", "holdout_rmse"]],
        on="label",
        how="inner",
        suffixes=("_hv2", "_hv1"),
    )
    if merged.empty:
        raise ValueError("No overlapping Hv1/Hv2 model labels are available for RMSE comparison.")

    x = np.arange(len(merged))
    width = 0.35
    plt.figure(figsize=(max(10, len(merged) * 0.8), 6))
    plt.bar(x - width / 2, merged["holdout_rmse_hv2"], width=width, label="H_v2")
    plt.bar(x + width / 2, merged["holdout_rmse_hv1"], width=width, label="H_v1")
    plt.xticks(x, merged["label"], rotation=40, ha="right")
    plt.ylabel("Holdout RMSE")
    plt.title("H_v1 vs H_v2 model RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> int:
    """Generate Chapter 4-ready figures. / 生成可用于第4章的图表。"""
    args = parse_args()
    figures_dir = args.output_dir / "figures"
    reports_dir = args.output_dir / "reports"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    hv2_table = read_csv_required(args.output_dir / "tables" / "hv2_model_comparison.csv")
    age_group_table = read_csv_required(args.output_dir / "tables" / "age_group_stability.csv")
    hv1_table = read_csv_required(args.output_dir / "tables" / "hv1_sensitivity_model_comparison.csv")

    full_predictions = read_csv_required(args.output_dir / "hv2_training" / "full" / "best_model_holdout_predictions.csv")
    reduced_predictions = read_csv_required(args.output_dir / "hv2_training" / "reduced" / "best_model_holdout_predictions.csv")

    report_lines = [
        "# Chapter 4 Figures Report",
        "",
        "This report lists the figures generated from saved experiment artifacts only.",
        "该报告只记录基于已保存实验产物生成的图，不伪造任何结果。",
        "",
    ]

    figure_specs = [
        (
            "outputs/figures/model_comparison_rmse.png",
            lambda: plot_model_comparison(
                metric="holdout_rmse",
                ylabel="Holdout RMSE",
                title="Hv2 model comparison (RMSE)",
                output_path=figures_dir / "model_comparison_rmse.png",
                hv2_table=hv2_table,
            ),
            "outputs/tables/hv2_model_comparison.csv",
            "Chapter 4 model-performance comparison section",
        ),
        (
            "outputs/figures/model_comparison_r2.png",
            lambda: plot_model_comparison(
                metric="holdout_r2",
                ylabel="Holdout R2",
                title="Hv2 model comparison (R2)",
                output_path=figures_dir / "model_comparison_r2.png",
                hv2_table=hv2_table,
            ),
            "outputs/tables/hv2_model_comparison.csv",
            "Chapter 4 model-performance comparison section",
        ),
        (
            "outputs/figures/predicted_vs_actual_full.png",
            lambda: plot_predicted_vs_actual(full_predictions, "Predicted vs actual H_v2 (full)", figures_dir / "predicted_vs_actual_full.png"),
            "outputs/hv2_training/full/best_model_holdout_predictions.csv",
            "Chapter 4 regression-fit visualization for the full feature set",
        ),
        (
            "outputs/figures/predicted_vs_actual_reduced.png",
            lambda: plot_predicted_vs_actual(reduced_predictions, "Predicted vs actual H_v2 (reduced)", figures_dir / "predicted_vs_actual_reduced.png"),
            "outputs/hv2_training/reduced/best_model_holdout_predictions.csv",
            "Chapter 4 regression-fit visualization for the reduced feature set",
        ),
        (
            "outputs/figures/residual_distribution_full.png",
            lambda: plot_residual_distribution(full_predictions, "Residual distribution (full)", figures_dir / "residual_distribution_full.png"),
            "outputs/hv2_training/full/best_model_holdout_predictions.csv",
            "Chapter 4 residual diagnostics for the full feature set",
        ),
        (
            "outputs/figures/residual_distribution_reduced.png",
            lambda: plot_residual_distribution(reduced_predictions, "Residual distribution (reduced)", figures_dir / "residual_distribution_reduced.png"),
            "outputs/hv2_training/reduced/best_model_holdout_predictions.csv",
            "Chapter 4 residual diagnostics for the reduced feature set",
        ),
        (
            "outputs/figures/residual_vs_predicted_full.png",
            lambda: plot_residual_vs_predicted(full_predictions, "Residual vs predicted (full)", figures_dir / "residual_vs_predicted_full.png"),
            "outputs/hv2_training/full/best_model_holdout_predictions.csv",
            "Chapter 4 residual-pattern check for the full feature set",
        ),
        (
            "outputs/figures/age_group_stability_mae.png",
            lambda: plot_age_group_mae(age_group_table, figures_dir / "age_group_stability_mae.png"),
            "outputs/tables/age_group_stability.csv",
            "Chapter 4 subgroup stability section",
        ),
        (
            "outputs/figures/hv1_vs_hv2_model_rmse.png",
            lambda: plot_hv1_vs_hv2_rmse(hv2_table, hv1_table, figures_dir / "hv1_vs_hv2_model_rmse.png"),
            "outputs/tables/hv2_model_comparison.csv and outputs/tables/hv1_sensitivity_model_comparison.csv",
            "Chapter 4 sensitivity-analysis comparison section",
        ),
    ]

    for figure_path, generator, source_csv, chapter_position in figure_specs:
        generator()
        report_lines.append(f"- `{figure_path}` generated from `{source_csv}`. Suggested Chapter 4 placement: {chapter_position}.")

    (reports_dir / "chapter4_figures_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote {reports_dir / 'chapter4_figures_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
