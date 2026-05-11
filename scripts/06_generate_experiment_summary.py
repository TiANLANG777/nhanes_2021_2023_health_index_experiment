"""Generate an artifact-based summary for the NHANES Hv2 experiment.
为 NHANES Hv2 实验生成基于产物的总结。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("/content/drive/MyDrive/nhanes_2021_2023_health_index_experiment/outputs")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. / 解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Generate an artifact-oriented experiment summary.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory containing experiment outputs.")
    return parser.parse_args()


def read_json_if_exists(path: Path) -> dict[str, object] | None:
    """Read JSON if present. / 如果存在则读取 JSON。"""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    """Read CSV if present. / 如果存在则读取 CSV。"""
    if not path.exists():
        return None
    return pd.read_csv(path)


def main() -> int:
    """Compile available artifacts into a markdown summary. / 将已有实验产物汇总为 Markdown。"""
    args = parse_args()
    summary_dir = args.output_dir / "experiment_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# NHANES Hv2 Experiment Summary",
        "",
        "This summary only reports files that already exist.",
        "该总结只报告已经实际生成的文件。",
        "",
    ]

    training_versions: dict[str, dict[str, object]] = {}
    for feature_set in ["full", "reduced"]:
        metadata = read_json_if_exists(args.output_dir / "hv2_training" / feature_set / "training_metadata.json")
        if metadata is not None:
            training_versions[feature_set] = metadata

    lines.extend(["## Runtime Versions", ""])
    if training_versions:
        for feature_set, metadata in training_versions.items():
            package_versions = metadata.get("package_versions", {})
            lines.append(
                f"- `{feature_set}` recorded `scikit-learn=={package_versions.get('scikit-learn', metadata.get('sklearn_version'))}`, "
                f"`xgboost=={package_versions.get('xgboost')}`, `lightgbm=={package_versions.get('lightgbm')}`."
            )
    else:
        lines.append("- No recorded runtime version metadata was found yet.")
    lines.append("")

    data_check_json = read_json_if_exists(args.output_dir / "data_check" / "data_check_summary.json")
    if data_check_json is None:
        lines.extend(["## Data Check", "", "- Data check artifacts were not found yet.", "- 尚未发现数据检查产物。", ""])
    else:
        lines.extend(["## Data Check", ""])
        for feature_item in data_check_json.get("feature_checks", []):
            lines.append(
                f"- `{feature_item['table_name']}` banned columns: {feature_item['banned_columns']}; overlap with targets: {feature_item['overlap_with_targets']}"
            )
        lines.append("")

    lines.extend(["## H_v2 Training", ""])
    training_found = False
    for feature_set in ["full", "reduced"]:
        leaderboard = read_csv_if_exists(args.output_dir / "hv2_training" / feature_set / "leaderboard.csv")
        if leaderboard is None or leaderboard.empty:
            lines.append(f"- `{feature_set}` leaderboard not found.")
            continue
        training_found = True
        if "selected_as_best_model_artifact" in leaderboard.columns:
            selected = leaderboard.loc[leaderboard["selected_as_best_model_artifact"] == True]  # noqa: E712
        else:
            selected = pd.DataFrame()
        best_row = selected.iloc[0] if not selected.empty else leaderboard.iloc[0]
        lines.append(
            f"- `{feature_set}` selected ensemble artifact: `{best_row.get('model_label', best_row['model'])}` with CV RMSE {best_row['cv_rmse_mean']:.4f}, "
            f"holdout RMSE {best_row['holdout_rmse']:.4f}, holdout R2 {best_row['holdout_r2']:.4f}."
        )
    if not training_found:
        lines.append("- No training leaderboard is available yet.")
    lines.append("")

    hv2_report_path = args.output_dir / "reports" / "hv2_regression_report.md"
    lines.extend(["## Hv2 Report", ""])
    if hv2_report_path.exists():
        lines.append(f"- Detailed regression report: `{hv2_report_path}`")
    else:
        lines.append("- Detailed Hv2 regression report was not found yet.")
    lines.append("")

    lines.extend(["## Age Group Stability", ""])
    stability_files = sorted((args.output_dir / "age_group_stability").glob("age_group_stability_*.csv"))
    if not stability_files:
        lines.extend(["- Age-group stability outputs were not found yet.", "- 尚未发现年龄组稳定性产物。", ""])
    else:
        for path in stability_files:
            table = pd.read_csv(path)
            overall = table.loc[table["age_group"] == "overall"]
            if overall.empty:
                lines.append(f"- `{path.name}` exists, but no overall row was found.")
            else:
                row = overall.iloc[0]
                lines.append(f"- `{path.name}` overall RMSE {row['rmse']:.4f}, MAE {row['mae']:.4f}, R2 {row['r2']:.4f}.")
        lines.append("")

    lines.extend(["## H_v1 Sensitivity", ""])
    sensitivity_files = sorted((args.output_dir / "hv1_sensitivity").glob("hv1_sensitivity_summary_*.csv"))
    if not sensitivity_files:
        lines.extend(["- H_v1 sensitivity outputs were not found yet.", "- 尚未发现 H_v1 敏感性分析产物。", ""])
    else:
        for path in sensitivity_files:
            table = pd.read_csv(path)
            for _, row in table.iterrows():
                lines.append(
                    f"- `{path.name}` target `{row['target']}`: RMSE {row['rmse']:.4f}, MAE {row['mae']:.4f}, R2 {row['r2']:.4f}."
                )
        lines.append("")

    lines.extend(["## SHAP", ""])
    shap_files = sorted((args.output_dir / "shap").glob("shap_importance_*.csv"))
    if not shap_files:
        lines.extend(["- SHAP artifacts were not found yet.", "- 尚未发现 SHAP 产物。", ""])
    else:
        for path in shap_files:
            table = pd.read_csv(path)
            top_rows = table.head(5)
            lines.append(f"- `{path.name}` top features:")
            for _, row in top_rows.iterrows():
                lines.append(f"  - `{row['feature']}`: {row['mean_abs_shap']:.6f}")
        lines.append("")

    lines.extend(
        [
            "## Scope Boundary",
            "",
            "- This file is an experiment artifact summary, not thesis Chapter 4 text.",
            "- 该文件是实验产物总结，不是论文第4章正文。",
        ]
    )

    output_path = summary_dir / "experiment_summary.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
