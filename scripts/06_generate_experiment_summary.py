"""Generate artifact-based experiment summaries and Chapter 4 tables.
生成基于实验产物的总总结和第4章表格。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiment_utils import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, FEATURE_FILES, round_numeric_table, safe_float_text


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. / 解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Generate artifact-oriented experiment summaries and Chapter 4 tables.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing experiment input files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory containing experiment outputs.")
    return parser.parse_args()


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    """Read CSV if present. / 如果存在则读取 CSV。"""
    if not path.exists():
        return None
    return pd.read_csv(path)


def read_json_if_exists(path: Path) -> dict[str, object] | None:
    """Read JSON if present. / 如果存在则读取 JSON。"""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_hv2_tables(output_dir: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Build Chapter 4 tables for Hv2. / 生成 Hv2 的第4章表格。"""
    hv2_path = output_dir / "tables" / "hv2_model_comparison.csv"
    hv2 = read_csv_if_exists(hv2_path)
    if hv2 is None or hv2.empty:
        return None, None

    hv2_success = hv2.loc[hv2["status"] == "success"].copy()
    if hv2_success.empty:
        return None, None

    chapter4_hv2 = hv2_success[
        [
            "feature_set",
            "model_label",
            "model_family",
            "cv_rmse_mean",
            "cv_mae_mean",
            "cv_r2_mean",
            "holdout_rmse",
            "holdout_mae",
            "holdout_r2",
            "holdout_spearman",
        ]
    ].copy()
    chapter4_hv2 = chapter4_hv2.rename(columns={"model_label": "model"})
    chapter4_hv2 = round_numeric_table(chapter4_hv2, digits=3)
    chapter4_hv2.to_csv(output_dir / "tables" / "chapter4_table_hv2_model_comparison.csv", index=False, encoding="utf-8-sig")

    selected = hv2_success.loc[hv2_success.get("selected_as_best_model_artifact", False) == True].copy()  # noqa: E712
    if selected.empty:
        selected = hv2_success.loc[hv2_success["model_family"] == "ensemble"].sort_values("cv_rmse_mean").groupby("feature_set", as_index=False).head(1)
    full_vs_reduced = selected[
        [
            "feature_set",
            "model_label",
            "cv_rmse_mean",
            "holdout_rmse",
            "holdout_mae",
            "holdout_r2",
            "holdout_spearman",
        ]
    ].copy()
    full_vs_reduced = full_vs_reduced.rename(columns={"model_label": "selected_model"})
    full_vs_reduced = round_numeric_table(full_vs_reduced, digits=3)
    full_vs_reduced.to_csv(output_dir / "tables" / "chapter4_table_full_vs_reduced.csv", index=False, encoding="utf-8-sig")
    return chapter4_hv2, full_vs_reduced


def build_age_group_table(output_dir: Path) -> pd.DataFrame | None:
    """Build Chapter 4 table for age-group stability. / 生成年龄组稳定性的第4章表格。"""
    age_group = read_csv_if_exists(output_dir / "tables" / "age_group_stability.csv")
    if age_group is None or age_group.empty:
        return None
    chapter4_age_group = round_numeric_table(age_group.copy(), digits=3)
    chapter4_age_group.to_csv(output_dir / "tables" / "chapter4_table_age_group_stability.csv", index=False, encoding="utf-8-sig")
    return chapter4_age_group


def build_hv1_table(output_dir: Path) -> pd.DataFrame | None:
    """Build Chapter 4 table for H_v1 sensitivity. / 生成 H_v1 敏感性分析的第4章表格。"""
    hv1 = read_csv_if_exists(output_dir / "tables" / "hv1_sensitivity_model_comparison.csv")
    if hv1 is None or hv1.empty:
        return None
    hv1_success = hv1.loc[hv1["status"] == "success"].copy()
    if hv1_success.empty:
        return None
    chapter4_hv1 = hv1_success[
        [
            "feature_set",
            "model_label",
            "model_family",
            "cv_rmse_mean",
            "holdout_rmse",
            "holdout_mae",
            "holdout_r2",
            "holdout_spearman",
        ]
    ].copy()
    chapter4_hv1 = chapter4_hv1.rename(columns={"model_label": "model"})
    chapter4_hv1 = round_numeric_table(chapter4_hv1, digits=3)
    chapter4_hv1.to_csv(output_dir / "tables" / "chapter4_table_hv1_sensitivity.csv", index=False, encoding="utf-8-sig")
    return chapter4_hv1


def build_shap_table(output_dir: Path) -> pd.DataFrame | None:
    """Build Chapter 4 table for SHAP top features. / 生成 SHAP 头部特征的第4章表格。"""
    shap_importance = read_csv_if_exists(output_dir / "tables" / "shap_feature_importance.csv")
    if shap_importance is None or shap_importance.empty:
        return None
    chapter4_shap = shap_importance.head(20).copy()
    chapter4_shap = round_numeric_table(chapter4_shap, digits=3)
    chapter4_shap.to_csv(output_dir / "tables" / "chapter4_table_shap_top_features.csv", index=False, encoding="utf-8-sig")
    return chapter4_shap


def main() -> int:
    """Compile available artifacts into a markdown summary. / 将已有实验产物汇总为 Markdown。"""
    args = parse_args()
    tables_dir = args.output_dir / "tables"
    reports_dir = args.output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    hv2_table, full_vs_reduced_table = build_hv2_tables(args.output_dir)
    age_group_table = build_age_group_table(args.output_dir)
    hv1_table = build_hv1_table(args.output_dir)
    shap_table = build_shap_table(args.output_dir)

    lines = [
        "# NHANES Experiment Summary",
        "",
        "This summary only reports files that already exist.",
        "该总结只报告已经实际生成的文件。",
        "",
        "It does not fabricate results.",
        "该总结不会伪造结果。",
        "",
        "It does not generate thesis Chapter 4 text.",
        "该总结不是论文第4章正文。",
        "",
        "## Runtime Metadata",
        "",
    ]

    for feature_set in FEATURE_FILES:
        metadata = read_json_if_exists(args.output_dir / "hv2_training" / feature_set / "training_metadata.json")
        if metadata is None:
            lines.append(f"- `{feature_set}` Hv2 metadata not found.")
            continue
        versions = metadata.get("package_versions", {})
        lines.append(
            f"- `{feature_set}` Hv2 versions: scikit-learn={versions.get('scikit-learn')}, xgboost={versions.get('xgboost')}, lightgbm={versions.get('lightgbm')}"
        )
    lines.append("")

    lines.extend(["## Canonical Outputs", ""])
    for relative_path in [
        "reports/data_integrity_report.md",
        "tables/leakage_check.csv",
        "tables/hv2_model_comparison.csv",
        "tables/age_group_stability.csv",
        "tables/hv1_sensitivity_model_comparison.csv",
        "tables/shap_feature_importance.csv",
        "reports/hv2_regression_report.md",
        "reports/hv1_sensitivity_report.md",
        "reports/age_group_stability_report.md",
        "reports/shap_report.md",
    ]:
        exists = (args.output_dir / relative_path).exists()
        lines.append(f"- `{relative_path}`: {exists}")
    lines.append("")

    lines.extend(["## Chapter 4 Tables", ""])
    for relative_path in [
        "tables/chapter4_table_hv2_model_comparison.csv",
        "tables/chapter4_table_full_vs_reduced.csv",
        "tables/chapter4_table_age_group_stability.csv",
        "tables/chapter4_table_hv1_sensitivity.csv",
        "tables/chapter4_table_shap_top_features.csv",
    ]:
        exists = (args.output_dir / relative_path).exists()
        lines.append(f"- `{relative_path}`: {exists}")
    lines.append("")

    if full_vs_reduced_table is not None and not full_vs_reduced_table.empty:
        lines.extend(["## Best Hv2 Models", ""])
        for _, row in full_vs_reduced_table.iterrows():
            lines.append(
                f"- `{row['feature_set']}` selected model `{row['selected_model']}` with holdout RMSE={safe_float_text(row['holdout_rmse'])}, holdout R2={safe_float_text(row['holdout_r2'])}."
            )
        lines.append("")

    if age_group_table is not None and not age_group_table.empty:
        lines.extend(["## Age Group Stability", ""])
        overall_rows = age_group_table.loc[age_group_table["age_group"] == "overall"]
        for _, row in overall_rows.iterrows():
            lines.append(
                f"- `{row['feature_set']}` overall age-group stability: MAE={safe_float_text(row['mae'])}, RMSE={safe_float_text(row['rmse'])}, R2={safe_float_text(row['r2'])}."
            )
        lines.append("")

    if shap_table is not None and not shap_table.empty:
        lines.extend(["## SHAP Top Features", ""])
        for _, row in shap_table.head(10).iterrows():
            lines.append(f"- `{row['feature']}`: {safe_float_text(row['mean_abs_shap'], digits=6)}")
        lines.append("")

    lines.extend(
        [
            "## Scope Boundary",
            "",
            "- This file is an experiment artifact summary, not thesis Chapter 4 text.",
            "- 该文件是实验产物总结，不是论文第4章正文。",
        ]
    )

    output_path = reports_dir / "experiment_summary.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
