"""Assess Hv2 prediction stability across age groups from saved artifacts.
基于已保存产物评估 Hv2 预测在不同年龄组中的稳定性。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from experiment_utils import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, FEATURE_FILES, TARGET_FILE, compute_rmse, derive_age_group, safe_float_text


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. / 解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run age-group stability checks from saved Hv2 prediction artifacts.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing input CSV files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated outputs.")
    return parser.parse_args()


def resolve_age_group(prediction_frame: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Resolve age_group by direct column or merge key. / 通过现有列或合并键解析 age_group。"""
    if "age_group" in prediction_frame.columns and prediction_frame["age_group"].notna().any():
        resolved = prediction_frame.copy()
        resolved["age_group"] = resolved["age_group"].fillna("missing")
        return resolved, "prediction_file.age_group"

    target_frame = targets.reset_index(drop=True).copy()
    target_frame["row_id"] = np.arange(len(target_frame))
    if "age_group" not in target_frame.columns:
        if "RIDAGEYR" not in target_frame.columns:
            raise ValueError("Cannot derive age_group because both age_group and RIDAGEYR are missing in adult_targets_v2.csv.")
        target_frame["age_group"] = derive_age_group(target_frame["RIDAGEYR"])
    else:
        target_frame["age_group"] = target_frame["age_group"].fillna("missing")

    for merge_key in ["row_id", "SEQN"]:
        if merge_key in prediction_frame.columns and merge_key in target_frame.columns:
            merged = prediction_frame.merge(target_frame[[merge_key, "age_group"]], on=merge_key, how="left", suffixes=("", "_target"))
            if merged["age_group"].notna().any():
                return merged, merge_key

    raise ValueError(
        "Unable to attach age_group to best_model_holdout_predictions.csv. "
        "Expected an existing age_group column or a merge key such as row_id or SEQN."
    )


def compute_group_metrics(frame: pd.DataFrame, feature_set: str, model_name: str) -> dict[str, object]:
    """Compute metrics for one age group. / 计算单个年龄组的指标。"""
    y_true = frame["H_v2_true"]
    y_pred = frame["H_v2_pred"]
    residual = y_true - y_pred
    return {
        "feature_set": feature_set,
        "model": model_name,
        "age_group": str(frame["age_group"].iloc[0]),
        "n": int(len(frame)),
        "rmse": compute_rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(frame) > 1 else float("nan"),
        "mean_residual": float(np.mean(residual)),
    }


def main() -> int:
    """Run age-group stability analysis without retraining. / 在不重训模型的前提下运行年龄组稳定性分析。"""
    args = parse_args()
    tables_dir = args.output_dir / "tables"
    reports_dir = args.output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    targets = pd.read_csv(args.data_dir / TARGET_FILE)
    rows: list[dict[str, object]] = []
    report_lines = [
        "# Age Group Stability Report",
        "",
        "This analysis only reads saved Hv2 prediction artifacts.",
        "该分析只读取已保存的 Hv2 预测产物，不会重新训练模型。",
        "",
    ]

    for feature_set in FEATURE_FILES:
        prediction_path = args.output_dir / "hv2_training" / feature_set / "best_model_holdout_predictions.csv"
        metadata_path = args.output_dir / "hv2_training" / feature_set / "training_metadata.json"
        if not prediction_path.exists():
            report_lines.append(f"- `{feature_set}`: missing `{prediction_path}`.")
            continue
        if not metadata_path.exists():
            report_lines.append(f"- `{feature_set}`: missing `{metadata_path}`.")
            continue

        prediction_frame = pd.read_csv(prediction_path)
        metadata = pd.read_json(metadata_path, typ="series")
        model_name = str(metadata.get("selected_model_name", "unknown"))
        resolved_frame, merge_source = resolve_age_group(prediction_frame, targets)
        if resolved_frame["age_group"].isna().all():
            raise ValueError(f"All age_group values are missing after merge for feature_set={feature_set}.")

        overall = resolved_frame.copy()
        overall["age_group"] = "overall"
        rows.append(compute_group_metrics(overall, feature_set, model_name))
        for _, group in resolved_frame.groupby("age_group", dropna=False):
            rows.append(compute_group_metrics(group, feature_set, model_name))
        report_lines.append(f"- `{feature_set}` merged age_group via `{merge_source}` using model `{model_name}`.")

    if not rows:
        raise FileNotFoundError("No best_model_holdout_predictions.csv files were found. Run Hv2 model training first.")

    summary = pd.DataFrame(rows)
    summary.to_csv(tables_dir / "age_group_stability.csv", index=False, encoding="utf-8-sig")

    report_lines.extend(["", "## Summary", ""])
    for feature_set in FEATURE_FILES:
        subset = summary.loc[(summary["feature_set"] == feature_set) & (summary["age_group"] == "overall")]
        if subset.empty:
            continue
        row = subset.iloc[0]
        report_lines.append(
            f"- `{feature_set}` overall MAE={safe_float_text(row['mae'])}, RMSE={safe_float_text(row['rmse'])}, R2={safe_float_text(row['r2'])}."
        )

    (reports_dir / "age_group_stability_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote {tables_dir / 'age_group_stability.csv'}")
    print(f"Wrote {reports_dir / 'age_group_stability_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
