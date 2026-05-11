"""Run H_v1 sensitivity analysis with resumable per-model artifacts.
运行支持断点续跑的 H_v1 敏感性分析。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split

from experiment_utils import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    FEATURE_FILES,
    MODEL_CATALOG,
    MODEL_CHOICES,
    MODEL_LOOKUP,
    RANDOM_STATE,
    REBUILD_MODEL_SENTINEL,
    build_model_registry,
    compute_rmse,
    compute_spearman,
    get_model_result_paths,
    make_prediction_frame,
    prepare_regression_inputs,
    safe_float_text,
    success_artifacts_exist,
)

LEADERBOARD_COLUMNS = [
    "feature_set",
    "target",
    "model",
    "model_label",
    "model_family",
    "status",
    "failure_reason",
    "selected_as_best_model_artifact",
    "cv_rmse_mean",
    "cv_rmse_std",
    "cv_mae_mean",
    "cv_r2_mean",
    "cv_spearman_mean",
    "holdout_rmse",
    "holdout_mae",
    "holdout_r2",
    "holdout_spearman",
    "mean_residual",
    "std_residual",
    "prediction_min",
    "prediction_max",
    "n_train",
    "n_test",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. / 解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run H_v1 sensitivity analysis with resumable artifacts.")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_FILES), default="full", help="Feature set to evaluate.")
    parser.add_argument(
        "--model",
        choices=[*MODEL_CHOICES, "all", REBUILD_MODEL_SENTINEL],
        default="all",
        help="Run one model, all models, or only rebuild the leaderboard.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun a model even if its saved artifacts already exist.")
    parser.add_argument(
        "--rebuild-leaderboard-only",
        action="store_true",
        help="Only rebuild leaderboard files from saved model artifacts without retraining.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing input CSV files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for experiment outputs.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds inside the training split.")
    return parser.parse_args()


def compute_holdout_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute holdout metrics. / 计算留出集评估指标。"""
    residual = y_true - y_pred
    return {
        "holdout_rmse": compute_rmse(y_true, y_pred),
        "holdout_mae": float(mean_absolute_error(y_true, y_pred)),
        "holdout_r2": float(r2_score(y_true, y_pred)),
        "holdout_spearman": compute_spearman(y_true, y_pred),
        "mean_residual": float(np.mean(residual)),
        "std_residual": float(np.std(residual)),
        "prediction_min": float(np.min(y_pred)),
        "prediction_max": float(np.max(y_pred)),
    }


def build_success_row(feature_set: str, model_name: str, model_label: str, model_family: str, cv_result: dict[str, np.ndarray], holdout_metrics: dict[str, float], n_train: int, n_test: int) -> dict[str, object]:
    """Build one successful leaderboard row. / 构造单个成功模型的排行榜记录。"""
    return {
        "feature_set": feature_set,
        "target": "H_v1",
        "model": model_name,
        "model_label": model_label,
        "model_family": model_family,
        "status": "success",
        "failure_reason": "",
        "selected_as_best_model_artifact": False,
        "cv_rmse_mean": float(-cv_result["test_rmse"].mean()),
        "cv_rmse_std": float(cv_result["test_rmse"].std()),
        "cv_mae_mean": float(-cv_result["test_mae"].mean()),
        "cv_r2_mean": float(cv_result["test_r2"].mean()),
        "cv_spearman_mean": float(np.nanmean(cv_result["test_spearman"])),
        "holdout_rmse": holdout_metrics["holdout_rmse"],
        "holdout_mae": holdout_metrics["holdout_mae"],
        "holdout_r2": holdout_metrics["holdout_r2"],
        "holdout_spearman": holdout_metrics["holdout_spearman"],
        "mean_residual": holdout_metrics["mean_residual"],
        "std_residual": holdout_metrics["std_residual"],
        "prediction_min": holdout_metrics["prediction_min"],
        "prediction_max": holdout_metrics["prediction_max"],
        "n_train": int(n_train),
        "n_test": int(n_test),
    }


def make_failure_row(feature_set: str, model_name: str, model_label: str, model_family: str, failure_reason: str, n_train: int, n_test: int) -> dict[str, object]:
    """Build one failed leaderboard row. / 构造单个失败模型的排行榜记录。"""
    row = {column: np.nan for column in LEADERBOARD_COLUMNS}
    row.update(
        {
            "feature_set": feature_set,
            "target": "H_v1",
            "model": model_name,
            "model_label": model_label,
            "model_family": model_family,
            "status": "failed",
            "failure_reason": failure_reason,
            "selected_as_best_model_artifact": False,
            "n_train": int(n_train),
            "n_test": int(n_test),
        }
    )
    return row


def load_saved_rows(feature_set: str, model_results_dir: Path, n_train: int, n_test: int) -> pd.DataFrame:
    """Load saved rows from model result artifacts. / 从模型结果产物中载入已保存记录。"""
    rows: list[dict[str, object]] = []
    for model_spec in MODEL_CATALOG:
        model_name = model_spec["model"]
        paths = get_model_result_paths(model_results_dir, model_name)
        if paths["metrics"].exists():
            row = json.loads(paths["metrics"].read_text(encoding="utf-8"))
            row["selected_as_best_model_artifact"] = False
            rows.append(row)
            continue
        if paths["error"].exists():
            failure_reason = paths["error"].read_text(encoding="utf-8").strip()
            rows.append(
                make_failure_row(
                    feature_set=feature_set,
                    model_name=model_name,
                    model_label=model_spec["model_label"],
                    model_family=model_spec["model_family"],
                    failure_reason=failure_reason,
                    n_train=n_train,
                    n_test=n_test,
                )
            )
    leaderboard = pd.DataFrame(rows) if rows else pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    for column in LEADERBOARD_COLUMNS:
        if column not in leaderboard.columns:
            leaderboard[column] = np.nan
    return leaderboard[LEADERBOARD_COLUMNS]


def rebuild_global_outputs(output_dir: Path) -> None:
    """Rebuild global Hv1 sensitivity outputs. / 重建全局 Hv1 敏感性分析汇总输出。"""
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    combined_frames: list[pd.DataFrame] = []
    metadata_by_feature_set: dict[str, dict[str, object]] = {}
    for feature_set in FEATURE_FILES:
        leaderboard_path = output_dir / "hv1_sensitivity" / feature_set / "leaderboard.csv"
        metadata_path = output_dir / "hv1_sensitivity" / feature_set / "training_metadata.json"
        if leaderboard_path.exists():
            combined_frames.append(pd.read_csv(leaderboard_path))
        if metadata_path.exists():
            metadata_by_feature_set[feature_set] = json.loads(metadata_path.read_text(encoding="utf-8"))

    combined = pd.concat(combined_frames, ignore_index=True) if combined_frames else pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    if not combined.empty:
        for column in LEADERBOARD_COLUMNS:
            if column not in combined.columns:
                combined[column] = np.nan
        combined = combined[LEADERBOARD_COLUMNS]
    combined.to_csv(tables_dir / "hv1_sensitivity_model_comparison.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# H_v1 Sensitivity Report",
        "",
        "This report is generated from saved experiment artifacts only.",
        "该报告只基于已保存的实验产物生成，不伪造任何结果。",
        "",
        "It does not generate thesis Chapter 4 text.",
        "该报告不是论文第4章正文。",
        "",
        "## Protocol",
        "",
        "- Target: `H_v1`.",
        "- Train/test split: 80/20 with `random_state=42`.",
        "- Cross-validation: 5-fold inside the training split only.",
        "- Missing-value imputation stays inside each pipeline.",
        "- Leakage variables are excluded from model inputs.",
        "",
    ]

    for feature_set in FEATURE_FILES:
        lines.append(f"## {feature_set}")
        leaderboard = combined.loc[combined["feature_set"] == feature_set].copy() if not combined.empty else pd.DataFrame()
        metadata = metadata_by_feature_set.get(feature_set)
        if leaderboard.empty:
            lines.append("- No saved leaderboard yet.")
            lines.append("")
            continue

        selected = leaderboard.loc[leaderboard["selected_as_best_model_artifact"] == True]  # noqa: E712
        if not selected.empty:
            best_row = selected.iloc[0]
            lines.append(
                f"- Selected ensemble artifact: `{best_row['model_label']}` with CV RMSE {safe_float_text(best_row['cv_rmse_mean'])}, holdout RMSE {safe_float_text(best_row['holdout_rmse'])}, holdout R2 {safe_float_text(best_row['holdout_r2'])}."
            )
        else:
            lines.append("- No selected ensemble artifact is available yet.")

        successful = leaderboard.loc[leaderboard["status"] == "success"]
        failed = leaderboard.loc[leaderboard["status"] == "failed"]
        lines.append(f"- Successful models: {successful['model_label'].tolist()}")
        if failed.empty:
            lines.append("- Failed models: none.")
        else:
            lines.append("- Failed models:")
            for _, row in failed.iterrows():
                lines.append(f"  - `{row['model_label']}`: {row['failure_reason']}")
        if metadata is not None:
            lines.append(f"- Dropped banned columns: {metadata.get('dropped_banned_columns', [])}")
        lines.append("")

    (reports_dir / "hv1_sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8")


def finalize_feature_set_outputs(output_dir: Path, feature_set: str, model_results_dir: Path, banned_columns: list[str], feature_names: list[str], n_train: int, n_test: int, args: argparse.Namespace, package_versions: dict[str, str | None], package_import_errors: dict[str, str | None], n_rows_after_target_filter: int) -> pd.DataFrame:
    """Finalize feature-set-level outputs. / 完成单个特征集的汇总输出。"""
    run_dir = output_dir / "hv1_sensitivity" / feature_set
    leaderboard = load_saved_rows(feature_set, model_results_dir, n_train=n_train, n_test=n_test)

    selected_model_name = None
    selected_model_family = None
    if not leaderboard.empty:
        successful_ensembles = leaderboard.loc[
            (leaderboard["status"] == "success") & (leaderboard["model_family"] == "ensemble")
        ].sort_values("cv_rmse_mean")
        if not successful_ensembles.empty:
            selected_model_name = str(successful_ensembles.iloc[0]["model"])
            selected_model_family = str(successful_ensembles.iloc[0]["model_family"])
            leaderboard.loc[leaderboard["model"] == selected_model_name, "selected_as_best_model_artifact"] = True
            selected_paths = get_model_result_paths(model_results_dir, selected_model_name)
            if selected_paths["artifact"].exists():
                shutil.copyfile(selected_paths["artifact"], run_dir / "best_model.joblib")
            if selected_paths["predictions"].exists():
                shutil.copyfile(selected_paths["predictions"], run_dir / "best_model_holdout_predictions.csv")

        sort_rank = {"success": 0, "failed": 1}
        leaderboard["_sort_status"] = leaderboard["status"].map(sort_rank).fillna(9)
        leaderboard = leaderboard.sort_values(["_sort_status", "cv_rmse_mean", "model_family", "model"]).drop(columns=["_sort_status"])

    leaderboard.to_csv(run_dir / "leaderboard.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": feature_names}).to_csv(run_dir / "feature_columns.csv", index=False, encoding="utf-8-sig")

    metadata_payload = {
        "feature_set": feature_set,
        "target_column": "H_v1",
        "selected_model_name": selected_model_name,
        "selected_model_family": selected_model_family,
        "dropped_banned_columns": banned_columns,
        "random_state": RANDOM_STATE,
        "train_test_split": "80/20",
        "cv_strategy": f"{args.n_splits}-fold on training split only",
        "requested_model_mode": args.model,
        "force_rerun": bool(args.force),
        "n_rows_after_target_filter": int(n_rows_after_target_filter),
        "n_features": int(len(feature_names)),
        "feature_names": feature_names,
        "package_versions": package_versions,
        "package_import_errors": package_import_errors,
        "model_statuses": leaderboard[["model", "model_family", "status", "failure_reason"]].to_dict(orient="records") if not leaderboard.empty else [],
    }
    (run_dir / "training_metadata.json").write_text(json.dumps(metadata_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    rebuild_global_outputs(output_dir)
    return leaderboard


def main() -> int:
    """Train H_v1 sensitivity models and persist artifacts. / 训练 H_v1 敏感性分析模型并保存产物。"""
    args = parse_args()
    run_dir = args.output_dir / "hv1_sensitivity" / args.feature_set
    model_results_dir = run_dir / "model_results"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_results_dir.mkdir(parents=True, exist_ok=True)

    X, y, metadata, banned_columns = prepare_regression_inputs(args.data_dir, args.feature_set, target_column="H_v1")
    feature_names = X.columns.tolist()
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X,
        y,
        metadata,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    registry, import_failures, package_versions, package_import_errors = build_model_registry()
    rebuild_only = bool(args.rebuild_leaderboard_only or args.model == REBUILD_MODEL_SENTINEL)
    if rebuild_only:
        leaderboard = finalize_feature_set_outputs(
            output_dir=args.output_dir,
            feature_set=args.feature_set,
            model_results_dir=model_results_dir,
            banned_columns=banned_columns,
            feature_names=feature_names,
            n_train=len(X_train),
            n_test=len(X_test),
            args=args,
            package_versions=package_versions,
            package_import_errors=package_import_errors,
            n_rows_after_target_filter=len(X),
        )
        print(leaderboard.to_string(index=False) if not leaderboard.empty else "No saved model results yet.")
        print("Rebuilt leaderboard only.")
        return 0

    requested_models = MODEL_CHOICES if args.model == "all" else [args.model]
    cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "rmse": make_scorer(compute_rmse, greater_is_better=False),
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
        "spearman": make_scorer(compute_spearman, greater_is_better=True),
    }

    for model_name in requested_models:
        model_spec = MODEL_LOOKUP[model_name]
        paths = get_model_result_paths(model_results_dir, model_name)
        if success_artifacts_exist(paths) and not args.force:
            print(f"Skipping completed model: feature_set={args.feature_set}, model={model_name}")
            continue

        print(f"Running H_v1 sensitivity: feature_set={args.feature_set}, model={model_name}")
        if model_name in import_failures:
            failure_reason = import_failures[model_name]
            paths["error"].write_text(failure_reason, encoding="utf-8")
            print(f"Model import failure: {model_name} -> {failure_reason}")
            finalize_feature_set_outputs(
                output_dir=args.output_dir,
                feature_set=args.feature_set,
                model_results_dir=model_results_dir,
                banned_columns=banned_columns,
                feature_names=feature_names,
                n_train=len(X_train),
                n_test=len(X_test),
                args=args,
                package_versions=package_versions,
                package_import_errors=package_import_errors,
                n_rows_after_target_filter=len(X),
            )
            continue

        pipeline = registry[model_name]["pipeline"]
        try:
            cv_result = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
            )
            holdout_pipeline = clone(pipeline)
            holdout_pipeline.fit(X_train, y_train)
            holdout_pred = pd.Series(holdout_pipeline.predict(X_test), name="H_v1_pred")
            holdout_metrics = compute_holdout_metrics(y_test, holdout_pred)
            result_row = build_success_row(
                feature_set=args.feature_set,
                model_name=model_name,
                model_label=model_spec["model_label"],
                model_family=model_spec["model_family"],
                cv_result=cv_result,
                holdout_metrics=holdout_metrics,
                n_train=len(X_train),
                n_test=len(X_test),
            )
            paths["metrics"].write_text(json.dumps(result_row, indent=2, ensure_ascii=False), encoding="utf-8")
            prediction_frame = make_prediction_frame(meta_test, y_test, holdout_pred, true_column="H_v1_true", pred_column="H_v1_pred")
            prediction_frame["feature_set"] = args.feature_set
            prediction_frame["model"] = model_name
            prediction_frame.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")
            fitted_full_pipeline = clone(pipeline)
            fitted_full_pipeline.fit(X, y)
            joblib.dump(fitted_full_pipeline, paths["artifact"])
            print(f"Saved model artifacts: {paths['metrics']}, {paths['predictions']}, {paths['artifact']}")
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            failure_reason = f"{type(exc).__name__}: {exc}"
            paths["error"].write_text(failure_reason, encoding="utf-8")
            print(f"Model failed during training: {model_name} -> {failure_reason}")

        finalize_feature_set_outputs(
            output_dir=args.output_dir,
            feature_set=args.feature_set,
            model_results_dir=model_results_dir,
            banned_columns=banned_columns,
            feature_names=feature_names,
            n_train=len(X_train),
            n_test=len(X_test),
            args=args,
            package_versions=package_versions,
            package_import_errors=package_import_errors,
            n_rows_after_target_filter=len(X),
        )

    leaderboard = finalize_feature_set_outputs(
        output_dir=args.output_dir,
        feature_set=args.feature_set,
        model_results_dir=model_results_dir,
        banned_columns=banned_columns,
        feature_names=feature_names,
        n_train=len(X_train),
        n_test=len(X_test),
        args=args,
        package_versions=package_versions,
        package_import_errors=package_import_errors,
        n_rows_after_target_filter=len(X),
    )
    print(leaderboard.to_string(index=False) if not leaderboard.empty else "No saved model results yet.")
    print(f"Wrote {run_dir / 'leaderboard.csv'}")
    print(f"Wrote {args.output_dir / 'tables' / 'hv1_sensitivity_model_comparison.csv'}")
    print(f"Wrote {args.output_dir / 'reports' / 'hv1_sensitivity_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
