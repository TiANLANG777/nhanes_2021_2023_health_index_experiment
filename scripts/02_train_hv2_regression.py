"""Train baseline and ensemble regressors for the NHANES Hv2 experiment.
训练 NHANES Hv2 实验的基线与集成回归模型。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost
    from xgboost import XGBRegressor
except Exception as exc:  # pragma: no cover - runtime environment dependent
    xgboost = None
    XGBRegressor = None
    XGBOOST_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    XGBOOST_IMPORT_ERROR = ""

try:
    import lightgbm
    from lightgbm import LGBMRegressor
except Exception as exc:  # pragma: no cover - runtime environment dependent
    lightgbm = None
    LGBMRegressor = None
    LIGHTGBM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    LIGHTGBM_IMPORT_ERROR = ""


DEFAULT_DATA_DIR = Path("/content/drive/MyDrive/nhanes_2021_2023_health_index_experiment/data")
DEFAULT_OUTPUT_DIR = Path("/content/drive/MyDrive/nhanes_2021_2023_health_index_experiment/outputs")
FEATURE_FILES = {
    "full": "adult_full_feature_set_v2.csv",
    "reduced": "adult_reduced_feature_set_v2.csv",
}
TARGET_FILE = "adult_targets_v2.csv"
STRICT_BANNED_COLUMNS = {
    "SEQN",
    "H_v1",
    "H_v2",
    "R",
    "R_v2",
    "H_grade",
    "H_grade_quantile",
    "available_risk_dimensions",
    "age_group",
}
STRICT_BANNED_PREFIXES = ("r_",)
RANDOM_STATE = 42
MODEL_CATALOG = [
    {"model": "random_forest", "model_label": "RandomForestRegressor", "model_family": "ensemble"},
    {"model": "gradient_boosting", "model_label": "GradientBoostingRegressor", "model_family": "ensemble"},
    {"model": "xgboost", "model_label": "XGBRegressor", "model_family": "ensemble"},
    {"model": "lightgbm", "model_label": "LGBMRegressor", "model_family": "ensemble"},
    {"model": "ridge", "model_label": "Ridge", "model_family": "baseline"},
    {"model": "elastic_net", "model_label": "ElasticNet", "model_family": "baseline"},
]
LEADERBOARD_COLUMNS = [
    "feature_set",
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
    parser = argparse.ArgumentParser(description="Train thesis-aligned regressors for H_v2.")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_FILES), default="full", help="Feature set to use.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing input CSV files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for experiment outputs.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds inside the training split.")
    return parser.parse_args()


def detect_banned_columns(columns: list[str]) -> list[str]:
    """Detect forbidden input columns. / 检测禁止进入模型的输入列。"""
    banned = [column for column in columns if column in STRICT_BANNED_COLUMNS or column.startswith(STRICT_BANNED_PREFIXES)]
    return sorted(set(banned))


def compute_rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """中文：计算 RMSE，避免依赖 sklearn 的 squared=False 参数。
    English: Compute RMSE without relying on sklearn's squared=False argument.
    """
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def compute_spearman(y_true: pd.Series, y_pred: pd.Series) -> float:
    """中文：计算 Spearman 相关系数，用于排序一致性评估。
    English: Compute Spearman correlation for rank-consistency evaluation.
    """
    result = spearmanr(y_true, y_pred)
    correlation = getattr(result, "correlation", result[0] if isinstance(result, tuple) else result)
    if correlation is None or np.isnan(correlation):
        return float("nan")
    return float(correlation)


def build_model_registry() -> tuple[dict[str, dict[str, object]], dict[str, str], dict[str, str | None]]:
    """Create model specifications and record import failures. / 创建模型规格并记录导入失败信息。"""
    registry: dict[str, dict[str, object]] = {
        "random_forest": {
            "model_family": "ensemble",
            "model_label": "RandomForestRegressor",
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=400,
                            min_samples_leaf=2,
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        },
        "gradient_boosting": {
            "model_family": "ensemble",
            "model_label": "GradientBoostingRegressor",
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                    ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
                ]
            ),
        },
        "ridge": {
            "model_family": "baseline",
            "model_label": "Ridge",
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
                ]
            ),
        },
        "elastic_net": {
            "model_family": "baseline",
            "model_label": "ElasticNet",
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=RANDOM_STATE)),
                ]
            ),
        },
    }
    failures: dict[str, str] = {}
    package_versions = {
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__ if xgboost is not None else None,
        "lightgbm": lightgbm.__version__ if lightgbm is not None else None,
    }

    if XGBRegressor is None:
        failures["xgboost"] = (
            "中文：XGBoost 导入失败，请确认 requirements_colab.txt 已安装。 "
            "English: XGBoost import failed. Please make sure requirements_colab.txt is installed. "
            f"Original error: {XGBOOST_IMPORT_ERROR}"
        )
    else:
        registry["xgboost"] = {
            "model_family": "ensemble",
            "model_label": "XGBRegressor",
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                    (
                        "model",
                        XGBRegressor(
                            objective="reg:squarederror",
                            n_estimators=400,
                            learning_rate=0.05,
                            max_depth=4,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                            verbosity=0,
                        ),
                    ),
                ]
            ),
        }

    if LGBMRegressor is None:
        failures["lightgbm"] = (
            "中文：LightGBM 导入失败，请确认 requirements_colab.txt 已安装。 "
            "English: LightGBM import failed. Please make sure requirements_colab.txt is installed. "
            f"Original error: {LIGHTGBM_IMPORT_ERROR}"
        )
    else:
        registry["lightgbm"] = {
            "model_family": "ensemble",
            "model_label": "LGBMRegressor",
            "pipeline": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                    (
                        "model",
                        LGBMRegressor(
                            n_estimators=400,
                            learning_rate=0.05,
                            num_leaves=31,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                            verbosity=-1,
                        ),
                    ),
                ]
            ),
        }

    return registry, failures, package_versions


def load_bundle(data_dir: Path, feature_set: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load features and targets, then strip leakage columns. / 载入特征与目标并剥离泄漏列。"""
    feature_path = data_dir / FEATURE_FILES[feature_set]
    target_path = data_dir / TARGET_FILE
    if not feature_path.exists() or not target_path.exists():
        raise FileNotFoundError(f"Missing input files: {feature_path}, {target_path}")

    features = pd.read_csv(feature_path)
    targets = pd.read_csv(target_path)
    banned_columns = detect_banned_columns(features.columns.tolist())
    cleaned_features = features.drop(columns=banned_columns, errors="ignore").copy()
    cleaned_features = cleaned_features.apply(pd.to_numeric, errors="coerce")
    return cleaned_features, targets, banned_columns


def align_features_and_targets(features: pd.DataFrame, targets: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Align feature rows with target rows. / 对齐特征行与目标行。"""
    if len(features) != len(targets):
        raise ValueError("Feature rows and target rows do not match. Please regenerate aligned CSV files.")

    aligned_targets = targets.reset_index(drop=True).copy()
    aligned_features = features.reset_index(drop=True).copy()
    y = pd.to_numeric(aligned_targets[target_column], errors="coerce")
    keep_mask = y.notna()

    metadata_columns = [column for column in ["SEQN", "RIDAGEYR", "age_group", "H_v1", "H_v2"] if column in aligned_targets.columns]
    metadata = aligned_targets.loc[keep_mask, metadata_columns].reset_index(drop=True)
    X = aligned_features.loc[keep_mask].reset_index(drop=True)
    y = y.loc[keep_mask].reset_index(drop=True)
    return X, y, metadata


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute regression metrics. / 计算回归评估指标。"""
    residual = y_true - y_pred
    return {
        "rmse": compute_rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": compute_spearman(y_true, y_pred),
        "mean_residual": float(np.mean(residual)),
        "std_residual": float(np.std(residual)),
        "prediction_min": float(np.min(y_pred)),
        "prediction_max": float(np.max(y_pred)),
    }


def make_failure_row(feature_set: str, model_name: str, model_label: str, model_family: str, failure_reason: str, n_train: int, n_test: int) -> dict[str, object]:
    """Create a standardized failure row. / 创建标准化失败记录。"""
    row = {column: np.nan for column in LEADERBOARD_COLUMNS}
    row.update(
        {
            "feature_set": feature_set,
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


def safe_float_text(value: object, digits: int = 4) -> str:
    """Format numeric values safely for markdown. / 为 Markdown 安全格式化数值。"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def update_aggregate_outputs(output_dir: Path) -> None:
    """Update the combined table and markdown report. / 更新汇总总表与 Markdown 报告。"""
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    combined_frames: list[pd.DataFrame] = []
    metadata_by_feature_set: dict[str, dict[str, object]] = {}
    for feature_set in FEATURE_FILES:
        leaderboard_path = output_dir / "hv2_training" / feature_set / "leaderboard.csv"
        if leaderboard_path.exists():
            combined_frames.append(pd.read_csv(leaderboard_path))
        metadata_path = output_dir / "hv2_training" / feature_set / "training_metadata.json"
        if metadata_path.exists():
            metadata_by_feature_set[feature_set] = json.loads(metadata_path.read_text(encoding="utf-8"))

    combined = pd.concat(combined_frames, ignore_index=True) if combined_frames else pd.DataFrame(columns=LEADERBOARD_COLUMNS)
    if not combined.empty:
        combined = combined[LEADERBOARD_COLUMNS]
    combined.to_csv(tables_dir / "hv2_model_comparison.csv", index=False, encoding="utf-8-sig")

    package_versions = None
    package_import_errors = None
    for metadata in metadata_by_feature_set.values():
        package_versions = metadata.get("package_versions", package_versions)
        package_import_errors = metadata.get("package_import_errors", package_import_errors)

    lines = [
        "# Hv2 Regression Report",
        "",
        "This report is generated from current experiment artifacts only.",
        "该报告仅基于当前真实实验产物生成，不包含任何伪造结果。",
        "",
        "It does not generate thesis Chapter 4 text.",
        "该报告不是论文第4章正文。",
        "",
        "## Model Pools",
        "",
        "- Ensemble main models: `RandomForestRegressor`, `GradientBoostingRegressor`, `XGBRegressor`, `LGBMRegressor`.",
        "- Baseline models: `Ridge`, `ElasticNet`.",
        "",
        "## Runtime Versions",
        "",
    ]
    if package_versions:
        lines.append(f"- `scikit-learn`: `{package_versions.get('scikit-learn')}`")
        lines.append(f"- `xgboost`: `{package_versions.get('xgboost')}`")
        lines.append(f"- `lightgbm`: `{package_versions.get('lightgbm')}`")
    else:
        lines.append("- Runtime package versions are not available yet.")
    if package_import_errors:
        if package_import_errors.get("xgboost"):
            lines.append(f"- XGBoost failure reason: {package_import_errors['xgboost']}")
        if package_import_errors.get("lightgbm"):
            lines.append(f"- LightGBM failure reason: {package_import_errors['lightgbm']}")
    lines.append("")

    for feature_set in FEATURE_FILES:
        lines.extend([f"## Feature Set: `{feature_set}`", ""])
        subset = combined.loc[combined["feature_set"] == feature_set].copy() if not combined.empty else pd.DataFrame()
        metadata = metadata_by_feature_set.get(feature_set)
        if subset.empty:
            lines.append("- This feature set has not been trained yet.")
            lines.append("")
            continue

        successful = subset.loc[subset["status"] == "success"].copy()
        selected = subset.loc[subset["selected_as_best_model_artifact"] == True].copy()  # noqa: E712
        if not selected.empty:
            best_row = selected.iloc[0]
            lines.append(
                "- Best ensemble model artifact: "
                f"`{best_row['model_label']}` with CV RMSE {safe_float_text(best_row['cv_rmse_mean'])}, "
                f"holdout RMSE {safe_float_text(best_row['holdout_rmse'])}, "
                f"holdout Spearman {safe_float_text(best_row['holdout_spearman'])}."
            )
        else:
            lines.append("- No successful ensemble model artifact is available yet.")

        successful_ensembles = successful.loc[successful["model_family"] == "ensemble"]
        successful_baselines = successful.loc[successful["model_family"] == "baseline"]
        lines.append(f"- Successful ensemble models: {successful_ensembles['model_label'].tolist()}")
        lines.append(f"- Successful baseline models: {successful_baselines['model_label'].tolist()}")

        failed_rows = subset.loc[subset["status"] == "failed"]
        if failed_rows.empty:
            lines.append("- Failure rows: none.")
        else:
            lines.append("- Failure rows:")
            for _, row in failed_rows.iterrows():
                lines.append(f"  - `{row['model_label']}`: {row['failure_reason']}")

        if metadata is not None:
            lines.append(f"- Leakage columns removed before fitting: {metadata.get('dropped_banned_columns', [])}")
            lines.append(f"- Train/test split: 80/20 with `random_state=42`.")
            lines.append(f"- Cross-validation: 5-fold inside the training split only.")
        lines.append("")

    if not combined.empty:
        contains_xgboost = bool((combined["model"] == "xgboost").any())
        contains_lightgbm = bool((combined["model"] == "lightgbm").any())
        lines.extend(
            [
                "## Protocol Checks",
                "",
                f"- Includes XGBoost rows: {contains_xgboost}",
                f"- Includes LightGBM rows: {contains_lightgbm}",
                "- Leakage variables were stripped before model fitting, including `SEQN`, `H_v1`, `H_v2`, `R`, `R_v2`, `H_grade`, `H_grade_quantile`, `available_risk_dimensions`, `age_group`, and all `r_*` columns.",
                "- Missing-value imputation remained inside each pipeline and was not run on the full dataset ahead of splitting.",
                "- The test set was used only for final evaluation, not for model selection inside cross-validation.",
                "",
            ]
        )

    report_path = reports_dir / "hv2_regression_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Train thesis-aligned regressors and save artifacts. / 按论文协议训练回归模型并保存产物。"""
    args = parse_args()
    run_dir = args.output_dir / "hv2_training" / args.feature_set
    run_dir.mkdir(parents=True, exist_ok=True)

    features, targets, banned_columns = load_bundle(args.data_dir, args.feature_set)
    X, y, metadata = align_features_and_targets(features, targets, target_column="H_v2")

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X,
        y,
        metadata,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    registry, import_failures, package_versions = build_model_registry()
    package_import_errors = {
        "xgboost": XGBOOST_IMPORT_ERROR or None,
        "lightgbm": LIGHTGBM_IMPORT_ERROR or None,
    }

    cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "rmse": make_scorer(compute_rmse, greater_is_better=False),
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
        "spearman": make_scorer(compute_spearman, greater_is_better=True),
    }

    rows: list[dict[str, object]] = []
    holdout_predictions: dict[str, pd.DataFrame] = {}

    for model_spec in MODEL_CATALOG:
        model_name = model_spec["model"]
        model_label = model_spec["model_label"]
        model_family = model_spec["model_family"]

        if model_name in import_failures:
            failure_reason = import_failures[model_name]
            rows.append(
                make_failure_row(
                    feature_set=args.feature_set,
                    model_name=model_name,
                    model_label=model_label,
                    model_family=model_family,
                    failure_reason=failure_reason,
                    n_train=len(X_train),
                    n_test=len(X_test),
                )
            )
            print(f"Model failed before training: {model_label} -> {failure_reason}")
            continue

        pipeline = registry[model_name]["pipeline"]
        try:
            # Keep CV strictly inside the training split. / 严格将交叉验证限制在训练集内部。
            cv_result = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=None,
                error_score="raise",
            )
            pipeline.fit(X_train, y_train)
            holdout_pred = pd.Series(pipeline.predict(X_test), index=y_test.index, name="H_v2_pred")
            holdout_metrics = compute_metrics(y_test, holdout_pred)

            row = {
                "feature_set": args.feature_set,
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
                "holdout_rmse": holdout_metrics["rmse"],
                "holdout_mae": holdout_metrics["mae"],
                "holdout_r2": holdout_metrics["r2"],
                "holdout_spearman": holdout_metrics["spearman"],
                "mean_residual": holdout_metrics["mean_residual"],
                "std_residual": holdout_metrics["std_residual"],
                "prediction_min": holdout_metrics["prediction_min"],
                "prediction_max": holdout_metrics["prediction_max"],
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
            }
            rows.append(row)

            prediction_frame = meta_test.reset_index(drop=True).copy()
            prediction_frame["H_v2_true"] = y_test.reset_index(drop=True)
            prediction_frame["H_v2_pred"] = holdout_pred.reset_index(drop=True)
            prediction_frame["abs_error"] = (prediction_frame["H_v2_true"] - prediction_frame["H_v2_pred"]).abs()
            prediction_frame["residual"] = prediction_frame["H_v2_true"] - prediction_frame["H_v2_pred"]
            holdout_predictions[model_name] = prediction_frame
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            failure_reason = f"{type(exc).__name__}: {exc}"
            rows.append(
                make_failure_row(
                    feature_set=args.feature_set,
                    model_name=model_name,
                    model_label=model_label,
                    model_family=model_family,
                    failure_reason=failure_reason,
                    n_train=len(X_train),
                    n_test=len(X_test),
                )
            )
            print(f"Model failed during training: {model_label} -> {failure_reason}")

    leaderboard = pd.DataFrame(rows)[LEADERBOARD_COLUMNS]
    successful_ensembles = leaderboard.loc[
        (leaderboard["status"] == "success") & (leaderboard["model_family"] == "ensemble")
    ].sort_values("cv_rmse_mean")

    selected_model_name = None
    selected_model_family = None
    if not successful_ensembles.empty:
        selected_model_name = str(successful_ensembles.iloc[0]["model"])
        selected_model_family = str(successful_ensembles.iloc[0]["model_family"])
        leaderboard.loc[leaderboard["model"] == selected_model_name, "selected_as_best_model_artifact"] = True

        fitted_full_pipeline = clone(registry[selected_model_name]["pipeline"])
        fitted_full_pipeline.fit(X, y)
        joblib.dump(fitted_full_pipeline, run_dir / "best_model.joblib")
        holdout_predictions[selected_model_name].to_csv(
            run_dir / "best_model_holdout_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        print("No successful ensemble model is available for best_model.joblib.")

    sort_rank = {"success": 0, "failed": 1}
    leaderboard["_sort_status"] = leaderboard["status"].map(sort_rank).fillna(9)
    leaderboard = leaderboard.sort_values(["_sort_status", "cv_rmse_mean", "model_family", "model"]).drop(columns=["_sort_status"])
    leaderboard.to_csv(run_dir / "leaderboard.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({"feature": X.columns.tolist()}).to_csv(run_dir / "feature_columns.csv", index=False, encoding="utf-8-sig")

    metadata_payload = {
        "feature_set": args.feature_set,
        "target_column": "H_v2",
        "selected_model_name": selected_model_name,
        "selected_model_family": selected_model_family,
        "dropped_banned_columns": banned_columns,
        "random_state": RANDOM_STATE,
        "train_test_split": "80/20",
        "cv_strategy": f"{args.n_splits}-fold on training split only",
        "n_rows_after_target_filter": int(len(X)),
        "n_features": int(X.shape[1]),
        "sklearn_version": sklearn.__version__,
        "package_versions": package_versions,
        "package_import_errors": package_import_errors,
        "model_statuses": leaderboard[["model", "model_family", "status", "failure_reason"]].to_dict(orient="records"),
    }
    (run_dir / "training_metadata.json").write_text(json.dumps(metadata_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    update_aggregate_outputs(args.output_dir)

    if selected_model_name is None:
        raise RuntimeError(
            "中文：没有任何主集成模型训练成功，因此不会让 baseline 替代主模型池。 "
            "English: No main ensemble model trained successfully, so baselines are not allowed to replace the main model pool. "
            f"See {args.output_dir / 'reports' / 'hv2_regression_report.md'} for failure reasons."
        )

    print(leaderboard.to_string(index=False))
    print(f"Selected best ensemble model artifact: {selected_model_name}")
    print(f"scikit-learn version: {sklearn.__version__}")
    print(f"xgboost version: {package_versions.get('xgboost')}")
    print(f"lightgbm version: {package_versions.get('lightgbm')}")
    print(f"Wrote {run_dir / 'leaderboard.csv'}")
    print(f"Wrote {run_dir / 'best_model.joblib'}")
    print(f"Wrote {run_dir / 'best_model_holdout_predictions.csv'}")
    print(f"Wrote {args.output_dir / 'tables' / 'hv2_model_comparison.csv'}")
    print(f"Wrote {args.output_dir / 'reports' / 'hv2_regression_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
