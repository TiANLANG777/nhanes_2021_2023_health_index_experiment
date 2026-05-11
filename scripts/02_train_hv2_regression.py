"""Train baseline regressors for the NHANES Hv2 experiment.
训练 NHANES Hv2 实验的基线回归模型。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. / 解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Train baseline regressors for H_v2.")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_FILES), default="full", help="Feature set to use.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing input CSV files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for experiment outputs.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds.")
    return parser.parse_args()


def detect_banned_columns(columns: list[str]) -> list[str]:
    """Detect forbidden input columns. / 检测禁止进入模型的输入列。"""
    banned = [column for column in columns if column in STRICT_BANNED_COLUMNS or column.startswith(STRICT_BANNED_PREFIXES)]
    return sorted(set(banned))


def build_model_registry() -> dict[str, Pipeline]:
    """Create baseline model pipelines. / 创建基线模型流水线。"""
    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    linear_imputer = SimpleImputer(strategy="constant", fill_value=0.0)

    return {
        "ridge": Pipeline(
            steps=[
                ("imputer", linear_imputer),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
            ]
        ),
        "elastic_net": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
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
        "gradient_boosting": Pipeline(
            steps=[
                ("imputer", imputer),
                ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
            ]
        ),
    }


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
    return {
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> int:
    """Train baseline regressors and save artifacts. / 训练基线回归器并保存实验产物。"""
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

    cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    rows: list[dict[str, object]] = []
    best_model_name = ""
    best_pipeline: Pipeline | None = None
    best_predictions: pd.DataFrame | None = None
    best_rank_value = float("inf")

    for model_name, pipeline in build_model_registry().items():
        # Evaluate each model with identical splits. / 用相同划分评估每个模型。
        cv_result = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=None)
        pipeline.fit(X_train, y_train)
        holdout_pred = pd.Series(pipeline.predict(X_test), index=y_test.index, name="y_pred")
        holdout_metrics = compute_metrics(y_test, holdout_pred)

        row = {
            "model": model_name,
            "cv_rmse_mean": float(-cv_result["test_rmse"].mean()),
            "cv_rmse_std": float(cv_result["test_rmse"].std()),
            "cv_mae_mean": float(-cv_result["test_mae"].mean()),
            "cv_r2_mean": float(cv_result["test_r2"].mean()),
            "holdout_rmse": holdout_metrics["rmse"],
            "holdout_mae": holdout_metrics["mae"],
            "holdout_r2": holdout_metrics["r2"],
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        }
        rows.append(row)

        if row["cv_rmse_mean"] < best_rank_value:
            best_rank_value = row["cv_rmse_mean"]
            best_model_name = model_name
            best_pipeline = pipeline
            best_predictions = meta_test.copy()
            best_predictions["H_v2_true"] = y_test.reset_index(drop=True)
            best_predictions["H_v2_pred"] = holdout_pred.reset_index(drop=True)
            best_predictions["abs_error"] = (best_predictions["H_v2_true"] - best_predictions["H_v2_pred"]).abs()

    leaderboard = pd.DataFrame(rows).sort_values("cv_rmse_mean").reset_index(drop=True)
    leaderboard.to_csv(run_dir / "leaderboard.csv", index=False, encoding="utf-8-sig")

    if best_pipeline is None or best_predictions is None:
        raise RuntimeError("No model was trained.")

    best_pipeline.fit(X, y)
    joblib.dump(best_pipeline, run_dir / "best_model.joblib")
    best_predictions.to_csv(run_dir / "best_model_holdout_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"feature": X.columns.tolist()}).to_csv(run_dir / "feature_columns.csv", index=False, encoding="utf-8-sig")

    metadata_payload = {
        "feature_set": args.feature_set,
        "target_column": "H_v2",
        "best_model": best_model_name,
        "dropped_banned_columns": banned_columns,
        "random_state": RANDOM_STATE,
        "n_rows_after_target_filter": int(len(X)),
        "n_features": int(X.shape[1]),
    }
    (run_dir / "training_metadata.json").write_text(json.dumps(metadata_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(leaderboard.to_string(index=False))
    print(f"Best model: {best_model_name}")
    print(f"Wrote {run_dir / 'leaderboard.csv'}")
    print(f"Wrote {run_dir / 'best_model.joblib'}")
    print(f"Wrote {run_dir / 'best_model_holdout_predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
