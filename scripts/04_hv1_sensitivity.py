"""Run H_v1 sensitivity analysis with the same feature inputs.
使用相同特征输入运行 H_v1 敏感性分析。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
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
    parser = argparse.ArgumentParser(description="Compare H_v2 and H_v1 with a shared regression pipeline.")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_FILES), default="full", help="Feature set to evaluate.")
    parser.add_argument(
        "--model",
        choices=["ridge", "elastic_net", "random_forest", "gradient_boosting"],
        default="random_forest",
        help="Model family for sensitivity analysis.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing input CSV files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated outputs.")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds.")
    return parser.parse_args()


def detect_banned_columns(columns: list[str]) -> list[str]:
    """Detect forbidden input columns. / 检测禁止进入模型的输入列。"""
    banned = [column for column in columns if column in STRICT_BANNED_COLUMNS or column.startswith(STRICT_BANNED_PREFIXES)]
    return sorted(set(banned))


def build_model_registry() -> dict[str, Pipeline]:
    """Create reusable model pipelines. / 创建可复用模型流水线。"""
    return {
        "ridge": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
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
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
            ]
        ),
    }


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute regression metrics. / 计算回归评估指标。"""
    return {
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> int:
    """Compare H_v2 and H_v1 under the same setup. / 在相同设置下比较 H_v2 与 H_v1。"""
    args = parse_args()
    output_dir = args.output_dir / "hv1_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.data_dir / FEATURE_FILES[args.feature_set])
    targets = pd.read_csv(args.data_dir / TARGET_FILE)
    banned_columns = detect_banned_columns(features.columns.tolist())
    features = features.drop(columns=banned_columns, errors="ignore").apply(pd.to_numeric, errors="coerce")

    if len(features) != len(targets):
        raise ValueError("Feature rows and target rows do not match. Please regenerate aligned CSV files.")

    cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=RANDOM_STATE)
    model = build_model_registry()[args.model]
    rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    for target_name in ["H_v2", "H_v1"]:
        if target_name not in targets.columns:
            continue

        y = pd.to_numeric(targets[target_name], errors="coerce")
        keep_mask = y.notna()
        if not keep_mask.any():
            continue

        X = features.loc[keep_mask].reset_index(drop=True)
        y = y.loc[keep_mask].reset_index(drop=True)
        metadata = targets.loc[keep_mask, [column for column in ["SEQN", "RIDAGEYR", "age_group"] if column in targets.columns]].reset_index(drop=True)

        # Use the same CV protocol for both targets. / 对两个目标使用同一套交叉验证方案。
        predictions = cross_val_predict(model, X, y, cv=cv, n_jobs=None)
        metrics = compute_metrics(y, pd.Series(predictions))
        rows.append(
            {
                "target": target_name,
                "model": args.model,
                "feature_set": args.feature_set,
                "n": int(len(y)),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
            }
        )

        prediction_frame = metadata.copy()
        prediction_frame["target"] = target_name
        prediction_frame["y_true"] = y
        prediction_frame["y_pred"] = predictions
        prediction_frame["abs_error"] = (prediction_frame["y_true"] - prediction_frame["y_pred"]).abs()
        prediction_rows.append(prediction_frame)

    summary = pd.DataFrame(rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()

    stem = f"{args.feature_set}_{args.model}"
    summary.to_csv(output_dir / f"hv1_sensitivity_summary_{stem}.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output_dir / f"hv1_sensitivity_predictions_{stem}.csv", index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print(f"Dropped banned columns: {banned_columns}")
    print(f"Wrote {output_dir / f'hv1_sensitivity_summary_{stem}.csv'}")
    print(f"Wrote {output_dir / f'hv1_sensitivity_predictions_{stem}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
