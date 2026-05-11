"""Assess Hv2 regression stability across age groups.
评估 Hv2 回归在不同年龄组中的稳定性。
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
    parser = argparse.ArgumentParser(description="Run age-group stability checks for H_v2 models.")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_FILES), default="full", help="Feature set to evaluate.")
    parser.add_argument(
        "--model",
        choices=["ridge", "elastic_net", "random_forest", "gradient_boosting"],
        default="random_forest",
        help="Model family for subgroup evaluation.",
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


def derive_age_group(age_series: pd.Series) -> pd.Series:
    """Derive age groups if they are missing. / 在缺少年龄组时重新派生年龄组。"""
    age = pd.to_numeric(age_series, errors="coerce")
    groups = pd.Series("missing", index=age.index, dtype="object")
    groups[(age >= 18) & (age < 35)] = "18-35"
    groups[(age >= 35) & (age <= 50)] = "35-50"
    groups[age > 50] = ">50"
    groups[age.isna()] = "missing"
    return groups


def compute_group_metrics(frame: pd.DataFrame) -> dict[str, float | int | str]:
    """Compute regression metrics for one subgroup. / 为单个子组计算回归指标。"""
    y_true = frame["H_v2_true"]
    y_pred = frame["H_v2_pred"]
    return {
        "age_group": str(frame["age_group"].iloc[0]),
        "n": int(len(frame)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(frame) > 1 else float("nan"),
        "mean_error": float((y_pred - y_true).mean()),
    }


def main() -> int:
    """Run age-group stability analysis. / 运行年龄组稳定性分析。"""
    args = parse_args()
    output_dir = args.output_dir / "age_group_stability"
    output_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.data_dir / FEATURE_FILES[args.feature_set])
    targets = pd.read_csv(args.data_dir / TARGET_FILE)
    banned_columns = detect_banned_columns(features.columns.tolist())
    features = features.drop(columns=banned_columns, errors="ignore").apply(pd.to_numeric, errors="coerce")

    if len(features) != len(targets):
        raise ValueError("Feature rows and target rows do not match. Please regenerate aligned CSV files.")

    y = pd.to_numeric(targets["H_v2"], errors="coerce")
    keep_mask = y.notna()
    X = features.loc[keep_mask].reset_index(drop=True)
    y = y.loc[keep_mask].reset_index(drop=True)
    metadata = targets.loc[keep_mask].reset_index(drop=True).copy()

    if "age_group" not in metadata.columns:
        metadata["age_group"] = derive_age_group(metadata["RIDAGEYR"])
    else:
        metadata["age_group"] = metadata["age_group"].fillna("missing")

    model = build_model_registry()[args.model]
    cv = KFold(n_splits=args.n_splits, shuffle=True, random_state=RANDOM_STATE)

    # Use out-of-fold predictions to avoid optimistic subgroup metrics. / 使用折外预测避免过于乐观的分组指标。
    predictions = cross_val_predict(model, X, y, cv=cv, n_jobs=None)
    prediction_frame = metadata[[column for column in ["SEQN", "RIDAGEYR", "age_group"] if column in metadata.columns]].copy()
    prediction_frame["H_v2_true"] = y
    prediction_frame["H_v2_pred"] = predictions
    prediction_frame["abs_error"] = (prediction_frame["H_v2_true"] - prediction_frame["H_v2_pred"]).abs()

    rows = [
        {
            "age_group": "overall",
            "n": int(len(prediction_frame)),
            "rmse": float(mean_squared_error(prediction_frame["H_v2_true"], prediction_frame["H_v2_pred"], squared=False)),
            "mae": float(mean_absolute_error(prediction_frame["H_v2_true"], prediction_frame["H_v2_pred"])),
            "r2": float(r2_score(prediction_frame["H_v2_true"], prediction_frame["H_v2_pred"])),
            "mean_error": float((prediction_frame["H_v2_pred"] - prediction_frame["H_v2_true"]).mean()),
        }
    ]
    rows.extend(compute_group_metrics(group) for _, group in prediction_frame.groupby("age_group", dropna=False))
    summary = pd.DataFrame(rows)

    stem = f"{args.feature_set}_{args.model}"
    prediction_frame.to_csv(output_dir / f"age_group_predictions_{stem}.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / f"age_group_stability_{stem}.csv", index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print(f"Dropped banned columns: {banned_columns}")
    print(f"Wrote {output_dir / f'age_group_predictions_{stem}.csv'}")
    print(f"Wrote {output_dir / f'age_group_stability_{stem}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
