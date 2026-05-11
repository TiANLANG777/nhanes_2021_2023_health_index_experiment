"""Compute SHAP values for the NHANES Hv2 tree model.
为 NHANES Hv2 树模型计算 SHAP 值。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


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
    parser = argparse.ArgumentParser(description="Run SHAP analysis for the H_v2 random forest model.")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_FILES), default="full", help="Feature set to explain.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing input CSV files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated outputs.")
    parser.add_argument("--sample-size", type=int, default=500, help="Maximum number of rows for SHAP calculation.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of features to show in the summary plot.")
    return parser.parse_args()


def detect_banned_columns(columns: list[str]) -> list[str]:
    """Detect forbidden input columns. / 检测禁止进入模型的输入列。"""
    banned = [column for column in columns if column in STRICT_BANNED_COLUMNS or column.startswith(STRICT_BANNED_PREFIXES)]
    return sorted(set(banned))


def main() -> int:
    """Fit a tree model and export SHAP artifacts. / 拟合树模型并导出 SHAP 产物。"""
    args = parse_args()
    output_dir = args.output_dir / "shap"
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
    metadata = targets.loc[keep_mask, [column for column in ["SEQN", "RIDAGEYR", "age_group"] if column in targets.columns]].reset_index(drop=True)

    pipeline = Pipeline(
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
    )
    pipeline.fit(X, y)

    sample_size = min(args.sample_size, len(X))
    sample_index = X.sample(n=sample_size, random_state=RANDOM_STATE).index
    X_sample = X.loc[sample_index].reset_index(drop=True)
    metadata_sample = metadata.loc[sample_index].reset_index(drop=True)

    imputer: SimpleImputer = pipeline.named_steps["imputer"]
    model: RandomForestRegressor = pipeline.named_steps["model"]
    X_sample_imputed = imputer.transform(X_sample)

    # Explain the fitted tree model on a sampled subset. / 在抽样子集上解释已拟合的树模型。
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample_imputed)
    shap_matrix = np.asarray(shap_values)
    importance = pd.DataFrame(
        {
            "feature": X.columns.tolist(),
            "mean_abs_shap": np.abs(shap_matrix).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    shap_frame = metadata_sample.copy()
    for column_index, column_name in enumerate(X.columns):
        shap_frame[f"shap__{column_name}"] = shap_matrix[:, column_index]
    shap_frame.to_csv(output_dir / f"shap_values_sample_{args.feature_set}.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(output_dir / f"shap_importance_{args.feature_set}.csv", index=False, encoding="utf-8-sig")

    top_importance = importance.head(args.top_n).iloc[::-1]
    plt.figure(figsize=(10, max(6, args.top_n * 0.35)))
    plt.barh(top_importance["feature"], top_importance["mean_abs_shap"])
    plt.xlabel("mean(|SHAP value|)")
    plt.ylabel("feature")
    plt.title(f"NHANES Hv2 SHAP summary ({args.feature_set})")
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_importance_{args.feature_set}.png", dpi=200)
    plt.close()

    metadata_payload = {
        "feature_set": args.feature_set,
        "sample_size": int(sample_size),
        "random_state": RANDOM_STATE,
        "dropped_banned_columns": banned_columns,
        "n_features": int(X.shape[1]),
        "n_rows_with_target": int(len(X)),
    }
    (output_dir / f"shap_metadata_{args.feature_set}.json").write_text(
        json.dumps(metadata_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(importance.head(args.top_n).to_string(index=False))
    print(f"Dropped banned columns: {banned_columns}")
    print(f"Wrote {output_dir / f'shap_values_sample_{args.feature_set}.csv'}")
    print(f"Wrote {output_dir / f'shap_importance_{args.feature_set}.csv'}")
    print(f"Wrote {output_dir / f'shap_importance_{args.feature_set}.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
