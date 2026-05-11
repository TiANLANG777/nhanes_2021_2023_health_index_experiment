"""Compute sampled SHAP outputs from saved Hv2 model artifacts.
基于已保存的 Hv2 模型产物计算抽样 SHAP 结果。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from experiment_utils import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    FEATURE_FILES,
    MODEL_CHOICES,
    MODEL_LOOKUP,
    RANDOM_STATE,
    prepare_regression_inputs,
    safe_float_text,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. / 解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run sampled SHAP analysis from saved H_v2 model artifacts.")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_FILES), default="full", help="Feature set to explain.")
    parser.add_argument("--model", choices=[*MODEL_CHOICES, "best"], default="best", help="Model artifact to explain.")
    parser.add_argument("--sample-size", type=int, default=500, help="Maximum number of rows for SHAP calculation.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top features to report.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing input CSV files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated outputs.")
    return parser.parse_args()


def resolve_model_name(args: argparse.Namespace, training_metadata: dict[str, object]) -> str:
    """Resolve the model artifact to explain. / 解析要解释的模型产物名称。"""
    if args.model == "best":
        selected_model_name = training_metadata.get("selected_model_name")
        if not selected_model_name:
            raise ValueError("No selected_model_name found in training_metadata.json. Run Hv2 training first.")
        return str(selected_model_name)
    return args.model


def main() -> int:
    """Load one saved Hv2 model and export sampled SHAP outputs. / 载入已保存的 Hv2 模型并导出抽样 SHAP 输出。"""
    args = parse_args()
    tables_dir = args.output_dir / "tables"
    figures_dir = args.output_dir / "figures"
    reports_dir = args.output_dir / "reports"
    detailed_dir = args.output_dir / "shap"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    detailed_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = args.output_dir / "hv2_training" / args.feature_set / "training_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError("Missing training_metadata.json. Run scripts/02_train_hv2_regression.py first.")
    training_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_name = resolve_model_name(args, training_metadata)
    model_family = MODEL_LOOKUP.get(model_name, {}).get("model_family")
    if model_family != "ensemble":
        raise ValueError("SHAP analysis is only supported for ensemble models. Baseline linear models are excluded here.")

    model_path = args.output_dir / "hv2_training" / args.feature_set / "model_results" / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing saved model artifact: {model_path}")

    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Loaded model artifact is not a scikit-learn Pipeline.")

    X, y, metadata, banned_columns = prepare_regression_inputs(args.data_dir, args.feature_set, target_column="H_v2")
    sample_size = min(int(args.sample_size), len(X))
    sample_index = X.sample(n=sample_size, random_state=RANDOM_STATE).index
    X_sample = X.loc[sample_index].reset_index(drop=True)
    metadata_sample = metadata.loc[sample_index].reset_index(drop=True)

    imputer = pipeline.named_steps["imputer"]
    model = pipeline.named_steps["model"]
    X_sample_imputed = imputer.transform(X_sample)

    report_lines = [
        "# SHAP Report",
        "",
        "This report explains saved model behavior only.",
        "该报告只解释已保存模型的行为，不重新训练模型。",
        "",
        "SHAP explains model behavior, not causal effects.",
        "SHAP 解释的是模型行为，而不是因果效应。",
        "",
        f"- Feature set: `{args.feature_set}`",
        f"- Model: `{model_name}`",
        f"- Sample size used for SHAP: `{sample_size}`",
        f"- Dropped banned columns before fitting: {banned_columns}",
        "",
    ]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample_imputed)
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "SHAP TreeExplainer failed. Please check the saved model object and package versions. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    shap_matrix = np.asarray(shap_values)
    if shap_matrix.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape: {shap_matrix.shape}")

    importance = pd.DataFrame(
        {
            "feature": X.columns.tolist(),
            "mean_abs_shap": np.abs(shap_matrix).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance.insert(0, "feature_set", args.feature_set)
    importance.insert(1, "model", model_name)
    importance.to_csv(tables_dir / "shap_feature_importance.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(detailed_dir / f"shap_feature_importance_{args.feature_set}_{model_name}.csv", index=False, encoding="utf-8-sig")

    shap_value_frame = metadata_sample.copy()
    for column_index, column_name in enumerate(X.columns):
        shap_value_frame[f"shap__{column_name}"] = shap_matrix[:, column_index]
    shap_value_frame.to_csv(detailed_dir / f"shap_values_sample_{args.feature_set}_{model_name}.csv", index=False, encoding="utf-8-sig")

    top_importance = importance.head(args.top_n).iloc[::-1]
    plt.figure(figsize=(10, max(6, len(top_importance) * 0.35)))
    plt.barh(top_importance["feature"], top_importance["mean_abs_shap"])
    plt.xlabel("mean(|SHAP value|)")
    plt.ylabel("feature")
    plt.title(f"SHAP importance ({args.feature_set}, {model_name})")
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_bar.png", dpi=300)
    plt.close()
    report_lines.append("- Wrote `outputs/figures/shap_bar.png`.")

    summary_error = None
    try:
        shap.summary_plot(shap_matrix, X_sample, show=False, max_display=min(args.top_n, X_sample.shape[1]))
        plt.tight_layout()
        plt.savefig(figures_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()
        report_lines.append("- Wrote `outputs/figures/shap_summary.png`.")
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        plt.close("all")
        summary_error = f"{type(exc).__name__}: {exc}"
        report_lines.append(f"- SHAP summary plot failed: {summary_error}")

    dependence_errors: list[str] = []
    for feature_name in importance["feature"].head(min(3, len(importance))):
        try:
            shap.dependence_plot(feature_name, shap_matrix, X_sample, interaction_index=None, show=False)
            plt.tight_layout()
            plt.savefig(figures_dir / f"shap_dependence_{feature_name}.png", dpi=300, bbox_inches="tight")
            plt.close()
            report_lines.append(f"- Wrote `outputs/figures/shap_dependence_{feature_name}.png`.")
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            plt.close("all")
            dependence_errors.append(f"{feature_name}: {type(exc).__name__}: {exc}")

    if dependence_errors:
        report_lines.append("- Some SHAP dependence plots failed:")
        for item in dependence_errors:
            report_lines.append(f"  - {item}")

    report_lines.extend(["", "## Top Features", ""])
    for _, row in importance.head(min(args.top_n, 10)).iterrows():
        report_lines.append(f"- `{row['feature']}`: {safe_float_text(row['mean_abs_shap'], digits=6)}")

    metadata_payload = {
        "feature_set": args.feature_set,
        "model": model_name,
        "sample_size": sample_size,
        "top_n": int(args.top_n),
        "summary_plot_error": summary_error,
        "dependence_plot_errors": dependence_errors,
        "interpretation_scope": "association_only_not_causal",
    }
    (detailed_dir / "shap_metadata.json").write_text(json.dumps(metadata_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (reports_dir / "shap_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(importance.head(args.top_n).to_string(index=False))
    print(f"Wrote {tables_dir / 'shap_feature_importance.csv'}")
    print(f"Wrote {figures_dir / 'shap_bar.png'}")
    print(f"Wrote {reports_dir / 'shap_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
