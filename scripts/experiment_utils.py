"""Shared utilities for NHANES experiment scripts.
NHANES 实验脚本的共享工具函数。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
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
MODEL_LOOKUP = {item["model"]: item for item in MODEL_CATALOG}
MODEL_CHOICES = [item["model"] for item in MODEL_CATALOG]
REBUILD_MODEL_SENTINEL = "rebuild_leaderboard"


def detect_banned_columns(columns: list[str]) -> list[str]:
    """Detect forbidden input columns. / 检测禁止进入模型的输入列。"""
    banned = [column for column in columns if column in STRICT_BANNED_COLUMNS or column.startswith(STRICT_BANNED_PREFIXES)]
    return sorted(set(banned))


def derive_age_group(age_series: pd.Series) -> pd.Series:
    """Derive age groups from age in years. / 根据年龄派生年龄组。"""
    age = pd.to_numeric(age_series, errors="coerce")
    groups = pd.Series("missing", index=age.index, dtype="object")
    groups[(age >= 18) & (age < 35)] = "18-34"
    groups[(age >= 35) & (age < 50)] = "35-49"
    groups[(age >= 50) & (age < 65)] = "50-64"
    groups[age >= 65] = "65+"
    groups[age.isna()] = "missing"
    return groups


def compute_rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """中文：计算 RMSE，避免依赖 sklearn 的 squared=False 参数。
    English: Compute RMSE without relying on sklearn's squared=False argument.
    """
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def compute_spearman(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    """中文：计算 Spearman 相关系数，用于排序一致性评估。
    English: Compute Spearman correlation for rank-consistency evaluation.
    """
    result = spearmanr(y_true, y_pred)
    correlation = getattr(result, "correlation", result[0] if isinstance(result, tuple) else result)
    if correlation is None or np.isnan(correlation):
        return float("nan")
    return float(correlation)


def safe_float_text(value: object, digits: int = 4) -> str:
    """Format numeric values safely for markdown. / 为 Markdown 安全格式化数值。"""
    if value is None:
        return "NA"
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(value_float):
        return "NA"
    return f"{value_float:.{digits}f}"


def round_numeric_table(frame: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    """Round only numeric columns. / 仅对数值列进行四舍五入。"""
    result = frame.copy()
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    result.loc[:, numeric_columns] = result.loc[:, numeric_columns].round(digits)
    return result


def load_raw_feature_and_target_frames(data_dir: Path, feature_set: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load raw feature and target frames. / 载入原始特征表与目标表。"""
    feature_path = data_dir / FEATURE_FILES[feature_set]
    target_path = data_dir / TARGET_FILE
    if not feature_path.exists() or not target_path.exists():
        raise FileNotFoundError(f"Missing input files: {feature_path}, {target_path}")

    feature_frame = pd.read_csv(feature_path)
    target_frame = pd.read_csv(target_path)
    banned_columns = detect_banned_columns(feature_frame.columns.tolist())
    return feature_frame, target_frame, banned_columns


def prepare_regression_inputs(data_dir: Path, feature_set: str, target_column: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    """Prepare aligned regression inputs. / 准备对齐后的回归输入。"""
    raw_features, raw_targets, banned_columns = load_raw_feature_and_target_frames(data_dir, feature_set)
    if len(raw_features) != len(raw_targets):
        raise ValueError("Feature rows and target rows do not match. Please regenerate aligned CSV files.")

    cleaned_features = raw_features.drop(columns=banned_columns, errors="ignore").copy()
    cleaned_features = cleaned_features.apply(pd.to_numeric, errors="coerce")

    targets = raw_targets.reset_index(drop=True).copy()
    targets["row_id"] = np.arange(len(targets))
    if "age_group" not in targets.columns and "RIDAGEYR" in targets.columns:
        targets["age_group"] = derive_age_group(targets["RIDAGEYR"])
    elif "age_group" in targets.columns:
        targets["age_group"] = targets["age_group"].fillna("missing")

    y = pd.to_numeric(targets[target_column], errors="coerce")
    keep_mask = y.notna()
    metadata_columns = [column for column in ["row_id", "SEQN", "RIDAGEYR", "age_group", "H_v1", "H_v2"] if column in targets.columns]
    metadata = targets.loc[keep_mask, metadata_columns].reset_index(drop=True)
    X = cleaned_features.loc[keep_mask].reset_index(drop=True)
    y = y.loc[keep_mask].reset_index(drop=True)
    return X, y, metadata, banned_columns


def make_prediction_frame(metadata: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series, true_column: str, pred_column: str) -> pd.DataFrame:
    """Create a standard prediction frame. / 创建标准预测结果表。"""
    prediction_frame = metadata.reset_index(drop=True).copy()
    prediction_frame[true_column] = y_true.reset_index(drop=True)
    prediction_frame[pred_column] = y_pred.reset_index(drop=True)
    prediction_frame["abs_error"] = (prediction_frame[true_column] - prediction_frame[pred_column]).abs()
    prediction_frame["residual"] = prediction_frame[true_column] - prediction_frame[pred_column]
    return prediction_frame


def get_model_result_paths(model_results_dir: Path, model_name: str) -> dict[str, Path]:
    """Get all file paths for one model result bundle. / 获取单模型结果文件路径。"""
    return {
        "metrics": model_results_dir / f"{model_name}_metrics.json",
        "predictions": model_results_dir / f"{model_name}_predictions.csv",
        "artifact": model_results_dir / f"{model_name}.joblib",
        "error": model_results_dir / f"{model_name}_error.txt",
    }


def success_artifacts_exist(paths: dict[str, Path]) -> bool:
    """Check whether a model already finished successfully. / 检查模型是否已成功完成。"""
    return paths["metrics"].exists() and paths["predictions"].exists() and paths["artifact"].exists()


def build_model_registry() -> tuple[dict[str, dict[str, object]], dict[str, str], dict[str, str | None], dict[str, str | None]]:
    """Create model pipelines, import failures, and package versions. / 创建模型流水线、导入失败信息和包版本。"""
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
                            n_estimators=150,
                            max_depth=12,
                            min_samples_leaf=5,
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
    import_failures: dict[str, str] = {}
    package_versions = {
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__ if xgboost is not None else None,
        "lightgbm": lightgbm.__version__ if lightgbm is not None else None,
    }
    package_import_errors = {
        "xgboost": XGBOOST_IMPORT_ERROR or None,
        "lightgbm": LIGHTGBM_IMPORT_ERROR or None,
    }

    if XGBRegressor is None:
        import_failures["xgboost"] = (
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
                            n_estimators=300,
                            learning_rate=0.05,
                            max_depth=4,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            n_jobs=2,
                            random_state=RANDOM_STATE,
                            verbosity=0,
                        ),
                    ),
                ]
            ),
        }

    if LGBMRegressor is None:
        import_failures["lightgbm"] = (
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
                            n_estimators=300,
                            learning_rate=0.05,
                            num_leaves=31,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            n_jobs=2,
                            random_state=RANDOM_STATE,
                            verbosity=-1,
                        ),
                    ),
                ]
            ),
        }

    return registry, import_failures, package_versions, package_import_errors
