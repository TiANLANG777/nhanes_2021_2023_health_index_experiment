# NHANES 2021–2023 综合健康指数 Colab 实验项目 / NHANES 2021–2023 Composite Health Index Colab Experiment

这个仓库只搭建实验代码，不包含伪造结果，不生成论文第4章正文。  
This repository only provides experiment code scaffolding. It does not fabricate results and does not generate thesis Chapter 4 text.

## 固定数据路径 / Fixed Data Path

```text
/content/drive/MyDrive/nhanes_2021_2023_health_index_experiment/data
```

必需输入文件 / Required input files:

- `adult_full_feature_set_v2.csv`
- `adult_reduced_feature_set_v2.csv`
- `adult_targets_v2.csv`

## 固定输出路径 / Fixed Output Path

```text
/content/drive/MyDrive/nhanes_2021_2023_health_index_experiment/outputs
```

## 项目结构 / Project Structure

- `notebooks/NHANES_Hv2_experiment_colab.ipynb`: Colab notebook entry.
- `scripts/01_data_check.py`: data integrity and leakage checks.
- `scripts/02_train_hv2_regression.py`: `H_v2` regression experiments.
- `scripts/03_age_group_stability.py`: subgroup stability checks by `age_group`.
- `scripts/04_hv1_sensitivity.py`: sensitivity analysis with `H_v1`.
- `scripts/05_shap_analysis.py`: SHAP analysis for the fitted tree model.
- `scripts/06_generate_experiment_summary.py`: artifact-oriented summary generator.
- `requirements_colab.txt`: Colab dependencies.

## 泄漏控制规则 / Leakage Control Rules

以下变量严格禁止进入模型输入。  
The following variables are strictly forbidden from entering model inputs.

- `SEQN`
- `H_v1`
- `H_v2`
- `R`
- `R_v2`
- `H_grade`
- `H_grade_quantile`
- `available_risk_dimensions`
- `age_group`
- all columns starting with `r_`

脚本会在训练前再次检测并删除这些列，避免误用。  
The scripts check and drop these columns again before training to prevent accidental leakage.

## sklearn 兼容性 / sklearn Compatibility

为了兼容不同版本的 Colab 运行环境，RMSE 计算不依赖 `mean_squared_error(..., squared=False)`。  
To stay compatible with different Colab runtime versions, RMSE computation does not rely on `mean_squared_error(..., squared=False)`.

当前实际运行时的 `scikit-learn` 版本会写入：  
The actual `scikit-learn` version used at runtime is recorded in:

- `outputs/hv2_training/full/training_metadata.json`
- `outputs/hv2_training/reduced/training_metadata.json`
- `outputs/experiment_summary/experiment_summary.md`

## H_v2 训练与续跑 / H_v2 Training and Resume Workflow

`scripts/02_train_hv2_regression.py` 现在支持单模型运行、断点续跑和分模型保存。  
`scripts/02_train_hv2_regression.py` now supports single-model execution, resumable runs, and per-model artifacts.

可选模型 / Available model values:

- `random_forest`
- `gradient_boosting`
- `xgboost`
- `lightgbm`
- `ridge`
- `elastic_net`
- `all`

默认值 / Default value:

- `all`

推荐在 Colab 中按模型运行。  
It is recommended to run models one by one in Colab.

示例命令 / Example commands:

```bash
python scripts/02_train_hv2_regression.py \
  --feature-set full \
  --model random_forest \
  --data-dir /content/drive/MyDrive/nhanes_2021_2023_health_index_experiment/data \
  --output-dir /content/drive/MyDrive/nhanes_2021_2023_health_index_experiment/outputs
```

```bash
python scripts/02_train_hv2_regression.py \
  --feature-set reduced \
  --model lightgbm \
  --data-dir /content/drive/MyDrive/nhanes_2021_2023_health_index_experiment/data \
  --output-dir /content/drive/MyDrive/nhanes_2021_2023_health_index_experiment/outputs
```

如果某个模型已经成功完成，且没有传入 `--force`，脚本会自动跳过它。  
If a model already finished successfully and `--force` is not provided, the script skips it automatically.

如需强制重跑某个模型，可使用：  
To rerun a finished model deliberately, use:

```bash
python scripts/02_train_hv2_regression.py \
  --feature-set full \
  --model xgboost \
  --force \
  --data-dir /content/drive/MyDrive/nhanes_2021_2023_health_index_experiment/data \
  --output-dir /content/drive/MyDrive/nhanes_2021_2023_health_index_experiment/outputs
```

每个模型的单独产物会保存到：  
Per-model artifacts are saved to:

- `outputs/hv2_training/{feature_set}/model_results/{model_name}_metrics.json`
- `outputs/hv2_training/{feature_set}/model_results/{model_name}_predictions.csv`
- `outputs/hv2_training/{feature_set}/model_results/{model_name}.joblib`
- `outputs/hv2_training/{feature_set}/model_results/{model_name}_error.txt`

成功模型结束后会自动更新：  
After each model finishes, the script automatically updates:

- `outputs/hv2_training/{feature_set}/leaderboard.csv`
- `outputs/tables/hv2_model_comparison.csv`
- `outputs/reports/hv2_regression_report.md`

## 当前 Colab 友好设置 / Current Colab-Friendly Settings

为了减少 Colab 被中断的概率，当前实现包含这些约束：  
To reduce the chance of Colab interruption, the current implementation includes these constraints:

- `train/test split` 保持 `80/20`。 / `train/test split` stays at `80/20`.
- `random_state` 固定为 `42`。 / `random_state` is fixed at `42`.
- `5-fold cross-validation` 只在训练集内部执行。 / `5-fold cross-validation` runs only inside the training split.
- `cross_validate(..., n_jobs=1)` 避免嵌套并行。 / `cross_validate(..., n_jobs=1)` avoids nested parallelism.
- `RandomForestRegressor` 使用中等复杂度参数。 / `RandomForestRegressor` uses a moderate-complexity configuration.
- 缺失值填补只在 `Pipeline` 内进行。 / Missing-value imputation stays inside the `Pipeline`.

## 建议的 Colab 使用方式 / Suggested Colab Workflow

1. Mount Google Drive.
2. Clone this repository to `/content/nhanes_2021_2023_health_index_experiment`.
3. Install `requirements_colab.txt`.
4. Run `scripts/01_data_check.py` first.
5. Run `scripts/02_train_hv2_regression.py` one model at a time for `full` and `reduced`.
6. After main training is complete, run age-group stability, `H_v1` sensitivity, SHAP, and summary scripts.

如果仓库被克隆到其他目录，请同步修改 notebook 中的 `REPO_DIR`。  
If the repository is cloned elsewhere, update `REPO_DIR` in the notebook accordingly.

## 当前范围边界 / Current Scope Boundary

- 只搭建代码，不承诺任何实验结果。 / Code scaffolding only, no promised experiment outcomes.
- 不自动生成论文正文。 / No automatic thesis prose generation.
- 不直接连接 Google Colab 页面。 / No direct interaction with the Google Colab page.
