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

## 建议的 Colab 使用方式 / Suggested Colab Workflow

1. Mount Google Drive.
2. Clone this repository to `/content/nhanes_2021_2023_health_index_experiment`.
3. Install `requirements_colab.txt`.
4. Run the notebook from top to bottom.

如果仓库被克隆到其他目录，请同步修改 notebook 中的 `PROJECT_ROOT`。  
If the repository is cloned elsewhere, update `PROJECT_ROOT` in the notebook accordingly.

## 当前范围边界 / Current Scope Boundary

- 只搭建代码，不承诺任何实验结果。 / Code scaffolding only, no promised experiment outcomes.
- 不自动生成论文正文。 / No automatic thesis prose generation.
- 不直接连接 Google Colab 页面。 / No direct interaction with the Google Colab page.
