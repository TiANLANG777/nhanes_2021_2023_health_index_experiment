"""Colab data integrity and leakage checks for NHANES Hv2 experiments.
NHANES Hv2 Colab 实验的数据完整性与泄漏检查脚本。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIR = Path("/content/drive/MyDrive/nhanes_2021_2023_health_index_experiment/data")
DEFAULT_OUTPUT_DIR = Path("/content/drive/MyDrive/nhanes_2021_2023_health_index_experiment/outputs")
REQUIRED_FILES = {
    "full_features": "adult_full_feature_set_v2.csv",
    "reduced_features": "adult_reduced_feature_set_v2.csv",
    "targets": "adult_targets_v2.csv",
}
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. / 解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run NHANES Hv2 data checks for Colab experiments.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing experiment CSV files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated check reports.")
    return parser.parse_args()


def detect_banned_columns(columns: list[str]) -> list[str]:
    """Return strictly forbidden feature columns. / 返回严格禁止的特征列。"""
    banned = [column for column in columns if column in STRICT_BANNED_COLUMNS or column.startswith(STRICT_BANNED_PREFIXES)]
    return sorted(set(banned))


def load_required_frames(data_dir: Path) -> tuple[dict[str, Path], dict[str, pd.DataFrame]]:
    """Load all required CSV files. / 载入全部必需 CSV 文件。"""
    paths = {name: data_dir / filename for name, filename in REQUIRED_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")

    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    return paths, frames


def summarise_feature_frame(name: str, frame: pd.DataFrame, target_columns: set[str]) -> dict[str, object]:
    """Summarise feature-table integrity and leakage. / 汇总特征表完整性与泄漏情况。"""
    banned_columns = detect_banned_columns(frame.columns.tolist())
    overlap_with_targets = sorted(set(frame.columns) & target_columns)
    return {
        "table_name": name,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "duplicate_rows": int(frame.duplicated().sum()),
        "banned_columns": banned_columns,
        "overlap_with_targets": overlap_with_targets,
        "missing_rate_mean": float(frame.isna().mean().mean()),
    }


def summarise_targets(frame: pd.DataFrame) -> dict[str, object]:
    """Summarise target-table integrity. / 汇总目标表完整性。"""
    summary: dict[str, object] = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "duplicate_seqn": 0,
        "age_group_counts": {},
    }

    if "SEQN" in frame.columns:
        summary["duplicate_seqn"] = int(frame["SEQN"].duplicated().sum())

    for target_column in ["H_v1", "H_v2"]:
        if target_column in frame.columns:
            series = pd.to_numeric(frame[target_column], errors="coerce")
            summary[target_column] = {
                "non_missing": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "min": None if series.dropna().empty else float(series.min()),
                "max": None if series.dropna().empty else float(series.max()),
                "mean": None if series.dropna().empty else float(series.mean()),
                "in_range_0_100": bool(series.dropna().between(0, 100).all()),
            }

    if "age_group" in frame.columns:
        counts = frame["age_group"].fillna("missing").value_counts(dropna=False)
        summary["age_group_counts"] = {str(index): int(value) for index, value in counts.items()}

    return summary


def build_markdown(summary: dict[str, object]) -> str:
    """Build a compact markdown report. / 生成精简的 Markdown 报告。"""
    lines = [
        "# NHANES Hv2 Data Check",
        "",
        "This report documents data integrity and leakage checks only.",
        "该报告仅记录数据完整性与泄漏检查。",
        "",
        "## Input Files",
        "",
    ]

    for name, path in summary["input_paths"].items():
        lines.append(f"- `{name}`: `{path}`")

    lines.extend(["", "## Feature Tables", ""])
    for feature_summary in summary["feature_checks"]:
        lines.extend(
            [
                f"### {feature_summary['table_name']}",
                "",
                f"- Rows: {feature_summary['rows']}",
                f"- Columns: {feature_summary['columns']}",
                f"- Duplicate rows: {feature_summary['duplicate_rows']}",
                f"- Mean missing rate: {feature_summary['missing_rate_mean']:.6f}",
                f"- Banned columns found: {feature_summary['banned_columns']}",
                f"- Overlap with target columns: {feature_summary['overlap_with_targets']}",
                "",
            ]
        )

    target_summary = summary["target_checks"]
    lines.extend(["## Targets", ""])
    lines.append(f"- Rows: {target_summary['rows']}")
    lines.append(f"- Columns: {target_summary['columns']}")
    lines.append(f"- Duplicate `SEQN`: {target_summary['duplicate_seqn']}")
    lines.append(f"- `age_group` counts: {target_summary['age_group_counts']}")

    for target_name in ["H_v1", "H_v2"]:
        if target_name in target_summary:
            stats = target_summary[target_name]
            lines.extend(
                [
                    f"- `{target_name}` non-missing: {stats['non_missing']}",
                    f"- `{target_name}` missing: {stats['missing']}",
                    f"- `{target_name}` min/max: {stats['min']} / {stats['max']}",
                    f"- `{target_name}` mean: {stats['mean']}",
                    f"- `{target_name}` within 0-100: {stats['in_range_0_100']}",
                ]
            )

    lines.extend(
        [
            "",
            "## Scope Boundary",
            "",
            "- No model is trained in this step.",
            "- 该步骤不训练任何模型。",
            "- No experimental result is fabricated in this step.",
            "- 该步骤不会伪造任何实验结果。",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """Run data checks and write reports. / 运行数据检查并写出报告。"""
    args = parse_args()
    output_dir = args.output_dir / "data_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths, frames = load_required_frames(args.data_dir)
    targets = frames["targets"]
    target_columns = set(targets.columns)

    feature_checks = [
        summarise_feature_frame("adult_full_feature_set_v2.csv", frames["full_features"], target_columns),
        summarise_feature_frame("adult_reduced_feature_set_v2.csv", frames["reduced_features"], target_columns),
    ]
    target_checks = summarise_targets(targets)

    summary = {
        "input_paths": {name: str(path) for name, path in input_paths.items()},
        "feature_checks": feature_checks,
        "target_checks": target_checks,
    }

    (output_dir / "data_check_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "data_check_report.md").write_text(build_markdown(summary), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {(output_dir / 'data_check_summary.json')}")
    print(f"Wrote {(output_dir / 'data_check_report.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
