"""Colab data integrity and leakage checks for NHANES experiments.
NHANES Colab 实验的数据完整性与泄漏检查脚本。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiment_utils import DEFAULT_DATA_DIR, DEFAULT_OUTPUT_DIR, FEATURE_FILES, TARGET_FILE, detect_banned_columns


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments. / 解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run NHANES data checks for Colab experiments.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing experiment CSV files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated check reports.")
    return parser.parse_args()


def summarise_feature_table(feature_set: str, feature_frame: pd.DataFrame, target_columns: set[str]) -> tuple[dict[str, object], pd.DataFrame]:
    """Summarise one feature table. / 汇总单个特征表。"""
    banned_columns = detect_banned_columns(feature_frame.columns.tolist())
    overlap_with_targets = sorted(set(feature_frame.columns) & target_columns)
    summary = {
        "feature_set": feature_set,
        "file_name": FEATURE_FILES[feature_set],
        "rows": int(len(feature_frame)),
        "columns": int(len(feature_frame.columns)),
        "duplicate_rows": int(feature_frame.duplicated().sum()),
        "mean_missing_rate": float(feature_frame.isna().mean().mean()),
        "banned_column_count": int(len(banned_columns)),
        "target_overlap_count": int(len(overlap_with_targets)),
    }
    leakage_rows = []
    for column_name in banned_columns:
        leakage_rows.append(
            {
                "feature_set": feature_set,
                "file_name": FEATURE_FILES[feature_set],
                "column_name": column_name,
                "issue_type": "banned_model_input",
            }
        )
    for column_name in overlap_with_targets:
        leakage_rows.append(
            {
                "feature_set": feature_set,
                "file_name": FEATURE_FILES[feature_set],
                "column_name": column_name,
                "issue_type": "target_overlap",
            }
        )
    leakage_frame = pd.DataFrame(leakage_rows)
    return summary, leakage_frame


def summarise_targets(target_frame: pd.DataFrame) -> dict[str, object]:
    """Summarise target-table integrity. / 汇总目标表完整性。"""
    summary: dict[str, object] = {
        "rows": int(len(target_frame)),
        "columns": int(len(target_frame.columns)),
        "duplicate_seqn": int(target_frame["SEQN"].duplicated().sum()) if "SEQN" in target_frame.columns else None,
    }
    for target_name in ["H_v1", "H_v2"]:
        if target_name in target_frame.columns:
            series = pd.to_numeric(target_frame[target_name], errors="coerce")
            summary[target_name] = {
                "non_missing": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "min": None if series.dropna().empty else float(series.min()),
                "max": None if series.dropna().empty else float(series.max()),
                "mean": None if series.dropna().empty else float(series.mean()),
            }
    if "age_group" in target_frame.columns:
        counts = target_frame["age_group"].fillna("missing").value_counts(dropna=False)
        summary["age_group_counts"] = {str(index): int(value) for index, value in counts.items()}
    return summary


def build_markdown_report(input_paths: dict[str, str], feature_summaries: list[dict[str, object]], target_summary: dict[str, object], leakage_frame: pd.DataFrame) -> str:
    """Build the markdown report. / 生成 Markdown 报告。"""
    lines = [
        "# NHANES Data Integrity Report",
        "",
        "This report checks data integrity and leakage only.",
        "该报告仅检查数据完整性与潜在泄漏，不训练任何模型。",
        "",
        "No experimental result is fabricated in this step.",
        "该步骤不会伪造任何实验结果。",
        "",
        "## Input Files",
        "",
    ]
    for name, path in input_paths.items():
        lines.append(f"- `{name}`: `{path}`")

    lines.extend(["", "## Feature Tables", ""])
    for item in feature_summaries:
        lines.extend(
            [
                f"### {item['feature_set']}",
                "",
                f"- File: `{item['file_name']}`",
                f"- Rows: {item['rows']}",
                f"- Columns: {item['columns']}",
                f"- Duplicate rows: {item['duplicate_rows']}",
                f"- Mean missing rate: {item['mean_missing_rate']:.6f}",
                f"- Banned model-input columns found: {item['banned_column_count']}",
                f"- Target-overlap columns found: {item['target_overlap_count']}",
                "",
            ]
        )

    lines.extend(["## Targets", ""])
    lines.append(f"- Rows: {target_summary['rows']}")
    lines.append(f"- Columns: {target_summary['columns']}")
    lines.append(f"- Duplicate `SEQN`: {target_summary['duplicate_seqn']}")
    if "age_group_counts" in target_summary:
        lines.append(f"- `age_group` counts: {target_summary['age_group_counts']}")
    for target_name in ["H_v1", "H_v2"]:
        if target_name in target_summary:
            stats = target_summary[target_name]
            lines.append(
                f"- `{target_name}` non-missing={stats['non_missing']}, missing={stats['missing']}, min={stats['min']}, max={stats['max']}, mean={stats['mean']}"
            )

    lines.extend(["", "## Leakage Table", ""])
    if leakage_frame.empty:
        lines.append("- No banned feature columns or target-overlap columns were detected in the raw feature files.")
    else:
        lines.append(f"- Leakage rows written to `outputs/tables/leakage_check.csv`: {len(leakage_frame)}")
    return "\n".join(lines)


def main() -> int:
    """Run data checks and write canonical outputs. / 运行数据检查并写出标准输出。"""
    args = parse_args()
    tables_dir = args.output_dir / "tables"
    reports_dir = args.output_dir / "reports"
    legacy_dir = args.output_dir / "data_check"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir.mkdir(parents=True, exist_ok=True)

    input_paths = {
        "full_features": str(args.data_dir / FEATURE_FILES["full"]),
        "reduced_features": str(args.data_dir / FEATURE_FILES["reduced"]),
        "targets": str(args.data_dir / TARGET_FILE),
    }
    feature_frames = {
        feature_set: pd.read_csv(args.data_dir / FEATURE_FILES[feature_set])
        for feature_set in FEATURE_FILES
    }
    target_frame = pd.read_csv(args.data_dir / TARGET_FILE)
    target_columns = set(target_frame.columns)

    feature_summaries: list[dict[str, object]] = []
    leakage_frames: list[pd.DataFrame] = []
    for feature_set, frame in feature_frames.items():
        summary, leakage_frame = summarise_feature_table(feature_set, frame, target_columns)
        feature_summaries.append(summary)
        if not leakage_frame.empty:
            leakage_frames.append(leakage_frame)

    leakage_check = pd.concat(leakage_frames, ignore_index=True) if leakage_frames else pd.DataFrame(columns=["feature_set", "file_name", "column_name", "issue_type"])
    target_summary = summarise_targets(target_frame)
    payload = {
        "input_paths": input_paths,
        "feature_summaries": feature_summaries,
        "target_summary": target_summary,
    }

    leakage_check.to_csv(tables_dir / "leakage_check.csv", index=False, encoding="utf-8-sig")
    report_text = build_markdown_report(input_paths, feature_summaries, target_summary, leakage_check)
    (reports_dir / "data_integrity_report.md").write_text(report_text, encoding="utf-8")
    (legacy_dir / "data_check_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (legacy_dir / "data_check_report.md").write_text(report_text, encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {tables_dir / 'leakage_check.csv'}")
    print(f"Wrote {reports_dir / 'data_integrity_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
