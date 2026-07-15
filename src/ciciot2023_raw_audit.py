"""Raw-CSV Phase-1 audit for CICIoT2023.

If the official CSV files are not present, this script writes a blocked-status
report instead of pretending evaluation happened. Once CSVs are placed under
data/ciciot2023/CSV/, the same script samples them and writes a real Phase-1
summary.

Run:
    .venv/bin/python src/ciciot2023_raw_audit.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import ciciot2023 as C
import data as D

RESULTS = D.REPO_ROOT / "results"
REPORT_PATH = RESULTS / "ciciot2023_raw_audit.md"


def numeric_quality(df: pd.DataFrame) -> dict[str, object]:
    """Basic quality diagnostics for a CICIoT2023 sample."""
    features = C.feature_columns(df)
    numeric = df[features].select_dtypes(include=[np.number])
    return {
        "feature_columns": len(features),
        "numeric_columns": numeric.shape[1],
        "missing_cells": int(df[features].isna().sum().sum()),
        "inf_cells": int(np.isinf(numeric.to_numpy()).sum()) if numeric.shape[1] else 0,
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_numeric": [c for c in numeric.columns if numeric[c].nunique(dropna=False) <= 1],
        "near_constant_numeric": [
            c
            for c in numeric.columns
            if numeric[c].nunique(dropna=False) > 1
            and numeric[c].value_counts(normalize=True).iloc[0] > 0.999
        ],
    }


def render_blocked() -> str:
    return "\n".join(
        [
            "# CICIoT2023 Raw CSV Audit",
            "",
            "**Status: blocked by missing local raw CSV files.**",
            "",
            f"Expected local layout: `{C.CSV_DIR.relative_to(D.REPO_ROOT)}/*.csv`.",
            f"Official source: <{C.DATASET_PAGE}>.",
            "",
            "No model or data-quality claim should be made about the official raw CSV "
            "release until files are present and this audit is rerun.",
            "",
        ]
    )


def render_sample_report(sample: pd.DataFrame, max_rows_per_file: int) -> str:
    summary = C.phase1_summary(sample)
    quality = numeric_quality(sample)
    lines = [
        "# CICIoT2023 Raw CSV Audit",
        "",
        f"Sample policy: first {max_rows_per_file:,} rows per discovered CSV file.",
        "",
        f"- sample rows: **{summary['rows']:,}**",
        f"- feature columns: **{summary['n_features']}**",
        f"- fine labels in sample: **{summary['n_fine_labels']}**",
        f"- coarse categories in sample: **{summary['n_categories']}**",
        f"- missing feature cells: **{quality['missing_cells']:,}**",
        f"- infinite numeric cells: **{quality['inf_cells']:,}**",
        f"- duplicate rows in sample: **{quality['duplicate_rows']:,}**",
        f"- constant numeric columns: **{len(quality['constant_numeric'])}** "
        f"{quality['constant_numeric'][:20]}",
        f"- near-constant numeric columns: **{len(quality['near_constant_numeric'])}** "
        f"{quality['near_constant_numeric'][:20]}",
        "",
        "## Coarse Category Counts",
        "",
        "| Category | Rows |",
        "| --- | ---: |",
    ]
    for category, count in summary["category_counts"].items():
        lines.append(f"| {category} | {int(count):,} |")
    lines += [
        "",
        "## Fine Label Counts",
        "",
        "| Fine label | Rows |",
        "| --- | ---: |",
    ]
    for label, count in summary["label_counts"].head(40).items():
        lines.append(f"| {label} | {int(count):,} |")
    lines.append("")
    return "\n".join(lines)


def run(max_rows_per_file: int = 25_000) -> str:
    """Write the raw CSV audit report and return its markdown."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not C.discover_csv_files():
        report = render_blocked()
    else:
        sample = C.load_csv_sample(max_rows_per_file=max_rows_per_file)
        report = render_sample_report(sample, max_rows_per_file=max_rows_per_file)
    REPORT_PATH.write_text(report, encoding="utf-8")
    return report


def main() -> int:
    report = run()
    print(report)
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
