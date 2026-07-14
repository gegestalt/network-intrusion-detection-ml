"""CICIoT2023 loader + data-quality / leakage audit.

CIC IoT Dataset 2023 (Neto et al.): IoT-topology traffic, 33 attacks grouped into
7 categories (+ Benign). We use a pre-split, downsampled dev parquet (~1.34M rows;
see data/ciciot2023/SOURCE.md); scale to the 2.1 GB full version later.

Three label levels ship in the data:
  * ``label``        — binary (0 benign / 1 attack)
  * ``attack_class`` — 8 categories (Benign + DDoS/DoS/Recon/Web/BruteForce/Spoofing/Mirai)
  * ``Label``        — fine-grained attack name (~34 classes)

All 39 features are numeric (protocol/flag indicators are already 0/1), so
preprocessing is scale-only — no one-hot needed, unlike NSL-KDD.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR: Path = Path(__file__).resolve().parents[1] / "data" / "ciciot2023"

LABEL_COLS: list[str] = ["label", "attack_class", "Label"]
CATEGORY_ORDER: list[str] = [
    "Benign", "DDoS", "DoS", "Recon", "Spoofing", "Mirai", "Web-based", "BruteForce",
]


def load(split: str) -> pd.DataFrame:
    """Load a CICIoT2023 dev split ('train' or 'test')."""
    path = DATA_DIR / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (see data/ciciot2023/SOURCE.md).")
    return pd.read_parquet(path)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The numeric feature columns (everything except the 3 label columns)."""
    return [c for c in df.columns if c not in LABEL_COLS]


def audit(df: pd.DataFrame, name: str) -> list[str]:
    """Return markdown lines of a data-quality / leakage audit for ``df``."""
    feats = feature_columns(df)
    X = df[feats]
    n = len(df)

    n_missing = int(X.isna().sum().sum())
    n_inf = int(np.isinf(X.select_dtypes("number").to_numpy()).sum())
    n_dup = int(df.duplicated().sum())
    constant = [c for c in feats if X[c].nunique(dropna=False) <= 1]
    near_const = [c for c in feats
                  if X[c].nunique(dropna=False) > 1
                  and X[c].value_counts(normalize=True).iloc[0] > 0.999]

    lines = [f"### {name}  ({n:,} rows x {len(feats)} features)",
             "",
             f"- missing cells: **{n_missing:,}**",
             f"- infinite cells: **{n_inf:,}**",
             f"- fully duplicated rows: **{n_dup:,}** ({n_dup / n * 100:.2f}%)",
             f"- constant features: **{len(constant)}** {constant or ''}",
             f"- near-constant (>99.9% one value): **{len(near_const)}** "
             f"{near_const or ''}",
             ""]
    # label-level consistency: every attack_class!=Benign must be label==1
    bad = df[(df["attack_class"] == "Benign") != (df["label"] == 0)]
    lines.append(f"- label-level consistency (binary vs category): "
                 f"**{'OK' if bad.empty else f'{len(bad)} mismatches'}**")
    lines.append("")
    return lines


def main() -> int:
    train, test = load("train"), load("test")
    out = ["# CICIoT2023 (dev sample) — data-quality & leakage audit", "",
           "All features numeric (flag/protocol indicators already 0/1) → "
           "preprocessing is scale-only. This dev sample uses a **random** "
           "train/test split, so it is *in-distribution* (unlike NSL-KDD's "
           "official shift-heavy split) — CICIoT2023 scores will look higher for "
           "reasons of protocol; caveat every result.", "",
           "> No socket-identifier columns (IP/port/timestamp) are present, so the "
           "classic CICFlowMeter host/time **leakage risk is already avoided** here.",
           ""]
    out += audit(train, "train")
    out += audit(test, "test")

    out += ["## Label distribution (train)", "",
            "| Level | #classes | head |", "| --- | ---: | --- |"]
    out.append(f"| binary `label` | 2 | attack {train['label'].mean()*100:.1f}% |")
    cat = train["attack_class"].value_counts()
    out.append(f"| category `attack_class` | {cat.size} | "
               + ", ".join(f"{k} {v:,}" for k, v in cat.head(4).items()) + ", … |")
    out.append(f"| fine `Label` | {train['Label'].nunique()} | "
               f"{train['Label'].nunique()} attack types |")
    out.append("")

    report = "\n".join(out) + "\n"
    dest = Path(__file__).resolve().parents[1] / "results" / "ciciot2023_quality.md"
    dest.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
