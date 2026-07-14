"""CICIoT2023 source-aware loading and Phase-1 EDA helpers.

The full CICIoT2023 release is large, so this module is deliberately chunk- and
sample-friendly. Place the downloaded CSV files under ``data/ciciot2023/CSV/``
or pass a custom path to the helpers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import data as D

REPO_ROOT: Path = D.REPO_ROOT
DATA_DIR: Path = REPO_ROOT / "data" / "ciciot2023"
CSV_DIR: Path = DATA_DIR / "CSV"

DATASET_PAGE = "https://www.unb.ca/cic/datasets/iotdataset-2023.html"
DOWNLOAD_PAGE = "https://www.unb.ca/cic/datasets/iotdataset-2023.html"

CATEGORY_ORDER: list[str] = [
    "Benign",
    "DDoS",
    "DoS",
    "Recon",
    "Web-based",
    "Brute Force",
    "Spoofing",
    "Mirai",
]

# Published dataset scale from Neto et al. (Sensors, 2023), Table 3 / Table 4.
# The feature table includes `ts`; the paper's ML pipeline removes timestamp and
# trains on the remaining behavior features.
EXPECTED_TOTAL_ROWS: int = 46_686_579
EXPECTED_FINE_LABELS: int = 34  # BenignTraffic + 33 attacks
EXPECTED_ATTACK_LABELS: int = 33
EXPECTED_CATEGORIES: int = 8  # Benign + 7 attack categories
EXPECTED_EXTRACTED_ATTRIBUTES: int = 47
EXPECTED_MODEL_FEATURES_AFTER_DROPPING_TS: int = 46
EXPECTED_IOT_DEVICES: int = 105

PUBLISHED_CATEGORY_ROWS: dict[str, int] = {
    "Benign": 1_098_195,
    "DDoS": 33_984_560,
    "DoS": 8_090_738,
    "Recon": 354_565,
    "Web-based": 24_829,
    "Brute Force": 13_064,
    "Spoofing": 486_504,
    "Mirai": 2_634_124,
}

WEB_ATTACK_TOKENS: set[str] = {
    "sqlinjection",
    "commandinjection",
    "backdoormalware",
    "uploadingattack",
    "uploading_attack",
    "xss",
    "browserhijacking",
}

LABEL_COLUMN_CANDIDATES: tuple[str, ...] = (
    "label",
    "Label",
    "class",
    "Class",
    "attack",
    "Attack",
    "Attack_type",
    "Attack Type",
)


def _compact(value: str) -> str:
    """Lowercase label with separators removed for robust matching."""
    return "".join(ch for ch in value.lower().strip() if ch.isalnum())


def attack_category(label: str) -> str:
    """Map a CICIoT2023 fine label to the 7 attack categories plus Benign.

    CICIoT2023 labels are commonly written with small separator/capitalization
    differences across mirrors, e.g. ``DDoS-UDP_Flood`` vs ``DDoS_UDP_Flood``.
    This mapper is deliberately tolerant about those separators.
    """
    raw = str(label).strip()
    compact = _compact(raw)

    if compact in {"benign", "benigntraffic", "normal"}:
        return "Benign"
    if compact.startswith("ddos"):
        return "DDoS"
    if compact.startswith("dos"):
        return "DoS"
    if compact.startswith("recon") or compact in {
        "vulnerabilityscan",
        "vulnerability_scan",
        "portsweep",
        "pingsweep",
        "osscan",
        "hostdiscovery",
    }:
        return "Recon"
    if compact.startswith("mirai"):
        return "Mirai"
    if "spoof" in compact:
        return "Spoofing"
    if "bruteforce" in compact or "dictionarybruteforce" in compact:
        return "Brute Force"
    if compact in WEB_ATTACK_TOKENS:
        return "Web-based"

    raise KeyError(f"Unmapped CICIoT2023 label: {label!r}")


def discover_csv_files(root: Path = CSV_DIR) -> list[Path]:
    """Return sorted CICIoT2023 CSV files below ``root``."""
    root = Path(root)
    if root.is_file() and root.suffix.lower() == ".csv":
        return [root]
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.csv") if p.is_file())


def find_label_column(columns: list[str]) -> str:
    """Find the label column in a CICIoT2023 CSV header."""
    exact = [c for c in LABEL_COLUMN_CANDIDATES if c in columns]
    if exact:
        return exact[0]
    lower_to_original = {c.lower(): c for c in columns}
    for candidate in LABEL_COLUMN_CANDIDATES:
        hit = lower_to_original.get(candidate.lower())
        if hit:
            return hit
    raise KeyError(
        "Could not find a CICIoT2023 label column. Expected one of "
        f"{LABEL_COLUMN_CANDIDATES}, got {columns}"
    )


def add_label_columns(df: pd.DataFrame, label_col: str | None = None) -> pd.DataFrame:
    """Add normalized ``label``, ``attack_category``, and ``binary_label``."""
    label_col = label_col or find_label_column(list(df.columns))
    out = df.copy()
    out["label"] = out[label_col].astype(str).str.strip()
    out["attack_category"] = out["label"].map(attack_category)
    out["binary_label"] = (out["attack_category"] != "Benign").astype(int)
    return out


def load_csv_sample(
    root: Path = CSV_DIR,
    max_rows_per_file: int = 25_000,
) -> pd.DataFrame:
    """Load a bounded sample from every CICIoT2023 CSV under ``root``."""
    files = discover_csv_files(root)
    if not files:
        raise FileNotFoundError(
            f"No CICIoT2023 CSV files found under {root}. Download from "
            f"{DOWNLOAD_PAGE} and place CSVs under data/ciciot2023/CSV/."
        )

    frames = []
    for path in files:
        df = pd.read_csv(path, nrows=max_rows_per_file)
        df.columns = [c.strip() for c in df.columns]
        df = add_label_columns(df)
        df["source_file"] = path.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model feature columns after dropping labels/provenance columns."""
    drop = {"label", "Label", "attack_category", "binary_label", "source_file"}
    return [c for c in df.columns if c not in drop]


def phase1_summary(df: pd.DataFrame) -> dict:
    """Compact Phase-1 EDA summary for notebooks/tests."""
    labelled = add_label_columns(df) if "attack_category" not in df.columns else df
    label_counts = labelled["label"].value_counts()
    category_counts = labelled["attack_category"].value_counts().reindex(
        CATEGORY_ORDER,
        fill_value=0,
    )
    return {
        "rows": int(len(labelled)),
        "n_features": len(feature_columns(labelled)),
        "n_fine_labels": int(label_counts.size),
        "n_categories": int((category_counts > 0).sum()),
        "label_counts": label_counts,
        "category_counts": category_counts,
    }


if __name__ == "__main__":
    try:
        sample = load_csv_sample()
    except FileNotFoundError as exc:
        print(exc)
    else:
        s = phase1_summary(sample)
        print(f"rows={s['rows']:,} features={s['n_features']} labels={s['n_fine_labels']}")
        print(s["category_counts"])
