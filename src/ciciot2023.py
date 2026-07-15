"""CICIoT2023 source-aware loading and Phase-1 EDA helpers.

The full CICIoT2023 release is large, so this module is deliberately chunk- and
sample-friendly. Place the downloaded CSV files under ``data/ciciot2023/CSV/``
or pass a custom path to the helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import data as D

REPO_ROOT: Path = D.REPO_ROOT
DATA_DIR: Path = REPO_ROOT / "data" / "ciciot2023"
CSV_DIR: Path = DATA_DIR / "CSV"
# Pre-split, downsampled parquet dev sample (see data/ciciot2023/SOURCE.md).
PARQUET_SPLITS: dict[str, str] = {"train": "train.parquet", "test": "test.parquet"}

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
    drop = {"label", "Label", "attack_category", "attack_class", "binary_label",
            "source_file"}
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


# --------------------------------------------------------------------------- #
# Parquet dev sample: canonical loader + scale-only preprocessing + audit
# --------------------------------------------------------------------------- #
LABEL_LEVELS: dict[str, str] = {
    "binary": "binary_label",       # 0 benign / 1 attack
    "category": "attack_category",  # 8 canonical CATEGORY_ORDER
    "fine": "label",                # ~34 fine attack names
}


def load_parquet(split: str) -> pd.DataFrame:
    """Load a CICIoT2023 parquet dev split with canonical label columns.

    Produces the same label convention as ``add_label_columns``: ``label`` (fine
    string), ``attack_category`` (canonical 8), ``binary_label`` (0/1). The
    parquet's own ``label`` (binary) and ``attack_class`` columns are re-derived
    so categories always come from the tolerant fine-label mapper.
    """
    fname = PARQUET_SPLITS.get(split)
    if fname is None:
        raise ValueError(f"split must be one of {list(PARQUET_SPLITS)}, got {split!r}")
    path = DATA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. See data/ciciot2023/SOURCE.md for the source.")
    raw = pd.read_parquet(path)
    feats = [c for c in raw.columns if c not in ("Label", "attack_class", "label")]
    out = raw[feats].copy()
    out["label"] = raw["Label"].astype(str).str.strip()        # fine
    out["attack_category"] = out["label"].map(attack_category)  # canonical 8
    out["binary_label"] = raw["label"].astype(int)             # 0 benign / 1 attack
    return out


@dataclass
class CicPrepared:
    """Model-ready arrays for one CICIoT2023 label level."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    classes: list[str]
    scaler: StandardScaler
    level: str

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def prepare(train_df: pd.DataFrame, test_df: pd.DataFrame,
            level: str = "binary") -> CicPrepared:
    """Scale-only, leakage-safe preprocessing for a chosen label level.

    All CICIoT2023 features are numeric → StandardScaler only (no one-hot).
    Inf→NaN, median imputation, and scaling are all fit on **train only**.
    """
    if level not in LABEL_LEVELS:
        raise ValueError(f"level must be one of {list(LABEL_LEVELS)}, got {level!r}")
    feats = feature_columns(train_df)

    Xtr = train_df[feats].replace([np.inf, -np.inf], np.nan)
    Xte = test_df[feats].replace([np.inf, -np.inf], np.nan)
    medians = Xtr.median()                        # train-only imputation
    Xtr, Xte = Xtr.fillna(medians), Xte.fillna(medians)

    scaler = StandardScaler().fit(Xtr)            # FIT ON TRAIN ONLY
    X_train, X_test = scaler.transform(Xtr), scaler.transform(Xte)

    col = LABEL_LEVELS[level]
    if level == "binary":
        classes = ["Benign", "Attack"]
        y_train = train_df[col].to_numpy(dtype=np.int64)
        y_test = test_df[col].to_numpy(dtype=np.int64)
    else:
        classes = list(CATEGORY_ORDER) if level == "category" \
            else sorted(train_df[col].unique())
        idx = {c: i for i, c in enumerate(classes)}

        def _enc(s: pd.Series, name: str) -> np.ndarray:
            m = s.map(idx)
            if m.isna().any():
                bad = sorted(set(s[m.isna()].astype(str)))
                raise ValueError(f"{name}: labels not in {classes}: {bad}")
            return m.to_numpy(dtype=np.int64)

        y_train = _enc(train_df[col], "train")
        y_test = _enc(test_df[col], "test")

    return CicPrepared(X_train, X_test, y_train, y_test, list(feats),
                       list(classes), scaler, level)


def data_quality_audit(df: pd.DataFrame, name: str) -> list[str]:
    """Markdown lines: missing / inf / duplicate / (near-)constant + consistency."""
    feats = feature_columns(df)
    X = df[feats]
    n = len(df)
    n_missing = int(X.isna().sum().sum())
    n_inf = int(np.isinf(X.select_dtypes("number").to_numpy()).sum())
    n_dup = int(df.duplicated().sum())
    constant = [c for c in feats if X[c].nunique(dropna=False) <= 1]
    near = [c for c in feats if X[c].nunique(dropna=False) > 1
            and X[c].value_counts(normalize=True).iloc[0] > 0.999]
    bad = df[(df["attack_category"] == "Benign") != (df["binary_label"] == 0)]
    return [f"### {name}  ({n:,} rows x {len(feats)} features)", "",
            f"- missing cells: **{n_missing:,}**",
            f"- infinite cells: **{n_inf:,}**",
            f"- duplicated rows: **{n_dup:,}** ({n_dup / n * 100:.2f}%)",
            f"- constant features: **{len(constant)}** {constant or ''}",
            f"- near-constant (>99.9% one value): **{len(near)}** {near or ''}",
            f"- binary/category consistency: "
            f"**{'OK' if bad.empty else f'{len(bad)} mismatches'}**", ""]


if __name__ == "__main__":
    if (DATA_DIR / PARQUET_SPLITS["train"]).exists():
        tr, te = load_parquet("train"), load_parquet("test")
        out = ["# CICIoT2023 (parquet dev) — data-quality & leakage audit", "",
               "All features numeric → scale-only preprocessing. This dev sample "
               "uses a **random** split (in-distribution — caveat every score). No "
               "IP/port/timestamp columns → host/time leakage already avoided.", ""]
        out += data_quality_audit(tr, "train")
        out += data_quality_audit(te, "test")
        report = "\n".join(out) + "\n"
        (REPO_ROOT / "results" / "ciciot2023_quality.md").write_text(report,
                                                                     encoding="utf-8")
        print(report)
    else:
        try:
            sample = load_csv_sample()
        except FileNotFoundError as exc:
            print(exc)
        else:
            s = phase1_summary(sample)
            print(f"rows={s['rows']:,} features={s['n_features']} "
                  f"labels={s['n_fine_labels']}")
            print(s["category_counts"])
