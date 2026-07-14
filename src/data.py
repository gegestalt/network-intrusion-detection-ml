"""NSL-KDD schema, attack-family mapping, and loaders.

Single source of truth for column names, feature groupings, and the mapping from
specific attack labels to the four attack families. Import from here everywhere
(notebooks and training scripts) so the dataset is described in exactly one place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Repo root = parent of this file's directory (<repo>/src/data.py -> <repo>).
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = REPO_ROOT / "data" / "nsl_kdd"

RANDOM_STATE: int = 42

# --- the 41 NSL-KDD features, in file order, then label + difficulty ---------
FEATURE_NAMES: list[str] = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
# Full column list as stored in the .txt files (43 columns).
COLUMN_NAMES: list[str] = [*FEATURE_NAMES, "label", "difficulty"]

# --- feature groupings -------------------------------------------------------
CATEGORICAL_COLS: list[str] = ["protocol_type", "service", "flag"]

# Genuinely binary (0/1) numerics — do NOT one-hot these; leave as 0/1.
BINARY_COLS: list[str] = [
    "land", "logged_in", "root_shell", "su_attempted",
    "is_host_login", "is_guest_login",
]

# Everything numeric that isn't categorical.
NUMERIC_COLS: list[str] = [c for c in FEATURE_NAMES if c not in CATEGORICAL_COLS]

# `num_outbound_cmds` is all-zero in NSL-KDD (zero variance); harmless but
# uninformative. Kept in NUMERIC_COLS so the schema matches the file; scalers
# handle constant columns fine. Flagged here for transparency.
ZERO_VARIANCE_COLS: list[str] = ["num_outbound_cmds"]

# --- attack label -> family --------------------------------------------------
# DoS = denial of service, Probe = surveillance/scanning,
# R2L = remote-to-local (unauthorized remote access),
# U2R = user-to-root (privilege escalation).
ATTACK_FAMILY_MAP: dict[str, str] = {
    "normal": "normal",
    # DoS
    "neptune": "DoS", "smurf": "DoS", "back": "DoS", "teardrop": "DoS",
    "pod": "DoS", "land": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "mailbomb": "DoS", "worm": "DoS",
    # Probe
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # R2L
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R",
    "perl": "U2R", "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}

FAMILY_ORDER: list[str] = ["normal", "DoS", "Probe", "R2L", "U2R"]


def load_nsl_kdd(split: str) -> pd.DataFrame:
    """Load a raw NSL-KDD split with proper column names.

    Parameters
    ----------
    split : {"train", "test", "test-21", "train-20"}
        Which official file to load: full train (``KDDTrain+``), full test
        (``KDDTest+``), the hard test subset (``KDDTest-21``), or the 20%
        training subset (``KDDTrain+_20Percent``).

    Returns
    -------
    pd.DataFrame
        43 columns (41 features + ``label`` + ``difficulty``), plus derived
        ``binary_label`` (normal/attack) and ``attack_family`` columns.
    """
    fname = {
        "train": "KDDTrain+.txt",
        "test": "KDDTest+.txt",
        "test-21": "KDDTest-21.txt",
        "train-20": "KDDTrain+_20Percent.txt",
    }[split]
    path = DATA_DIR / fname
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/download_data.py` first."
        )
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    df = add_label_columns(df)
    return df


def add_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``binary_label`` and ``attack_family`` derived from ``label``.

    Raises
    ------
    KeyError
        If any raw label is not present in ``ATTACK_FAMILY_MAP`` — the test set
        introduces new attack names and each must be mapped deliberately, never
        silently dropped.
    """
    unknown = set(df["label"].unique()) - set(ATTACK_FAMILY_MAP)
    if unknown:
        raise KeyError(
            f"Unmapped attack labels (add to ATTACK_FAMILY_MAP): {sorted(unknown)}"
        )
    df = df.copy()
    df["attack_family"] = df["label"].map(ATTACK_FAMILY_MAP)
    df["binary_label"] = (df["label"] != "normal").astype(int)  # 1 = attack
    return df


if __name__ == "__main__":
    # Smoke test: load both splits and print a compact summary.
    for sp in ("train", "test"):
        d = load_nsl_kdd(sp)
        fam = d["attack_family"].value_counts().reindex(FAMILY_ORDER)
        print(f"{sp:>5}: {len(d):>7,} rows | families -> "
              + ", ".join(f"{k}={int(v):,}" for k, v in fam.items()))
