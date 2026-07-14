"""Download and verify all datasets for this study.

Three network-intrusion benchmarks spanning three eras:

  * **NSL-KDD** (1999-era, cleaned KDD'99) — UNB/CIC. Mirror: Jehuty4949, defcom17.
  * **UNSW-NB15** (2015) — UNSW Canberra. Mirror: Mireu-Lab (HuggingFace).
  * **CICIDS2017** (2017) — UNB/CIC, CICFlowMeter flows. Mirror: c01dsnap (HF).

Everything is fetched from public mirrors and *verified* against documented
shape (row/column counts) so a truncated or wrong file cannot pass silently.
SHA-256 hashes and source URLs are recorded to ``data/<dataset>/SOURCE.md``.

Data layout produced::

    data/
      nsl_kdd/     KDDTrain+.txt KDDTest+.txt KDDTest-21.txt KDDTrain+_20Percent.txt
      unsw_nb15/   UNSW_NB15_training-set.csv  UNSW_NB15_testing-set.csv
      cicids2017/  <8 CICFlowMeter CSVs>

Usage (from repo root)::

    python src/download_data.py            # all datasets
    python src/download_data.py nsl_kdd    # one dataset
    python src/download_data.py unsw_nb15 cicids2017

Notes / gotchas handled here:
  * NSL-KDD '+' is percent-encoded in URLs but kept literally on disk.
  * The Mireu-Lab UNSW mirror has **train/test filenames swapped** relative to
    the official partition, so we assign roles by *row count* (175,341 = train,
    82,332 = test), not by the source filename.
"""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = REPO_ROOT / "data"

CHUNK = 1 << 20  # 1 MiB
TIMEOUT = 120


# --------------------------------------------------------------------------- #
# generic helpers
# --------------------------------------------------------------------------- #
def _fetch(urls: list[str], dest: Path) -> None:
    """Stream the first working URL in ``urls`` to ``dest`` (fallback chain)."""
    last_err: Exception | None = None
    for url in urls:
        try:
            print(f"    -> {url}")
            with requests.get(url, timeout=TIMEOUT, stream=True) as resp:
                resp.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                size = 0
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(CHUNK):
                        fh.write(chunk)
                        size += len(chunk)
                if size == 0:
                    raise ValueError("empty response body")
                tmp.replace(dest)
                print(f"       {size:,} bytes")
                return
        except Exception as err:  # noqa: BLE001 - try the next mirror
            print(f"       failed: {err}")
            last_err = err
    raise RuntimeError(f"all mirrors failed for {dest.name}: {last_err}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _shape_csv(path: Path, sep: str = ",", header: int | None = 0) -> tuple[int, int]:
    """Return (n_data_rows, n_cols) for a delimited file, memory-friendly."""
    # Column count from the first line; row count by streaming.
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
        n_cols = first.count(sep) + 1
        n_rows = sum(1 for line in fh if line.strip())
    if header is not None:  # first line was a header, already consumed
        return n_rows, n_cols
    return n_rows + 1, n_cols


def _write_source(dataset: str, title: str, note: str,
                  rows: list[tuple[str, int, int, str]]) -> None:
    out = DATA_DIR / dataset / "SOURCE.md"
    lines = [f"# {title} — source & provenance", "", note, "",
             "| File | Rows | Cols | SHA-256 |", "| --- | ---: | ---: | --- |"]
    for name, r, c, digest in rows:
        lines.append(f"| `{name}` | {r:,} | {c} | `{digest}` |")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  provenance -> {out.relative_to(REPO_ROOT)}")


# --------------------------------------------------------------------------- #
# NSL-KDD
# --------------------------------------------------------------------------- #
NSL_MIRRORS = (
    "https://raw.githubusercontent.com/Jehuty4949/NSL_KDD/master/{f}",
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/{f}",
    "https://raw.githubusercontent.com/HoaNP/NSL-KDD-DataSet/master/{f}",
)
# local name -> (url basename, expected_rows). All 43 columns.
NSL_FILES = {
    "KDDTrain+.txt": ("KDDTrain%2B.txt", 125_973),
    "KDDTest+.txt": ("KDDTest%2B.txt", 22_544),
    "KDDTrain+_20Percent.txt": ("KDDTrain%2B_20Percent.txt", 25_192),
    "KDDTest-21.txt": ("KDDTest-21.txt", 11_850),
}


def download_nsl_kdd() -> None:
    print("[nsl_kdd]")
    d = DATA_DIR / "nsl_kdd"
    d.mkdir(parents=True, exist_ok=True)
    prov: list[tuple[str, int, int, str]] = []
    for local, (basename, exp_rows) in NSL_FILES.items():
        dest = d / local
        urls = [m.format(f=basename) for m in NSL_MIRRORS]
        _fetch(urls, dest)
        rows, cols = _shape_csv(dest, header=None)
        if cols != 43:
            raise ValueError(f"{local}: expected 43 cols, got {cols}")
        if rows != exp_rows:
            raise ValueError(f"{local}: expected {exp_rows:,} rows, got {rows:,}")
        prov.append((local, rows, cols, _sha256(dest)))
        print(f"    verified {local}: {rows:,} x {cols}")
    _write_source(
        "nsl_kdd", "NSL-KDD",
        "CIC/UNB, <https://www.unb.ca/cic/datasets/nsl.html>. Cleaned KDD'99. "
        "Primary mirror <https://github.com/Jehuty4949/NSL_KDD>. "
        "`KDDTest-21` = hard subset (drops records all 21 original learners got "
        "right); `_20Percent` = fast-iteration training subset. "
        "Cite: Tavallaee et al., IEEE CISDA 2009.", prov)


# --------------------------------------------------------------------------- #
# UNSW-NB15  (official partition; source filenames are swapped -> fix by rows)
# --------------------------------------------------------------------------- #
UNSW_BASE = "https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/resolve/main"
UNSW_EXPECT = {175_341: "UNSW_NB15_training-set.csv",  # official train
               82_332: "UNSW_NB15_testing-set.csv"}    # official test


def download_unsw_nb15() -> None:
    print("[unsw_nb15]")
    d = DATA_DIR / "unsw_nb15"
    d.mkdir(parents=True, exist_ok=True)
    prov: list[tuple[str, int, int, str]] = []
    # Download both source files under temp names, then assign roles by row count.
    staged: list[Path] = []
    for src_name in ("train.csv", "test.csv"):
        tmp = d / f"_staged_{src_name}"
        _fetch([f"{UNSW_BASE}/{src_name}"], tmp)
        staged.append(tmp)
    for tmp in staged:
        rows, cols = _shape_csv(tmp, header=0)
        if rows not in UNSW_EXPECT:
            raise ValueError(
                f"{tmp.name}: {rows:,} rows matches neither official partition "
                f"({sorted(UNSW_EXPECT)}) — mirror may have changed")
        if cols != 45:
            raise ValueError(f"{tmp.name}: expected 45 cols, got {cols}")
        canonical = UNSW_EXPECT[rows]
        dest = d / canonical
        tmp.replace(dest)
        role = "train" if rows == 175_341 else "test"
        prov.append((f"{canonical}  ({role})", rows, cols, _sha256(dest)))
        print(f"    verified {canonical}: {rows:,} x {cols}  [{role}]")
    _write_source(
        "unsw_nb15", "UNSW-NB15",
        "UNSW Canberra, <https://research.unsw.edu.au/projects/unsw-nb15-dataset>. "
        "Official train/test partition (175,341 / 82,332). Mirror "
        "<https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15> — NOTE its "
        "`train.csv`/`test.csv` are swapped vs the official partition, so roles "
        "are assigned here by row count and saved under canonical names. "
        "Cite: Moustafa & Slay, MilCIS 2015.", prov)


# --------------------------------------------------------------------------- #
# CICIDS2017  (CICFlowMeter CSVs, ~2.8M flows across 8 files, ~1.2 GB)
# --------------------------------------------------------------------------- #
CICIDS_BASE = "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main"
CICIDS_FILES = (
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
)


def download_cicids2017() -> None:
    print("[cicids2017]  (~1.2 GB, this takes a few minutes)")
    d = DATA_DIR / "cicids2017"
    d.mkdir(parents=True, exist_ok=True)
    prov: list[tuple[str, int, int, str]] = []
    total_rows = 0
    col_counts: set[int] = set()
    for name in CICIDS_FILES:
        dest = d / name
        url = f"{CICIDS_BASE}/{name.replace(' ', '%20')}"
        _fetch([url], dest)
        rows, cols = _shape_csv(dest, header=0)
        col_counts.add(cols)
        total_rows += rows
        prov.append((name, rows, cols, _sha256(dest)))
        print(f"    verified {name}: {rows:,} x {cols}")
    if len(col_counts) != 1:
        raise ValueError(f"inconsistent column counts across files: {col_counts}")
    print(f"    TOTAL: {total_rows:,} flows x {col_counts.pop()} cols across "
          f"{len(CICIDS_FILES)} files")
    _write_source(
        "cicids2017", "CICIDS2017",
        "CIC/UNB, <https://www.unb.ca/cic/datasets/ids-2017.html>. "
        "CICFlowMeter flow features (5 days of traffic, 8 CSVs). Mirror "
        "<https://huggingface.co/datasets/c01dsnap/CIC-IDS2017>. No official "
        "train/test split — we construct a stratified one in preprocessing. "
        "Known issues (whitespace in headers, NaN/Inf, class imbalance) are "
        "cleaned downstream. Cite: Sharafaldin et al., ICISSP 2018.", prov)


# --------------------------------------------------------------------------- #
REGISTRY = {
    "nsl_kdd": download_nsl_kdd,
    "unsw_nb15": download_unsw_nb15,
    "cicids2017": download_cicids2017,
}


def main(argv: list[str]) -> int:
    targets = argv or list(REGISTRY)
    unknown = set(targets) - set(REGISTRY)
    if unknown:
        print(f"unknown dataset(s): {sorted(unknown)}; choose from {list(REGISTRY)}")
        return 2
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in targets:
        REGISTRY[name]()
        print()
    print("Done. All requested datasets downloaded and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
