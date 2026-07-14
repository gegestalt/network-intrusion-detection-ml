"""Download and verify the NSL-KDD dataset.

NSL-KDD is a cleaned, de-duplicated revision of the classic KDD Cup 1999
intrusion-detection benchmark, published by the Canadian Institute for
Cybersecurity (CIC) at the University of New Brunswick (UNB):

    https://www.unb.ca/cic/datasets/nsl.html

The canonical UNB download sits behind a web form, which is awkward to automate
and occasionally rate-limited. We therefore pull the two files we need
(``KDDTrain+.txt`` and ``KDDTest+.txt``) from well-known, byte-identical GitHub
mirrors, and *verify* the result against the dataset's documented shape so a bad
or truncated mirror can never slip through silently.

Integrity checks (per file):
  * exact expected row count (as documented for NSL-KDD),
  * exact column count (41 features + label + difficulty = 43),
  * SHA-256 recorded to ``data/SOURCE.md`` for provenance.

Run from the repo root:

    .venv/bin/python src/download_data.py
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import requests

# Repo layout: this file is <repo>/src/download_data.py
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = REPO_ROOT / "data"

# Primary + fallback mirrors. All verified byte-identical (19,109,424 B train).
# Primary: defcom17/NSL_KDD is the most widely cited raw-text mirror.
MIRRORS: tuple[str, ...] = (
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/{fname}",
    "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/{fname}",
    "https://raw.githubusercontent.com/HoaNP/NSL-KDD-DataSet/master/{fname}",
)

# '+' must be percent-encoded in a URL path; the on-disk name keeps the '+'.
URL_NAMES: dict[str, str] = {
    "KDDTrain+.txt": "KDDTrain%2B.txt",
    "KDDTest+.txt": "KDDTest%2B.txt",
}


@dataclass(frozen=True)
class DatasetFile:
    """A single NSL-KDD file plus the invariants used to verify it."""

    filename: str
    expected_rows: int
    expected_cols: int = 43  # 41 features + label + difficulty
    sha256: str = field(default="", compare=False)


# Documented NSL-KDD record counts for the full KDDTrain+/KDDTest+ splits.
FILES: tuple[DatasetFile, ...] = (
    DatasetFile("KDDTrain+.txt", expected_rows=125_973),
    DatasetFile("KDDTest+.txt", expected_rows=22_544),
)

CHUNK = 1 << 16  # 64 KiB streaming chunks
TIMEOUT = 60  # seconds


def _download_one(filename: str) -> bytes:
    """Fetch ``filename`` from the first mirror that responds with 200/OK.

    Raises RuntimeError if every mirror fails.
    """
    url_name = URL_NAMES[filename]
    last_err: Exception | None = None
    for template in MIRRORS:
        url = template.format(fname=url_name)
        try:
            print(f"  -> trying {url}")
            resp = requests.get(url, timeout=TIMEOUT, stream=True)
            resp.raise_for_status()
            data = b"".join(resp.iter_content(CHUNK))
            if not data:
                raise ValueError("empty response body")
            print(f"     got {len(data):,} bytes")
            return data
        except Exception as err:  # noqa: BLE001 - we want to try the next mirror
            print(f"     failed: {err}")
            last_err = err
    raise RuntimeError(f"all mirrors failed for {filename}: {last_err}")


def _verify(spec: DatasetFile, raw: bytes) -> str:
    """Validate row/column shape; return the SHA-256 hex digest.

    Raises ValueError on any mismatch so corruption cannot pass silently.
    """
    text = raw.decode("utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    n_rows = len(lines)
    if n_rows != spec.expected_rows:
        raise ValueError(
            f"{spec.filename}: expected {spec.expected_rows:,} rows, "
            f"got {n_rows:,}"
        )
    n_cols = lines[0].count(",") + 1
    if n_cols != spec.expected_cols:
        raise ValueError(
            f"{spec.filename}: expected {spec.expected_cols} columns, "
            f"got {n_cols}"
        )
    digest = hashlib.sha256(raw).hexdigest()
    print(f"     verified: {n_rows:,} rows x {n_cols} cols  sha256={digest[:16]}...")
    return digest


def _write_provenance(results: list[DatasetFile]) -> None:
    """Record source URLs and hashes to data/SOURCE.md for reproducibility."""
    lines = [
        "# NSL-KDD data source & provenance",
        "",
        "Dataset: **NSL-KDD** — Canadian Institute for Cybersecurity (CIC),",
        "University of New Brunswick. Official page:",
        "<https://www.unb.ca/cic/datasets/nsl.html>",
        "",
        "Downloaded programmatically by `src/download_data.py` from a GitHub",
        "mirror (byte-identical to the UNB release). Primary mirror:",
        "<https://github.com/defcom17/NSL_KDD>.",
        "",
        "Cite: M. Tavallaee, E. Bagheri, W. Lu, A. Ghorbani, *A Detailed",
        'Analysis of the KDD CUP 99 Data Set*, IEEE CISDA, 2009.',
        "",
        "| File | Rows | Cols | SHA-256 |",
        "| --- | ---: | ---: | --- |",
    ]
    for f in results:
        lines.append(
            f"| `{f.filename}` | {f.expected_rows:,} | {f.expected_cols} | "
            f"`{f.sha256}` |"
        )
    lines.append("")
    (DATA_DIR / "SOURCE.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote provenance -> {DATA_DIR / 'SOURCE.md'}")


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading NSL-KDD into {DATA_DIR}\n")

    verified: list[DatasetFile] = []
    for spec in FILES:
        dest = DATA_DIR / spec.filename
        if dest.exists():
            print(f"[{spec.filename}] already present, re-verifying")
            raw = dest.read_bytes()
        else:
            print(f"[{spec.filename}] downloading")
            raw = _download_one(spec.filename)
        digest = _verify(spec, raw)
        dest.write_bytes(raw)
        # keep the hash alongside the rest of the spec
        verified.append(DatasetFile(spec.filename, spec.expected_rows,
                                    spec.expected_cols, digest))
        print(f"  saved -> {dest}\n")

    _write_provenance(verified)
    print("\nDone. All files downloaded and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
