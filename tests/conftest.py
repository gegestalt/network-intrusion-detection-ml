"""Shared pytest fixtures / markers.

Some tests need the downloaded NSL-KDD files; others are pure logic. Tests that
touch data use the ``needs_data`` marker (auto-skipped if the files are absent,
so `pytest` still passes on a fresh clone before `download_data.py` runs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import data as D  # noqa: E402  (after sys.path tweak)

_NSL_TRAIN = D.DATA_DIR / "KDDTrain+.txt"
DATA_PRESENT = _NSL_TRAIN.exists()

needs_data = pytest.mark.skipif(
    not DATA_PRESENT,
    reason="NSL-KDD files not downloaded (run `python src/download_data.py nsl_kdd`)",
)
