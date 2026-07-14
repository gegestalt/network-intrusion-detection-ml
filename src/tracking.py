"""Lightweight repository-local experiment tracking.

This is intentionally simpler than MLflow: every run is one JSON object in
``experiments/runs/*.jsonl`` plus optional CSV/Markdown summaries elsewhere.
The goal is observability without a server dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json

import data as D

EXPERIMENT_DIR = D.REPO_ROOT / "experiments" / "runs"


@dataclass
class ExperimentRun:
    """Machine-readable metadata for one experiment run."""

    experiment_id: str
    dataset: str
    task: str
    model_family: str
    run_name: str
    seed: int
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    notes: str = ""
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def stable_experiment_id(*parts: object) -> str:
    """Stable short identifier from descriptive parts."""
    raw = "::".join(str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def append_run(run: ExperimentRun, filename: str = "runs.jsonl") -> Path:
    """Append one run record to experiments/runs/filename."""
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPERIMENT_DIR / filename
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(run), sort_keys=True) + "\n")
    return path


def read_runs(filename: str = "runs.jsonl") -> list[dict[str, Any]]:
    """Read tracked runs; return [] if the log does not exist."""
    path = EXPERIMENT_DIR / filename
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
