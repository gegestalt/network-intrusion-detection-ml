"""Small tuning helpers for audit and model-selection experiments.

These functions intentionally know nothing about NSL-KDD test features. They
operate on validation labels/scores or already-made predictions, which keeps the
fine-tuning protocol honest: choose on train/validation, then evaluate on test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ValidationSplit:
    """Integer row indices for a train/validation split of the training set."""

    train_idx: np.ndarray
    val_idx: np.ndarray


def make_validation_split(
    y_train: np.ndarray,
    validation_size: float = 0.2,
    seed: int = 42,
) -> ValidationSplit:
    """Return stratified train/validation indices for tuning.

    The input is only ``y_train`` by design. Callers cannot accidentally pass the
    official test labels/features into the split helper.
    """
    indices = np.arange(len(y_train))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=validation_size,
        stratify=y_train,
        random_state=seed,
    )
    return ValidationSplit(train_idx=train_idx, val_idx=val_idx)


def select_best_candidate(
    rows: list[dict[str, Any]],
    metric: str,
    maximize: bool = True,
) -> dict[str, Any]:
    """Select the best tuning row by ``metric``.

    Ties keep the earlier row, so candidate ordering remains meaningful and
    deterministic.
    """
    if not rows:
        raise ValueError("cannot select from an empty candidate list")
    best = rows[0]
    for row in rows[1:]:
        better = row[metric] > best[metric] if maximize else row[metric] < best[metric]
        if better:
            best = row
    return dict(best)


def threshold_sweep(
    y_val: np.ndarray,
    positive_scores: np.ndarray,
    thresholds: np.ndarray | None = None,
    beta: float = 2.0,
    pos_label: int = 1,
) -> dict[str, Any]:
    """Choose a binary threshold on validation scores using F-beta."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        pred = (positive_scores >= threshold).astype(int)
        score = fbeta_score(
            y_val,
            pred,
            beta=beta,
            pos_label=pos_label,
            zero_division=0,
        )
        rows.append({"threshold": float(threshold), "score": float(score)})
    best = select_best_candidate(rows, metric="score")
    return {"best": best, "rows": rows}


def metric_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    """Return the metric bundle used by tuning/audit tables."""
    labels = list(range(len(classes)))
    _, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    row: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    for i, name in enumerate(classes):
        row[f"recall:{name}"] = float(recall[i])
        row[f"f1:{name}"] = float(f1[i])
        row[f"support:{name}"] = int(support[i])
    return row


def rare_recall_mean(metric: dict[str, Any], rare_classes: list[str]) -> float:
    """Average recall for classes that define the rare-class objective."""
    if not rare_classes:
        raise ValueError("rare_classes must not be empty")
    recalls = [float(metric[f"recall:{name}"]) for name in rare_classes]
    return float(np.mean(recalls))
