"""Preprocessing pipeline: raw dataframes -> model-ready numeric matrices.

LEARNING SCAFFOLD
-----------------
This file is intentionally incomplete. The plumbing (the ``PreparedData``
container, imports, and the ``__main__`` self-check) is written for you. YOU
implement the three functions marked ``# TODO`` — they are the high-value core.

Concepts recap (full teach-up was in chat):
* Categorical features (text) -> **OneHotEncoder** so we don't invent a fake
  ordering. Use ``handle_unknown="ignore"`` so unseen test categories become an
  all-zero block instead of raising.
* Numeric features -> **StandardScaler** (mean 0, sd 1) so the MLP trains well;
  harmless to trees. One shared pipeline keeps every model apples-to-apples.
* **Leakage rule:** ``fit`` (or ``fit_transform``) on TRAIN only, then
  ``transform`` TEST with the statistics learned from train. Never fit on test.

Definition of done: ``python src/preprocess.py`` runs and prints, for both
schemes, ``X_train (125973, 122)`` / ``X_test (22544, 122)``, correct class
counts, and no NaNs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import data as D  # src/data.py (NSL-KDD schema + loaders)


@dataclass
class PreparedData:
    """Everything a model needs, plus metadata for interpretation. (Given.)"""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]      # names after one-hot expansion, in column order
    classes: list[str]            # label order (index i <-> integer label i)
    preprocessor: ColumnTransformer
    scheme: str                   # "binary" or "multiclass"

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def build_preprocessor(categorical: list[str],
                       numeric: list[str]) -> ColumnTransformer:
    """Return an (unfitted) ColumnTransformer.

    It must one-hot-encode the ``categorical`` columns and standard-scale the
    ``numeric`` columns, dropping everything else.

    TODO(you):
      * Build a ``ColumnTransformer`` with two transformers:
          - "cat": OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                   applied to ``categorical``
          - "num": StandardScaler() applied to ``numeric``
      * Set ``remainder="drop"`` (label/difficulty columns must not leak in).
      * Hint: pass ``verbose_feature_names_out=False`` so feature names stay
        clean (e.g. "protocol_type_tcp" not "cat__protocol_type_tcp").
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _feature_names(pre: ColumnTransformer) -> list[str]:
    """Column names after fitting (one-hot names + numeric names). (Given.)"""
    return list(pre.get_feature_names_out())


def prepare(train_df: pd.DataFrame,
            test_df: pd.DataFrame,
            categorical: list[str],
            numeric: list[str],
            y_train_labels: pd.Series,
            y_test_labels: pd.Series,
            class_order: list[str],
            scheme: str) -> PreparedData:
    """Fit the pipeline on train, transform both splits, encode labels.

    ``class_order`` fixes the integer mapping (index i <-> class_order[i]) so it
    is stable and readable across models and confusion matrices.

    TODO(you):
      * pre = build_preprocessor(categorical, numeric)
      * X_train = pre.fit_transform(train_df)   # FIT ON TRAIN ONLY
      * X_test  = pre.transform(test_df)         # transform test w/ train stats
      * Encode labels to integers using ``class_order``:
          label_to_int = {c: i for i, c in enumerate(class_order)}
          y_train = y_train_labels.map(label_to_int).to_numpy()
          (same for y_test)
      * Return a PreparedData(...) with feature_names=_feature_names(pre).
    """
    pre = build_preprocessor(categorical, numeric)
    X_train = pre.fit_transform(train_df)
    X_test = pre.transform(test_df)

    label_to_int = {label: i for i, label in enumerate(class_order)}
    y_train_mapped = y_train_labels.map(label_to_int)
    y_test_mapped = y_test_labels.map(label_to_int)
    # Fail loud on any label absent from class_order (else it becomes a silent
    # NaN and a cryptic int-cast error downstream).
    for split_name, mapped, raw in (("train", y_train_mapped, y_train_labels),
                                    ("test", y_test_mapped, y_test_labels)):
        if mapped.isna().any():
            bad = sorted(set(raw[mapped.isna()].astype(str)))
            raise ValueError(
                f"{split_name}: labels absent from class_order {class_order}: {bad}"
            )
    y_train = y_train_mapped.to_numpy(dtype=np.int64)
    y_test = y_test_mapped.to_numpy(dtype=np.int64)

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=_feature_names(pre),
        classes=list(class_order),  # defensive copy; don't alias the caller's list
        preprocessor=pre,
        scheme=scheme,
    )


def prepare_nsl_kdd(scheme: str, test_split: str = "test") -> PreparedData:
    """Load NSL-KDD and return model-ready arrays for the chosen label scheme.

    Parameters
    ----------
    scheme : {"binary", "multiclass"}
        ``binary`` = normal(0) vs attack(1); ``multiclass`` = the 5 families.
    test_split : {"test", "test-21"}
        Full official test set, or the hard KDDTest-21 subset.

    TODO(you):
      * train = D.load_nsl_kdd("train"); test = D.load_nsl_kdd(test_split)
      * For scheme == "binary":
          - build y label Series of strings "normal"/"attack" from the
            "binary_label" column (0->"normal", 1->"attack")
          - class_order = ["normal", "attack"]
      * For scheme == "multiclass":
          - y labels come straight from the "attack_family" column
          - class_order = D.FAMILY_ORDER
      * else: raise ValueError.
      * Return prepare(train, test, D.CATEGORICAL_COLS, D.NUMERIC_COLS,
                       y_tr, y_te, class_order, scheme)
    """
    train = D.load_nsl_kdd("train")
    test = D.load_nsl_kdd(test_split)

    if scheme == "binary":
        label_names = {0: "normal", 1: "attack"}
        y_tr = train["binary_label"].map(label_names)
        y_te = test["binary_label"].map(label_names)
        class_order = ["normal", "attack"]
    elif scheme == "multiclass":
        y_tr = train["attack_family"]
        y_te = test["attack_family"]
        class_order = D.FAMILY_ORDER
    else:
        raise ValueError(
            f"unknown scheme {scheme!r}; expected 'binary' or 'multiclass'"
        )

    return prepare(
        train,
        test,
        D.CATEGORICAL_COLS,
        D.NUMERIC_COLS,
        y_tr,
        y_te,
        class_order,
        scheme,
    )


if __name__ == "__main__":
    # Self-check harness (given). Run: python src/preprocess.py
    print("=" * 70)
    for scheme in ("binary", "multiclass"):
        pd_ = prepare_nsl_kdd(scheme)
        print(f"\n[{scheme}]  X_train {pd_.X_train.shape}  X_test {pd_.X_test.shape}")
        print(f"  features after one-hot: {pd_.n_features} "
              f"(from {len(D.FEATURE_NAMES)} raw)")
        print(f"  classes (int order): {list(enumerate(pd_.classes))}")
        for name, y in (("train", pd_.y_train), ("test", pd_.y_test)):
            counts = np.bincount(y, minlength=len(pd_.classes))
            pretty = ", ".join(f"{pd_.classes[i]}={c:,}" for i, c in enumerate(counts))
            print(f"  {name} counts: {pretty}")
        assert not np.isnan(pd_.X_train).any(), "NaNs in X_train"
        assert not np.isnan(pd_.X_test).any(), "NaNs in X_test"
    print("\nAll checks passed." )
    print("=" * 70)
