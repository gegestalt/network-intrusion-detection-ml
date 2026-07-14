"""Preprocessing pipeline: raw dataframes -> model-ready numeric matrices.

Design goals
------------
* **Leakage-safe by construction.** The transformer is *fit on training data
  only* and then applied unchanged to test data. Nothing about the test set
  informs any scaling statistic or category vocabulary.
* **Apples-to-apples.** One shared pipeline feeds every model (RF, LightGBM,
  MLP), so differences in results come from the models, not the inputs.
* **Dataset-agnostic core.** ``build_preprocessor`` / ``prepare`` take explicit
  column lists, so the same code will serve UNSW-NB15 and CICIDS2017 later. For
  now only the NSL-KDD convenience wrapper is wired up (deliverable-first).

Categorical features -> OneHotEncoder(handle_unknown="ignore"): unseen test-set
categories become an all-zero block instead of raising. Numeric features ->
StandardScaler (mean 0, sd 1): needed by the MLP, harmless to trees.
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
    """Everything a model needs, plus metadata for interpretation."""

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
    """One-hot the categoricals, standard-scale the numerics; drop the rest."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _feature_names(pre: ColumnTransformer) -> list[str]:
    """Column names after fitting (one-hot names + numeric names, in order)."""
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
    """
    pre = build_preprocessor(categorical, numeric)
    X_train = pre.fit_transform(train_df)          # FIT on train only
    X_test = pre.transform(test_df)                # transform test with train's stats

    label_to_int = {c: i for i, c in enumerate(class_order)}
    y_train = y_train_labels.map(label_to_int).to_numpy()
    y_test = y_test_labels.map(label_to_int).to_numpy()

    return PreparedData(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        feature_names=_feature_names(pre),
        classes=list(class_order),
        preprocessor=pre,
        scheme=scheme,
    )


# --------------------------------------------------------------------------- #
# NSL-KDD convenience wrapper (the only dataset wired up for now)
# --------------------------------------------------------------------------- #
def prepare_nsl_kdd(scheme: str, test_split: str = "test") -> PreparedData:
    """Load NSL-KDD and return model-ready arrays for the chosen label scheme.

    Parameters
    ----------
    scheme : {"binary", "multiclass"}
        ``binary`` = normal(0) vs attack(1); ``multiclass`` = the 5 families.
    test_split : {"test", "test-21"}
        Evaluate on the full official test set or the hard KDDTest-21 subset.
    """
    train = D.load_nsl_kdd("train")
    test = D.load_nsl_kdd(test_split)

    if scheme == "binary":
        y_tr = train["binary_label"].map({0: "normal", 1: "attack"})
        y_te = test["binary_label"].map({0: "normal", 1: "attack"})
        class_order = ["normal", "attack"]
    elif scheme == "multiclass":
        y_tr = train["attack_family"]
        y_te = test["attack_family"]
        class_order = D.FAMILY_ORDER  # normal, DoS, Probe, R2L, U2R
    else:
        raise ValueError(f"scheme must be 'binary' or 'multiclass', got {scheme!r}")

    return prepare(
        train_df=train, test_df=test,
        categorical=D.CATEGORICAL_COLS,
        numeric=D.NUMERIC_COLS,
        y_train_labels=y_tr, y_test_labels=y_te,
        class_order=class_order, scheme=scheme,
    )


if __name__ == "__main__":
    # Verification / smoke test — prints shapes, class balances, and proves the
    # leakage-safe encoder handles test-only categories.
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

    # Prove handle_unknown works: services present in test but never in train.
    tr = D.load_nsl_kdd("train")
    te = D.load_nsl_kdd("test")
    unseen = set(te["service"].unique()) - set(tr["service"].unique())
    n_rows_unseen = te["service"].isin(unseen).sum()
    print(f"\nLeakage-safe check: {len(unseen)} service value(s) appear only in "
          f"test ({sorted(unseen)}), affecting {n_rows_unseen:,} rows.")
    print("  -> encoded as all-zero service block (no crash, no train leakage).")
    print("=" * 70)
