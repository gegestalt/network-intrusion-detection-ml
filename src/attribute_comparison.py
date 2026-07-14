"""Attribute-by-attribute audit for the initial NSL-KDD dataset.

This is the deeper comparison layer: one row per original raw attribute, plus
one row per encoded model feature. It combines distribution checks, class
separation, correlations, outliers, redundancy, and model-based importance.

Run:
    .venv/bin/python src/attribute_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression

import data as D
import preprocess as P

RESULTS = D.REPO_ROOT / "results"
FIGURES = RESULTS / "figures"
ATTR_CSV = RESULTS / "attribute_comparison_nsl_kdd.csv"
ENCODED_CSV = RESULTS / "encoded_feature_comparison_nsl_kdd.csv"
REPORT_PATH = RESULTS / "attribute_comparison_nsl_kdd.md"


ATTR_COLUMNS = [
    "dataset",
    "attribute",
    "raw_type",
    "encoded_feature_count",
    "train_missing_rate",
    "test_missing_rate",
    "train_unique",
    "test_unique",
    "unseen_test_categories",
    "train_mean",
    "test_mean",
    "train_std",
    "test_std",
    "train_test_shift_std",
    "normal_mean",
    "attack_mean",
    "class_separation",
    "max_category_attack_rate",
    "target_corr_max_abs",
    "target_corr_mean_abs",
    "outlier_rate_max",
    "outlier_rate_mean",
    "redundant_pair_count",
    "logreg_importance_sum",
    "logreg_importance_max",
    "rf_importance_sum",
    "rf_importance_max",
    "mutual_info_sum",
    "mutual_info_max",
    "importance_consensus",
    "rank_by_consensus",
    "audit_flags",
    "engineering_recommendation",
]


def _sample_indices(y: np.ndarray, max_rows: int, seed: int = D.RANDOM_STATE) -> np.ndarray:
    """Deterministic stratified cap for attribution models."""
    if len(y) <= max_rows:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        n = max(1, int(round(max_rows * len(cls_idx) / len(y))))
        n = min(n, len(cls_idx))
        parts.append(rng.choice(cls_idx, size=n, replace=False))
    out = np.sort(np.concatenate(parts))
    if len(out) > max_rows:
        out = np.sort(rng.choice(out, size=max_rows, replace=False))
    return out


def encoded_to_attribute(feature: str) -> str:
    """Map one-hot/scaled model feature names back to the raw attribute."""
    for cat in D.CATEGORICAL_COLS:
        if feature.startswith(f"{cat}_"):
            return cat
    return feature


def raw_type(attribute: str) -> str:
    if attribute in D.CATEGORICAL_COLS:
        return "categorical"
    if attribute in D.BINARY_COLS:
        return "binary_numeric"
    return "numeric"


def _safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V without scipy."""
    table = pd.crosstab(x.astype(str), y.astype(str))
    if table.empty:
        return 0.0
    observed = table.to_numpy(dtype=float)
    n = observed.sum()
    if n == 0:
        return 0.0
    row = observed.sum(axis=1, keepdims=True)
    col = observed.sum(axis=0, keepdims=True)
    expected = row @ col / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((observed - expected) ** 2 / expected)
    phi2 = chi2 / n
    r, k = observed.shape
    if n <= 1:
        return 0.0
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0


def _numeric_class_separation(values: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    normal = values[y == 0].astype(float)
    attack = values[y == 1].astype(float)
    normal_mean = _safe_float(normal.mean())
    attack_mean = _safe_float(attack.mean())
    pooled = np.sqrt((normal.var(ddof=0) + attack.var(ddof=0)) / 2)
    if not np.isfinite(pooled) or pooled == 0:
        sep = 0.0
    else:
        sep = abs(attack_mean - normal_mean) / pooled
    return normal_mean, attack_mean, float(sep)


def _categorical_attack_rate(values: pd.Series, y: pd.Series) -> float:
    tmp = pd.DataFrame({"value": values.astype(str), "target": y.to_numpy()})
    rates = tmp.groupby("value")["target"].mean()
    return _safe_float(rates.max()) if len(rates) else 0.0


def _normalise(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").fillna(0.0)
    max_value = float(series.max())
    if max_value <= 0:
        return series * 0.0
    return series / max_value


def encoded_feature_audit(
    prep: P.PreparedData,
    max_rows_model: int = 60_000,
    max_rows_mi: int = 40_000,
) -> pd.DataFrame:
    """Compute model and statistical signals for every encoded feature."""
    model_idx = _sample_indices(prep.y_train, max_rows_model)
    mi_idx = _sample_indices(prep.y_train, max_rows_mi, seed=D.RANDOM_STATE + 1)

    logreg = LogisticRegression(class_weight="balanced", max_iter=2000)
    logreg.fit(prep.X_train[model_idx], prep.y_train[model_idx])
    logreg_importance = np.abs(logreg.coef_[0])

    rf = RandomForestClassifier(
        n_estimators=80,
        max_depth=18,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=D.RANDOM_STATE,
    )
    rf.fit(prep.X_train[model_idx], prep.y_train[model_idx])
    rf_importance = rf.feature_importances_

    mi = mutual_info_classif(
        prep.X_train[mi_idx],
        prep.y_train[mi_idx],
        discrete_features=False,
        random_state=D.RANDOM_STATE,
    )

    y_float = prep.y_train.astype(float)
    rows = []
    for i, feature in enumerate(prep.feature_names):
        col = prep.X_train[:, i].astype(float)
        if col.std() == 0 or y_float.std() == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(col, y_float)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        q1, q3 = np.quantile(col, [0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            outlier_rate = 0.0
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_rate = float(((col < lower) | (col > upper)).mean())
        rows.append(
            {
                "dataset": "nsl_kdd",
                "encoded_feature": feature,
                "attribute": encoded_to_attribute(feature),
                "target_corr": corr,
                "abs_target_corr": abs(corr),
                "outlier_rate_iqr": outlier_rate,
                "logreg_importance": float(logreg_importance[i]),
                "rf_importance": float(rf_importance[i]),
                "mutual_info": float(mi[i]),
            }
        )
    out = pd.DataFrame(rows)
    out["importance_consensus"] = (
        _normalise(out["logreg_importance"])
        + _normalise(out["rf_importance"])
        + _normalise(out["mutual_info"])
    ) / 3.0
    return out.sort_values("importance_consensus", ascending=False)


def high_corr_redundancy(prep: P.PreparedData, threshold: float = 0.95, max_rows: int = 50_000) -> pd.DataFrame:
    idx = _sample_indices(np.zeros(len(prep.X_train), dtype=int), max_rows)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(prep.X_train[idx], rowvar=False)
    corr = np.nan_to_num(corr)
    rows = []
    for i in range(len(prep.feature_names)):
        for j in range(i + 1, len(prep.feature_names)):
            value = float(corr[i, j])
            if abs(value) >= threshold:
                rows.append(
                    {
                        "feature_a": prep.feature_names[i],
                        "feature_b": prep.feature_names[j],
                        "attribute_a": encoded_to_attribute(prep.feature_names[i]),
                        "attribute_b": encoded_to_attribute(prep.feature_names[j]),
                        "corr": value,
                    }
                )
    return pd.DataFrame(rows)


def attribute_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return raw-attribute and encoded-feature audit tables for NSL-KDD."""
    train = D.load_nsl_kdd("train")
    test = D.load_nsl_kdd("test")
    prep = P.prepare_nsl_kdd("binary")
    encoded = encoded_feature_audit(prep)
    redundancy = high_corr_redundancy(prep)

    encoded_groups = encoded.groupby("attribute")
    if redundancy.empty:
        redundancy_counts = pd.Series(dtype=int)
    else:
        redundancy_counts = pd.concat([redundancy["attribute_a"], redundancy["attribute_b"]]).value_counts()

    rows = []
    y = train["binary_label"]
    for attr in D.FEATURE_NAMES:
        kind = raw_type(attr)
        train_col = train[attr]
        test_col = test[attr]
        encoded_part = encoded_groups.get_group(attr) if attr in encoded_groups.groups else pd.DataFrame()

        train_missing = float(train_col.isna().mean())
        test_missing = float(test_col.isna().mean())
        train_unique = int(train_col.nunique(dropna=True))
        test_unique = int(test_col.nunique(dropna=True))
        unseen = sorted(set(test_col.dropna().astype(str)) - set(train_col.dropna().astype(str)))

        train_mean = test_mean = train_std = test_std = shift = np.nan
        normal_mean = attack_mean = class_sep = np.nan
        max_attack_rate = np.nan

        if kind == "categorical":
            class_sep = _cramers_v(train_col, y)
            max_attack_rate = _categorical_attack_rate(train_col, y)
        else:
            train_values = train_col.astype(float)
            test_values = test_col.astype(float)
            train_mean = _safe_float(train_values.mean())
            test_mean = _safe_float(test_values.mean())
            train_std = _safe_float(train_values.std(ddof=0))
            test_std = _safe_float(test_values.std(ddof=0))
            if train_std and np.isfinite(train_std):
                shift = abs(test_mean - train_mean) / train_std
            else:
                shift = 0.0
            normal_mean, attack_mean, class_sep = _numeric_class_separation(train_values, y)

        rows.append(
            {
                "dataset": "nsl_kdd",
                "attribute": attr,
                "raw_type": kind,
                "encoded_feature_count": int(len(encoded_part)),
                "train_missing_rate": train_missing,
                "test_missing_rate": test_missing,
                "train_unique": train_unique,
                "test_unique": test_unique,
                "unseen_test_categories": len(unseen) if kind == "categorical" else 0,
                "train_mean": train_mean,
                "test_mean": test_mean,
                "train_std": train_std,
                "test_std": test_std,
                "train_test_shift_std": shift,
                "normal_mean": normal_mean,
                "attack_mean": attack_mean,
                "class_separation": class_sep,
                "max_category_attack_rate": max_attack_rate,
                "target_corr_max_abs": _safe_float(encoded_part["abs_target_corr"].max()) if len(encoded_part) else 0.0,
                "target_corr_mean_abs": _safe_float(encoded_part["abs_target_corr"].mean()) if len(encoded_part) else 0.0,
                "outlier_rate_max": _safe_float(encoded_part["outlier_rate_iqr"].max()) if len(encoded_part) else 0.0,
                "outlier_rate_mean": _safe_float(encoded_part["outlier_rate_iqr"].mean()) if len(encoded_part) else 0.0,
                "redundant_pair_count": int(redundancy_counts.get(attr, 0)),
                "logreg_importance_sum": _safe_float(encoded_part["logreg_importance"].sum()) if len(encoded_part) else 0.0,
                "logreg_importance_max": _safe_float(encoded_part["logreg_importance"].max()) if len(encoded_part) else 0.0,
                "rf_importance_sum": _safe_float(encoded_part["rf_importance"].sum()) if len(encoded_part) else 0.0,
                "rf_importance_max": _safe_float(encoded_part["rf_importance"].max()) if len(encoded_part) else 0.0,
                "mutual_info_sum": _safe_float(encoded_part["mutual_info"].sum()) if len(encoded_part) else 0.0,
                "mutual_info_max": _safe_float(encoded_part["mutual_info"].max()) if len(encoded_part) else 0.0,
                "importance_consensus": 0.0,
            }
        )

    attr_df = pd.DataFrame(rows)
    attr_df["importance_consensus"] = (
        _normalise(attr_df["logreg_importance_sum"])
        + _normalise(attr_df["rf_importance_sum"])
        + _normalise(attr_df["mutual_info_sum"])
    ) / 3.0
    attr_df["rank_by_consensus"] = attr_df["importance_consensus"].rank(
        method="min",
        ascending=False,
    ).astype(int)
    attr_df["audit_flags"] = attr_df.apply(attribute_flags, axis=1)
    attr_df["engineering_recommendation"] = attr_df.apply(attribute_recommendation, axis=1)
    attr_df = attr_df[ATTR_COLUMNS].sort_values("rank_by_consensus")
    return attr_df, encoded


def attribute_flags(row: pd.Series) -> str:
    flags = []
    if row["train_missing_rate"] > 0 or row["test_missing_rate"] > 0:
        flags.append("missing_values")
    if row["raw_type"] == "categorical" and row["unseen_test_categories"] > 0:
        flags.append("unseen_test_categories")
    if row["train_unique"] <= 1:
        flags.append("constant_or_near_constant")
    if row["train_test_shift_std"] >= 1:
        flags.append("large_train_test_shift")
    if row["outlier_rate_max"] >= 0.10:
        flags.append("many_iqr_outliers")
    if row["redundant_pair_count"] >= 2:
        flags.append("highly_redundant")
    if row["target_corr_max_abs"] >= 0.5 or row["class_separation"] >= 0.8:
        flags.append("strong_target_signal")
    if not flags:
        flags.append("ordinary")
    return "; ".join(flags)


def attribute_recommendation(row: pd.Series) -> str:
    if "constant_or_near_constant" in row["audit_flags"]:
        return "Drop or keep only for schema compatibility; it adds no learning signal here."
    if row["raw_type"] == "categorical":
        if row["unseen_test_categories"] > 0:
            return "Keep one-hot with handle_unknown='ignore'; inspect unseen categories before deployment."
        return "Keep one-hot encoded; compare category-level attack rates before collapsing."
    if "many_iqr_outliers" in row["audit_flags"] and row["importance_consensus"] > 0.05:
        return "Keep, but prefer robust scaling/winsorization ablation because extreme values may carry signal."
    if "highly_redundant" in row["audit_flags"]:
        return "Keep for tree baselines; test feature selection or regularization to reduce redundancy."
    if row["importance_consensus"] <= 0.01 and row["target_corr_max_abs"] <= 0.05:
        return "Candidate for pruning in a compact model, but verify with ablation before removing."
    return "Keep in the baseline; use ablation/permutation importance before changing it."


def _format(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format(row[col]) for col in columns) + " |")
    lines.append("")
    return lines


def render_report(attr: pd.DataFrame, encoded: pd.DataFrame) -> str:
    key_cols = [
        "attribute",
        "raw_type",
        "encoded_feature_count",
        "class_separation",
        "target_corr_max_abs",
        "outlier_rate_max",
        "redundant_pair_count",
        "importance_consensus",
        "rank_by_consensus",
        "audit_flags",
        "engineering_recommendation",
    ]
    drift_cols = [
        "attribute",
        "raw_type",
        "train_unique",
        "test_unique",
        "unseen_test_categories",
        "train_test_shift_std",
        "train_missing_rate",
        "test_missing_rate",
        "audit_flags",
    ]
    encoded_cols = [
        "encoded_feature",
        "attribute",
        "abs_target_corr",
        "outlier_rate_iqr",
        "logreg_importance",
        "rf_importance",
        "mutual_info",
        "importance_consensus",
    ]
    lines = [
        "# NSL-KDD Attribute-By-Attribute Comparison",
        "",
        "This is not a model leaderboard. It is a forensic table over the 41 raw "
        "NSL-KDD attributes and the 122 encoded model features.",
        "",
        "## What Each Row Means",
        "",
        "- **class_separation**: numeric standardized normal-vs-attack mean gap; categorical Cramer's V.",
        "- **target_corr_max_abs**: strongest absolute encoded-feature correlation with the binary attack target.",
        "- **outlier_rate_max**: largest IQR outlier rate among the encoded children of that attribute.",
        "- **redundant_pair_count**: number of high-correlation encoded pairs touching the attribute.",
        "- **importance_consensus**: average-normalized signal from Logistic Regression, Random Forest, and mutual information.",
        "",
        "## All Raw Attributes Ranked By Consensus Importance",
        "",
    ]
    lines += _markdown_table(attr[key_cols], key_cols)
    lines += [
        "## Distribution And Drift Checks",
        "",
    ]
    lines += _markdown_table(attr.sort_values("train_test_shift_std", ascending=False)[drift_cols], drift_cols)
    lines += [
        "## Top Encoded Features",
        "",
    ]
    lines += _markdown_table(encoded.head(40)[encoded_cols], encoded_cols)
    lines += [
        "## Referee Notes",
        "",
        "- A high rank does not mean the feature is safe for deployment; it means the current benchmark uses it strongly.",
        "- Categorical attributes are expanded into many one-hot columns, so their consensus score is aggregated back to the raw attribute.",
        "- Outlier-heavy traffic features should not be deleted automatically; rare extreme flows can be the attack signal.",
        "- Highly redundant attributes are prime candidates for compact-model ablations, not automatic removal.",
        "",
    ]
    return "\n".join(lines)


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    attr, encoded = attribute_audit()
    attr.to_csv(ATTR_CSV, index=False)
    encoded.to_csv(ENCODED_CSV, index=False)
    REPORT_PATH.write_text(render_report(attr, encoded), encoding="utf-8")
    return attr, encoded


def main() -> int:
    attr, encoded = run()
    print(f"raw attributes: {len(attr)}")
    print(f"encoded features: {len(encoded)}")
    print("top raw attributes")
    print(attr.head(15)[["attribute", "raw_type", "importance_consensus", "audit_flags"]].to_string(index=False))
    print(f"Wrote {ATTR_CSV}")
    print(f"Wrote {ENCODED_CSV}")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
