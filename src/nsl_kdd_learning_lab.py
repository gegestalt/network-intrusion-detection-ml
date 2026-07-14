"""First-dataset learning laboratory for NSL-KDD.

This script applies the requested learning roadmap directly to the initial
dataset. It is deliberately first-dataset-only: every row is NSL-KDD, using
KDDTrain+ for development and KDDTest+ for final evaluation.

Run:
    .venv/bin/python src/nsl_kdd_learning_lab.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM
from sklearn.tree import DecisionTreeClassifier

import data as D
import preprocess as P

RESULTS = D.REPO_ROOT / "results"
LAB_CSV = RESULTS / "nsl_kdd_learning_lab.csv"
GROUP_CSV = RESULTS / "nsl_kdd_feature_group_ablation.csv"
BOOTSTRAP_CSV = RESULTS / "nsl_kdd_bootstrap_ci.csv"
MATRIX_CSV = RESULTS / "nsl_kdd_concept_application_matrix.csv"
REPORT_PATH = RESULTS / "nsl_kdd_learning_lab.md"


@dataclass(frozen=True)
class SplitBundle:
    X_dev: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_dev: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    classes: list[str]


def _binary_labels(df: pd.DataFrame) -> pd.Series:
    return df["binary_label"].map({0: "normal", 1: "attack"})


def load_split_bundle(val_size: float = 0.20) -> SplitBundle:
    """Fit preprocessing on an inner development split, never on validation/test."""
    raw_train = D.load_nsl_kdd("train")
    raw_test = D.load_nsl_kdd("test")
    y_train = raw_train["binary_label"].to_numpy()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=D.RANDOM_STATE)
    dev_idx, val_idx = next(splitter.split(raw_train[D.FEATURE_NAMES], y_train))
    dev = raw_train.iloc[dev_idx].reset_index(drop=True)
    val = raw_train.iloc[val_idx].reset_index(drop=True)

    pre = P.build_preprocessor(D.CATEGORICAL_COLS, D.NUMERIC_COLS)
    X_dev = pre.fit_transform(dev)
    X_val = pre.transform(val)
    X_test = pre.transform(raw_test)
    return SplitBundle(
        X_dev=X_dev,
        X_val=X_val,
        X_test=X_test,
        y_dev=dev["binary_label"].to_numpy(dtype=np.int64),
        y_val=val["binary_label"].to_numpy(dtype=np.int64),
        y_test=raw_test["binary_label"].to_numpy(dtype=np.int64),
        feature_names=list(pre.get_feature_names_out()),
        classes=["normal", "attack"],
    )


def stratified_cap(y: np.ndarray, max_rows: int, seed: int = D.RANDOM_STATE) -> np.ndarray:
    if len(y) <= max_rows:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    parts = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        n = max(1, int(round(max_rows * len(cls_idx) / len(y))))
        parts.append(rng.choice(cls_idx, size=min(n, len(cls_idx)), replace=False))
    out = np.sort(np.concatenate(parts))
    if len(out) > max_rows:
        out = np.sort(rng.choice(out, size=max_rows, replace=False))
    return out


def metrics_row(
    *,
    level: str,
    family: str,
    method: str,
    variant: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_seconds: float,
    notes: str,
) -> dict[str, object]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    false_positive_rate = fp / max(fp + tn, 1)
    false_negative_rate = fn / max(fn + tp, 1)
    return {
        "dataset": "nsl_kdd",
        "level": level,
        "family": family,
        "method": method,
        "variant": variant,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_attack": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "normal_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "attack_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "fp_per_10k_benign": float(false_positive_rate * 10_000),
        "fn_per_10k_attack": float(false_negative_rate * 10_000),
        "train_seconds": float(train_seconds),
        "notes": notes,
    }


def positive_scores(model: object, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        score = model.decision_function(X)
        score = np.asarray(score, dtype=float)
        return (score - score.min()) / max(float(score.max() - score.min()), 1e-12)
    pred = model.predict(X)
    return np.asarray(pred, dtype=float)


def tune_threshold(y_val: np.ndarray, scores: np.ndarray, beta: float = 1.0) -> tuple[float, float]:
    best_t = 0.5
    best_score = -1.0
    beta2 = beta * beta
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = (scores >= threshold).astype(int)
        tp = float(((pred == 1) & (y_val == 1)).sum())
        fp = float(((pred == 1) & (y_val == 0)).sum())
        fn = float(((pred == 0) & (y_val == 1)).sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        score = (1 + beta2) * precision * recall / max((beta2 * precision) + recall, 1e-12)
        if score > best_score:
            best_score = score
            best_t = float(threshold)
    return best_t, best_score


def run_model_lab(bundle: SplitBundle) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    train_idx = stratified_cap(bundle.y_dev, 70_000)
    X_fit = bundle.X_dev[train_idx]
    y_fit = bundle.y_dev[train_idx]

    model_specs = [
        ("Level 0", "naive_baseline", "DummyClassifier", "most_frequent", DummyClassifier(strategy="most_frequent"), "lowest meaningful benchmark"),
        ("Level 0", "naive_baseline", "DummyClassifier", "stratified", DummyClassifier(strategy="stratified", random_state=D.RANDOM_STATE), "random label baseline preserving class balance"),
        ("Level 1", "logistic_regression", "LogReg", "l2_C1_unweighted", LogisticRegression(max_iter=2000, C=1.0), "transparent linear classifier"),
        ("Level 1", "logistic_regression", "LogReg", "l2_C1_balanced", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"), "class weighting changes false-negative pressure"),
        ("Level 1", "logistic_regression", "LogReg", "l2_C0.1_balanced", LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced"), "stronger L2 regularization"),
        ("Level 1", "logistic_regression", "LogReg", "l1_C0.1_balanced", LogisticRegression(max_iter=2000, C=0.1, penalty="l1", solver="liblinear", class_weight="balanced"), "sparse coefficient baseline"),
        ("Level 1", "logistic_regression", "LogReg", "elasticnet_balanced", LogisticRegression(max_iter=2000, C=0.1, penalty="elasticnet", solver="saga", l1_ratio=0.5, class_weight="balanced"), "mixed L1/L2 regularization"),
        ("Level 4", "tree_family", "DecisionTree", "balanced_depth18", DecisionTreeClassifier(max_depth=18, min_samples_leaf=3, class_weight="balanced", random_state=D.RANDOM_STATE), "single interpretable tree"),
        ("Level 4", "tree_family", "RandomForest", "balanced_depth18", RandomForestClassifier(n_estimators=120, max_depth=18, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=D.RANDOM_STATE), "bagged tree baseline"),
        ("Level 4", "tree_family", "ExtraTrees", "balanced_depth18", ExtraTreesClassifier(n_estimators=120, max_depth=18, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=D.RANDOM_STATE), "more randomized ensemble"),
        ("Level 4", "boosting", "HistGradientBoosting", "default", HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=D.RANDOM_STATE), "sklearn histogram boosting baseline"),
        ("Level 4", "linear_online", "SGDClassifier", "log_loss_balanced", SGDClassifier(loss="log_loss", class_weight="balanced", alpha=1e-4, max_iter=1500, random_state=D.RANDOM_STATE), "linear online-capable classifier"),
        ("Level 4", "instance_based", "KNN", "k5_train_cap", KNeighborsClassifier(n_neighbors=5), "local-neighbour baseline capped for runtime"),
    ]

    for level, family, method, variant, model, notes in model_specs:
        start = time.perf_counter()
        model.fit(X_fit, y_fit)
        elapsed = time.perf_counter() - start
        pred = model.predict(bundle.X_test)
        rows.append(
            metrics_row(
                level=level,
                family=family,
                method=method,
                variant=variant,
                y_true=bundle.y_test,
                y_pred=pred,
                train_seconds=elapsed,
                notes=notes,
            )
        )

        if method == "LogReg" and variant == "l2_C1_balanced":
            val_scores = positive_scores(model, bundle.X_val)
            test_scores = positive_scores(model, bundle.X_test)
            for name, beta in (("validation_F1_threshold", 1.0), ("validation_F2_attack_weighted", 2.0)):
                threshold, _ = tune_threshold(bundle.y_val, val_scores, beta=beta)
                threshold_pred = (test_scores >= threshold).astype(int)
                rows.append(
                    metrics_row(
                        level="Level 1",
                        family="threshold_policy",
                        method="LogReg",
                        variant=f"{name}_{threshold:.2f}",
                        y_true=bundle.y_test,
                        y_pred=threshold_pred,
                        train_seconds=elapsed,
                        notes="threshold selected on validation split only",
                    )
                )

    return pd.DataFrame(rows)


def feature_group_masks(feature_names: list[str]) -> dict[str, list[int]]:
    groups = {
        "all_features": list(range(len(feature_names))),
        "categorical_one_hot": [],
        "numeric_only": [],
        "protocol_service_flag": [],
        "byte_volume": [],
        "connection_count_rates": [],
        "dst_host_behavior": [],
        "content_login_shell": [],
    }
    content = {
        "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell",
        "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    }
    for i, name in enumerate(feature_names):
        raw = name
        for cat in D.CATEGORICAL_COLS:
            if name.startswith(f"{cat}_"):
                raw = cat
                groups["categorical_one_hot"].append(i)
                groups["protocol_service_flag"].append(i)
                break
        else:
            groups["numeric_only"].append(i)
        if raw in {"src_bytes", "dst_bytes"}:
            groups["byte_volume"].append(i)
        if raw in content:
            groups["content_login_shell"].append(i)
        if raw.startswith("dst_host"):
            groups["dst_host_behavior"].append(i)
        if raw in {
            "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
            "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        }:
            groups["connection_count_rates"].append(i)
    return {name: idx for name, idx in groups.items() if idx}


def run_feature_group_ablation(bundle: SplitBundle) -> pd.DataFrame:
    rows = []
    train_idx = stratified_cap(bundle.y_dev, 70_000, seed=D.RANDOM_STATE + 2)
    for group, idx in feature_group_masks(bundle.feature_names).items():
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        start = time.perf_counter()
        model.fit(bundle.X_dev[train_idx][:, idx], bundle.y_dev[train_idx])
        elapsed = time.perf_counter() - start
        pred = model.predict(bundle.X_test[:, idx])
        rows.append(
            {
                **metrics_row(
                    level="Feature groups",
                    family="feature_ablation",
                    method="LogReg",
                    variant=group,
                    y_true=bundle.y_test,
                    y_pred=pred,
                    train_seconds=elapsed,
                    notes=f"{len(idx)} encoded features",
                ),
                "encoded_feature_count": len(idx),
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False)


class SmallAutoEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16) -> None:
        super().__init__()
        hidden = min(96, max(24, input_dim // 2))
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, latent_dim),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def autoencoder_anomaly(bundle: SplitBundle) -> dict[str, object]:
    torch.manual_seed(D.RANDOM_STATE)
    normal_idx = np.flatnonzero(bundle.y_dev == 0)
    normal_idx = normal_idx[: min(18_000, len(normal_idx))]
    X_normal = bundle.X_dev[normal_idx].astype(np.float32)
    model = SmallAutoEncoder(bundle.X_dev.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    tensor = torch.from_numpy(X_normal)
    start = time.perf_counter()
    for _ in range(6):
        order = torch.randperm(len(tensor))
        for batch_start in range(0, len(tensor), 512):
            batch = tensor[order[batch_start:batch_start + 512]]
            opt.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            opt.step()
    elapsed = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        val_recon = model(torch.from_numpy(bundle.X_val.astype(np.float32))).numpy()
        test_recon = model(torch.from_numpy(bundle.X_test.astype(np.float32))).numpy()
    val_err = ((bundle.X_val - val_recon) ** 2).mean(axis=1)
    test_err = ((bundle.X_test - test_recon) ** 2).mean(axis=1)
    threshold = float(np.quantile(val_err[bundle.y_val == 0], 0.95))
    pred = (test_err >= threshold).astype(int)
    row = metrics_row(
        level="Security ML",
        family="benign_only_anomaly",
        method="AutoEncoder",
        variant="normal_only_q0.95",
        y_true=bundle.y_test,
        y_pred=pred,
        train_seconds=elapsed,
        notes="autoencoder trained only on normal development rows; threshold from normal validation errors",
    )
    row["threshold"] = threshold
    return row


def one_class_svm_anomaly(bundle: SplitBundle) -> dict[str, object]:
    normal_idx = np.flatnonzero(bundle.y_dev == 0)
    normal_idx = normal_idx[: min(5_000, len(normal_idx))]
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    start = time.perf_counter()
    model.fit(bundle.X_dev[normal_idx])
    elapsed = time.perf_counter() - start
    pred = (model.predict(bundle.X_test) == -1).astype(int)
    return metrics_row(
        level="Security ML",
        family="benign_only_anomaly",
        method="OneClassSVM",
        variant="normal_only_nu0.05",
        y_true=bundle.y_test,
        y_pred=pred,
        train_seconds=elapsed,
        notes="one-class boundary trained only on normal development rows; capped for runtime",
    )


def bootstrap_ci(bundle: SplitBundle, models: dict[str, object], n_boot: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(D.RANDOM_STATE)
    rows = []
    n = len(bundle.y_test)
    for name, model in models.items():
        pred = model.predict(bundle.X_test)
        macro_values = []
        attack_values = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            macro_values.append(f1_score(bundle.y_test[idx], pred[idx], average="macro", zero_division=0))
            attack_values.append(recall_score(bundle.y_test[idx], pred[idx], pos_label=1, zero_division=0))
        rows.append(
            {
                "dataset": "nsl_kdd",
                "model": name,
                "metric": "macro_f1",
                "mean": float(np.mean(macro_values)),
                "ci_low": float(np.quantile(macro_values, 0.025)),
                "ci_high": float(np.quantile(macro_values, 0.975)),
                "n_bootstrap": n_boot,
            }
        )
        rows.append(
            {
                "dataset": "nsl_kdd",
                "model": name,
                "metric": "attack_recall",
                "mean": float(np.mean(attack_values)),
                "ci_low": float(np.quantile(attack_values, 0.025)),
                "ci_high": float(np.quantile(attack_values, 0.975)),
                "n_bootstrap": n_boot,
            }
        )
    return pd.DataFrame(rows)


def concept_matrix() -> pd.DataFrame:
    rows = [
        ("Level 0 data validation", "APPLIED", "src/data.py, notebooks/04, results/attribute_comparison_nsl_kdd.csv", "schema, labels, class balance, missing/unique/constant checks"),
        ("Naive baselines", "APPLIED", "results/nsl_kdd_learning_lab.csv", "most-frequent and stratified dummy baselines"),
        ("Logistic Regression variants", "APPLIED", "results/nsl_kdd_learning_lab.csv", "unweighted, balanced, L1, L2, Elastic Net, threshold variants"),
        ("Feature-group ablation", "APPLIED", "results/nsl_kdd_feature_group_ablation.csv", "numeric, categorical, byte-volume, host, connection-rate, content groups"),
        ("Regularization", "APPLIED", "results/nsl_kdd_learning_lab.csv", "C=1 vs C=0.1, L1, L2, Elastic Net"),
        ("Tree and ensemble baselines", "APPLIED", "results/nsl_kdd_learning_lab.csv", "DecisionTree, RandomForest, ExtraTrees, HistGradientBoosting"),
        ("Online-capable linear model", "APPLIED", "results/nsl_kdd_learning_lab.csv", "SGDClassifier; true drift is not claimed"),
        ("Bootstrap confidence intervals", "APPLIED", "results/nsl_kdd_bootstrap_ci.csv", "test-set bootstrap intervals for selected models"),
        ("Benign-only security anomaly", "APPLIED", "results/nsl_kdd_learning_lab.csv", "OneClassSVM and normal-only autoencoder"),
        ("Artificial neuron foundations", "APPLIED", "results/neural_foundations.md", "activation/loss/update demo"),
        ("Feed-forward ANN ablations", "APPLIED", "results/neural_ablation.md", "depth, dropout, label smoothing, gradient tracking"),
        ("CNN / pooling", "NOT_APPLIED", "docs/deep_learning_taxonomy.md", "not scientifically justified on isolated tabular rows without sequence/window representation"),
        ("Cross-dataset transfer", "NOT_APPLIED", "docs/audits/experimental_lab_prompt_audit.md", "needs another complete local dataset pipeline"),
    ]
    return pd.DataFrame(rows, columns=["concept", "status", "evidence", "first_dataset_interpretation"])


def _fit_selected_models(bundle: SplitBundle) -> dict[str, object]:
    train_idx = stratified_cap(bundle.y_dev, 70_000, seed=D.RANDOM_STATE + 3)
    models = {
        "LogReg_l2_balanced": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "ExtraTrees_balanced": ExtraTreesClassifier(n_estimators=120, max_depth=18, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=D.RANDOM_STATE),
    }
    for model in models.values():
        model.fit(bundle.X_dev[train_idx], bundle.y_dev[train_idx])
    return models


def run() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    bundle = load_split_bundle()
    lab = run_model_lab(bundle)
    anomaly = pd.DataFrame([one_class_svm_anomaly(bundle), autoencoder_anomaly(bundle)])
    lab = pd.concat([lab, anomaly], ignore_index=True)
    groups = run_feature_group_ablation(bundle)
    ci = bootstrap_ci(bundle, _fit_selected_models(bundle))
    matrix = concept_matrix()
    lab.to_csv(LAB_CSV, index=False)
    groups.to_csv(GROUP_CSV, index=False)
    ci.to_csv(BOOTSTRAP_CSV, index=False)
    matrix.to_csv(MATRIX_CSV, index=False)
    REPORT_PATH.write_text(render_report(lab, groups, ci, matrix), encoding="utf-8")
    return lab, groups, ci, matrix


def md_table(df: pd.DataFrame, columns: list[str], n: int | None = None) -> list[str]:
    show = df[columns].head(n) if n is not None else df[columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in show.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def render_report(lab: pd.DataFrame, groups: pd.DataFrame, ci: pd.DataFrame, matrix: pd.DataFrame) -> str:
    metric_cols = [
        "level",
        "family",
        "method",
        "variant",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "precision_attack",
        "attack_recall",
        "normal_recall",
        "mcc",
        "fp_per_10k_benign",
        "fn_per_10k_attack",
        "notes",
    ]
    group_cols = [
        "variant",
        "encoded_feature_count",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "precision_attack",
        "attack_recall",
        "normal_recall",
        "fp_per_10k_benign",
        "fn_per_10k_attack",
        "notes",
    ]
    ci_cols = ["model", "metric", "mean", "ci_low", "ci_high", "n_bootstrap"]
    matrix_cols = ["concept", "status", "evidence", "first_dataset_interpretation"]
    lines = [
        "# NSL-KDD First-Dataset Learning Lab",
        "",
        "This report exists to answer one question: are the requested learning methods "
        "actually applied to the first dataset? For every row below, the dataset is NSL-KDD.",
        "",
        "## Concept Application Matrix",
        "",
    ]
    lines += md_table(matrix, matrix_cols)
    lines += [
        "## Model And Training-Method Experiments",
        "",
    ]
    lines += md_table(lab.sort_values("macro_f1", ascending=False), metric_cols)
    lines += [
        "## Feature-Group Ablation",
        "",
    ]
    lines += md_table(groups, group_cols)
    lines += [
        "## Bootstrap Confidence Intervals",
        "",
    ]
    lines += md_table(ci, ci_cols)
    lines += [
        "## Referee Interpretation",
        "",
        "- CNN/pooling is explicitly not applied to NSL-KDD rows because there is no valid local/temporal structure yet.",
        "- Logistic Regression is not a single baseline here; it is varied by weighting, penalty, regularization strength, and threshold policy.",
        "- The anomaly rows are security-style normal-only experiments, not ordinary supervised classifiers.",
        "- Feature-group rows show what happens when the first dataset is deliberately restricted to different information families.",
        "- The lab uses an inner development split from KDDTrain+ for fitting/tuning, then KDDTest+ once for evaluation; encoded feature counts can differ slightly from the full-train preprocessor.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    lab, groups, ci, matrix = run()
    print(f"lab rows: {len(lab)}")
    print(f"feature-group rows: {len(groups)}")
    print(f"bootstrap rows: {len(ci)}")
    print(f"concept rows: {len(matrix)}")
    print(lab.sort_values("macro_f1", ascending=False).head(10).to_string(index=False))
    print(f"Wrote {LAB_CSV}")
    print(f"Wrote {GROUP_CSV}")
    print(f"Wrote {BOOTSTRAP_CSV}")
    print(f"Wrote {MATRIX_CSV}")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
