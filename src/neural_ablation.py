"""Controlled neural ablations on NSL-KDD.

This is the first reproducible neural architecture-ablation phase. It uses a
bounded training subset for speed, keeps the official test split untouched, and
changes one major factor at a time around a small MLP baseline.

Run:
    .venv/bin/python src/neural_ablation.py
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os
import time

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, matthews_corrcoef

import data as D
import evaluate as E
import preprocess as P
import tracking

RESULTS = D.REPO_ROOT / "results"
FIGURES = RESULTS / "figures"
RESULTS_CSV = RESULTS / "neural_ablation.csv"
CURVES_CSV = RESULTS / "neural_ablation_curves.csv"
REPORT_PATH = RESULTS / "neural_ablation.md"


@dataclass(frozen=True)
class NeuralConfig:
    """One controlled neural ablation config."""

    name: str
    hidden: tuple[int, ...] = (64,)
    activation: str = "relu"
    dropout: float = 0.0
    normalization: str = "none"
    weighted_loss: bool = False
    focal_loss: bool = False
    label_smoothing: float = 0.0
    lr: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 14
    patience: int = 4


CONFIGS = [
    NeuralConfig(name="baseline_relu_1x64"),
    NeuralConfig(name="activation_tanh", activation="tanh"),
    NeuralConfig(name="activation_gelu", activation="gelu"),
    NeuralConfig(name="dropout_0.30", dropout=0.30),
    NeuralConfig(name="batchnorm", normalization="batchnorm"),
    NeuralConfig(name="deeper_2x128_64", hidden=(128, 64)),
    NeuralConfig(name="weighted_cross_entropy", weighted_loss=True),
    NeuralConfig(name="focal_loss", focal_loss=True),
    NeuralConfig(name="label_smoothing_0.05", label_smoothing=0.05),
]


def set_seed(seed: int = D.RANDOM_STATE) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def activation_layer(name: str) -> torch.nn.Module:
    if name == "relu":
        return torch.nn.ReLU()
    if name == "tanh":
        return torch.nn.Tanh()
    if name == "gelu":
        return torch.nn.GELU()
    if name == "leaky_relu":
        return torch.nn.LeakyReLU(0.01)
    raise ValueError(f"unknown activation {name!r}")


class AblationMLP(torch.nn.Module):
    """Configurable MLP for controlled ablations."""

    def __init__(self, in_dim: int, n_classes: int, config: NeuralConfig):
        super().__init__()
        layers: list[torch.nn.Module] = []
        prev = in_dim
        for width in config.hidden:
            layers.append(torch.nn.Linear(prev, width))
            if config.normalization == "batchnorm":
                layers.append(torch.nn.BatchNorm1d(width))
            elif config.normalization == "layernorm":
                layers.append(torch.nn.LayerNorm(width))
            layers.append(activation_layer(config.activation))
            if config.dropout > 0:
                layers.append(torch.nn.Dropout(config.dropout))
            prev = width
        layers.append(torch.nn.Linear(prev, n_classes))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FocalLoss(torch.nn.Module):
    """Multiclass focal loss for class-imbalance ablation."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = torch.nn.functional.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes)
    return (len(y) / (n_classes * np.clip(counts, 1, None))).astype(np.float32)


def stratified_subset(y: np.ndarray, max_rows: int, seed: int = D.RANDOM_STATE) -> np.ndarray:
    if len(y) <= max_rows:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    pieces = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        n = max(1, int(round(max_rows * len(idx) / len(y))))
        pieces.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    out = np.sort(np.concatenate(pieces))
    if len(out) > max_rows:
        out = np.sort(rng.choice(out, size=max_rows, replace=False))
    return out


def train_val_indices(y: np.ndarray, val_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Simple deterministic stratified train/validation split."""
    rng = np.random.default_rng(D.RANDOM_STATE)
    train_parts = []
    val_parts = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(val_parts))


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
    ds = torch.utils.data.TensorDataset(
        torch.as_tensor(X, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(D.RANDOM_STATE)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train_one(
    config: NeuralConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
) -> tuple[AblationMLP, list[dict[str, float]], float, int]:
    set_seed()
    device = torch.device("cpu")
    model = AblationMLP(X_train.shape[1], n_classes, config).to(device)
    weight = None
    if config.weighted_loss or config.focal_loss:
        weight = torch.as_tensor(class_weights(y_train, n_classes), dtype=torch.float32).to(device)
    if config.focal_loss:
        criterion: torch.nn.Module = FocalLoss(weight=weight)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=config.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    loader = make_loader(X_train, y_train, config.batch_size, shuffle=True)
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32).to(device)

    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    curves: list[dict[str, float]] = []
    start_time = time.perf_counter()

    for epoch in range(config.max_epochs):
        model.train()
        losses = []
        grad_norms = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            grad_norms.append(float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm))

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_pred = val_logits.argmax(1).cpu().numpy()
            val_loss = float(criterion(val_logits, torch.as_tensor(y_val, dtype=torch.long)).detach().cpu())
        val_f1 = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
        scheduler.step(val_f1)
        lr = float(optimizer.param_groups[0]["lr"])
        curves.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)),
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
                "grad_norm": float(np.mean(grad_norms)),
                "lr": lr,
            }
        )
        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, curves, time.perf_counter() - start_time, best_epoch


def predict(model: torch.nn.Module, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X, dtype=torch.float32))
        proba = torch.softmax(logits, dim=1).numpy()
    return proba.argmax(1), proba, time.perf_counter() - start


def plot_curves(curves: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for metric in ("train_loss", "val_loss", "val_macro_f1", "grad_norm"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, group in curves.groupby("config"):
            ax.plot(group["epoch"], group[metric], label=name)
        ax.set_title(f"Neural ablation: {metric}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGURES / f"neural_ablation_{metric}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def run(max_train_rows: int = 30_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    prep = P.prepare_nsl_kdd("binary")
    subset = stratified_subset(prep.y_train, max_train_rows)
    X_sub = prep.X_train[subset].astype(np.float32)
    y_sub = prep.y_train[subset]
    tr_idx, val_idx = train_val_indices(y_sub)
    X_tr, X_val = X_sub[tr_idx], X_sub[val_idx]
    y_tr, y_val = y_sub[tr_idx], y_sub[val_idx]

    result_rows = []
    curve_rows = []
    for config in CONFIGS:
        model, curves, train_seconds, best_epoch = train_one(
            config,
            X_tr,
            y_tr,
            X_val,
            y_val,
            n_classes=len(prep.classes),
        )
        y_pred, proba, inference_seconds = predict(model, prep.X_test.astype(np.float32))
        metrics = E.compute_metrics(prep.y_test, y_pred, prep.classes, y_score=proba[:, 1])
        row = {
            "dataset": "nsl_kdd",
            "task": "binary",
            "config": config.name,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "mcc": float(matthews_corrcoef(prep.y_test, y_pred)),
            "normal_recall": metrics["per_class"]["normal"]["recall"],
            "attack_recall": metrics["per_class"]["attack"]["recall"],
            "param_count": count_parameters(model),
            "best_epoch": best_epoch,
            "epochs_ran": len(curves),
            "train_seconds": train_seconds,
            "inference_seconds": inference_seconds,
            **asdict(config),
        }
        result_rows.append(row)
        for curve in curves:
            curve_rows.append({"config": config.name, **curve})

        run_id = tracking.stable_experiment_id("neural_ablation", config.name, D.RANDOM_STATE)
        tracking.append_run(
            tracking.ExperimentRun(
                experiment_id=run_id,
                dataset="nsl_kdd",
                task="binary",
                model_family="MLP",
                run_name=config.name,
                seed=D.RANDOM_STATE,
                status="complete",
                metrics={
                    "macro_f1": row["macro_f1"],
                    "attack_recall": row["attack_recall"],
                    "mcc": row["mcc"],
                },
                params=asdict(config),
                artifacts={
                    "results_csv": str(RESULTS_CSV.relative_to(D.REPO_ROOT)),
                    "curves_csv": str(CURVES_CSV.relative_to(D.REPO_ROOT)),
                },
                notes="Bounded NSL-KDD train subset; official KDDTest+ final evaluation.",
            ),
            filename="neural_ablation.jsonl",
        )

    results = pd.DataFrame(result_rows)
    curves = pd.DataFrame(curve_rows)
    results.to_csv(RESULTS_CSV, index=False)
    curves.to_csv(CURVES_CSV, index=False)
    plot_curves(curves)
    REPORT_PATH.write_text(render_markdown(results), encoding="utf-8")
    return results, curves


def render_markdown(results: pd.DataFrame) -> str:
    lines = [
        "# NSL-KDD — Controlled Neural Ablations",
        "",
        "This phase changes one major neural-training factor at a time around a small "
        "MLP baseline. Training uses a bounded stratified subset of the training set; "
        "the official KDDTest+ split is evaluated once per frozen configuration.",
        "",
        "| Config | Activation | Hidden | Dropout | Norm | Loss variant | Params | Epochs | Macro-F1 | Attack recall | MCC | Train sec |",
        "| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in results.sort_values("macro_f1", ascending=False).iterrows():
        loss = "focal" if r["focal_loss"] else ("weighted CE" if r["weighted_loss"] else "CE")
        if r["label_smoothing"] > 0:
            loss += f" + smooth {r['label_smoothing']:.2f}"
        lines.append(
            f"| {r['config']} | {r['activation']} | {r['hidden']} | {r['dropout']:.2f} | "
            f"{r['normalization']} | {loss} | {int(r['param_count']):,} | "
            f"{int(r['epochs_ran'])} | {r['macro_f1']:.4f} | {r['attack_recall']:.4f} | "
            f"{r['mcc']:.4f} | {r['train_seconds']:.2f} |"
        )
    lines += [
        "",
        "## Interpretation Guardrails",
        "",
        "- These are bounded ablations, not final neural-network rankings.",
        "- If a config improves validation or test metrics by a tiny amount, seed stability is still required.",
        "- CNN/RNN/LSTM/GRU are not used here because NSL-KDD rows have no valid temporal or spatial locality.",
        "- Future CNN/RNN work needs packet sequences, flow windows, host timelines, or another justified representation.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    results, _ = run()
    print(results.sort_values("macro_f1", ascending=False).to_string(index=False))
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
