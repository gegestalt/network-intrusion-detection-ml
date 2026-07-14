"""Artificial-neuron and activation-function foundations.

This phase exists because the project should not jump straight to neural
architectures as magic boxes. It demonstrates one neuron's weighted input,
activation, loss, gradient, and parameter update, then records common activation
functions and derivatives.

Run:
    .venv/bin/python src/neural_foundations.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data as D

RESULTS = D.REPO_ROOT / "results"
FIGURES = RESULTS / "figures"
ACTIVATION_CSV = RESULTS / "activation_functions.csv"
NEURON_CSV = RESULTS / "single_neuron_demo.csv"
REPORT_PATH = RESULTS / "neural_foundations.md"
FIG_PATH = FIGURES / "neural_foundations_activation_functions.png"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def activation_values(x: np.ndarray) -> pd.DataFrame:
    """Return activation values and derivatives for common functions."""
    sig = sigmoid(x)
    tanh = np.tanh(x)
    relu = np.maximum(0.0, x)
    leaky = np.where(x > 0, x, 0.01 * x)
    elu = np.where(x > 0, x, np.exp(x) - 1.0)
    # GELU tanh approximation used by many implementations.
    gelu = 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    return pd.DataFrame(
        {
            "x": x,
            "linear": x,
            "linear_derivative": np.ones_like(x),
            "sigmoid": sig,
            "sigmoid_derivative": sig * (1 - sig),
            "tanh": tanh,
            "tanh_derivative": 1 - tanh**2,
            "relu": relu,
            "relu_derivative": (x > 0).astype(float),
            "leaky_relu": leaky,
            "leaky_relu_derivative": np.where(x > 0, 1.0, 0.01),
            "elu": elu,
            "elu_derivative": np.where(x > 0, 1.0, np.exp(x)),
            "gelu": gelu,
            "gelu_derivative_numeric": np.gradient(gelu, x),
        }
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def single_neuron_step(
    x: np.ndarray | None = None,
    w: np.ndarray | None = None,
    bias: float = -0.1,
    y_true: float = 1.0,
    lr: float = 0.1,
) -> dict[str, object]:
    """One binary sigmoid neuron update under binary cross-entropy."""
    x = np.array([0.8, -1.2, 0.4], dtype=float) if x is None else x.astype(float)
    w = np.array([0.3, -0.2, 0.1], dtype=float) if w is None else w.astype(float)
    z = float(x @ w + bias)
    y_hat = float(sigmoid(np.array([z]))[0])
    eps = 1e-12
    loss = float(-(y_true * np.log(y_hat + eps) + (1 - y_true) * np.log(1 - y_hat + eps)))

    # For sigmoid + BCE, dL/dz = y_hat - y.
    dz = y_hat - y_true
    grad_w = dz * x
    grad_b = dz
    w_next = w - lr * grad_w
    b_next = bias - lr * grad_b
    return {
        "x": x,
        "w_before": w,
        "bias_before": bias,
        "linear_z": z,
        "prediction": y_hat,
        "loss": loss,
        "grad_w": grad_w,
        "grad_b": grad_b,
        "w_after": w_next,
        "bias_after": b_next,
        "learning_rate": lr,
    }


def render_report(activation_df: pd.DataFrame, neuron: dict[str, object]) -> str:
    z = neuron["linear_z"]
    pred = neuron["prediction"]
    loss = neuron["loss"]
    return "\n".join(
        [
            "# Neural Foundations: One Neuron and Activations",
            "",
            "This is an educational foundation phase. It explains the machinery used by "
            "later MLP/CNN/RNN experiments before those models are treated as black boxes.",
            "",
            "## Single Neuron Step",
            "",
            f"- weighted sum `z`: **{z:.6f}**",
            f"- sigmoid prediction: **{pred:.6f}**",
            f"- binary cross-entropy loss: **{loss:.6f}**",
            f"- gradient wrt weights: `{np.round(neuron['grad_w'], 6).tolist()}`",
            f"- gradient wrt bias: **{neuron['grad_b']:.6f}**",
            f"- updated weights: `{np.round(neuron['w_after'], 6).tolist()}`",
            f"- updated bias: **{neuron['bias_after']:.6f}**",
            "",
            "## Activation Interpretation",
            "",
            "- **Sigmoid** is bounded and useful for binary output probabilities, but it saturates.",
            "- **Tanh** is zero-centered but still saturates at large magnitude.",
            "- **ReLU** is simple and efficient, but negative units can become inactive.",
            "- **Leaky ReLU / ELU / GELU** keep some signal for negative inputs or smooth the transition.",
            "- **Softmax** converts multiclass logits into class probabilities whose sum is one.",
            "",
            f"Activation table saved to `{ACTIVATION_CSV.relative_to(D.REPO_ROOT)}`.",
            f"Activation plot saved to `{FIG_PATH.relative_to(D.REPO_ROOT)}`.",
            "",
        ]
    )


def plot_activations(df: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for col in ["linear", "sigmoid", "tanh", "relu", "leaky_relu", "elu", "gelu"]:
        axes[0].plot(df["x"], df[col], label=col)
    axes[0].set_title("activation functions")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].legend(fontsize=8)

    for col in [
        "sigmoid_derivative",
        "tanh_derivative",
        "relu_derivative",
        "leaky_relu_derivative",
        "elu_derivative",
        "gelu_derivative_numeric",
    ]:
        axes[1].plot(df["x"], df[col], label=col)
    axes[1].set_title("activation derivatives")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("df/dx")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-5, 5, 401)
    activations = activation_values(x)
    activations.to_csv(ACTIVATION_CSV, index=False)
    plot_activations(activations)

    neuron = single_neuron_step()
    neuron_df = pd.DataFrame(
        [
            {
                "linear_z": neuron["linear_z"],
                "prediction": neuron["prediction"],
                "loss": neuron["loss"],
                "grad_w": np.round(neuron["grad_w"], 8).tolist(),
                "grad_b": neuron["grad_b"],
                "w_after": np.round(neuron["w_after"], 8).tolist(),
                "bias_after": neuron["bias_after"],
            }
        ]
    )
    neuron_df.to_csv(NEURON_CSV, index=False)
    REPORT_PATH.write_text(render_report(activations, neuron), encoding="utf-8")
    return activations, neuron_df


def main() -> int:
    _, neuron = run()
    print(neuron.to_string(index=False))
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
