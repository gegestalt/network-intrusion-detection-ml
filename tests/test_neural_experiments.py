"""Tests for neural foundations, taxonomy, ablations, and tracking."""

from __future__ import annotations

import numpy as np
import torch

import deep_learning_taxonomy as TAX
import neural_ablation as NA
import neural_foundations as NF
import tracking


def test_sigmoid_and_softmax_basic_properties():
    x = np.array([-1.0, 0.0, 1.0])
    s = NF.sigmoid(x)
    assert np.all((s > 0) & (s < 1))
    assert s[1] == np.float64(0.5)

    p = NF.softmax(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(p.sum(), 1.0)
    assert p.argmax() == 2


def test_single_neuron_step_updates_against_error_direction():
    out = NF.single_neuron_step()

    assert out["prediction"] < 1.0
    assert out["loss"] > 0
    assert out["w_before"].shape == out["w_after"].shape
    assert np.isfinite(out["grad_w"]).all()


def test_activation_values_include_derivatives():
    df = NF.activation_values(np.linspace(-1, 1, 5))

    assert {"sigmoid", "relu", "gelu", "sigmoid_derivative"} <= set(df.columns)
    assert len(df) == 5


def test_ablation_mlp_forward_shape_and_parameter_count():
    config = NA.NeuralConfig(name="test", hidden=(8, 4), activation="gelu", dropout=0.1)
    model = NA.AblationMLP(in_dim=6, n_classes=2, config=config)
    out = model(torch.zeros(3, 6))

    assert out.shape == (3, 2)
    assert NA.count_parameters(model) > 0


def test_focal_loss_returns_scalar():
    loss_fn = NA.FocalLoss()
    logits = torch.tensor([[2.0, 0.1], [0.2, 1.5]])
    y = torch.tensor([0, 1])

    loss = loss_fn(logits, y)

    assert loss.ndim == 0
    assert float(loss) >= 0


def test_tracking_stable_id_and_read_write(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "EXPERIMENT_DIR", tmp_path)
    run = tracking.ExperimentRun(
        experiment_id=tracking.stable_experiment_id("a", 1),
        dataset="toy",
        task="binary",
        model_family="MLP",
        run_name="smoke",
        seed=7,
        status="complete",
        metrics={"macro_f1": 0.5},
    )
    path = tracking.append_run(run, filename="test.jsonl")
    rows = tracking.read_runs("test.jsonl")

    assert path.exists()
    assert rows[0]["dataset"] == "toy"
    assert tracking.stable_experiment_id("a", 1) == tracking.stable_experiment_id("a", 1)


def test_taxonomy_mentions_invalid_static_drl():
    text = TAX.render()

    assert "Deep RL" in text
    assert "not for static classification" in text
    assert "CNN/RNN/GAN/DRL" in text
