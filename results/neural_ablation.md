# NSL-KDD — Controlled Neural Ablations

This phase changes one major neural-training factor at a time around a small MLP baseline. Training uses a bounded stratified subset of the training set; the official KDDTest+ split is evaluated once per frozen configuration.

| Config | Activation | Hidden | Dropout | Norm | Loss variant | Params | Epochs | Macro-F1 | Attack recall | MCC | Train sec |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| label_smoothing_0.05 | relu | (64,) | 0.00 | none | CE + smooth 0.05 | 8,002 | 14 | 0.7759 | 0.6624 | 0.5929 | 0.79 |
| focal_loss | relu | (64,) | 0.00 | none | focal | 8,002 | 14 | 0.7699 | 0.6520 | 0.5836 | 0.70 |
| batchnorm | relu | (64,) | 0.00 | batchnorm | CE | 8,130 | 14 | 0.7676 | 0.6468 | 0.5810 | 0.73 |
| deeper_2x128_64 | relu | (128, 64) | 0.00 | none | CE | 24,130 | 14 | 0.7661 | 0.6438 | 0.5788 | 0.76 |
| weighted_cross_entropy | relu | (64,) | 0.00 | none | weighted CE | 8,002 | 14 | 0.7609 | 0.6370 | 0.5698 | 0.72 |
| activation_tanh | tanh | (64,) | 0.00 | none | CE | 8,002 | 14 | 0.7573 | 0.6309 | 0.5642 | 0.80 |
| baseline_relu_1x64 | relu | (64,) | 0.00 | none | CE | 8,002 | 14 | 0.7532 | 0.6233 | 0.5586 | 0.72 |
| dropout_0.30 | relu | (64,) | 0.30 | none | CE | 8,002 | 14 | 0.7521 | 0.6211 | 0.5571 | 0.69 |
| activation_gelu | gelu | (64,) | 0.00 | none | CE | 8,002 | 14 | 0.7503 | 0.6190 | 0.5538 | 0.81 |

## Interpretation Guardrails

- These are bounded ablations, not final neural-network rankings.
- If a config improves validation or test metrics by a tiny amount, seed stability is still required.
- CNN/RNN/LSTM/GRU are not used here because NSL-KDD rows have no valid temporal or spatial locality.
- Future CNN/RNN work needs packet sequences, flow windows, host timelines, or another justified representation.
