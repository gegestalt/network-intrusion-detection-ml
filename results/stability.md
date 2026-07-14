# NSL-KDD — multi-seed stability (5 seeds, CPU)

Seeds [0, 1, 2, 3, 4]. Mean ± std over seeds. CPU for reproducibility. Preprocessing is deterministic; only model init/splits vary.

## binary

| Model | Accuracy | Macro-F1 | attack_recall |
| --- | ---: | ---: | ---: |
| RandomForest | 0.7718 ± 0.0057 | 0.7708 ± 0.0059 | 0.6195 ± 0.0100 |
| LightGBM | 0.7850 ± 0.0000 | 0.7845 ± 0.0000 | 0.6438 ± 0.0000 |
| MLP-unweighted | 0.7961 ± 0.0049 | 0.7959 ± 0.0049 | 0.6703 ± 0.0126 |
| MLP-weighted | 0.7921 ± 0.0087 | 0.7919 ± 0.0087 | 0.6751 ± 0.0144 |

## multiclass

| Model | Accuracy | Macro-F1 | R2L_recall | U2R_recall |
| --- | ---: | ---: | ---: | ---: |
| RandomForest | 0.7473 ± 0.0025 | 0.5018 ± 0.0077 | 0.0393 ± 0.0111 | 0.0478 ± 0.0146 |
| LightGBM | 0.5614 ± 0.0000 | 0.2806 ± 0.0000 | 0.0007 ± 0.0000 | 0.0000 ± 0.0000 |
| MLP-unweighted | 0.7698 ± 0.0087 | 0.5772 ± 0.0462 | 0.0841 ± 0.0300 | 0.2119 ± 0.1165 |
| MLP-weighted | 0.7734 ± 0.0110 | 0.5682 ± 0.0119 | 0.1528 ± 0.0146 | 0.5075 ± 0.0647 |

## Takeaway
Report these mean ± std, not single-run numbers. Where std is a large fraction of the gap between two models, the ranking is **not** statistically meaningful on one run — a lesson the MLP-on-MPS swing taught us directly.
