# NSL-KDD — Phase 3 baseline results (RF vs LightGBM)

Official split; metrics on the held-out test set. Leading with **macro-F1** and per-class recall (accuracy is inflated by the easy majority classes).

| Model | Task | Test set | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC | PR-AUC | Best params |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| RandomForest | binary | KDDTest+ | 0.7772 | 0.7763 | 0.7744 | 0.9620 | 0.9651 | `{'max_depth': None, 'n_estimators': 300}` |
| RandomForest | binary | KDDTest-21 | 0.5761 | 0.5462 | 0.6204 | 0.8177 | 0.9447 | `{'max_depth': None, 'n_estimators': 300}` |
| LightGBM | binary | KDDTest+ | 0.7850 | 0.7845 | 0.7829 | 0.9608 | 0.9628 | `{'n_estimators': 400, 'num_leaves': 31}` |
| LightGBM | binary | KDDTest-21 | 0.5911 | 0.5579 | 0.6351 | 0.7889 | 0.9321 | `{'n_estimators': 400, 'num_leaves': 31}` |
| RandomForest | multiclass | KDDTest+ | 0.7430 | 0.5038 | 0.7019 | — | — | `{'max_depth': 25, 'n_estimators': 300}` |
| LightGBM | multiclass | KDDTest+ | 0.6267 | 0.4102 | 0.6082 | — | — | `{'n_estimators': 400, 'num_leaves': 31}` |

## Per-class breakdown (KDDTest+)

**RandomForest — binary**

| class | precision | recall | f1 | support |
| --- | ---: | ---: | ---: | ---: |
| normal | 0.6649 | 0.9733 | 0.7901 | 9,711 |
| attack | 0.9689 | 0.6288 | 0.7626 | 12,833 |

**LightGBM — binary**

| class | precision | recall | f1 | support |
| --- | ---: | ---: | ---: | ---: |
| normal | 0.6737 | 0.9717 | 0.7957 | 9,711 |
| attack | 0.9678 | 0.6438 | 0.7732 | 12,833 |

**RandomForest — multiclass**

| class | precision | recall | f1 | support |
| --- | ---: | ---: | ---: | ---: |
| normal | 0.6431 | 0.9737 | 0.7746 | 9,711 |
| DoS | 0.9611 | 0.7651 | 0.8520 | 7,460 |
| Probe | 0.8223 | 0.5964 | 0.6914 | 2,421 |
| R2L | 0.9857 | 0.0478 | 0.0912 | 2,885 |
| U2R | 0.6667 | 0.0597 | 0.1096 | 67 |

**LightGBM — multiclass**

| class | precision | recall | f1 | support |
| --- | ---: | ---: | ---: | ---: |
| normal | 0.6447 | 0.8302 | 0.7258 | 9,711 |
| DoS | 0.7798 | 0.6058 | 0.6819 | 7,460 |
| Probe | 0.4592 | 0.5527 | 0.5016 | 2,421 |
| R2L | 0.6536 | 0.0693 | 0.1254 | 2,885 |
| U2R | 0.0088 | 0.1343 | 0.0165 | 67 |


# NSL-KDD — Phase 4: MLP + class-weighting ablation

Same preprocessing and evaluation as Phase 3. The point of this phase is the **with vs without class-weighting** contrast on the rare classes.

| Variant | Task | Accuracy | Macro-F1 | Rare-class recall |
| --- | --- | ---: | ---: | --- |
| MLP (unweighted) | binary | 0.7893 | 0.7893 | attack 0.678 |
| MLP (weighted) | binary | 0.8105 | 0.8104 | attack 0.689 |
| MLP (unweighted) | multiclass | 0.7816 | 0.6086 | R2L 0.080 / U2R 0.284 |
| MLP (weighted) | multiclass | 0.7907 | 0.7081 | R2L 0.477 / U2R 0.418 |
