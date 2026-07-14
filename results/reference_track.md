# NSL-KDD — Reference Track (supervised benchmark)

`class_weight='balanced'` where supported. Single seed; see `stability.md` for seed variance on the headline models.

## binary

| Model | Accuracy | Macro-F1 | Rare-class recall |
| --- | ---: | ---: | --- |
| Dummy (most_frequent) | 0.4308 | 0.3011 | attack 0.000 |
| LogReg (balanced) | 0.7543 | 0.7539 | attack 0.625 |
| RandomForest (balanced) | 0.7674 | 0.7662 | attack 0.612 |
| ExtraTrees (balanced) | 0.7882 | 0.7877 | attack 0.649 |
| HistGradientBoosting (balanced) | 0.7972 | 0.7968 | attack 0.665 |
| LightGBM (balanced) | 0.7842 | 0.7835 | attack 0.642 |

## multiclass

| Model | Accuracy | Macro-F1 | Rare-class recall |
| --- | ---: | ---: | --- |
| Dummy (most_frequent) | 0.4308 | 0.1204 | R2L 0.000 / U2R 0.000 |
| LogReg (balanced) | 0.7913 | 0.6069 | R2L 0.236 / U2R 0.552 |
| RandomForest (balanced) | 0.7437 | 0.5128 | R2L 0.017 / U2R 0.104 |
| ExtraTrees (balanced) | 0.7653 | 0.5325 | R2L 0.047 / U2R 0.090 |
| HistGradientBoosting (balanced) | 0.7895 | 0.6382 | R2L 0.248 / U2R 0.284 |
| LightGBM (balanced) | 0.7629 | 0.5792 | R2L 0.079 / U2R 0.239 |

