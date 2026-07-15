# NSL-KDD — Normal-Only Anomaly Detection

Models are trained on normal training traffic only. Thresholds are chosen from normal-only training score quantiles, so attack labels do not tune the detector.

| Model | Normal quantile | Accuracy | Macro-F1 | Normal recall | Attack recall | DoS | Probe | R2L | U2R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IsolationForest | 0.90 | 0.8319 | 0.8314 | 0.9027 | 0.7783 | 0.9146 | 0.9946 | 0.2506 | 0.5075 |
| IsolationForest | 0.95 | 0.8096 | 0.8095 | 0.9189 | 0.7269 | 0.8790 | 0.9897 | 0.1213 | 0.3731 |
| IsolationForest | 0.99 | 0.7968 | 0.7963 | 0.9802 | 0.6580 | 0.8472 | 0.8501 | 0.0215 | 0.0597 |
| LocalOutlierFactor | 0.90 | 0.7398 | 0.7397 | 0.8692 | 0.6418 | 0.6462 | 0.7914 | 0.5033 | 0.7015 |
| LocalOutlierFactor | 0.95 | 0.6276 | 0.6159 | 0.9309 | 0.3981 | 0.3668 | 0.4304 | 0.4475 | 0.5970 |
| LocalOutlierFactor | 0.99 | 0.4388 | 0.3213 | 0.9923 | 0.0199 | 0.0004 | 0.0000 | 0.0849 | 0.1194 |
| KMeans_distance | 0.90 | 0.8419 | 0.8413 | 0.9079 | 0.7919 | 0.9233 | 0.9719 | 0.3043 | 0.6567 |
| KMeans_distance | 0.95 | 0.7888 | 0.7888 | 0.9244 | 0.6861 | 0.7760 | 0.8992 | 0.2756 | 0.6567 |
| KMeans_distance | 0.99 | 0.4676 | 0.3758 | 0.9880 | 0.0739 | 0.0021 | 0.1640 | 0.1712 | 0.6119 |

## Interpretation

This is not expected to beat supervised classifiers on known NSL-KDD labels. Its purpose is different: measure whether a normal-behaviour model catches attack families without being trained on attack examples.
