# CICIoT2023 — supervised baselines (dev parquet)

Train subsampled to 200,000 (stratified); full test split. `class_weight='balanced'` where supported. Random split → in-distribution (caveat vs NSL-KDD's official shift).

## binary

| Model | Accuracy | Balanced-acc | Macro-F1 | ROC-AUC | PR-AUC | Rare-class recall |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Dummy (stratified) | 0.7451 | 0.4983 | 0.4983 | 0.4983 | 0.8506 | attack 0.850 |
| LogReg (balanced) | 0.8167 | 0.8862 | 0.7477 | 0.9324 | 0.9884 | attack 0.787 |
| RandomForest (balanced) | 0.9172 | 0.9073 | 0.8563 | 0.9695 | 0.9948 | attack 0.921 |
| HistGB (balanced) | 0.8903 | 0.9273 | 0.8291 | 0.9694 | 0.9948 | attack 0.875 |
| LightGBM (balanced) | 0.8998 | 0.9285 | 0.8401 | 0.9714 | 0.9952 | attack 0.888 |

## category

| Model | Accuracy | Balanced-acc | Macro-F1 | Rare-class recall |
| --- | ---: | ---: | ---: | --- |
| Dummy (stratified) | 0.1995 | 0.1250 | 0.1250 | Web-based 0.016 / Brute Force 0.007 |
| LogReg (balanced) | 0.6812 | 0.6286 | 0.5693 | Web-based 0.462 / Brute Force 0.404 |
| RandomForest (balanced) | 0.8318 | 0.7211 | 0.7361 | Web-based 0.325 / Brute Force 0.354 |
| HistGB (balanced) | 0.8076 | 0.7633 | 0.7021 | Web-based 0.627 / Brute Force 0.597 |
| LightGBM (balanced) | 0.8235 | 0.7647 | 0.7236 | Web-based 0.594 / Brute Force 0.532 |

