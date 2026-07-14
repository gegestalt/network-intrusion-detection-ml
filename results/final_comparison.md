# Final Consolidated Comparison

This report consolidates the saved experiment artifacts. It is meant to stop the project from becoming a folder of unrelated tables.

## Coverage Matrix

| experiment_area | status | dataset_scope | evidence | limitation |
| --- | --- | --- | --- | --- |
| Raw schema and preprocessing | PROVEN | NSL-KDD | src/data.py, src/preprocess.py | CICIoT raw CSV absent |
| Classical supervised reference | PROVEN | NSL-KDD | results/reference_track.md | single seed for some models |
| Tree/boosting official split | PROVEN | NSL-KDD | results/metrics.md | RF/LightGBM only |
| MLP weighted/unweighted | PROVEN | NSL-KDD | results/metrics.md, results/stability.md | more ablations now separate |
| Feature engineering/learning | PROVEN | NSL-KDD, CICIoT2023 dev | results/feature_learning.md | not full raw CICIoT |
| Attribute-by-attribute audit | PROVEN | NSL-KDD | results/attribute_comparison_nsl_kdd.md | binary target attribution only |
| Row-level outlier datapoints | PROVEN | NSL-KDD | results/nsl_kdd_outlier_datapoints.md | numeric IQR outliers only; outlier is not automatically attack |
| First-dataset learning lab | PROVEN | NSL-KDD | results/nsl_kdd_learning_lab.md | CNN/pooling intentionally not applied to isolated tabular rows |
| Neural foundations/ablations | PROVEN | NSL-KDD | results/neural_foundations.md, results/neural_ablation.md | bounded single-seed MLP ablations |
| Threshold tuning | PROVEN | NSL-KDD | results/threshold_ablation.md | binary only |
| SOC workload simulation | PROVEN | NSL-KDD | results/soc_simulation.md | one traffic scenario |
| Normal-only anomaly | PROVEN | NSL-KDD | results/anomaly_detection.md | not modern zero-day split |
| Semi-supervised label budget | PROVEN | NSL-KDD | results/semi_supervised.md | binary and self-training only |
| Online learning | PARTIAL | NSL-KDD | results/online_learning.md | not true chronological drift |
| CICIoT2023 dev | PARTIAL | CICIoT2023 | results/ciciot2023_quality.md | random dev sample only |
| CICIoT2023 raw CSV | BLOCKED | CICIoT2023 | results/ciciot2023_raw_audit.md | data/ciciot2023/CSV is empty |
| TON_IoT | BLOCKED | TON_IoT | docs/datasets/catalog.md | local files absent |
| CSE-CIC-IDS2018 | BLOCKED | CSE-CIC-IDS2018 | docs/datasets/catalog.md | local files absent |
| Temporal CNN/RNN/LSTM/GRU | MISSING | needs timestamped/sequence data | docs/deep_learning_taxonomy.md | invalid on isolated NSL-KDD rows |
| Graph ML | MISSING | needs IP/device graph data | docs/audits/experimental_lab_prompt_audit.md | no graph representation |
| Calibration/explainability | PARTIAL | NSL-KDD | feature importance exists | Brier/ECE/SHAP not done |

## Best NSL-KDD Binary Rows By Macro-F1

| track | method | variant | comparison_family | macro_f1 | attack_recall | normal_recall | limitation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| normal_only_anomaly | KMeans_distance | q=0.90 | unsupervised_anomaly | 0.8413 | 0.7919 | 0.9079 | normal-only training; threshold from normal quantiles |
| normal_only_anomaly | IsolationForest | q=0.90 | unsupervised_anomaly | 0.8314 | 0.7783 | 0.9027 | normal-only training; threshold from normal quantiles |
| soc_simulation | HistGB_balanced | validation_F2_attack_recall_weighted | operational_policy | 0.8230 | 0.7123 | 0.9694 | fixed scenario: 1M flows/day and 0.5% malicious |
| threshold_tuning | HistGB_balanced | validation_F2_attack_recall_weighted | threshold_policy | 0.8230 | 0.7123 | 0.9694 | threshold selected on one validation split |
| soc_simulation | HistGB_balanced | validation_F1 | operational_policy | 0.8159 | 0.6987 | 0.9710 | fixed scenario: 1M flows/day and 0.5% malicious |
| threshold_tuning | HistGB_balanced | validation_F1 | threshold_policy | 0.8159 | 0.6987 | 0.9710 | threshold selected on one validation split |
| soc_simulation | ExtraTrees_balanced | validation_F2_attack_recall_weighted | operational_policy | 0.8126 | 0.7315 | 0.9200 | fixed scenario: 1M flows/day and 0.5% malicious |
| threshold_tuning | ExtraTrees_balanced | validation_F2_attack_recall_weighted | threshold_policy | 0.8126 | 0.7315 | 0.9200 | threshold selected on one validation split |
| phase4_mlp_weighting | MLP (unweighted) | weighted/unweighted | neural_mlp | 0.8104 | 0.6880 |  | single run; stability reported separately |
| normal_only_anomaly | IsolationForest | q=0.95 | unsupervised_anomaly | 0.8095 | 0.7269 | 0.9189 | normal-only training; threshold from normal quantiles |
| first_dataset_learning_lab | DecisionTree | balanced_depth18 | tree_family | 0.8067 | 0.7212 | 0.9198 | inner KDDTrain+ dev split for fitting/tuning; KDDTest+ final evaluation |
| first_dataset_learning_lab | AutoEncoder | normal_only_q0.95 | benign_only_anomaly | 0.8018 | 0.7103 | 0.9229 | inner KDDTrain+ dev split for fitting/tuning; KDDTest+ final evaluation |

## Best NSL-KDD Binary Rows By Attack Recall

| track | method | variant | comparison_family | attack_recall | macro_f1 | normal_recall | limitation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| normal_only_anomaly | KMeans_distance | q=0.90 | unsupervised_anomaly | 0.7919 | 0.8413 | 0.9079 | normal-only training; threshold from normal quantiles |
| normal_only_anomaly | IsolationForest | q=0.90 | unsupervised_anomaly | 0.7783 | 0.8314 | 0.9027 | normal-only training; threshold from normal quantiles |
| first_dataset_feature_ablation | LogReg | content_login_shell | feature_group_ablation | 0.7609 | 0.7562 | 0.7565 | feature groups tested with balanced Logistic Regression only |
| soc_simulation | ExtraTrees_balanced | validation_F2_attack_recall_weighted | operational_policy | 0.7315 | 0.8126 | 0.9200 | fixed scenario: 1M flows/day and 0.5% malicious |
| threshold_tuning | ExtraTrees_balanced | validation_F2_attack_recall_weighted | threshold_policy | 0.7315 | 0.8126 | 0.9200 | threshold selected on one validation split |
| normal_only_anomaly | IsolationForest | q=0.95 | unsupervised_anomaly | 0.7269 | 0.8095 | 0.9189 | normal-only training; threshold from normal quantiles |
| first_dataset_learning_lab | DecisionTree | balanced_depth18 | tree_family | 0.7212 | 0.8067 | 0.9198 | inner KDDTrain+ dev split for fitting/tuning; KDDTest+ final evaluation |
| soc_simulation | HistGB_balanced | validation_F2_attack_recall_weighted | operational_policy | 0.7123 | 0.8230 | 0.9694 | fixed scenario: 1M flows/day and 0.5% malicious |
| threshold_tuning | HistGB_balanced | validation_F2_attack_recall_weighted | threshold_policy | 0.7123 | 0.8230 | 0.9694 | threshold selected on one validation split |
| first_dataset_learning_lab | AutoEncoder | normal_only_q0.95 | benign_only_anomaly | 0.7103 | 0.8018 | 0.9229 | inner KDDTrain+ dev split for fitting/tuning; KDDTest+ final evaluation |
| phase4_mlp_weighting | MLP (weighted) | weighted/unweighted | neural_mlp | 0.7000 | 0.7975 |  | single run; stability reported separately |
| threshold_tuning | HistGB_balanced | validation_F1 | threshold_policy | 0.6987 | 0.8159 | 0.9710 | threshold selected on one validation split |

## Best Row Per Experiment Family

| comparison_family | track | method | variant | macro_f1 | attack_recall | limitation |
| --- | --- | --- | --- | --- | --- | --- |
| unsupervised_anomaly | normal_only_anomaly | KMeans_distance | q=0.90 | 0.8413 | 0.7919 | normal-only training; threshold from normal quantiles |
| operational_policy | soc_simulation | HistGB_balanced | validation_F2_attack_recall_weighted | 0.8230 | 0.7123 | fixed scenario: 1M flows/day and 0.5% malicious |
| threshold_policy | threshold_tuning | HistGB_balanced | validation_F2_attack_recall_weighted | 0.8230 | 0.7123 | threshold selected on one validation split |
| neural_mlp | phase4_mlp_weighting | MLP (unweighted) | weighted/unweighted | 0.8104 | 0.6880 | single run; stability reported separately |
| tree_family | first_dataset_learning_lab | DecisionTree | balanced_depth18 | 0.8067 | 0.7212 | inner KDDTrain+ dev split for fitting/tuning; KDDTest+ final evaluation |
| benign_only_anomaly | first_dataset_learning_lab | AutoEncoder | normal_only_q0.95 | 0.8018 | 0.7103 | inner KDDTrain+ dev split for fitting/tuning; KDDTest+ final evaluation |
| classical_ml | supervised_reference | HistGradientBoosting (balanced) | balanced where supported | 0.7968 | 0.6650 | single seed for several reference models |
| boosting | first_dataset_learning_lab | HistGradientBoosting | default | 0.7948 | 0.6608 | inner KDDTrain+ dev split for fitting/tuning; KDDTest+ final evaluation |
| feature_group_ablation | first_dataset_feature_ablation | LogReg | dst_host_behavior | 0.7888 | 0.6666 | feature groups tested with balanced Logistic Regression only |
| tree_boosting | phase3_tree_boosting | LightGBM | KDDTest+ | 0.7845 |  | only RF/LightGBM in this artifact |
| neural_mlp_ablation | neural_ablation | MLP | label_smoothing_0.05 | 0.7759 | 0.6624 | bounded train subset; single seed |
| instance_based | first_dataset_learning_lab | KNN | k5_train_cap | 0.7670 | 0.6140 | inner KDDTrain+ dev split for fitting/tuning; KDDTest+ final evaluation |
| online_partial_fit | online_learning_proxy | SGD_log_loss_balanced | chunk=10000 | 0.7642 | 0.6437 | NSL-KDD file order is not chronological drift |
| linear_online | first_dataset_learning_lab | SGDClassifier | log_loss_balanced | 0.7619 | 0.6405 | inner KDDTrain+ dev split for fitting/tuning; KDDTest+ final evaluation |
| feature_representation | feature_learning | LogisticRegression | autoencoder_embedding_16 | 0.7576 | 0.6285 | CICIoT2023 rows are dev-sample only when dataset=ciciot2023_dev |
| label_budget | semi_supervised | labelled_only_logreg | 0.10 labels | 0.7542 | 0.6268 | binary only; self-training only |
| logistic_regression | first_dataset_learning_lab | LogReg | l2_C0.1_balanced | 0.7541 | 0.6252 | inner KDDTrain+ dev split for fitting/tuning; KDDTest+ final evaluation |
| naive_baseline | first_dataset_learning_lab | DummyClassifier | stratified | 0.4974 | 0.4689 | inner KDDTrain+ dev split for fitting/tuning; KDDTest+ final evaluation |

## Lowest Operational Alert Burden

| method | variant | macro_f1 | attack_recall | false_alerts_per_day | missed_attacks_per_day | total_alerts_per_day | operational_precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ExtraTrees_balanced | validation_F1 | 0.7853 | 0.6447 | 27,460 | 1,776 | 30,683 | 0.1051 |
| ExtraTrees_balanced | default_0.50 | 0.7876 | 0.6487 | 27,562 | 1,756 | 30,806 | 0.1053 |
| HistGB_balanced | default_0.50 | 0.7968 | 0.6650 | 28,074 | 1,675 | 31,399 | 0.1059 |
| HistGB_balanced | validation_F1 | 0.8159 | 0.6987 | 28,894 | 1,506 | 32,388 | 0.1079 |
| HistGB_balanced | validation_F2_attack_recall_weighted | 0.8230 | 0.7123 | 30,431 | 1,438 | 33,992 | 0.1048 |
| LogReg | default_0.50 | 0.7535 | 0.6238 | 73,567 | 1,881 | 76,686 | 0.0407 |
| LogReg_balanced | default_0.50 | 0.7539 | 0.6251 | 74,489 | 1,874 | 77,615 | 0.0403 |
| LogReg | validation_F1 | 0.7585 | 0.6361 | 78,690 | 1,820 | 81,871 | 0.0388 |
| LogReg_balanced | validation_F1 | 0.7594 | 0.6383 | 79,612 | 1,809 | 82,804 | 0.0385 |
| ExtraTrees_balanced | validation_F2_attack_recall_weighted | 0.8126 | 0.7315 | 79,612 | 1,343 | 83,270 | 0.0439 |
| LogReg_balanced | validation_F2_attack_recall_weighted | 0.7791 | 0.6813 | 91,190 | 1,594 | 94,597 | 0.0360 |
| LogReg | validation_F2_attack_recall_weighted | 0.7830 | 0.6905 | 94,162 | 1,548 | 97,614 | 0.0354 |

## Referee Interpretation

- The highest macro-F1 rows are not necessarily the lowest-alert operational rows.
- Threshold and SOC rows answer a different question than classifier macro-F1 rows.
- Anomaly detection is valuable as a different learning paradigm, not because it automatically beats supervised learning.
- Feature-learning and neural-ablation rows are bounded experiments; they need seed stability before strong ranking claims.
- CICIoT2023 dev rows are useful but cannot be sold as full raw CICIoT2023 results.
