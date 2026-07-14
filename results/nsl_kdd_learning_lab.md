# NSL-KDD First-Dataset Learning Lab

This report exists to answer one question: are the requested learning methods actually applied to the first dataset? For every row below, the dataset is NSL-KDD.

## Concept Application Matrix

| concept | status | evidence | first_dataset_interpretation |
| --- | --- | --- | --- |
| Level 0 data validation | APPLIED | src/data.py, notebooks/04, results/attribute_comparison_nsl_kdd.csv | schema, labels, class balance, missing/unique/constant checks |
| Naive baselines | APPLIED | results/nsl_kdd_learning_lab.csv | most-frequent and stratified dummy baselines |
| Logistic Regression variants | APPLIED | results/nsl_kdd_learning_lab.csv | unweighted, balanced, L1, L2, Elastic Net, threshold variants |
| Feature-group ablation | APPLIED | results/nsl_kdd_feature_group_ablation.csv | numeric, categorical, byte-volume, host, connection-rate, content groups |
| Regularization | APPLIED | results/nsl_kdd_learning_lab.csv | C=1 vs C=0.1, L1, L2, Elastic Net |
| Tree and ensemble baselines | APPLIED | results/nsl_kdd_learning_lab.csv | DecisionTree, RandomForest, ExtraTrees, HistGradientBoosting |
| Online-capable linear model | APPLIED | results/nsl_kdd_learning_lab.csv | SGDClassifier; true drift is not claimed |
| Bootstrap confidence intervals | APPLIED | results/nsl_kdd_bootstrap_ci.csv | test-set bootstrap intervals for selected models |
| Benign-only security anomaly | APPLIED | results/nsl_kdd_learning_lab.csv | OneClassSVM and normal-only autoencoder |
| Artificial neuron foundations | APPLIED | results/neural_foundations.md | activation/loss/update demo |
| Feed-forward ANN ablations | APPLIED | results/neural_ablation.md | depth, dropout, label smoothing, gradient tracking |
| CNN / pooling | NOT_APPLIED | docs/deep_learning_taxonomy.md | not scientifically justified on isolated tabular rows without sequence/window representation |
| Cross-dataset transfer | NOT_APPLIED | docs/audits/experimental_lab_prompt_audit.md | needs another complete local dataset pipeline |

## Model And Training-Method Experiments

| level | family | method | variant | accuracy | balanced_accuracy | macro_f1 | weighted_f1 | precision_attack | attack_recall | normal_recall | mcc | fp_per_10k_benign | fn_per_10k_attack | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Level 4 | tree_family | DecisionTree | balanced_depth18 | 0.8067 | 0.8205 | 0.8067 | 0.8071 | 0.9224 | 0.7212 | 0.9198 | 0.6387 | 802.1831 | 2788.1244 | single interpretable tree |
| Security ML | benign_only_anomaly | AutoEncoder | normal_only_q0.95 | 0.8019 | 0.8166 | 0.8018 | 0.8020 | 0.9241 | 0.7103 | 0.9229 | 0.6320 | 771.2903 | 2897.2181 | autoencoder trained only on normal development rows; threshold from normal validation errors |
| Level 4 | boosting | HistGradientBoosting | default | 0.7952 | 0.8168 | 0.7948 | 0.7936 | 0.9697 | 0.6608 | 0.9727 | 0.6438 | 272.8864 | 3392.0362 | sklearn histogram boosting baseline |
| Level 1 | threshold_policy | LogReg | validation_F2_attack_weighted_0.15 | 0.7860 | 0.8008 | 0.7860 | 0.7861 | 0.9087 | 0.6938 | 0.9079 | 0.6011 | 920.6055 | 3062.4172 | threshold selected on validation split only |
| Security ML | benign_only_anomaly | OneClassSVM | normal_only_nu0.05 | 0.7830 | 0.7997 | 0.7830 | 0.7827 | 0.9181 | 0.6794 | 0.9199 | 0.6010 | 801.1533 | 3205.7976 | one-class boundary trained only on normal development rows; capped for runtime |
| Level 4 | instance_based | KNN | k5_train_cap | 0.7681 | 0.7929 | 0.7670 | 0.7648 | 0.9665 | 0.6140 | 0.9719 | 0.6038 | 281.1245 | 3860.3600 | local-neighbour baseline capped for runtime |
| Level 1 | threshold_policy | LogReg | validation_F1_threshold_0.35 | 0.7625 | 0.7816 | 0.7623 | 0.7613 | 0.9138 | 0.6434 | 0.9198 | 0.5691 | 802.1831 | 3565.8069 | threshold selected on validation split only |
| Level 4 | linear_online | SGDClassifier | log_loss_balanced | 0.7622 | 0.7817 | 0.7619 | 0.7608 | 0.9165 | 0.6405 | 0.9229 | 0.5700 | 771.2903 | 3594.6388 | linear online-capable classifier |
| Level 4 | tree_family | RandomForest | balanced_depth18 | 0.7632 | 0.7887 | 0.7619 | 0.7594 | 0.9669 | 0.6048 | 0.9726 | 0.5971 | 273.9162 | 3952.3104 | bagged tree baseline |
| Level 4 | tree_family | ExtraTrees | balanced_depth18 | 0.7582 | 0.7787 | 0.7578 | 0.7564 | 0.9196 | 0.6303 | 0.9272 | 0.5659 | 728.0404 | 3697.4986 | more randomized ensemble |
| Level 1 | logistic_regression | LogReg | l2_C0.1_balanced | 0.7545 | 0.7753 | 0.7541 | 0.7526 | 0.9172 | 0.6252 | 0.9254 | 0.5595 | 745.5463 | 3748.1493 | stronger L2 regularization |
| Level 1 | logistic_regression | LogReg | l2_C1_balanced | 0.7544 | 0.7751 | 0.7540 | 0.7525 | 0.9164 | 0.6256 | 0.9246 | 0.5590 | 753.7844 | 3744.2531 | class weighting changes false-negative pressure |
| Level 1 | logistic_regression | LogReg | l1_C0.1_balanced | 0.7543 | 0.7751 | 0.7538 | 0.7523 | 0.9170 | 0.6249 | 0.9252 | 0.5590 | 747.6058 | 3751.2663 | sparse coefficient baseline |
| Level 1 | logistic_regression | LogReg | l2_C1_unweighted | 0.7534 | 0.7743 | 0.7529 | 0.7514 | 0.9172 | 0.6230 | 0.9257 | 0.5579 | 743.4868 | 3769.9681 | transparent linear classifier |
| Level 1 | logistic_regression | LogReg | elasticnet_balanced | 0.7434 | 0.7655 | 0.7427 | 0.7408 | 0.9143 | 0.6060 | 0.9249 | 0.5424 | 750.6951 | 3939.8426 | mixed L1/L2 regularization |
| Level 0 | naive_baseline | DummyClassifier | stratified | 0.4980 | 0.5027 | 0.4974 | 0.4999 | 0.5721 | 0.4689 | 0.5366 | 0.0054 | 4633.9203 | 5311.3068 | random label baseline preserving class balance |
| Level 0 | naive_baseline | DummyClassifier | most_frequent | 0.4308 | 0.5000 | 0.3011 | 0.2594 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 10000.0000 | lowest meaningful benchmark |

## Feature-Group Ablation

| variant | encoded_feature_count | accuracy | balanced_accuracy | macro_f1 | precision_attack | attack_recall | normal_recall | fp_per_10k_benign | fn_per_10k_attack | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dst_host_behavior | 10 | 0.7890 | 0.8087 | 0.7888 | 0.9471 | 0.6666 | 0.9508 | 492.2253 | 3333.5931 | 10 encoded features |
| categorical_one_hot | 83 | 0.7817 | 0.7963 | 0.7817 | 0.9026 | 0.6911 | 0.9015 | 985.4804 | 3088.9114 | 83 encoded features |
| protocol_service_flag | 83 | 0.7817 | 0.7963 | 0.7817 | 0.9026 | 0.6911 | 0.9015 | 985.4804 | 3088.9114 | 83 encoded features |
| connection_count_rates | 9 | 0.7661 | 0.7870 | 0.7657 | 0.9313 | 0.6360 | 0.9380 | 619.9156 | 3639.8348 | 9 encoded features |
| content_login_shell | 13 | 0.7590 | 0.7587 | 0.7562 | 0.8050 | 0.7609 | 0.7565 | 2435.3826 | 2390.7114 | 13 encoded features |
| all_features | 121 | 0.7545 | 0.7752 | 0.7540 | 0.9169 | 0.6253 | 0.9251 | 748.6356 | 3746.5908 | 121 encoded features |
| numeric_only | 38 | 0.7486 | 0.7706 | 0.7479 | 0.9198 | 0.6118 | 0.9295 | 705.3856 | 3882.1788 | 38 encoded features |
| byte_volume | 2 | 0.4583 | 0.5225 | 0.3601 | 0.8515 | 0.0585 | 0.9865 | 134.8986 | 9414.7900 | 2 encoded features |

## Bootstrap Confidence Intervals

| model | metric | mean | ci_low | ci_high | n_bootstrap |
| --- | --- | --- | --- | --- | --- |
| LogReg_l2_balanced | macro_f1 | 0.7520 | 0.7462 | 0.7575 | 300 |
| LogReg_l2_balanced | attack_recall | 0.6223 | 0.6136 | 0.6298 | 300 |
| ExtraTrees_balanced | macro_f1 | 0.7744 | 0.7691 | 0.7799 | 300 |
| ExtraTrees_balanced | attack_recall | 0.6277 | 0.6189 | 0.6361 | 300 |

## Referee Interpretation

- CNN/pooling is explicitly not applied to NSL-KDD rows because there is no valid local/temporal structure yet.
- Logistic Regression is not a single baseline here; it is varied by weighting, penalty, regularization strength, and threshold policy.
- The anomaly rows are security-style normal-only experiments, not ordinary supervised classifiers.
- Feature-group rows show what happens when the first dataset is deliberately restricted to different information families.
- The lab uses an inner development split from KDDTrain+ for fitting/tuning, then KDDTest+ once for evaluation; encoded feature counts can differ slightly from the full-train preprocessor.
