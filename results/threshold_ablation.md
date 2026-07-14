# NSL-KDD — Threshold Tuning Ablation

Thresholds are selected on a stratified validation split from the training set. The official KDDTest+ set is used only after the threshold is fixed.

| Model | Threshold rule | Threshold | Accuracy | Macro-F1 | Attack precision | Attack recall | Normal recall | FP / 10k benign |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LogReg | default_0.50 | 0.50 | 0.7540 | 0.7535 | 0.9177 | 0.6238 | 0.9261 | 739.4 |
| LogReg | validation_F1 | 0.37 | 0.7588 | 0.7585 | 0.9140 | 0.6361 | 0.9209 | 790.9 |
| LogReg | validation_F2_attack_recall_weighted | 0.13 | 0.7830 | 0.7830 | 0.9060 | 0.6905 | 0.9054 | 946.3 |
| LogReg_balanced | default_0.50 | 0.50 | 0.7543 | 0.7539 | 0.9169 | 0.6251 | 0.9251 | 748.6 |
| LogReg_balanced | validation_F1 | 0.37 | 0.7596 | 0.7594 | 0.9134 | 0.6383 | 0.9200 | 800.1 |
| LogReg_balanced | validation_F2_attack_recall_weighted | 0.17 | 0.7791 | 0.7791 | 0.9076 | 0.6813 | 0.9084 | 916.5 |
| ExtraTrees_balanced | default_0.50 | 0.50 | 0.7881 | 0.7876 | 0.9687 | 0.6487 | 0.9723 | 277.0 |
| ExtraTrees_balanced | validation_F1 | 0.51 | 0.7859 | 0.7853 | 0.9686 | 0.6447 | 0.9724 | 276.0 |
| ExtraTrees_balanced | validation_F2_attack_recall_weighted | 0.31 | 0.8127 | 0.8126 | 0.9236 | 0.7315 | 0.9200 | 800.1 |
| HistGB_balanced | default_0.50 | 0.50 | 0.7972 | 0.7968 | 0.9689 | 0.6650 | 0.9718 | 282.2 |
| HistGB_balanced | validation_F1 | 0.32 | 0.8160 | 0.8159 | 0.9695 | 0.6987 | 0.9710 | 290.4 |
| HistGB_balanced | validation_F2_attack_recall_weighted | 0.24 | 0.8231 | 0.8230 | 0.9685 | 0.7123 | 0.9694 | 305.8 |

## Reading this table

Lowering the threshold usually raises attack recall but also creates more false alerts from benign flows. The operational column, FP / 10k benign, is the bridge from ML score to SOC workload.
