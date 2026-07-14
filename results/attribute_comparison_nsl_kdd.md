# NSL-KDD Attribute-By-Attribute Comparison

This is not a model leaderboard. It is a forensic table over the 41 raw NSL-KDD attributes and the 122 encoded model features.

## What Each Row Means

- **class_separation**: numeric standardized normal-vs-attack mean gap; categorical Cramer's V.
- **target_corr_max_abs**: strongest absolute encoded-feature correlation with the binary attack target.
- **outlier_rate_max**: largest IQR outlier rate among the encoded children of that attribute.
- **redundant_pair_count**: number of high-correlation encoded pairs touching the attribute.
- **importance_consensus**: average-normalized signal from Logistic Regression, Random Forest, and mutual information.

## All Raw Attributes Ranked By Consensus Importance

| attribute | raw_type | encoded_feature_count | class_separation | target_corr_max_abs | outlier_rate_max | redundant_pair_count | importance_consensus | rank_by_consensus | audit_flags | engineering_recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| service | categorical | 70 | 0.8600 | 0.5623 | 0.0000 | 0 | 0.8383 | 1 | strong_target_signal | Keep one-hot encoded; compare category-level attack rates before collapsing. |
| src_bytes | numeric | 1 | 0.0115 | 0.0059 | 0.1099 | 0 | 0.6330 | 2 | many_iqr_outliers | Keep, but prefer robust scaling/winsorization ablation because extreme values may carry signal. |
| flag | categorical | 11 | 0.7749 | 0.7563 | 0.0000 | 4 | 0.6052 | 3 | highly_redundant; strong_target_signal | Keep one-hot encoded; compare category-level attack rates before collapsing. |
| dst_bytes | numeric | 1 | 0.0080 | 0.0041 | 0.1872 | 0 | 0.4953 | 4 | many_iqr_outliers | Keep, but prefer robust scaling/winsorization ablation because extreme values may carry signal. |
| same_srv_rate | numeric | 1 | 2.2252 | 0.7519 | 0.0000 | 0 | 0.2863 | 5 | strong_target_signal | Keep in the baseline; use ablation/permutation importance before changing it. |
| dst_host_srv_count | numeric | 1 | 2.1324 | 0.7225 | 0.0000 | 0 | 0.2788 | 6 | strong_target_signal | Keep in the baseline; use ablation/permutation importance before changing it. |
| dst_host_same_srv_rate | numeric | 1 | 1.9318 | 0.6938 | 0.0000 | 0 | 0.2625 | 7 | strong_target_signal | Keep in the baseline; use ablation/permutation importance before changing it. |
| diff_srv_rate | numeric | 1 | 0.4122 | 0.2037 | 0.0633 | 0 | 0.2577 | 8 | ordinary | Keep in the baseline; use ablation/permutation importance before changing it. |
| dst_host_serror_rate | numeric | 1 | 1.6669 | 0.6518 | 0.0000 | 4 | 0.2486 | 9 | highly_redundant; strong_target_signal | Keep for tree baselines; test feature selection or regularization to reduce redundancy. |
| count | numeric | 1 | 1.3805 | 0.5764 | 0.0251 | 0 | 0.2441 | 10 | strong_target_signal | Keep in the baseline; use ablation/permutation importance before changing it. |
| serror_rate | numeric | 1 | 1.6617 | 0.6507 | 0.0000 | 4 | 0.2140 | 11 | highly_redundant; strong_target_signal | Keep for tree baselines; test feature selection or regularization to reduce redundancy. |
| dst_host_diff_srv_rate | numeric | 1 | 0.4928 | 0.2429 | 0.0837 | 0 | 0.2113 | 12 | ordinary | Keep in the baseline; use ablation/permutation importance before changing it. |
| dst_host_srv_serror_rate | numeric | 1 | 1.6782 | 0.6550 | 0.0000 | 4 | 0.2104 | 13 | highly_redundant; strong_target_signal | Keep for tree baselines; test feature selection or regularization to reduce redundancy. |
| srv_serror_rate | numeric | 1 | 1.6506 | 0.6483 | 0.0000 | 4 | 0.2034 | 14 | highly_redundant; strong_target_signal | Keep for tree baselines; test feature selection or regularization to reduce redundancy. |
| logged_in | binary_numeric | 1 | 1.9594 | 0.6902 | 0.0000 | 0 | 0.1907 | 15 | strong_target_signal | Keep in the baseline; use ablation/permutation importance before changing it. |
| dst_host_same_src_port_rate | numeric | 1 | 0.1840 | 0.0924 | 0.2007 | 0 | 0.1466 | 16 | many_iqr_outliers | Keep, but prefer robust scaling/winsorization ablation because extreme values may carry signal. |
| dst_host_srv_diff_host_rate | numeric | 1 | 0.1224 | 0.0623 | 0.1169 | 0 | 0.1451 | 17 | many_iqr_outliers | Keep, but prefer robust scaling/winsorization ablation because extreme values may carry signal. |
| protocol_type | categorical | 3 | 0.2821 | 0.2172 | 0.0000 | 0 | 0.1233 | 18 | ordinary | Keep one-hot encoded; compare category-level attack rates before collapsing. |
| dst_host_count | numeric | 1 | 0.8180 | 0.3751 | 0.0000 | 0 | 0.1122 | 19 | strong_target_signal | Keep in the baseline; use ablation/permutation importance before changing it. |
| srv_diff_host_rate | numeric | 1 | 0.2420 | 0.1194 | 0.0000 | 0 | 0.0690 | 20 | ordinary | Keep in the baseline; use ablation/permutation importance before changing it. |
| srv_count | numeric | 1 | 0.0015 | 0.0008 | 0.0973 | 0 | 0.0671 | 21 | ordinary | Keep in the baseline; use ablation/permutation importance before changing it. |
| dst_host_rerror_rate | numeric | 1 | 0.5126 | 0.2526 | 0.0000 | 0 | 0.0626 | 22 | ordinary | Keep in the baseline; use ablation/permutation importance before changing it. |
| dst_host_srv_rerror_rate | numeric | 1 | 0.5137 | 0.2534 | 0.0000 | 2 | 0.0576 | 23 | highly_redundant | Keep for tree baselines; test feature selection or regularization to reduce redundancy. |
| rerror_rate | numeric | 1 | 0.5143 | 0.2534 | 0.0000 | 2 | 0.0471 | 24 | highly_redundant | Keep for tree baselines; test feature selection or regularization to reduce redundancy. |
| srv_rerror_rate | numeric | 1 | 0.5144 | 0.2535 | 0.0000 | 2 | 0.0415 | 25 | highly_redundant | Keep for tree baselines; test feature selection or regularization to reduce redundancy. |
| num_compromised | numeric | 1 | 0.0211 | 0.0102 | 0.0000 | 1 | 0.0368 | 26 | ordinary | Keep in the baseline; use ablation/permutation importance before changing it. |
| hot | numeric | 1 | 0.0264 | 0.0131 | 0.0000 | 0 | 0.0249 | 27 | ordinary | Keep in the baseline; use ablation/permutation importance before changing it. |
| duration | numeric | 1 | 0.0953 | 0.0488 | 0.0000 | 0 | 0.0220 | 28 | ordinary | Keep in the baseline; use ablation/permutation importance before changing it. |
| num_root | numeric | 1 | 0.0237 | 0.0115 | 0.0000 | 1 | 0.0207 | 29 | ordinary | Keep in the baseline; use ablation/permutation importance before changing it. |
| wrong_fragment | numeric | 1 | 0.1864 | 0.0959 | 0.0000 | 0 | 0.0198 | 30 | ordinary | Keep in the baseline; use ablation/permutation importance before changing it. |
| is_guest_login | binary_numeric | 1 | 0.0799 | 0.0393 | 0.0000 | 0 | 0.0122 | 31 | ordinary | Keep in the baseline; use ablation/permutation importance before changing it. |
| num_failed_logins | numeric | 1 | 0.0076 | 0.0038 | 0.0000 | 0 | 0.0043 | 32 | ordinary | Candidate for pruning in a compact model, but verify with ablation before removing. |
| su_attempted | binary_numeric | 1 | 0.0465 | 0.0224 | 0.0000 | 0 | 0.0033 | 33 | ordinary | Candidate for pruning in a compact model, but verify with ablation before removing. |
| num_access_files | numeric | 1 | 0.0761 | 0.0367 | 0.0000 | 0 | 0.0029 | 34 | ordinary | Candidate for pruning in a compact model, but verify with ablation before removing. |
| root_shell | binary_numeric | 1 | 0.0415 | 0.0203 | 0.0000 | 0 | 0.0027 | 35 | ordinary | Candidate for pruning in a compact model, but verify with ablation before removing. |
| num_outbound_cmds | numeric | 1 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0027 | 36 | constant_or_near_constant | Drop or keep only for schema compatibility; it adds no learning signal here. |
| num_shells | numeric | 1 | 0.0192 | 0.0095 | 0.0000 | 0 | 0.0019 | 37 | ordinary | Candidate for pruning in a compact model, but verify with ablation before removing. |
| num_file_creations | numeric | 1 | 0.0440 | 0.0213 | 0.0000 | 0 | 0.0010 | 38 | ordinary | Candidate for pruning in a compact model, but verify with ablation before removing. |
| land | binary_numeric | 1 | 0.0142 | 0.0072 | 0.0000 | 0 | 0.0004 | 39 | ordinary | Candidate for pruning in a compact model, but verify with ablation before removing. |
| is_host_login | binary_numeric | 1 | 0.0054 | 0.0026 | 0.0000 | 0 | 0.0003 | 40 | ordinary | Candidate for pruning in a compact model, but verify with ablation before removing. |
| urgent | numeric | 1 | 0.0057 | 0.0028 | 0.0000 | 0 | 0.0000 | 41 | ordinary | Candidate for pruning in a compact model, but verify with ablation before removing. |

## Distribution And Drift Checks

| attribute | raw_type | train_unique | test_unique | unseen_test_categories | train_test_shift_std | train_missing_rate | test_missing_rate | audit_flags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| num_failed_logins | numeric | 6 | 5 | 0 | 0.4515 | 0.0000 | 0.0000 | ordinary |
| dst_host_serror_rate | numeric | 101 | 99 | 0 | 0.4196 | 0.0000 | 0.0000 | highly_redundant; strong_target_signal |
| serror_rate | numeric | 89 | 88 | 0 | 0.4067 | 0.0000 | 0.0000 | highly_redundant; strong_target_signal |
| dst_host_srv_serror_rate | numeric | 100 | 101 | 0 | 0.4018 | 0.0000 | 0.0000 | highly_redundant; strong_target_signal |
| srv_serror_rate | numeric | 86 | 82 | 0 | 0.4001 | 0.0000 | 0.0000 | highly_redundant; strong_target_signal |
| dst_host_rerror_rate | numeric | 101 | 101 | 0 | 0.3737 | 0.0000 | 0.0000 | ordinary |
| rerror_rate | numeric | 82 | 90 | 0 | 0.3698 | 0.0000 | 0.0000 | highly_redundant |
| srv_rerror_rate | numeric | 62 | 93 | 0 | 0.3522 | 0.0000 | 0.0000 | highly_redundant |
| dst_host_srv_rerror_rate | numeric | 101 | 100 | 0 | 0.3332 | 0.0000 | 0.0000 | highly_redundant |
| dst_host_srv_count | numeric | 256 | 256 | 0 | 0.2267 | 0.0000 | 0.0000 | strong_target_signal |
| is_guest_login | binary_numeric | 2 | 2 | 0 | 0.1968 | 0.0000 | 0.0000 | ordinary |
| dst_host_same_srv_rate | numeric | 101 | 101 | 0 | 0.1949 | 0.0000 | 0.0000 | strong_target_signal |
| same_srv_rate | numeric | 101 | 75 | 0 | 0.1806 | 0.0000 | 0.0000 | strong_target_signal |
| diff_srv_rate | numeric | 95 | 99 | 0 | 0.1720 | 0.0000 | 0.0000 | ordinary |
| is_host_login | binary_numeric | 2 | 2 | 0 | 0.1704 | 0.0000 | 0.0000 | ordinary |
| dst_host_count | numeric | 256 | 256 | 0 | 0.1181 | 0.0000 | 0.0000 | strong_target_signal |
| dst_host_srv_diff_host_rate | numeric | 75 | 58 | 0 | 0.1146 | 0.0000 | 0.0000 | many_iqr_outliers |
| logged_in | binary_numeric | 2 | 2 | 0 | 0.0950 | 0.0000 | 0.0000 | strong_target_signal |
| wrong_fragment | numeric | 3 | 3 | 0 | 0.0562 | 0.0000 | 0.0000 | ordinary |
| dst_host_same_src_port_rate | numeric | 101 | 101 | 0 | 0.0522 | 0.0000 | 0.0000 | many_iqr_outliers |
| srv_count | numeric | 509 | 457 | 0 | 0.0466 | 0.0000 | 0.0000 | ordinary |
| hot | numeric | 28 | 16 | 0 | 0.0461 | 0.0000 | 0.0000 | ordinary |
| count | numeric | 512 | 495 | 0 | 0.0444 | 0.0000 | 0.0000 | strong_target_signal |
| urgent | numeric | 4 | 4 | 0 | 0.0417 | 0.0000 | 0.0000 | ordinary |
| dst_host_diff_srv_rate | numeric | 101 | 101 | 0 | 0.0402 | 0.0000 | 0.0000 | ordinary |
| num_shells | numeric | 3 | 4 | 0 | 0.0334 | 0.0000 | 0.0000 | ordinary |
| root_shell | binary_numeric | 2 | 2 | 0 | 0.0300 | 0.0000 | 0.0000 | ordinary |
| duration | numeric | 2981 | 624 | 0 | 0.0262 | 0.0000 | 0.0000 | ordinary |
| su_attempted | binary_numeric | 3 | 3 | 0 | 0.0185 | 0.0000 | 0.0000 | ordinary |
| num_file_creations | numeric | 35 | 9 | 0 | 0.0081 | 0.0000 | 0.0000 | ordinary |
| land | binary_numeric | 2 | 2 | 0 | 0.0080 | 0.0000 | 0.0000 | ordinary |
| num_root | numeric | 82 | 20 | 0 | 0.0077 | 0.0000 | 0.0000 | ordinary |
| num_compromised | numeric | 88 | 23 | 0 | 0.0067 | 0.0000 | 0.0000 | ordinary |
| src_bytes | numeric | 3341 | 1149 | 0 | 0.0060 | 0.0000 | 0.0000 | many_iqr_outliers |
| num_access_files | numeric | 10 | 5 | 0 | 0.0055 | 0.0000 | 0.0000 | ordinary |
| dst_bytes | numeric | 9326 | 3650 | 0 | 0.0044 | 0.0000 | 0.0000 | many_iqr_outliers |
| srv_diff_host_rate | numeric | 60 | 84 | 0 | 0.0030 | 0.0000 | 0.0000 | ordinary |
| num_outbound_cmds | numeric | 1 | 1 | 0 | 0.0000 | 0.0000 | 0.0000 | constant_or_near_constant |
| service | categorical | 70 | 64 | 0 |  | 0.0000 | 0.0000 | strong_target_signal |
| flag | categorical | 11 | 11 | 0 |  | 0.0000 | 0.0000 | highly_redundant; strong_target_signal |
| protocol_type | categorical | 3 | 3 | 0 |  | 0.0000 | 0.0000 | ordinary |

## Top Encoded Features

| encoded_feature | attribute | abs_target_corr | outlier_rate_iqr | logreg_importance | rf_importance | mutual_info | importance_consensus |
| --- | --- | --- | --- | --- | --- | --- | --- |
| src_bytes | src_bytes | 0.0059 | 0.1099 | 0.1349 | 0.1500 | 0.5625 | 0.6727 |
| dst_bytes | dst_bytes | 0.0041 | 0.1872 | 1.5638 | 0.1157 | 0.4359 | 0.5855 |
| flag_SF | flag | 0.7563 | 0.0000 | 1.8749 | 0.0544 | 0.3322 | 0.4019 |
| flag_S0 | flag | 0.6502 | 0.0000 | 4.2207 | 0.0231 | 0.2605 | 0.3950 |
| dst_host_srv_count | dst_host_srv_count | 0.7225 | 0.0000 | 1.8510 | 0.0426 | 0.3319 | 0.3744 |
| count | count | 0.5764 | 0.0251 | 2.0672 | 0.0427 | 0.2643 | 0.3443 |
| same_srv_rate | same_srv_rate | 0.7519 | 0.0000 | 0.8012 | 0.0408 | 0.3617 | 0.3409 |
| service_IRC | service | 0.0356 | 0.0000 | 7.4290 | 0.0002 | 0.0017 | 0.3348 |
| srv_serror_rate | srv_serror_rate | 0.6483 | 0.0000 | 2.7785 | 0.0240 | 0.2605 | 0.3324 |
| dst_host_same_srv_rate | dst_host_same_srv_rate | 0.6938 | 0.0000 | 0.8570 | 0.0438 | 0.3039 | 0.3159 |
| diff_srv_rate | diff_srv_rate | 0.2037 | 0.0633 | 0.1909 | 0.0297 | 0.3590 | 0.2873 |
| service_private | service | 0.4497 | 0.0000 | 3.9076 | 0.0132 | 0.1172 | 0.2742 |
| dst_host_serror_rate | dst_host_serror_rate | 0.6518 | 0.0000 | 0.0871 | 0.0441 | 0.2824 | 0.2693 |
| dst_host_srv_serror_rate | dst_host_srv_serror_rate | 0.6550 | 0.0000 | 0.9538 | 0.0268 | 0.2760 | 0.2660 |
| serror_rate | serror_rate | 0.6507 | 0.0000 | 0.8024 | 0.0302 | 0.2703 | 0.2631 |
| service_http | service | 0.5623 | 0.0000 | 1.3324 | 0.0330 | 0.1872 | 0.2441 |
| dst_host_diff_srv_rate | dst_host_diff_srv_rate | 0.2429 | 0.0837 | 0.2107 | 0.0273 | 0.2817 | 0.2370 |
| logged_in | logged_in | 0.6902 | 0.0000 | 0.6262 | 0.0177 | 0.2799 | 0.2332 |
| flag_REJ | flag | 0.1849 | 0.0000 | 4.6233 | 0.0020 | 0.0167 | 0.2218 |
| num_compromised | num_compromised | 0.0102 | 0.0000 | 4.3668 | 0.0073 | 0.0045 | 0.2149 |
| num_root | num_root | 0.0115 | 0.0000 | 4.4519 | 0.0002 | 0.0033 | 0.2022 |
| dst_host_same_src_port_rate | dst_host_same_src_port_rate | 0.0924 | 0.2007 | 0.9535 | 0.0325 | 0.1326 | 0.1935 |
| service_smtp | service | 0.2123 | 0.0000 | 3.4379 | 0.0005 | 0.0299 | 0.1731 |
| service_telnet | service | 0.0401 | 0.0000 | 3.8271 | 0.0004 | 0.0000 | 0.1727 |
| dst_host_srv_diff_host_rate | dst_host_srv_diff_host_rate | 0.0623 | 0.1169 | 0.3084 | 0.0194 | 0.1894 | 0.1692 |
| service_domain_u | service | 0.2589 | 0.0000 | 2.6278 | 0.0067 | 0.0491 | 0.1619 |
| dst_host_count | dst_host_count | 0.3751 | 0.0000 | 0.8518 | 0.0156 | 0.1390 | 0.1554 |
| service_pop_3 | service | 0.0156 | 0.0000 | 3.1611 | 0.0000 | 0.0007 | 0.1422 |
| service_urp_i | service | 0.0640 | 0.0000 | 3.0346 | 0.0014 | 0.0043 | 0.1417 |
| flag_S1 | flag | 0.0491 | 0.0000 | 3.0803 | 0.0003 | 0.0019 | 0.1400 |
| flag_RSTR | flag | 0.1331 | 0.0000 | 2.7627 | 0.0025 | 0.0088 | 0.1348 |
| service_ecr_i | service | 0.1500 | 0.0000 | 2.1937 | 0.0116 | 0.0134 | 0.1322 |
| service_eco_i | service | 0.1661 | 0.0000 | 1.8689 | 0.0134 | 0.0161 | 0.1233 |
| srv_rerror_rate | srv_rerror_rate | 0.2535 | 0.0000 | 1.8481 | 0.0056 | 0.0402 | 0.1192 |
| service_ftp_data | service | 0.0924 | 0.0000 | 2.1233 | 0.0061 | 0.0081 | 0.1136 |
| service_auth | service | 0.0504 | 0.0000 | 2.2789 | 0.0001 | 0.0033 | 0.1043 |
| service_other | service | 0.0238 | 0.0000 | 2.1793 | 0.0012 | 0.0000 | 0.1006 |
| service_domain | service | 0.0632 | 0.0000 | 2.1855 | 0.0000 | 0.0021 | 0.0993 |
| dst_host_rerror_rate | dst_host_rerror_rate | 0.2526 | 0.0000 | 0.7543 | 0.0186 | 0.0343 | 0.0954 |
| srv_count | srv_count | 0.0008 | 0.0973 | 0.5757 | 0.0137 | 0.0646 | 0.0945 |

## Referee Notes

- A high rank does not mean the feature is safe for deployment; it means the current benchmark uses it strongly.
- Categorical attributes are expanded into many one-hot columns, so their consensus score is aggregated back to the raw attribute.
- Outlier-heavy traffic features should not be deleted automatically; rare extreme flows can be the attack signal.
- Highly redundant attributes are prime candidates for compact-model ablations, not automatic removal.
