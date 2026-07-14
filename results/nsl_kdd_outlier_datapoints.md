# NSL-KDD Row-Level Outlier Datapoints

IQR fences are fitted on KDDTrain+ numeric features only, then applied to train and KDDTest+.
This identifies actual rows whose numeric values sit outside the training distribution.

## Outputs

- Row summary: `results/nsl_kdd_outlier_datapoints.csv`
- Feature-frequency table: `results/nsl_kdd_outlier_feature_frequency.csv`
- Long row-feature matrix: `results/nsl_kdd_top_outlier_feature_matrix.csv`
- Distribution plot: `results/figures/nsl_outlier_score_distribution.png`
- Top-row plot: `results/figures/nsl_top_outlier_datapoints.png`
- Row-feature heatmap: `results/figures/nsl_top_outlier_feature_heatmap.png`
- Feature-frequency plot: `results/figures/nsl_outlier_feature_frequency.png`

## Top Outlier Datapoints

| split | row_index | label | family | outlier_feature_count | max_abs_robust_z | top_outlier_features |
| --- | ---: | --- | --- | ---: | ---: | --- |
| train | 23954 | normal | normal | 12 | 20252.00 | duration, num_root, num_compromised, dst_bytes, src_bytes, num_access_files, su_attempted, srv_rerror_rate |
| train | 17927 | normal | normal | 12 | 16754.00 | duration, dst_bytes, num_compromised, num_root, dst_host_srv_diff_host_rate, src_bytes, dst_host_diff_srv_rate, su_attempted |
| train | 47186 | normal | normal | 12 | 13368.00 | duration, dst_bytes, num_file_creations, src_bytes, dst_host_srv_diff_host_rate, dst_host_diff_srv_rate, num_root, num_compromised |
| test | 11765 | multihop | R2L | 12 | 1776.00 | duration, dst_bytes, dst_host_srv_diff_host_rate, num_root, hot, num_compromised, src_bytes, dst_host_diff_srv_rate |
| test | 2421 | xterm | U2R | 12 | 293.00 | duration, dst_bytes, num_root, num_compromised, src_bytes, dst_host_diff_srv_rate, dst_host_same_src_port_rate, num_access_files |
| test | 20555 | buffer_overflow | U2R | 12 | 192.00 | duration, num_compromised, num_root, dst_bytes, src_bytes, dst_host_srv_diff_host_rate, hot, num_access_files |
| test | 3621 | xterm | U2R | 12 | 184.00 | duration, dst_host_srv_diff_host_rate, num_root, dst_host_same_src_port_rate, num_compromised, src_bytes, dst_bytes, hot |
| train | 63940 | normal | normal | 11 | 15183.00 | duration, num_root, num_compromised, dst_bytes, dst_host_diff_srv_rate, num_access_files, src_bytes, su_attempted |
| train | 31166 | normal | normal | 11 | 14943.00 | duration, num_root, num_compromised, dst_bytes, dst_host_srv_diff_host_rate, dst_host_diff_srv_rate, dst_host_same_src_port_rate, num_access_files |
| train | 108454 | normal | normal | 11 | 14538.00 | duration, dst_bytes, num_compromised, num_root, src_bytes, num_file_creations, su_attempted, root_shell |
| train | 73694 | normal | normal | 11 | 5930.00 | duration, dst_bytes, num_root, su_attempted, num_compromised, srv_rerror_rate, rerror_rate, root_shell |
| train | 53128 | normal | normal | 11 | 1299.40 | src_bytes, duration, diff_srv_rate, dst_host_diff_srv_rate, dst_host_same_src_port_rate, num_root, dst_host_srv_rerror_rate, srv_diff_host_rate |
| test | 20734 | rootkit | U2R | 11 | 988.00 | duration, dst_bytes, num_compromised, num_root, num_file_creations, src_bytes, num_shells, hot |
| test | 10995 | rootkit | U2R | 11 | 804.00 | duration, num_root, dst_bytes, num_compromised, hot, src_bytes, urgent, num_file_creations |
| train | 3005 | multihop | R2L | 11 | 718.00 | duration, num_root, dst_bytes, num_compromised, dst_host_same_src_port_rate, hot, src_bytes, num_file_creations |
| test | 21192 | buffer_overflow | U2R | 11 | 684.00 | duration, dst_bytes, dst_host_diff_srv_rate, src_bytes, num_compromised, num_file_creations, num_root, num_failed_logins |
| test | 9082 | xterm | U2R | 11 | 70.66 | dst_bytes, duration, num_compromised, src_bytes, num_root, urgent, num_access_files, root_shell |
| train | 5930 | normal | normal | 10 | 16800.00 | duration, dst_bytes, num_root, num_compromised, src_bytes, num_file_creations, su_attempted, num_failed_logins |
| train | 103773 | normal | normal | 10 | 15377.00 | duration, num_root, num_compromised, dst_bytes, num_access_files, src_bytes, su_attempted, root_shell |
| train | 111751 | normal | normal | 10 | 15211.00 | duration, dst_bytes, num_root, num_compromised, dst_host_srv_diff_host_rate, dst_host_diff_srv_rate, dst_host_same_src_port_rate, su_attempted |
| train | 65609 | normal | normal | 10 | 14857.00 | duration, num_root, num_compromised, dst_bytes, num_access_files, src_bytes, su_attempted, root_shell |
| train | 75655 | normal | normal | 10 | 14743.00 | duration, num_root, num_compromised, dst_bytes, src_bytes, su_attempted, num_access_files, root_shell |
| train | 19953 | normal | normal | 10 | 13417.00 | duration, num_root, num_compromised, dst_bytes, src_bytes, num_access_files, su_attempted, root_shell |
| train | 25095 | ipsweep | Probe | 10 | 12743.00 | duration, dst_bytes, dst_host_srv_diff_host_rate, num_file_creations, src_bytes, dst_host_diff_srv_rate, num_root, num_compromised |
| train | 4677 | normal | normal | 10 | 12039.00 | duration, dst_bytes, num_root, num_compromised, num_access_files, src_bytes, su_attempted, root_shell |

## Most Frequent Outlier Features

| split | feature | outlier_rows | mean_abs_robust_z |
| --- | --- | ---: | ---: |
| test | dst_host_rerror_rate | 9175 | 0.57 |
| test | dst_host_srv_rerror_rate | 7251 | 0.70 |
| test | rerror_rate | 5763 | 0.93 |
| test | srv_rerror_rate | 5648 | 0.94 |
| test | srv_diff_host_rate | 4975 | 0.44 |
| test | dst_bytes | 4327 | 19.55 |
| test | dst_host_same_src_port_rate | 3610 | 12.92 |
| test | duration | 3526 | 1399.31 |
| test | src_bytes | 2737 | 307.15 |
| test | srv_count | 2271 | 14.42 |
| test | diff_srv_rate | 2062 | 14.66 |
| test | dst_host_diff_srv_rate | 2059 | 10.14 |
| train | srv_diff_host_rate | 28399 | 0.43 |
| train | dst_host_same_src_port_rate | 25052 | 11.60 |
| train | dst_bytes | 23579 | 203.73 |
| train | dst_host_rerror_rate | 22795 | 0.66 |
| train | dst_host_srv_rerror_rate | 19357 | 0.78 |
| train | srv_rerror_rate | 16206 | 0.94 |
| train | rerror_rate | 16190 | 0.93 |
| train | src_bytes | 13840 | 1499.43 |
| train | srv_count | 12054 | 12.32 |
| train | dst_host_srv_diff_host_rate | 11682 | 14.53 |
| train | dst_host_diff_srv_rate | 10550 | 8.88 |
| train | duration | 10018 | 3610.75 |

## Interpretation Guardrails

- Outlier datapoints are not automatically bad data. In intrusion detection, rare/extreme rows can be the actual attack signal.
- The key question is whether outlier rows cluster in certain labels, families, or features.
- Since fences are fitted on train only, test rows are judged against the learned training distribution rather than their own distribution.
