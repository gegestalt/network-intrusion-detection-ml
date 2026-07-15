# Feature Analysis and Representation Learning

This report covers correlation analysis, outlier mapping, feature selection, and learned representations. Every transform is fit on training data only.

## Dataset Caveats

- **nsl_kdd**: Official NSL-KDD split; old synthetic benchmark with strong train/test shift.
- **ciciot2023_dev**: Downsampled random dev split; not the full official raw CSV release.

## Representation Results

| Dataset | Representation | Features | Accuracy | Macro-F1 | MCC | Normal/benign recall | Attack recall | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| nsl_kdd | raw_all_features | 122 | 0.7543 | 0.7539 | 0.5591 | 0.9251 | 0.6251 |  |
| nsl_kdd | mutual_info_top_30 | 30 | 0.7467 | 0.7461 | 0.5464 | 0.9229 | 0.6134 |  |
| nsl_kdd | pca_30 | 30 | 0.7199 | 0.7182 | 0.5084 | 0.9272 | 0.5631 | explained variance 0.981 |
| nsl_kdd | l1_selected | 55 | 0.7513 | 0.7508 | 0.5543 | 0.9246 | 0.6202 |  |
| nsl_kdd | autoencoder_embedding_16 | 16 | 0.7580 | 0.7576 | 0.5666 | 0.9292 | 0.6285 |  |
| ciciot2023_dev | raw_all_features | 39 | 0.8212 | 0.7521 | 0.5916 | 0.9844 | 0.7926 |  |
| ciciot2023_dev | mutual_info_top_30 | 30 | 0.8205 | 0.7515 | 0.5908 | 0.9847 | 0.7917 |  |
| ciciot2023_dev | pca_30 | 30 | 0.8186 | 0.7495 | 0.5879 | 0.9841 | 0.7897 | explained variance 0.992 |
| ciciot2023_dev | l1_selected | 35 | 0.8209 | 0.7519 | 0.5913 | 0.9846 | 0.7923 |  |
| ciciot2023_dev | autoencoder_embedding_16 | 16 | 0.8350 | 0.7627 | 0.5948 | 0.9498 | 0.8149 |  |

## Top Feature-Target Correlations

| Dataset | Feature | Correlation | |corr| |
| --- | --- | ---: | ---: |
| nsl_kdd | flag_SF | -0.7563 | 0.7563 |
| nsl_kdd | same_srv_rate | -0.7519 | 0.7519 |
| nsl_kdd | dst_host_srv_count | -0.7225 | 0.7225 |
| nsl_kdd | dst_host_same_srv_rate | -0.6938 | 0.6938 |
| nsl_kdd | logged_in | -0.6902 | 0.6902 |
| nsl_kdd | dst_host_srv_serror_rate | 0.6550 | 0.6550 |
| nsl_kdd | dst_host_serror_rate | 0.6518 | 0.6518 |
| nsl_kdd | serror_rate | 0.6507 | 0.6507 |
| nsl_kdd | flag_S0 | 0.6502 | 0.6502 |
| nsl_kdd | srv_serror_rate | 0.6483 | 0.6483 |
| ciciot2023_dev | HTTPS | -0.5311 | 0.5311 |
| ciciot2023_dev | Number | 0.5079 | 0.5079 |
| ciciot2023_dev | ack_flag_number | -0.4446 | 0.4446 |
| ciciot2023_dev | Time_To_Live | -0.3504 | 0.3504 |
| ciciot2023_dev | Header_Length | -0.3419 | 0.3419 |
| ciciot2023_dev | Std | -0.2762 | 0.2762 |
| ciciot2023_dev | psh_flag_number | -0.2599 | 0.2599 |
| ciciot2023_dev | Max | -0.2393 | 0.2393 |
| ciciot2023_dev | Tot sum | 0.2281 | 0.2281 |
| ciciot2023_dev | Variance | -0.2074 | 0.2074 |

## Highest IQR Outlier Rates

| Dataset | Feature | Outlier rate |
| --- | --- | ---: |
| nsl_kdd | dst_host_same_src_port_rate | 0.2007 |
| nsl_kdd | dst_bytes | 0.1872 |
| nsl_kdd | dst_host_srv_diff_host_rate | 0.1169 |
| nsl_kdd | src_bytes | 0.1099 |
| nsl_kdd | srv_count | 0.0973 |
| nsl_kdd | dst_host_diff_srv_rate | 0.0837 |
| nsl_kdd | diff_srv_rate | 0.0633 |
| nsl_kdd | count | 0.0251 |
| nsl_kdd | flag_S1 | 0.0000 |
| nsl_kdd | flag_RSTO | 0.0000 |
| ciciot2023_dev | syn_count | 0.1911 |
| ciciot2023_dev | syn_flag_number | 0.1728 |
| ciciot2023_dev | Min | 0.1557 |
| ciciot2023_dev | IAT | 0.1522 |
| ciciot2023_dev | Time_To_Live | 0.1476 |
| ciciot2023_dev | UDP | 0.1404 |
| ciciot2023_dev | psh_flag_number | 0.1329 |
| ciciot2023_dev | HTTPS | 0.1279 |
| ciciot2023_dev | ack_count | 0.1241 |
| ciciot2023_dev | Tot sum | 0.1123 |

## High-Correlation Feature Pairs

| Dataset | Feature A | Feature B | Correlation |
| --- | --- | --- | ---: |
| nsl_kdd | num_compromised | num_root | 0.9986 |
| nsl_kdd | serror_rate | srv_serror_rate | 0.9933 |
| nsl_kdd | rerror_rate | srv_rerror_rate | 0.9892 |
| nsl_kdd | srv_serror_rate | dst_host_srv_serror_rate | 0.9864 |
| nsl_kdd | dst_host_serror_rate | dst_host_srv_serror_rate | 0.9855 |
| nsl_kdd | flag_S0 | srv_serror_rate | 0.9836 |
| nsl_kdd | serror_rate | dst_host_srv_serror_rate | 0.9817 |
| nsl_kdd | flag_S0 | dst_host_srv_serror_rate | 0.9810 |
| nsl_kdd | serror_rate | dst_host_serror_rate | 0.9805 |
| nsl_kdd | flag_S0 | serror_rate | 0.9803 |
| nsl_kdd | srv_serror_rate | dst_host_serror_rate | 0.9789 |
| nsl_kdd | flag_S0 | dst_host_serror_rate | 0.9765 |
| nsl_kdd | srv_rerror_rate | dst_host_srv_rerror_rate | 0.9705 |
| nsl_kdd | rerror_rate | dst_host_srv_rerror_rate | 0.9643 |
| ciciot2023_dev | IPv | LLC | 1.0000 |
| ciciot2023_dev | AVG | Tot size | 1.0000 |
| ciciot2023_dev | ARP | IPv | -1.0000 |
| ciciot2023_dev | ARP | LLC | -1.0000 |

## Interpretation Guardrails

- Correlation is not causation; it is a triage tool for feature inspection.
- Outlier flags are not automatically bad rows; in security data, rare values can be the signal.
- PCA and autoencoder embeddings are feature-learning baselines, not proof that deep learning is better.
- A representation only matters if it beats or clarifies a strong simple baseline.
