# Visual Comparison Dashboard

This dashboard exists because tables alone do not make the experiment work visible.

## Generated Figures

- `results/figures/nsl_metric_heatmap.png`: Core model metrics across the strongest NSL-KDD model rows.
- `results/figures/nsl_macro_attack_scatter.png`: Macro-F1 versus attack recall; point size shows benign false-positive burden.
- `results/figures/nsl_feature_group_ablation_bars.png`: Feature-group ablation showing which information families carry signal.
- `results/figures/nsl_bootstrap_confidence_intervals.png`: Bootstrap uncertainty for selected model/metric pairs.
- `results/figures/nsl_attribute_consensus_top20.png`: Top raw attributes by consensus signal across correlation, MI, LR, and RF signals.
- `results/figures/nsl_threshold_tradeoff.png`: How threshold policy moves normal recall against attack recall.
- `results/figures/nsl_concept_coverage_status.png`: Which first-dataset requested concepts are applied versus intentionally not applied.
- `results/figures/nsl_outlier_score_distribution.png`: Distribution of row-level outlier feature counts by normal versus attack.
- `results/figures/nsl_top_outlier_datapoints.png`: Actual split/row IDs with the most numeric outlier features.
- `results/figures/nsl_top_outlier_feature_heatmap.png`: Which features caused the top outlier rows to be flagged.
- `results/figures/nsl_outlier_feature_frequency.png`: Numeric features most often responsible for row-level outlier flags.

## Metric Coverage

- Metric summary rows: **25**
- NSL-KDD model-lab rows: **17**
- NSL-KDD feature-group rows: **8**
- Attribute-audit rows: **41**

## Referee Reading

- Do not use one metric alone. Macro-F1, attack recall, precision, and false-positive burden tell different stories.
- The scatter plot is usually more honest than a sorted leaderboard because it shows tradeoffs.
- Feature-group bars show what information families are doing, not merely which final model won.
- Bootstrap intervals show that tiny score differences should not be overclaimed.

## Top Method Rows

| model | macro_f1 | attack_recall | precision_attack | fp_per_10k_benign |
| --- | ---: | ---: | ---: | ---: |
| DecisionTree / balanced_depth18 | 0.8067 | 0.7212 | 0.9224 | 802.2 |
| AutoEncoder / normal_only_q0.95 | 0.8018 | 0.7103 | 0.9241 | 771.3 |
| HistGradientBoosting / default | 0.7948 | 0.6608 | 0.9697 | 272.9 |
| LogReg /<br>validation_F2_attack_weighted_0.15 | 0.7860 | 0.6938 | 0.9087 | 920.6 |
| OneClassSVM / normal_only_nu0.05 | 0.7830 | 0.6794 | 0.9181 | 801.2 |
